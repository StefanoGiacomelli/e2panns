"""
CLAP (Contrastive Language-Audio Pretraining)
==============================================
Large-scale Contrastive Language-Audio Pretraining with Feature Fusion and Keyword-to-Caption Augmentation.

Original repository: https://github.com/LAION-AI/CLAP

This is a standalone implementation of CLAP for AudioSet tagging and zero-shot classification.
"""

import math
import warnings
from typing import List, Tuple
import collections.abc
from itertools import repeat

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch.utils.checkpoint as checkpoint

from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation

from transformers import RobertaModel, RobertaTokenizer


# ============================================
# CONSTANTS
# ============================================
SAMPLE_RATE = 48000
CLASSES_NUM = 527
EMBED_DIM = 768  # Audio embedding dimension (pre-projection)
JOINT_EMBED_DIM = 512  # Joint audio-text embedding dimension (post-projection)

# Audio config (HTSAT-tiny with fusion)
AUDIO_LENGTH = 1024
CLIP_SAMPLES = 480000  # 10 seconds at 48kHz
MEL_BINS = 64
WINDOW_SIZE = 1024
HOP_SIZE = 480
FMIN = 50
FMAX = 14000

# HTSAT architecture
SPEC_SIZE = 256
PATCH_SIZE = 4
PATCH_STRIDE = (4, 4)
HTSAT_EMBED_DIM = 96
HTSAT_DEPTHS = [2, 2, 6, 2]
HTSAT_NUM_HEADS = [4, 8, 16, 32]
HTSAT_WINDOW_SIZE = 8

# Text config
TEXT_CONTEXT_LENGTH = 77
TEXT_MODEL = "roberta-base"


# ============================================
# Helper Functions
# ============================================
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse

to_2tuple = _ntuple(2)


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Stochastic Depth per sample."""
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


def do_mixup(x, mixup_lambda):
    """Mixup augmentation."""
    out = x * mixup_lambda.reshape(x.shape[0], 1, 1, 1) + \
          torch.flip(x, dims=[0]) * (1 - mixup_lambda.reshape(x.shape[0], 1, 1, 1))
    return out


def interpolate(x, ratio):
    """Interpolate feature maps."""
    (batch_size, time_steps, classes_num) = x.shape
    upsampled = x[:, :, None, :].repeat(1, 1, ratio, 1)
    upsampled = upsampled.reshape(batch_size, time_steps * ratio, classes_num)
    return upsampled


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    """Truncated normal initialization."""
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in trunc_normal_", stacklevel=2)

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


# ============================================
# FEATURE FUSION MODULES
# ============================================
class DAF(nn.Module):
    """Direct Add Fusion."""
    def __init__(self):
        super().__init__()

    def forward(self, x, residual):
        return x + residual


class AFF(nn.Module):
    """Attentional Feature Fusion."""
    def __init__(self, channels=64, r=4, type='2D'):
        super().__init__()
        inter_channels = int(channels // r)

        if type == '1D':
            self.local_att = nn.Sequential(
                nn.Conv1d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm1d(inter_channels),
                nn.ReLU(inplace=True),
                nn.Conv1d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm1d(channels),
            )
            self.global_att = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Conv1d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm1d(inter_channels),
                nn.ReLU(inplace=True),
                nn.Conv1d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm1d(channels),
            )
        elif type == '2D':
            self.local_att = nn.Sequential(
                nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(inter_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(channels),
            )
            self.global_att = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(inter_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(channels),
            )
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        flag = False
        xa = x + residual
        if xa.size(0) == 1:
            xa = torch.cat([xa, xa], dim=0)
            flag = True
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        xo = 2 * x * wei + 2 * residual * (1 - wei)
        if flag:
            xo = xo[0].unsqueeze(0)
        return xo


class iAFF(nn.Module):
    """Iterative Attentional Feature Fusion."""
    def __init__(self, channels=64, r=4, type='2D'):
        super().__init__()
        inter_channels = int(channels // r)

        if type == '1D':
            self.local_att = nn.Sequential(
                nn.Conv1d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm1d(inter_channels),
                nn.ReLU(inplace=True),
                nn.Conv1d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm1d(channels),
            )
            self.global_att = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Conv1d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm1d(inter_channels),
                nn.ReLU(inplace=True),
                nn.Conv1d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm1d(channels),
            )
            self.local_att2 = nn.Sequential(
                nn.Conv1d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm1d(inter_channels),
                nn.ReLU(inplace=True),
                nn.Conv1d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm1d(channels),
            )
            self.global_att2 = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Conv1d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm1d(inter_channels),
                nn.ReLU(inplace=True),
                nn.Conv1d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm1d(channels),
            )
        elif type == '2D':
            self.local_att = nn.Sequential(
                nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(inter_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(channels),
            )
            self.global_att = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(inter_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(channels),
            )
            self.local_att2 = nn.Sequential(
                nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(inter_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(channels),
            )
            self.global_att2 = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(inter_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(channels),
            )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        flag = False
        xa = x + residual
        if xa.size(0) == 1:
            xa = torch.cat([xa, xa], dim=0)
            flag = True
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        xi = x * wei + residual * (1 - wei)

        xl2 = self.local_att2(xi)
        xg2 = self.global_att(xi)
        xlg2 = xl2 + xg2
        wei2 = self.sigmoid(xlg2)
        xo = x * wei2 + residual * (1 - wei2)
        if flag:
            xo = xo[0].unsqueeze(0)
        return xo


# ============================================
# DROP PATH
# ============================================
class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


# ============================================
# MLP
# ============================================
class Mlp(nn.Module):
    """MLP as used in Vision Transformer."""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# ============================================
# MLP LAYERS (for projection)
# ============================================
class MLPLayers(nn.Module):
    """MLP layers for audio/text projection."""
    def __init__(self, units=[512, 512, 512], dropout=0.1):
        super().__init__()
        layers = []
        for i in range(len(units) - 1):
            layers.append(nn.Linear(units[i], units[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


# ============================================
# PATCH EMBEDDING
# ============================================
class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding with optional fusion support."""
    def __init__(self, img_size=224, patch_size=16, in_chans=1, embed_dim=768, 
                 norm_layer=None, flatten=True, patch_stride=16,
                 enable_fusion=False, fusion_type='None'):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patch_stride = to_2tuple(patch_stride)
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.grid_size = (img_size[0] // patch_stride[0], img_size[1] // patch_stride[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.enable_fusion = enable_fusion
        self.fusion_type = fusion_type
        
        padding = ((patch_size[0] - patch_stride[0]) // 2, (patch_size[1] - patch_stride[1]) // 2)

        if enable_fusion and fusion_type == 'channel_map':
            self.proj = nn.Conv2d(in_chans * 4, embed_dim, kernel_size=patch_size, 
                                  stride=patch_stride, padding=padding)
        else:
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, 
                                  stride=patch_stride, padding=padding)
        
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

        if enable_fusion and fusion_type in ['daf_2d', 'aff_2d', 'iaff_2d']:
            self.mel_conv2d = nn.Conv2d(in_chans, embed_dim, 
                                        kernel_size=(patch_size[0], patch_size[1] * 3),
                                        stride=(patch_stride[0], patch_stride[1] * 3), 
                                        padding=padding)
            if fusion_type == 'daf_2d':
                self.fusion_model = DAF()
            elif fusion_type == 'aff_2d':
                self.fusion_model = AFF(channels=embed_dim, type='2D')
            elif fusion_type == 'iaff_2d':
                self.fusion_model = iAFF(channels=embed_dim, type='2D')

    def forward(self, x, longer_idx=None):
        if self.enable_fusion and self.fusion_type in ['daf_2d', 'aff_2d', 'iaff_2d']:
            global_x = x[:, 0:1, :, :]
            
            B, C, H, W = global_x.shape
            assert H == self.img_size[0] and W == self.img_size[1], \
                f"Input size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
            global_x = self.proj(global_x)
            TW = global_x.size(-1)
            
            if longer_idx is not None and len(longer_idx) > 0:
                local_x = x[longer_idx, 1:, :, :].contiguous()
                B, C, H, W = local_x.shape
                local_x = local_x.view(B * C, 1, H, W)
                local_x = self.mel_conv2d(local_x)
                local_x = local_x.view(B, C, local_x.size(1), local_x.size(2), local_x.size(3))
                local_x = local_x.permute((0, 2, 3, 1, 4)).contiguous().flatten(3)
                TB, TC, TH, _ = local_x.size()
                if local_x.size(-1) < TW:
                    local_x = torch.cat([local_x, torch.zeros((TB, TC, TH, TW - local_x.size(-1)), 
                                        device=global_x.device)], dim=-1)
                else:
                    local_x = local_x[:, :, :, :TW]
                
                global_x[longer_idx] = self.fusion_model(global_x[longer_idx], local_x)
            x = global_x
        else:
            B, C, H, W = x.shape
            assert H == self.img_size[0] and W == self.img_size[1], \
                f"Input size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
            x = self.proj(x)
        
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


# ============================================
# WINDOW FUNCTIONS
# ============================================
def window_partition(x, window_size):
    """Partition into windows."""
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """Reverse window partition."""
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


# ============================================
# WINDOW ATTENTION
# ============================================
class WindowAttention(nn.Module):
    """Window based multi-head self attention with relative position bias."""
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, 
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # Relative position bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        # Get pair-wise relative position index
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], 
            self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


# ============================================
# SWIN TRANSFORMER BLOCK
# ============================================
class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block."""
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., 
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, 
                 norm_before_mlp='ln'):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.norm_before_mlp = norm_before_mlp
        
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        if self.norm_before_mlp == 'ln':
            self.norm2 = nn.LayerNorm(dim)
        elif self.norm_before_mlp == 'bn':
            self.norm2 = lambda x: nn.BatchNorm1d(dim)(x.transpose(1, 2)).transpose(1, 2)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # Partition windows
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # W-MSA/SW-MSA
        attn_windows, attn = self.attn(x_windows, mask=self.attn_mask)

        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x, attn


# ============================================
# PATCH MERGING
# ============================================
class PatchMerging(nn.Module):
    """Patch Merging Layer."""
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W
        assert H % 2 == 0 and W % 2 == 0

        x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)

        x = self.norm(x)
        x = self.reduction(x)
        return x


# ============================================
# BASIC LAYER
# ============================================
class BasicLayer(nn.Module):
    """A basic Swin Transformer layer for one stage."""
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, 
                 use_checkpoint=False, norm_before_mlp='ln'):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim, input_resolution=input_resolution,
                num_heads=num_heads, window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer, norm_before_mlp=norm_before_mlp)
            for i in range(depth)])

        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        attns = []
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x, attn = blk(x)
                if not self.training:
                    attns.append(attn.unsqueeze(0))
        if self.downsample is not None:
            x = self.downsample(x)
        if not self.training and attns:
            attn = torch.cat(attns, dim=0)
            attn = torch.mean(attn, dim=0)
        else:
            attn = None
        return x, attn


# ============================================
# HTSAT SWIN TRANSFORMER (Audio Encoder)
# ============================================
class HTSAT_Swin_Transformer(nn.Module):
    """Hierarchical Token-Semantic Audio Transformer based on Swin Transformer."""
    
    def __init__(self, spec_size=256, patch_size=4, patch_stride=(4, 4),
                 in_chans=1, num_classes=527, embed_dim=96, 
                 depths=[2, 2, 6, 2], num_heads=[4, 8, 16, 32],
                 window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, norm_before_mlp='ln',
                 audio_cfg=None, enable_fusion=False, fusion_type='None'):
        super().__init__()

        self.audio_cfg = audio_cfg
        self.spec_size = spec_size
        self.patch_stride = patch_stride
        self.patch_size = patch_size
        self.window_size = window_size
        self.embed_dim = embed_dim
        self.depths = depths
        self.ape = ape
        self.in_chans = in_chans
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.num_layers = len(depths)
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))

        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.patch_norm = patch_norm
        self.norm_layer = norm_layer if patch_norm else None
        self.norm_before_mlp = norm_before_mlp
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint
        self.enable_fusion = enable_fusion
        self.fusion_type = fusion_type

        # Mel spectrogram config
        self.freq_ratio = spec_size // audio_cfg['mel_bins']
        self.interpolate_ratio = 32

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(
            n_fft=audio_cfg['window_size'], 
            hop_length=audio_cfg['hop_size'],
            win_length=audio_cfg['window_size'], 
            window='hann', 
            center=True, 
            pad_mode='reflect',
            freeze_parameters=True
        )
        
        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(
            sr=audio_cfg['sample_rate'], 
            n_fft=audio_cfg['window_size'],
            n_mels=audio_cfg['mel_bins'], 
            fmin=audio_cfg['fmin'], 
            fmax=audio_cfg['fmax'],
            ref=1.0, 
            amin=1e-10, 
            top_db=None,
            freeze_parameters=True
        )
        
        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(
            time_drop_width=64, time_stripes_num=2,
            freq_drop_width=8, freq_stripes_num=2
        )
        
        self.bn0 = nn.BatchNorm2d(audio_cfg['mel_bins'])

        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=spec_size, patch_size=patch_size, in_chans=in_chans,
            embed_dim=embed_dim, norm_layer=self.norm_layer, patch_stride=patch_stride,
            enable_fusion=enable_fusion, fusion_type=fusion_type
        )

        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.grid_size
        self.patches_resolution = patches_resolution

        # Absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # Build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                  patches_resolution[1] // (2 ** i_layer)),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
                norm_before_mlp=norm_before_mlp
            )
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.maxpool = nn.AdaptiveMaxPool1d(1)

        SF = spec_size // (2 ** (len(depths) - 1)) // patch_stride[0] // self.freq_ratio
        self.tscam_conv = nn.Conv2d(
            in_channels=self.num_features,
            out_channels=num_classes,
            kernel_size=(SF, 3),
            padding=(0, 1)
        )
        self.head = nn.Linear(num_classes, num_classes)

        # 1D fusion support
        if enable_fusion and fusion_type in ['daf_1d', 'aff_1d', 'iaff_1d']:
            self.mel_conv1d = nn.Sequential(
                nn.Conv1d(64, 64, kernel_size=5, stride=3, padding=2),
                nn.BatchNorm1d(64)
            )
            if fusion_type == 'daf_1d':
                self.fusion_model = DAF()
            elif fusion_type == 'aff_1d':
                self.fusion_model = AFF(channels=64, type='1D')
            elif fusion_type == 'iaff_1d':
                self.fusion_model = iAFF(channels=64, type='1D')

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def reshape_wav2img(self, x):
        """Reshape spectrogram to image size for Swin input."""
        B, C, T, F = x.shape
        target_T = int(self.spec_size * self.freq_ratio)
        target_F = self.spec_size // self.freq_ratio
        
        assert T <= target_T and F <= target_F
        
        if T < target_T:
            x = nn.functional.interpolate(x, (target_T, x.shape[3]), mode="bicubic", align_corners=True)
        if F < target_F:
            x = nn.functional.interpolate(x, (x.shape[2], target_F), mode="bicubic", align_corners=True)
        
        x = x.permute(0, 1, 3, 2).contiguous()
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2], self.freq_ratio, x.shape[3] // self.freq_ratio)
        x = x.permute(0, 1, 3, 2, 4).contiguous()
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3], x.shape[4])
        return x

    def forward_features(self, x, longer_idx=None):
        """Forward through feature extraction layers."""
        frames_num = x.shape[2]
        x = self.patch_embed(x, longer_idx=longer_idx)
        
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        
        for layer in self.layers:
            x, attn = layer(x)
        
        x = self.norm(x)
        B, N, C = x.shape
        SF = frames_num // (2 ** (len(self.depths) - 1)) // self.patch_stride[0]
        ST = frames_num // (2 ** (len(self.depths) - 1)) // self.patch_stride[1]
        x = x.permute(0, 2, 1).contiguous().reshape(B, C, SF, ST)
        
        B, C, F, T = x.shape
        c_freq_bin = F // self.freq_ratio
        x = x.reshape(B, C, F // c_freq_bin, c_freq_bin, T)
        x = x.permute(0, 1, 3, 2, 4).contiguous().reshape(B, C, c_freq_bin, -1)
        
        # Get latent output (embedding)
        fine_grained_latent_output = torch.mean(x, dim=2)
        fine_grained_latent_output = interpolate(
            fine_grained_latent_output.permute(0, 2, 1).contiguous(), 
            8 * self.patch_stride[1]
        )
        
        latent_output = self.avgpool(torch.flatten(x, 2))
        latent_output = torch.flatten(latent_output, 1)

        # Classification head
        x = self.tscam_conv(x)
        x = torch.flatten(x, 2)
        
        fpx = interpolate(torch.sigmoid(x).permute(0, 2, 1).contiguous(), 8 * self.patch_stride[1])
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        output_dict = {
            'framewise_output': fpx,
            'clipwise_output': torch.sigmoid(x),
            'fine_grained_embedding': fine_grained_latent_output,
            'embedding': latent_output
        }
        return output_dict

    def forward(self, x: torch.Tensor, mixup_lambda=None, device=None):
        """Forward pass."""
        if self.enable_fusion and x["longer"].sum() == 0:
            if self.training:
                x["longer"][torch.randint(0, x["longer"].shape[0], (1,))] = True
            else:
                x = x["mel_fusion"].to(device=device, non_blocking=True)
                x = x.transpose(1, 3)
                x = self.bn0(x)
                x = x.transpose(1, 3)
                x = self.reshape_wav2img(x)
                output_dict = self.forward_features(x, longer_idx=[])
                return output_dict

        if not self.enable_fusion:
            x = x["waveform"].to(device=device, non_blocking=True)
            x = self.spectrogram_extractor(x)
            x = self.logmel_extractor(x)
            x = x.transpose(1, 3)
            x = self.bn0(x)
            x = x.transpose(1, 3)
            if self.training:
                x = self.spec_augmenter(x)
            if self.training and mixup_lambda is not None:
                x = do_mixup(x, mixup_lambda)
            x = self.reshape_wav2img(x)
            output_dict = self.forward_features(x)
        else:
            longer_list = x["longer"].to(device=device, non_blocking=True)
            x = x["mel_fusion"].to(device=device, non_blocking=True)
            x = x.transpose(1, 3)
            x = self.bn0(x)
            x = x.transpose(1, 3)
            longer_list_idx = torch.where(longer_list)[0]
            
            if self.fusion_type in ['daf_1d', 'aff_1d', 'iaff_1d']:
                new_x = x[:, 0:1, :, :].clone().contiguous()
                if len(longer_list_idx) > 0:
                    fusion_x_local = x[longer_list_idx, 1:, :, :].clone().contiguous()
                    FB, FC, FT, FF = fusion_x_local.size()
                    fusion_x_local = fusion_x_local.view(FB * FC, FT, FF)
                    fusion_x_local = torch.permute(fusion_x_local, (0, 2, 1)).contiguous()
                    fusion_x_local = self.mel_conv1d(fusion_x_local)
                    fusion_x_local = fusion_x_local.view(FB, FC, FF, fusion_x_local.size(-1))
                    fusion_x_local = torch.permute(fusion_x_local, (0, 2, 1, 3)).contiguous().flatten(2)
                    if fusion_x_local.size(-1) < FT:
                        fusion_x_local = torch.cat([fusion_x_local, 
                            torch.zeros((FB, FF, FT - fusion_x_local.size(-1)), device=device)], dim=-1)
                    else:
                        fusion_x_local = fusion_x_local[:, :, :FT]
                    new_x = new_x.squeeze(1).permute((0, 2, 1)).contiguous()
                    new_x[longer_list_idx] = self.fusion_model(new_x[longer_list_idx], fusion_x_local)
                    x = new_x.permute((0, 2, 1)).contiguous()[:, None, :, :]
                else:
                    x = new_x
            elif self.fusion_type in ['daf_2d', 'aff_2d', 'iaff_2d', 'channel_map']:
                x = x

            if self.training:
                x = self.spec_augmenter(x)
            if self.training and mixup_lambda is not None:
                x = do_mixup(x, mixup_lambda)

            x = self.reshape_wav2img(x)
            output_dict = self.forward_features(x, longer_idx=longer_list_idx)

        return output_dict


def create_htsat_model(audio_cfg, enable_fusion=False, fusion_type='None'):
    """Create HTSAT model based on config."""
    model_name = audio_cfg.get('model_name', 'tiny')
    
    if model_name == "tiny":
        return HTSAT_Swin_Transformer(
            spec_size=256, patch_size=4, patch_stride=(4, 4),
            num_classes=audio_cfg['class_num'],
            embed_dim=96, depths=[2, 2, 6, 2], num_heads=[4, 8, 16, 32],
            window_size=8, audio_cfg=audio_cfg,
            enable_fusion=enable_fusion, fusion_type=fusion_type
        )
    elif model_name == "base":
        return HTSAT_Swin_Transformer(
            spec_size=256, patch_size=4, patch_stride=(4, 4),
            num_classes=audio_cfg['class_num'],
            embed_dim=128, depths=[2, 2, 12, 2], num_heads=[4, 8, 16, 32],
            window_size=8, audio_cfg=audio_cfg,
            enable_fusion=enable_fusion, fusion_type=fusion_type
        )
    elif model_name == "large":
        return HTSAT_Swin_Transformer(
            spec_size=256, patch_size=4, patch_stride=(4, 4),
            num_classes=audio_cfg['class_num'],
            embed_dim=256, depths=[2, 2, 12, 2], num_heads=[4, 8, 16, 32],
            window_size=8, audio_cfg=audio_cfg,
            enable_fusion=enable_fusion, fusion_type=fusion_type
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")


# ============================================
# CLAP MODEL (Full Audio-Text)
# ============================================
class CLAPModel(nn.Module):
    """
    CLAP (Contrastive Language-Audio Pretraining) model.
    
    Combines HTSAT audio encoder with RoBERTa text encoder for
    audio-text contrastive learning and zero-shot classification.
    """
    
    def __init__(self, embed_dim=768, audio_cfg=None, text_cfg=None,
                 enable_fusion=False, fusion_type='None', 
                 joint_embed_shape=512, mlp_act='relu'):
        super().__init__()
        
        self.audio_cfg = audio_cfg
        self.text_cfg = text_cfg
        self.enable_fusion = enable_fusion
        self.fusion_type = fusion_type
        self.joint_embed_shape = joint_embed_shape

        # Audio branch (HTSAT)
        self.audio_branch = create_htsat_model(audio_cfg, enable_fusion, fusion_type)
        
        # Text branch (RoBERTa)
        self.text_branch = RobertaModel.from_pretrained('roberta-base')
        self.text_branch_type = "roberta"
        
        # MLP activation
        if mlp_act == 'relu':
            mlp_act_layer = nn.ReLU()
        elif mlp_act == 'gelu':
            mlp_act_layer = nn.GELU()
        else:
            mlp_act_layer = nn.ReLU()

        # Audio projection (768 -> 512)
        self.audio_projection = nn.Sequential(
            nn.Linear(embed_dim, joint_embed_shape),
            mlp_act_layer,
            nn.Linear(joint_embed_shape, joint_embed_shape)
        )
        
        # Text projection (768 -> 512)
        self.text_projection = nn.Sequential(
            nn.Linear(768, joint_embed_shape),
            mlp_act_layer,
            nn.Linear(joint_embed_shape, joint_embed_shape)
        )
        
        # Transform layers
        self.audio_transform = MLPLayers(
            units=[joint_embed_shape, joint_embed_shape, joint_embed_shape], 
            dropout=0.1
        )
        self.text_transform = MLPLayers(
            units=[joint_embed_shape, joint_embed_shape, joint_embed_shape], 
            dropout=0.1
        )

        # Logit scales
        self.logit_scale_a = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_scale_t = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def encode_audio(self, audio, device=None):
        """Encode audio to embedding."""
        return self.audio_branch(audio, mixup_lambda=None, device=device)

    def encode_text(self, text, device=None):
        """Encode text to embedding."""
        x = self.text_branch(
            input_ids=text["input_ids"].to(device=device, non_blocking=True),
            attention_mask=text["attention_mask"].to(device=device, non_blocking=True),
        )["pooler_output"]
        x = self.text_projection(x)
        return x

    def forward(self, audio, text=None, device=None):
        """Forward pass for training."""
        if device is None:
            device = next(self.parameters()).device
            
        if audio is None and text is None:
            return self.logit_scale_a.exp(), self.logit_scale_t.exp()
        elif audio is None:
            return self.encode_text(text, device=device)
        elif text is None:
            return self.audio_projection(self.encode_audio(audio, device=device)["embedding"])
        
        # Both audio and text
        audio_features = self.audio_projection(self.encode_audio(audio, device=device)["embedding"])
        audio_features = F.normalize(audio_features, dim=-1)
        
        text_features = self.encode_text(text, device=device)
        text_features = F.normalize(text_features, dim=-1)
        
        audio_features_mlp = self.audio_transform(audio_features)
        text_features_mlp = self.text_transform(text_features)
        
        return (
            audio_features,
            text_features,
            audio_features_mlp,
            text_features_mlp,
            self.logit_scale_a.exp(),
            self.logit_scale_t.exp(),
        )

    def get_audio_embedding(self, data):
        """Get normalized audio embedding (512 dim, projected)."""
        device = next(self.parameters()).device
        input_dict = {}
        keys = data[0].keys()
        for k in keys:
            input_dict[k] = torch.cat([d[k].unsqueeze(0) for d in data], dim=0).to(device)
        audio_embeds = self.encode_audio(input_dict, device=device)["embedding"]
        audio_embeds = self.audio_projection(audio_embeds)
        audio_embeds = F.normalize(audio_embeds, dim=-1)
        return audio_embeds

    def get_text_embedding(self, data):
        """Get normalized text embedding (512 dim, projected)."""
        device = next(self.parameters()).device
        for k in data:
            data[k] = data[k].to(device)
        text_embeds = self.encode_text(data, device=device)
        text_embeds = F.normalize(text_embeds, dim=-1)
        return text_embeds


# ============================================
# CLAP WRAPPER (Standard API)
# ============================================
class CLAP(nn.Module):
    """
    CLAP wrapper with standard API for audio classification and embedding extraction.
    
    Standard API (compatible with other audio models):
        - forward(waveform) -> probs (527 AudioSet classes)
        - forward_with_embedding(waveform) -> (probs, embedding)
        - get_embedding(waveform) -> embedding (768 dim, pre-projection)
    
    CLAP-specific API (zero-shot capabilities):
        - get_audio_embedding(waveform) -> embedding (512 dim, projected)
        - get_text_embedding(texts) -> embedding (512 dim, projected)
        - zero_shot_classify(waveform, prompts) -> scores
    """
    
    def __init__(self, sample_rate: int = SAMPLE_RATE):
        super().__init__()
        self.sample_rate = sample_rate
        
        # Default audio config (HTSAT-tiny with fusion)
        self.audio_cfg = {
            'audio_length': AUDIO_LENGTH,
            'clip_samples': CLIP_SAMPLES,
            'mel_bins': MEL_BINS,
            'sample_rate': SAMPLE_RATE,
            'window_size': WINDOW_SIZE,
            'hop_size': HOP_SIZE,
            'fmin': FMIN,
            'fmax': FMAX,
            'class_num': CLASSES_NUM,
            'model_type': 'HTSAT',
            'model_name': 'tiny'
        }
        
        # Text config
        self.text_cfg = {
            'context_length': TEXT_CONTEXT_LENGTH,
            'model_type': 'roberta'
        }
        
        # Model will be initialized on load_pretrained
        self.model = None
        self.tokenizer = None
        self.enable_fusion = True
        self.fusion_type = 'aff_2d'
    
    def load_pretrained(self, checkpoint_path: str) -> None:
        """Load pretrained CLAP checkpoint."""
        # Create model
        self.model = CLAPModel(
            embed_dim=EMBED_DIM,
            audio_cfg=self.audio_cfg,
            text_cfg=self.text_cfg,
            enable_fusion=self.enable_fusion,
            fusion_type=self.fusion_type,
            joint_embed_shape=JOINT_EMBED_DIM,
            mlp_act='relu'
        )
        
        # Load tokenizer
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        
        # Load checkpoint
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
        
        # Remove 'module.' prefix if present
        if any(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        # Load weights
        self.model.load_state_dict(state_dict, strict=False)
        
        print(f"Loaded pretrained weights from {checkpoint_path}")
    
    def _prepare_audio(self, waveform: Tensor, device) -> dict:
        """Prepare audio input for HTSAT."""
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        batch_size = waveform.shape[0]
        
        # Pad or truncate to clip_samples
        if waveform.shape[1] < CLIP_SAMPLES:
            waveform = F.pad(waveform, (0, CLIP_SAMPLES - waveform.shape[1]))
        else:
            waveform = waveform[:, :CLIP_SAMPLES]
        
        # For fusion mode, we need mel_fusion tensor
        # Extract mel spectrogram
        spec_extractor = self.model.audio_branch.spectrogram_extractor
        logmel_extractor = self.model.audio_branch.logmel_extractor
        
        waveform_device = waveform.to(device)
        spec = spec_extractor(waveform_device)  # (B, 1, T, F)
        mel = logmel_extractor(spec)  # (B, 1, T, mel_bins)
        
        # For fusion, we need [global_mel, local_mel_1, local_mel_2, local_mel_3]
        # Simplified: just use global mel repeated
        if self.enable_fusion:
            mel_fusion = mel.repeat(1, 4, 1, 1)  # (B, 4, T, mel_bins)
            longer = torch.zeros(batch_size, dtype=torch.bool, device=device)
            return {
                'mel_fusion': mel_fusion,
                'longer': longer
            }
        else:
            return {
                'waveform': waveform_device
            }
    
    def _tokenize_text(self, texts: List[str]) -> dict:
        """Tokenize text input."""
        result = self.tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=TEXT_CONTEXT_LENGTH,
            return_tensors='pt'
        )
        return result
    
    # ═══════════════════════════════════════════════════
    # STANDARD API (compatible with other audio models)
    # ═══════════════════════════════════════════════════
    
    def forward(self, waveform: Tensor) -> Tensor:
        """
        Forward pass returning AudioSet class probabilities.
        
        Args:
            waveform: (batch, samples) or (samples,) at 48kHz
        Returns:
            probs: (batch, 527) AudioSet class probabilities
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        device = next(self.model.parameters()).device
        audio_input = self._prepare_audio(waveform, device)
        
        output = self.model.audio_branch(audio_input, device=device)
        return output['clipwise_output']
    
    def forward_with_embedding(self, waveform: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward pass returning both probabilities and embedding.
        
        Args:
            waveform: (batch, samples) or (samples,) at 48kHz
        Returns:
            probs: (batch, 527) AudioSet class probabilities
            embedding: (batch, 768) feature embedding (pre-projection)
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        device = next(self.model.parameters()).device
        audio_input = self._prepare_audio(waveform, device)
        
        output = self.model.audio_branch(audio_input, device=device)
        return output['clipwise_output'], output['embedding']
    
    def get_embedding(self, waveform: Tensor) -> Tensor:
        """
        Extract embedding from raw waveform (pre-projection, 768 dim).
        
        Args:
            waveform: (batch, samples) or (samples,) at 48kHz
        Returns:
            embedding: (batch, 768) feature embedding
        """
        _, embedding = self.forward_with_embedding(waveform)
        return embedding
    
    # ═══════════════════════════════════════════════════
    # CLAP-SPECIFIC API (zero-shot capabilities)
    # ═══════════════════════════════════════════════════
    
    def get_audio_embedding_clap(self, waveform: Tensor) -> Tensor:
        """
        Get audio embedding in joint space (post-projection, 512 dim).
        
        Args:
            waveform: (batch, samples) or (samples,) at 48kHz
        Returns:
            embedding: (batch, 512) normalized audio embedding
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        device = next(self.model.parameters()).device
        audio_input = self._prepare_audio(waveform, device)
        
        output = self.model.audio_branch(audio_input, device=device)
        audio_emb = self.model.audio_projection(output['embedding'])
        audio_emb = F.normalize(audio_emb, dim=-1)
        return audio_emb
    
    def get_text_embedding_clap(self, texts: List[str]) -> Tensor:
        """
        Get text embedding in joint space (512 dim).
        
        Args:
            texts: List of text prompts
        Returns:
            embedding: (num_texts, 512) normalized text embedding
        """
        device = next(self.model.parameters()).device
        text_input = self._tokenize_text(texts)
        
        text_emb = self.model.encode_text(text_input, device=device)
        text_emb = F.normalize(text_emb, dim=-1)
        return text_emb
    
    def zero_shot_classify(self, waveform: Tensor, prompts: List[str]) -> Tensor:
        """
        Zero-shot audio classification using text prompts.
        
        Args:
            waveform: (batch, samples) or (samples,) at 48kHz
            prompts: List of text prompts (e.g., ["sound of rain", "dog barking"])
        Returns:
            scores: (batch, num_prompts) similarity scores
        """
        audio_emb = self.get_audio_embedding_clap(waveform)  # (B, 512)
        text_emb = self.get_text_embedding_clap(prompts)  # (num_prompts, 512)
        
        # Compute cosine similarity
        scores = audio_emb @ text_emb.T  # (B, num_prompts)
        return scores


# ============================================
# Demo
# ============================================
if __name__ == "__main__":
    import os
    import csv
    import soundfile as sf
    import torchaudio.functional as F_audio
    
    print("=" * 60)
    print("CLAP (Contrastive Language-Audio Pretraining) - Demo")
    print("=" * 60)
    
    # 1. Device selection
    print("\n1. Selecting device...")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"   Using device: {device}")
    
    # 2. Load AudioSet labels
    print("\n2. Loading AudioSet labels...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    labels_path = os.path.join(script_dir, "..", "..", "datasets", "AudioSet_EV_v2PANNs_2020", "audioset_metadata", "class_labels_indices.csv")
    with open(labels_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        labels = {i: row[2] for i, row in enumerate(reader)}
    print(f"   Loaded {len(labels)} labels")
    
    # 3. Load audio
    print("\n3. Loading audio...")
    audio_path = os.path.join(script_dir, "..", "..", "datasets", "AudioSet_EV_v2PANNs_2020", "Positive_files", "unbalanced", "_1f3piTQtmo.wav")
    waveform, sr = sf.read(audio_path)
    waveform = torch.from_numpy(waveform).float()
    if waveform.dim() == 2:
        waveform = waveform.mean(dim=1)
    
    # Resample if needed
    if sr != SAMPLE_RATE:
        waveform = F_audio.resample(waveform, sr, SAMPLE_RATE)
        print(f"   Resampled {sr} Hz → {SAMPLE_RATE} Hz")
    
    print(f"   Duration: {waveform.shape[0]/SAMPLE_RATE:.2f}s")
    waveform = waveform.to(device)
    
    # 4. Create model and load checkpoint
    print("\n4. Creating model...")
    checkpoint_path = os.path.join(script_dir, "630k-audioset-fusion-best.pt")
    model = CLAP(sample_rate=SAMPLE_RATE)
    model.load_pretrained(checkpoint_path)
    model = model.to(device)
    model.eval()
    print(f"   Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    # 5. Inference (AudioSet classification)
    print("\n5. Running inference (AudioSet)...")
    with torch.no_grad():
        probs = model(waveform)
    print(f"   Output shape: {probs.shape}")
    
    # 6. Top-10 predictions
    print("\n6. Top-10 AudioSet predictions:")
    probs_np = probs[0].cpu().numpy()
    top_indices = probs_np.argsort()[::-1][:10]
    for i, idx in enumerate(top_indices):
        print(f"   {i+1:2d}. {labels[idx]:<40} {probs_np[idx]:.4f}")
    
    # 7. Zero-shot classification
    print("\n7. Zero-shot classification...")
    prompts = [
        "a person speaking",
        "music playing",
        "telephone ringing",
        "dog barking",
        "rain falling"
    ]
    print(f"   Prompts: {prompts}")
    with torch.no_grad():
        scores = model.zero_shot_classify(waveform, prompts)
    
    print("\n   Zero-shot scores:")
    scores_np = scores[0].cpu().numpy()
    for prompt, score in zip(prompts, scores_np):
        print(f"   - {prompt:<30} {score:.4f}")
    
    # 8. Embedding extraction
    print("\n8. Extracting embeddings...")
    with torch.no_grad():
        embedding_768 = model.get_embedding(waveform)
        embedding_512 = model.get_audio_embedding_clap(waveform)
    print(f"   Pre-projection embedding shape: {embedding_768.shape} (768 dim)")
    print(f"   Post-projection embedding shape: {embedding_512.shape} (512 dim)")
    
    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)


