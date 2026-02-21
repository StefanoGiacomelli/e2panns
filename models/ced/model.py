"""
CED (Consistent Ensemble Distillation)
========================================
Consistent Ensemble Distillation for Audio Tagging.

Original repository: https://github.com/RicherMans/CED

This is a standalone implementation of CED-Base for AudioSet tagging.
"""

import math
import collections
from functools import partial
from typing import Any, Callable, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as audio_transforms
from einops import rearrange
from einops.layers.torch import Rearrange


# ============================================
# CONSTANTS (hardcoded for CED-Base / AudioSet)
# ============================================
SAMPLE_RATE = 16000
CLASSES_NUM = 527

# CED-Base architecture params
EMBED_DIM = 768
DEPTH = 12
NUM_HEADS = 12
MLP_RATIO = 4
PATCH_SIZE = 16
PATCH_STRIDE = 16

# Frontend params
N_MELS = 64
N_FFT = 512
HOP_SIZE = 160
WIN_SIZE = 512
F_MIN = 0
F_MAX = 8000
CENTER = True

# Other defaults
TARGET_LENGTH = 1012
POOLING = 'mean'
EVAL_AVG = 'mean'


# ============================================
# Helper Functions
# ============================================
def to_2tuple(x: Any) -> Tuple[Any, Any]:
    """Convert input to 2-tuple."""
    if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
        return tuple(x)
    return (x, x)


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    """
    Truncated normal initialization.
    Fills the input Tensor with values drawn from a truncated normal distribution.
    From timm.
    """
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """
    Drop paths (Stochastic Depth) per sample.
    From timm.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


# ============================================
# Building Blocks
# ============================================
class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample. From timm."""

    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob, 3):0.3f}'


class Mlp(nn.Module):
    """MLP as used in Vision Transformer."""

    def __init__(self,
                 in_features: int,
                 hidden_features: Optional[int] = None,
                 out_features: Optional[int] = None,
                 act_layer: Callable = nn.GELU,
                 drop: float = 0.):
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


class AudioPatchEmbed(nn.Module):
    """Audio Patch Embedding using Conv2d."""

    def __init__(self,
                 input_size: Tuple[int, int] = (64, 1012),
                 patch_size: int = 16,
                 patch_stride: int = 16,
                 in_chans: int = 1,
                 embed_dim: int = 768,
                 norm_layer: Optional[Callable] = None,
                 flatten: bool = False):
        super().__init__()
        self.input_size = to_2tuple(input_size)
        self.patch_size = to_2tuple(patch_size)
        self.patch_stride = to_2tuple(patch_stride)
        self.grid_size = (self.input_size[0] // self.patch_stride[0],
                          self.input_size[1] // self.patch_stride[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans,
                              embed_dim,
                              kernel_size=patch_size,
                              stride=patch_stride)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        if self.flatten:
            x = rearrange(x, 'b c f t -> b (f t) c')
        x = self.norm(x)
        return x


class Attention(nn.Module):
    """Multi-head Self Attention."""

    def __init__(self,
                 dim: int,
                 num_heads: int = 8,
                 qkv_bias: bool = False,
                 attn_drop: float = 0.,
                 proj_drop: float = 0.,
                 causal: bool = False):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.causal = causal

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if self.causal:
            mask_value = -torch.finfo(attn.dtype).max
            i, j = attn.shape[-2:]
            mask = torch.ones(i, j, device=q.device, dtype=torch.bool).triu(j - i + 1)
            attn = attn.masked_fill(mask, mask_value)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    """Transformer Block."""

    def __init__(self,
                 dim: int,
                 num_heads: int,
                 mlp_ratio: float = 4.,
                 qkv_bias: bool = False,
                 drop: float = 0.,
                 attn_drop: float = 0.,
                 drop_path: float = 0.,
                 act_layer: Callable = nn.GELU,
                 norm_layer: Callable = nn.LayerNorm,
                 attention_type: Callable = Attention,
                 attention_kwargs: dict = {},
                 **kwargs):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attention_type(dim,
                                   num_heads=num_heads,
                                   qkv_bias=qkv_bias,
                                   attn_drop=attn_drop,
                                   proj_drop=drop,
                                   **attention_kwargs)
        self.ls1 = nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=int(dim * mlp_ratio),
                       act_layer=act_layer,
                       drop=drop)
        self.ls2 = nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


# ============================================
# FRONTEND (Mel Spectrogram)
# ============================================
class FrontEnd(nn.Sequential):
    """Audio frontend: MelSpectrogram + AmplitudeToDB."""

    def __init__(self,
                 f_min: int = 0,
                 sample_rate: int = 16000,
                 win_size: int = 512,
                 center: bool = True,
                 n_fft: int = 512,
                 f_max: Optional[int] = None,
                 hop_size: int = 160,
                 n_mels: int = 64):
        self.f_min = f_min
        self.sample_rate = sample_rate
        self.win_size = win_size
        self.center = center
        self.n_fft = n_fft
        self.f_max = f_max
        self.hop_size = hop_size
        self.n_mels = n_mels

        super().__init__(audio_transforms.MelSpectrogram(f_min=self.f_min,
                                                         sample_rate=self.sample_rate,
                                                         win_length=self.win_size,
                                                         center=self.center,
                                                         n_fft=self.n_fft,
                                                         f_max=self.f_max,
                                                         hop_length=self.hop_size,
                                                         n_mels=self.n_mels),
                        audio_transforms.AmplitudeToDB(top_db=120))

    def forward(self, x):
        # Disable autocast for numerical stability
        with torch.amp.autocast(device_type='cuda', enabled=False):
            with torch.amp.autocast(device_type='cpu', enabled=False):
                return super().forward(x.float())


# ============================================
# Main Model
# ============================================
class CEDBase(nn.Module):
    """
    CED-Base Audio Transformer for AudioSet tagging.
    
    Consistent Ensemble Distillation model achieving 50.0% mAP on AudioSet.
    
    Args:
        sample_rate: Expected sample rate of input audio (default: 16000)
        classes_num: Number of output classes (default: 527 for AudioSet)
    
    Input:
        waveform: (batch, samples) tensor @ 16000 Hz
        
    Output:
        probabilities: (batch, 527) sigmoid probabilities
    """

    def __init__(self,
                 sample_rate: int = SAMPLE_RATE,
                 classes_num: int = CLASSES_NUM,
                 embed_dim: int = EMBED_DIM,
                 depth: int = DEPTH,
                 num_heads: int = NUM_HEADS,
                 mlp_ratio: float = MLP_RATIO,
                 patch_size: int = PATCH_SIZE,
                 patch_stride: int = PATCH_STRIDE,
                 n_mels: int = N_MELS,
                 n_fft: int = N_FFT,
                 hop_size: int = HOP_SIZE,
                 win_size: int = WIN_SIZE,
                 f_min: int = F_MIN,
                 f_max: int = F_MAX,
                 center: bool = CENTER,
                 target_length: int = TARGET_LENGTH,
                 pooling: str = POOLING,
                 eval_avg: str = EVAL_AVG,
                 qkv_bias: bool = True,
                 drop_rate: float = 0.,
                 attn_drop_rate: float = 0.,
                 drop_path_rate: float = 0.):
        super().__init__()

        assert pooling in ('mean', 'token', 'dm', 'logit')
        
        self.sample_rate = sample_rate
        self.classes_num = classes_num
        self.embed_dim = embed_dim
        self.patch_stride = patch_stride
        self.patch_size = patch_size
        self.n_mels = n_mels
        self.hop_size = hop_size
        self.win_size = win_size
        self.center = center
        self.pooling = pooling
        self.eval_avg = eval_avg
        self.target_length = target_length
        self.maximal_allowed_length = target_length
        self.pad_last = True

        # Frontend: waveform -> mel spectrogram
        self.front_end = FrontEnd(f_min=f_min,
                                  f_max=f_max,
                                  center=self.center,
                                  win_size=self.win_size,
                                  hop_size=self.hop_size,
                                  sample_rate=sample_rate,
                                  n_fft=n_fft,
                                  n_mels=self.n_mels)

        # Batch normalization
        self.init_bn = nn.Sequential(Rearrange('b c f t -> b f c t'),
                                     nn.BatchNorm2d(self.n_mels, momentum=0.01),
                                     Rearrange('b f c t -> b c f t'))

        # Patch embedding
        self.patch_embed = AudioPatchEmbed(input_size=(self.n_mels, target_length),
                                           embed_dim=self.embed_dim,
                                           patch_size=self.patch_size,
                                           flatten=False,
                                           patch_stride=self.patch_stride)

        # Positional embeddings
        self.time_pos_embed = nn.Parameter(torch.randn(1, embed_dim, 1, self.patch_embed.grid_size[1]) * .02)
        self.freq_pos_embed = nn.Parameter(torch.randn(1, embed_dim, self.patch_embed.grid_size[0], 1) * .02)

        # CLS token for token pooling
        if self.pooling == 'token':
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.token_pos_embed = nn.Parameter(torch.randn(1, embed_dim) * .02)

        # Transformer blocks
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        act_layer = nn.GELU
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.blocks = nn.Sequential(*[Block(dim=embed_dim,
                                            num_heads=num_heads,
                                            mlp_ratio=mlp_ratio,
                                            qkv_bias=qkv_bias,
                                            drop=drop_rate,
                                            attn_drop=attn_drop_rate,
                                            drop_path=dpr[i],
                                            norm_layer=norm_layer,
                                            act_layer=act_layer,
                                            attention_type=Attention) for i in range(depth)])

        # Output layers
        self.norm = norm_layer(embed_dim)
        self.outputlayer = nn.Sequential(nn.LayerNorm(self.embed_dim), nn.Linear(self.embed_dim, classes_num))

        # Initialize weights
        self.apply(self._init_weights)
        if hasattr(self, 'cls_token'):
            nn.init.normal_(self.cls_token, std=1e-6)

    def _init_weights(self, module):
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def load_pretrained(self, checkpoint_path: str) -> None:
        """
        Load pretrained CED-Base weights.
        
        Args:
            checkpoint_path: Path to audiotransformer_base_mAP_4999.pt
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Handle wrapped checkpoint
        if 'model' in checkpoint:
            checkpoint = checkpoint['model']

        # Handle positional embedding size mismatch
        if 'time_pos_embed' in checkpoint:
            if self.time_pos_embed.shape != checkpoint['time_pos_embed'].shape:
                self._resize_pos_embed(checkpoint)

        self.load_state_dict(checkpoint, strict=True)
        print(f"Loaded pretrained weights from {checkpoint_path}")

    def _resize_pos_embed(self, state_dict):
        """Resize positional embeddings if needed."""
        target_time_len = self.time_pos_embed.shape[-1]
        target_freq_len = self.freq_pos_embed.shape[-2]

        pretrained_time = state_dict['time_pos_embed']
        pretrained_freq = state_dict['freq_pos_embed']

        if target_time_len <= pretrained_time.shape[-1]:
            state_dict['time_pos_embed'] = pretrained_time[..., :target_time_len]
        else:
            state_dict['time_pos_embed'] = F.interpolate(pretrained_time,
                                                         size=(1, target_time_len),
                                                         align_corners=False,
                                                         mode='bilinear')

        if target_freq_len <= pretrained_freq.shape[-2]:
            state_dict['freq_pos_embed'] = pretrained_freq[:, :, :target_freq_len, :]
        else:
            state_dict['freq_pos_embed'] = F.interpolate(pretrained_freq,
                                                         size=(target_freq_len, 1),
                                                         align_corners=False,
                                                         mode='bilinear')

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from spectrogram."""
        x = self.patch_embed(x)
        b, c, f, t = x.shape
        
        # Add positional embeddings
        x = x + self.time_pos_embed[:, :, :, :t]
        x = x + self.freq_pos_embed[:, :, :, :]
        
        # Reshape to sequence
        x = rearrange(x, 'b c f t -> b (f t) c')
        
        # Add CLS token if using token pooling
        if self.pooling == 'token':
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            cls_token = cls_token + self.token_pos_embed
            x = torch.cat((cls_token, x), dim=1)
        
        x = self.pos_drop(x)
        x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward_head(self, x: torch.Tensor) -> torch.Tensor:
        """Apply pooling and output layer."""
        if self.pooling == 'token':
            x = x[:, 0]
            return self.outputlayer(x).sigmoid()
        elif self.pooling == 'mean':
            x = x.mean(1)
            return self.outputlayer(x).sigmoid()
        elif self.pooling == 'logit':
            x = x.mean(1)
            return self.outputlayer(x)
        elif self.pooling == 'dm':
            x = rearrange(x, 'b (f t) d -> b f t d', f=self.patch_embed.grid_size[0])
            x = self.outputlayer(x.mean(1)).sigmoid()
            return x.mean(1)
        else:
            return x.mean(1)

    def forward_spectrogram(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass on spectrogram, handling long audio with chunking."""
        x = rearrange(x, 'b f t -> b 1 f t')
        x = self.init_bn(x)
        
        if x.shape[-1] > self.maximal_allowed_length:
            # Split long audio into chunks
            splits = x.split(self.target_length, -1)

            if splits[-1].shape[-1] < self.target_length:
                if self.pad_last:
                    pad = torch.zeros(*x.shape[:-1], self.target_length, device=x.device)
                    pad[..., :splits[-1].shape[-1]] = splits[-1]
                    splits = torch.stack((*splits[:-1], pad), dim=0)
                else:
                    splits = torch.stack(splits[:-1], dim=0)
            else:
                splits = torch.stack(splits[:-1], dim=0)
            
            n_splits = len(splits)
            x = rearrange(splits, 'spl b c f t -> (spl b) c f t')
            x = self.forward_head(self.forward_features(x))
            x = rearrange(x, '(spl b) d -> spl b d', spl=n_splits)
            
            if self.eval_avg == 'mean':
                x = x.mean(0)
            elif self.eval_avg == 'max':
                x = x.max(0)[0]
            else:
                raise ValueError(f'Unknown eval average function ({self.eval_avg})')
        else:
            x = self.forward_features(x)
            x = self.forward_head(x)
        
        return x

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            waveform: (batch, samples) @ 16000 Hz
            
        Returns:
            probabilities: (batch, 527) sigmoid probabilities
        """
        x = self.front_end(waveform)
        x = self.forward_spectrogram(x)
        return x

    def get_embedding(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Extract audio embedding without classification.
        
        Args:
            waveform: (batch, samples) @ 16000 Hz
            
        Returns:
            embedding: (batch, 768) - Pre-classification embedding
        """
        x = self.front_end(waveform)
        x = rearrange(x, 'b f t -> b 1 f t')
        x = self.init_bn(x)
        x = self.forward_features(x)
        # Mean pooling (pre-classification)
        embedding = x.mean(1)
        return embedding


# ============================================
# Demo
# ============================================
if __name__ == "__main__":
    import os
    import csv
    import numpy as np
    import soundfile as sf
    import torchaudio

    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    CHECKPOINT_PATH = os.path.join(script_dir, "audiotransformer_base_mAP_4999.pt")
    AUDIO_PATH = os.path.join(script_dir, "..", "..", "datasets", "AudioSet_EV_v2PANNs_2020", "Positive_files", "unbalanced", "_1f3piTQtmo.wav")
    LABELS_PATH = os.path.join(script_dir, "..", "..", "datasets", "AudioSet_EV_v2PANNs_2020", "audioset_metadata", "class_labels_indices.csv")

    print("=" * 60)
    print("CED (Consistent Ensemble Distillation) - AudioSet Tagging Demo")
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

    # 2. Loading AudioSet labels
    print("\n2. Loading AudioSet labels...")
    labels = {}
    with open(LABELS_PATH, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels[int(row['index'])] = row['display_name']
    print(f"   Loaded {len(labels)} labels")

    # 3. Loading audio
    print("\n3. Loading audio...")
    audio, sr = sf.read(AUDIO_PATH)
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)  # stereo to mono
    waveform = torch.from_numpy(audio).float()
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
        print(f"   Resampled {sr} Hz → {SAMPLE_RATE} Hz")
    waveform = waveform.unsqueeze(0).to(device)  # (1, samples)
    print(f"   Duration: {waveform.shape[1] / SAMPLE_RATE:.2f}s")

    # 4. Creating model
    print("\n4. Creating model...")
    model = CEDBase()
    model.load_pretrained(CHECKPOINT_PATH)
    model = model.to(device)
    model.eval()
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # 5. Running inference
    print("\n5. Running inference...")
    with torch.no_grad():
        probs = model(waveform)
    print(f"   Output shape: {probs.shape}")

    # 6. Top-10 predictions
    print("\n6. Top-10 predictions:")
    probs_np = probs[0].cpu().numpy()
    top_indices = probs_np.argsort()[::-1][:10]
    for i, idx in enumerate(top_indices):
        print(f"   {i+1:2d}. {labels[idx]:<40} {probs_np[idx]:.4f}")

    # 7. Embedding extraction
    print("\n7. Extracting embedding...")
    with torch.no_grad():
        embedding = model.get_embedding(waveform)
    print(f"   Embedding shape: {embedding.shape}")

    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)
