"""
E-PANNs (Efficient PANNs)
========================================
Efficient Pre-trained Audio Neural Networks via Passive Filter Pruning.

Original repository: https://github.com/Arshdeep-Singh-Boparai/E-PANNs

This is a standalone implementation of E-PANNs (pruned CNN14) for AudioSet tagging.
"""

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation


# ============================================
# CONSTANTS (hardcoded for E-PANNs / AudioSet)
# ============================================
SAMPLE_RATE = 32000
CLASSES_NUM = 527
EMBED_DIM = 2048  # fc1 output dimension

# Spectrogram params
WINDOW_SIZE = 1024
HOP_SIZE = 320
MEL_BINS = 64
F_MIN = 50
F_MAX = 14000

# Pruning ratios (50% pruning on blocks 4, 5, 6)
# p1-p6 = 0 (no pruning), p7-p12 = 0.5 (50% pruning)
PRUNING_RATIOS = {'p1': 0, 'p2': 0,         # conv_block1
                  'p3': 0, 'p4': 0,         # conv_block2
                  'p5': 0, 'p6': 0,         # conv_block3
                  'p7': 0.5, 'p8': 0.5,     # conv_block4
                  'p9': 0.5, 'p10': 0.5,    # conv_block5
                  'p11': 0.5, 'p12': 0.5}   # conv_block6


# ============================================
# Helper Functions
# ============================================
def init_layer(layer: nn.Module):
    """Initialize a Linear or Convolutional layer."""
    nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)


def init_bn(bn: nn.Module):
    """Initialize a BatchNorm layer."""
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


# ============================================
# Building Blocks
# ============================================
class ConvBlock(nn.Module):
    """Standard convolutional block with two conv layers."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)
        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.init_weight()

    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, x: torch.Tensor, pool_size: Tuple[int, int] = (2, 2),
                pool_type: str = 'avg') -> torch.Tensor:
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise ValueError(f"Unknown pool_type: {pool_type}")
        
        return x


class ConvBlockPruned(nn.Module):
    """Pruned convolutional block with different input/output channels."""
    
    def __init__(self, in_channels: int, out_channels_1: int, out_channels_2: int):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels_1,
                               kernel_size=(3, 3), 
                               stride=(1, 1),
                               padding=(1, 1), 
                               bias=False)
        self.conv2 = nn.Conv2d(in_channels=out_channels_1,
                               out_channels=out_channels_2,
                               kernel_size=(3, 3), 
                               stride=(1, 1),
                               padding=(1, 1), 
                               bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels_1)
        self.bn2 = nn.BatchNorm2d(out_channels_2)
        
        self.init_weight()

    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, x: torch.Tensor, pool_size: Tuple[int, int] = (2, 2), pool_type: str = 'avg') -> torch.Tensor:
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise ValueError(f"Unknown pool_type: {pool_type}")
        
        return x


# ============================================
# Main Model
# ============================================
class Cnn14Pruned(nn.Module):
    """
    E-PANNs: Efficient CNN14 with 50% filter pruning on deeper blocks.
    
    Architecture:
        - Spectrogram + LogMel extraction (via torchlibrosa)
        - 6 ConvBlocks (blocks 4-6 are pruned by 50%)
        - Global pooling (mean + max)
        - FC layers: 1024 -> 2048 -> 527
    """
    
    def __init__(self,
                 sample_rate: int = SAMPLE_RATE,
                 window_size: int = WINDOW_SIZE,
                 hop_size: int = HOP_SIZE,
                 mel_bins: int = MEL_BINS,
                 fmin: int = F_MIN,
                 fmax: int = F_MAX,
                 classes_num: int = CLASSES_NUM,
                 # Pruning ratios
                 p1: float = 0, p2: float = 0,
                 p3: float = 0, p4: float = 0,
                 p5: float = 0, p6: float = 0,
                 p7: float = 0.5, p8: float = 0.5,
                 p9: float = 0.5, p10: float = 0.5,
                 p11: float = 0.5, p12: float = 0.5):
        super().__init__()
        
        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, 
                                                 hop_length=hop_size,
                                                 win_length=window_size, 
                                                 window=window,
                                                 center=center, 
                                                 pad_mode=pad_mode,
                                                 freeze_parameters=True)

        # LogMel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, 
                                                 n_fft=window_size,
                                                 n_mels=mel_bins, 
                                                 fmin=fmin, 
                                                 fmax=fmax,
                                                 ref=ref, 
                                                 amin=amin, 
                                                 top_db=top_db,
                                                 freeze_parameters=True)

        # Spec augmentation (only used in training)
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2, freq_drop_width=8, freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(mel_bins)

        # Conv blocks with pruning
        # Blocks 1-3: no pruning (standard ConvBlock-like but using pruned class for consistency)
        self.conv_block1 = ConvBlockPruned(in_channels=1, out_channels_1=int(64 * (1 - p1)), out_channels_2=int(64 * (1 - p2)))
        self.conv_block2 = ConvBlockPruned(in_channels=int(64 * (1 - p2)), out_channels_1=int(128 * (1 - p3)), out_channels_2=int(128 * (1 - p4)))
        self.conv_block3 = ConvBlockPruned(in_channels=int(128 * (1 - p4)), out_channels_1=int(256 * (1 - p5)), out_channels_2=int(256 * (1 - p6)))
        # Blocks 4-6: 50% pruning
        self.conv_block4 = ConvBlockPruned(in_channels=int(256 * (1 - p6)), out_channels_1=int(512 * (1 - p7)), out_channels_2=int(512 * (1 - p8)))
        self.conv_block5 = ConvBlockPruned(in_channels=int(512 * (1 - p8)), out_channels_1=int(1024 * (1 - p9)), out_channels_2=int(1024 * (1 - p10)))
        self.conv_block6 = ConvBlockPruned(in_channels=int(1024 * (1 - p10)), out_channels_1=int(2048 * (1 - p11)), out_channels_2=int(2048 * (1 - p12)))

        # FC layers
        self.fc1 = nn.Linear(int(2048 * (1 - p12)), 2048, bias=True)
        self.fc_audioset = nn.Linear(2048, classes_num, bias=True)

        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: (batch_size, data_length) raw waveform at 32kHz
            
        Returns:
            probs: (batch_size, 527) sigmoid probabilities
        """
        # Spectrogram extraction
        x = self.spectrogram_extractor(x)  # (batch, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)       # (batch, 1, time_steps, mel_bins)

        # BatchNorm on mel dimension
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        # Spec augmentation (training only)
        if self.training:
            x = self.spec_augmenter(x)

        # Conv blocks
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)

        # Global pooling
        x = torch.mean(x, dim=3)  # (batch, channels, time)
        x1, _ = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2  # (batch, channels)

        # FC layers
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        
        # Output with sigmoid (multi-label)
        clipwise_output = torch.sigmoid(self.fc_audioset(x))

        return clipwise_output

    def forward_with_embedding(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning both probabilities and embedding.
        
        Args:
            x: (batch_size, data_length) raw waveform at 32kHz
            
        Returns:
            probs: (batch_size, 527) sigmoid probabilities
            embedding: (batch_size, 2048) feature embedding
        """
        # Spectrogram extraction
        x = self.spectrogram_extractor(x)
        x = self.logmel_extractor(x)

        # BatchNorm on mel dimension
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        # Spec augmentation (training only)
        if self.training:
            x = self.spec_augmenter(x)

        # Conv blocks
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)

        # Global pooling
        x = torch.mean(x, dim=3)
        x1, _ = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2

        # FC layers
        x = F.dropout(x, p=0.5, training=self.training)
        embedding = F.relu_(self.fc1(x))
        x = F.dropout(embedding, p=0.5, training=self.training)
        
        clipwise_output = torch.sigmoid(self.fc_audioset(x))

        return clipwise_output, embedding


# ============================================
# MAIN CLASS WITH PREPROCESSING
# ============================================
class EPANNs(nn.Module):
    """
    E-PANNs wrapper (preprocessing is built into Cnn14Pruned).
    
    This is a thin wrapper for consistency with other vendorized models.
    """
    
    def __init__(self, sample_rate: int = SAMPLE_RATE):
        super().__init__()
        self.sample_rate = sample_rate
        
        # Build model with default pruning (50% on blocks 4-6)
        self.model = Cnn14Pruned(sample_rate=sample_rate,
                                 window_size=WINDOW_SIZE,
                                 hop_size=HOP_SIZE,
                                 mel_bins=MEL_BINS,
                                 fmin=F_MIN,
                                 fmax=F_MAX,
                                 classes_num=CLASSES_NUM,
                                 **PRUNING_RATIOS)

    def load_pretrained(self, checkpoint_path: str) -> None:
        """Load pretrained weights from checkpoint."""
        state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        self.model.load_state_dict(state_dict)
        print(f"Loaded pretrained weights from {checkpoint_path}")

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Forward pass from raw waveform to class probabilities.
        
        Args:
            waveform: (batch, samples) or (samples,) at self.sample_rate
            
        Returns:
            probs: (batch, 527) sigmoid probabilities
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        return self.model(waveform)

    def forward_with_embedding(self, waveform: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning both probabilities and embedding.
        
        Args:
            waveform: (batch, samples) or (samples,) at self.sample_rate
            
        Returns:
            probs: (batch, 527) sigmoid probabilities
            embedding: (batch, 2048) feature embedding
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        return self.model.forward_with_embedding(waveform)

    def get_embedding(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Extract embedding from raw waveform.
        
        Args:
            waveform: (batch, samples) or (samples,) at self.sample_rate
            
        Returns:
            embedding: (batch, 2048) raw feature embedding from fc1
        """
        _, embedding = self.forward_with_embedding(waveform)
        return embedding


# ============================================
# Demo
# ============================================
if __name__ == "__main__":
    import os
    import csv
    import soundfile as sf
    import torchaudio.functional as F_audio
    
    print("=" * 60)
    print("E-PANNs (Efficient PANNs) - AudioSet Tagging Demo")
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
    labels_path = "/Users/stefano/Documents/PhD_main_project/datasets_gui_data/youtube/audioset/class_labels_indices.csv"
    with open(labels_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        labels = {i: row[2] for i, row in enumerate(reader)}
    print(f"   Loaded {len(labels)} labels")
    
    # 3. Load audio
    print("\n3. Loading audio...")
    audio_path = "/Users/stefano/Documents/PhD_main_project/utils/R9_ZSCveAHg_7s.wav"
    waveform, sr = sf.read(audio_path)
    waveform = torch.from_numpy(waveform).float()
    if waveform.dim() == 2:
        waveform = waveform.mean(dim=1)  # mono
    
    # Resample if needed
    if sr != SAMPLE_RATE:
        waveform = F_audio.resample(waveform, sr, SAMPLE_RATE)
        print(f"   Resampled {sr} Hz → {SAMPLE_RATE} Hz")
    
    print(f"   Duration: {waveform.shape[0]/SAMPLE_RATE:.2f}s")
    waveform = waveform.to(device)
    
    # 4. Create model and load checkpoint
    print("\n4. Creating model...")
    # Load checkpoint
    script_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(script_dir, "checkpoint_closeto_.44.pt")
    model = EPANNs(sample_rate=SAMPLE_RATE)
    model.load_pretrained(checkpoint_path)
    model = model.to(device)
    model.eval()
    print(f"   Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    # 5. Inference
    print("\n5. Running inference...")
    with torch.no_grad():
        probs = model(waveform)
    print(f"   Output shape: {probs.shape}")
    
    # 6. Top-10 predictions
    print("\n6. Top-10 predictions:")
    probs_cpu = probs[0].cpu().numpy()
    top_indices = probs_cpu.argsort()[::-1][:10]
    for i, idx in enumerate(top_indices):
        print(f"   {i+1:2d}. {labels[idx]:<40} {probs_cpu[idx]:.4f}")
    
    # 7. Embedding extraction
    print("\n7. Extracting embedding...")
    with torch.no_grad():
        embedding = model.get_embedding(waveform)
    print(f"   Embedding shape: {embedding.shape}")
    
    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)
