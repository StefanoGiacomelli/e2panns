"""
KineScaper Emergency Vehicles Dataset Dataloader
=================================================
PyTorch Dataset and Lightning DataModule for KineScaper EV binary and multiclass classification.

Dataset structure (external mount):
- /mnt/ssd/Kinescaper_EV/dataset/
  - audio/           (61,600 files x 40s each)
  - json/            (metadata.json)
  - csv/             (metadata.tsv)
  - config_siren.yaml

After chunking (4 x 10s chunks per file):
- Positive chunks: ~234,269 (95%)
- Negative chunks: ~12,131 (5%)
Total: 246,400 chunks

Features:
- Non-overlapping chunking (0-10s, 10-20s, 20-30s, 30-40s)
- Overlap-based labeling (≥0.5s overlap → positive)
- Hierarchical negative balancing with augmentation
- Binary and multiclass classification support
- Detection mode with temporal label tracks (40s full samples)
- Seed-controlled for reproducibility

Modes:
- "train": train/val/test split (80/10/10) after chunking
- "benchmark": entire dataset as test set
- "detection": full 40s samples with temporal label tracks

Label types:
- "binary": 0=negative, 1=positive
- "multiclass": 0-6=siren types, 7=negative

Siren classes (multiclass):
  0: hi-lo
  1: two-tone
  2: wail
  3: phaser
  4: piercer
  5: rumbler
  6: yelp
  7: negative

Author: Stefano Giacomelli - Ph.D. candidate in ICT (DISIM dpt. - University of L'Aquila)
"""

import os
import json
import random
import re
from collections import Counter
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import torchaudio
import pytorch_lightning as pl
from pytorch_lightning import seed_everything


# =============================================================================
# CONSTANTS
# =============================================================================

SIREN_CLASS_MAPPING = {
    'hi-lo': 0,
    'two-tone': 1,
    'wail': 2,
    'phaser': 3,
    'piercer': 4,
    'rumbler': 5,
    'yelp': 6,
    'negative': 7
}

SIREN_CLASS_NAMES = [
    'hi-lo', 'two-tone', 'wail', 'phaser',
    'piercer', 'rumbler', 'yelp', 'negative'
]

# Filename pattern: {class}_{type}_{waveform}_{iter}_{onset}_{offset}_i0.wav
FILENAME_PATTERN = re.compile(
    r'^([^_]+)_([^_]+)_([^_]+)_(\d+)_([\d.]+)_([\d.]+)_i0\.wav$'
)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def calculate_overlap(chunk_start: float, chunk_end: float, event_start: float, event_end: float) -> float:
    """Calculate overlap duration between chunk and event."""
    overlap_start = max(chunk_start, event_start)
    overlap_end = min(chunk_end, event_end)
    return max(0, overlap_end - overlap_start)


def parse_kinescaper_filename(filename: str) -> Optional[Dict]:
    """
    Parse KineScaper filename to extract metadata.
    
    Format: {siren_class}_{siren_type}_{waveform}_{iteration}_{onset}_{offset}_i0.wav
    Example: hi-lo_electronic_sawtooth_00_3.164_34.409_i0.wav
    
    Returns:
        Dictionary with parsed fields or None if parsing fails
    """
    match = FILENAME_PATTERN.match(filename)
    if match:
        siren_class, siren_type, waveform, iteration, onset, offset = match.groups()
        return {
            'siren_class': siren_class,
            'siren_type': siren_type,
            'waveform': waveform,
            'iteration': int(iteration),
            'onset': float(onset),
            'offset': float(offset)
        }
    return None


# =============================================================================
# NEGATIVE POOL MANAGER
# =============================================================================

class NegativePoolManager:
    """
    Manages hierarchical negative sample pool with augmentation.
    
    Hierarchy (in order):
      1. KineScaper negative chunks
      2. AudioSet_EV_v2 negatives
      3. FSD50K negatives
      4. UrbanSound8K negatives
      5. ESC50 negatives
      6. LSSiren negatives
    
    Features:
    - Load and cache all negative samples
    - Sample with priority to original samples
    - Apply augmentation on-the-fly to fill gaps
    """
    
    def __init__(self,
                 kinescaper_negatives: List[Tuple[str, int]],
                 use_audioset_v2: bool = True,
                 use_other_datasets: bool = True,
                 target_sr: int = 32000,
                 target_size: int = 320000,
                 augmentation_prob: float = 0.7,
                 seed: int = 42):
        """
        Initialize negative pool manager.
        
        Args:
            kinescaper_negatives: List of (audio_path, label) from KineScaper
            use_audioset_v2: Whether to include AudioSet_EV_v2 negatives
            use_other_datasets: Whether to include other dataset negatives
            target_sr: Target sample rate
            target_size: Target audio size in samples
            augmentation_prob: Probability of applying each augmentation
            seed: Random seed
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        self.kinescaper_negatives = kinescaper_negatives
        self.target_sr = target_sr
        self.target_size = target_size
        self.augmentation_prob = augmentation_prob
        self.seed = seed
        
        # Build negative pool
        self.negative_pool = []
        self._load_negatives(use_audioset_v2, use_other_datasets)
        
        # Define augmentations
        self.augmentations = self._define_augmentations()
        
        print(f"  NegativePoolManager initialized with {len(self.negative_pool):,} samples")
    
    def _load_negatives(self, use_audioset_v2: bool, use_other_datasets: bool):
        """Load negatives from all sources in hierarchy order."""
        # 1. KineScaper negatives
        self.negative_pool.extend(self.kinescaper_negatives)
        print(f"    Loaded {len(self.kinescaper_negatives):,} KineScaper negatives")
        
        # 2. AudioSet_EV_v2
        if use_audioset_v2:
            audioset_v2_negatives = self._load_audioset_v2_negatives()
            self.negative_pool.extend(audioset_v2_negatives)
            print(f"    Loaded {len(audioset_v2_negatives):,} AudioSet_EV_v2 negatives")
        
        # 3. Other datasets
        if use_other_datasets:
            fsd50k_negatives = self._load_fsd50k_negatives()
            us8k_negatives = self._load_us8k_negatives()
            esc50_negatives = self._load_esc50_negatives()
            lssiren_negatives = self._load_lssiren_negatives()
            
            self.negative_pool.extend(fsd50k_negatives)
            self.negative_pool.extend(us8k_negatives)
            self.negative_pool.extend(esc50_negatives)
            self.negative_pool.extend(lssiren_negatives)
            
            print(f"    Loaded {len(fsd50k_negatives):,} FSD50K negatives")
            print(f"    Loaded {len(us8k_negatives):,} UrbanSound8K negatives")
            print(f"    Loaded {len(esc50_negatives):,} ESC50 negatives")
            print(f"    Loaded {len(lssiren_negatives):,} LSSiren negatives")
    
    def _load_audioset_v2_negatives(self) -> List[Tuple[str, int]]:
        """Load AudioSet_EV_v2 negative samples (lazy - no existence check)."""
        negatives = []
        base_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                "AudioSet_EV_v2PANNs_2020")
        
        csv_path = os.path.join(base_dir, "EV_Negatives.csv")
        if not os.path.exists(csv_path):
            return negatives
        
        # Only read downloaded column to speed up
        df = pd.read_csv(csv_path, usecols=['yt_id', 'segment_type', 'downloaded'])
        df_downloaded = df[df['downloaded'] == True]
        
        # Segment mapping
        segment_mapping = {
            'balanced_train': 'balanced_train',
            'eval': 'eval'
        }
        
        # Build path list WITHOUT checking existence (lazy verification)
        for _, row in df_downloaded.iterrows():
            yt_id = row['yt_id']
            segment_type = row.get('segment_type', 'eval')
            mapped_segment = segment_mapping.get(segment_type, segment_type)
            
            audio_path = os.path.join(base_dir, "Negative_files", mapped_segment, f"{yt_id}.wav")
            negatives.append((audio_path, 0))  # Verification done in __getitem__
        
        return negatives
    
    def _load_fsd50k_negatives(self) -> List[Tuple[str, int]]:
        """Load FSD50K negative samples (lazy - no existence check)."""
        negatives = []
        base_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "FSD50K")
        
        # Dev negatives
        dev_csv = os.path.join(base_dir, "FSD-dev_negatives.csv")
        if os.path.exists(dev_csv):
            df_dev = pd.read_csv(dev_csv, usecols=['fname'])
            for _, row in df_dev.iterrows():
                fname = row['fname']
                audio_path = os.path.join(base_dir, "FSD50K.dev_audio", f"{fname}.wav")
                negatives.append((audio_path, 0))
        
        # Eval negatives
        eval_csv = os.path.join(base_dir, "FSD-eval_negatives.csv")
        if os.path.exists(eval_csv):
            df_eval = pd.read_csv(eval_csv, usecols=['fname'])
            for _, row in df_eval.iterrows():
                fname = row['fname']
                audio_path = os.path.join(base_dir, "FSD50K.eval_audio", f"{fname}.wav")
                negatives.append((audio_path, 0))
        
        return negatives
    
    def _load_us8k_negatives(self) -> List[Tuple[str, int]]:
        """Load UrbanSound8K negative samples (lazy - no existence check)."""
        negatives = []
        base_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "UrbanSound8K")
        
        metadata_path = os.path.join(base_dir, "metadata", "UrbanSound8K.csv")
        if not os.path.exists(metadata_path):
            return negatives
        
        df = pd.read_csv(metadata_path, usecols=['fold', 'slice_file_name', 'classID'])
        # Class 9 is siren (positive), rest are negatives
        df_neg = df[df['classID'] != 9]
        
        for _, row in df_neg.iterrows():
            fold = f"fold{row['fold']}"
            fname = row['slice_file_name']
            audio_path = os.path.join(base_dir, "audio", fold, fname)
            negatives.append((audio_path, 0))
        
        return negatives
    
    def _load_esc50_negatives(self) -> List[Tuple[str, int]]:
        """Load ESC50 negative samples (lazy - no existence check)."""
        negatives = []
        base_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ESC50")
        
        metadata_path = os.path.join(base_dir, "esc50.csv")
        if not os.path.exists(metadata_path):
            return negatives
        
        df = pd.read_csv(metadata_path, usecols=['filename', 'category'])
        # Only 'siren' is positive
        df_neg = df[df['category'] != 'siren']
        
        for _, row in df_neg.iterrows():
            fname = row['filename']
            audio_path = os.path.join(base_dir, "original_audio", fname)
            negatives.append((audio_path, 0))
        
        return negatives
    
    def _load_lssiren_negatives(self) -> List[Tuple[str, int]]:
        """Load LSSiren (road noises) negative samples (lazy - no existence check)."""
        negatives = []
        base_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "LSSiren")
        
        csv_path = os.path.join(base_dir, "Road_final.csv")
        if not os.path.exists(csv_path):
            return negatives
        
        # CSV has no header, first column is filename
        df = pd.read_csv(csv_path, header=None, usecols=[0])
        for _, row in df.iterrows():
            filename = row[0]  # First column
            audio_path = os.path.join(base_dir, "Road_Noises", filename)
            negatives.append((audio_path, 0))
        
        return negatives
    
    def _define_augmentations(self) -> dict:
        """Define augmentation functions."""
        return {
            "add_noise": self._add_random_noise,
            "time_roll": self._time_roll,
            "polarity_inversion": self._polarity_inversion,
            "rand_amp_scaling": self._random_amplification,
        }
    
    def _add_random_noise(self, waveform: torch.Tensor, scale: float = 0.1) -> torch.Tensor:
        """Add random noise to waveform."""
        noise_type = random.choice(["white", "gaussian"])
        noise = torch.randn_like(waveform) if noise_type == "white" else torch.normal(0, 1, size=waveform.shape)
        noisy = waveform + noise * scale
        max_val = torch.max(torch.abs(noisy))
        if max_val > 0:
            return noisy / max_val
        return noisy
    
    def _time_roll(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply circular time shift."""
        if waveform.ndim == 1 and waveform.size(0) > 1:
            shift = random.randint(1, waveform.size(0))
            return torch.roll(waveform, shifts=shift, dims=0)
        elif waveform.ndim == 2 and waveform.size(1) > 1:
            shift = random.randint(1, waveform.size(1))
            return torch.roll(waveform, shifts=shift, dims=1)
        return waveform
    
    def _polarity_inversion(self, waveform: torch.Tensor) -> torch.Tensor:
        """Invert waveform polarity."""
        return waveform * -1
    
    def _random_amplification(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply random amplitude scaling."""
        if random.random() > 0.5:
            scalar = random.uniform(0.1, 1.0)
            return waveform * scalar
        else:
            # Handle both 1D and 2D tensors
            if waveform.ndim == 1:
                vector = torch.rand(waveform.size(0))
                return waveform * vector
            else:
                vector = torch.rand(waveform.size(1))
                return waveform * vector.unsqueeze(0)
    
    def _apply_augmentations(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply random augmentations to waveform."""
        augment_order = list(self.augmentations.keys())
        random.shuffle(augment_order)
        
        for aug_name in augment_order:
            if random.random() < self.augmentation_prob:
                waveform = self.augmentations[aug_name](waveform)
        
        return waveform
    
    def sample_original(self) -> Tuple[torch.Tensor, int]:
        """Sample a random original negative (no augmentation)."""
        audio_path, label = random.choice(self.negative_pool)
        waveform = self._load_audio(audio_path)
        return waveform, label
    
    def sample_augmented(self) -> Tuple[torch.Tensor, int]:
        """Sample a random negative and apply augmentation."""
        audio_path, label = random.choice(self.negative_pool)
        waveform = self._load_audio(audio_path)
        waveform = self._apply_augmentations(waveform)
        return waveform, label
    
    def _load_audio(self, audio_path: str) -> torch.Tensor:
        """Load and process audio to target format (with lazy existence check)."""
        # Check existence only when loading
        if not os.path.exists(audio_path):
            print(f"Warning: File not found: {audio_path}")
            return torch.zeros(self.target_size)
        
        try:
            waveform, sr = sf.read(audio_path, dtype='float32')
            waveform = torch.from_numpy(waveform).unsqueeze(0)
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Resample if needed
            if sr != self.target_sr:
                resampler = torchaudio.transforms.Resample(sr, self.target_sr)
                waveform = resampler(waveform)
            
            # Pad or trim to target size
            if waveform.shape[1] < self.target_size:
                waveform = F.pad(waveform, (0, self.target_size - waveform.shape[1]))
            elif waveform.shape[1] > self.target_size:
                waveform = waveform[:, :self.target_size]
            
            return waveform.squeeze(0)  # Return 1D tensor
        
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            return torch.zeros(self.target_size)
    
    def __len__(self):
        return len(self.negative_pool)


# =============================================================================
# KINESCAPER CHUNK DATASET (Train/Benchmark modes)
# =============================================================================

class KineScaper_EV_ChunkDataset(Dataset):
    """
    KineScaper EV dataset with 10s chunking for classification.
    
    Features:
    - Non-overlapping chunks (0-10s, 10-20s, 20-30s, 30-40s)
    - Overlap-based labeling (≥0.5s → positive)
    - Binary or multiclass labels
    - Data augmentation support
    """
    
    def __init__(self,
                 dataset_root: str,
                 metadata_format: str = "json",
                 label_type: str = "binary",
                 min_overlap: float = 0.5,
                 target_sr: int = 32000,
                 chunk_duration: float = 10.0,
                 augmentation: bool = False,
                 aug_prob: float = 0.7,
                 seed: int = 42):
        """
        Initialize KineScaper chunk dataset.
        
        Args:
            dataset_root: Path to dataset root (e.g., /mnt/ssd/Kinescaper_EV/dataset/)
            metadata_format: "json" or "tsv"
            label_type: "binary" or "multiclass"
            min_overlap: Minimum overlap for positive label (seconds)
            target_sr: Target sample rate
            chunk_duration: Chunk duration in seconds
            augmentation: Whether to apply augmentation
            aug_prob: Augmentation probability
            seed: Random seed
        """
        super().__init__()
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        self.dataset_root = dataset_root
        self.metadata_format = metadata_format
        self.label_type = label_type
        self.min_overlap = min_overlap
        self.target_sr = target_sr
        self.chunk_duration = chunk_duration
        self.target_size = int(target_sr * chunk_duration)
        self.augmentation = augmentation
        self.aug_prob = aug_prob
        self.seed = seed
        
        if self.augmentation:
            self.augmentations = self._define_augmentations()
        
        # Load metadata and create chunk list
        self.chunks = self._load_and_chunk_metadata()
        
        print(f"  Loaded {len(self.chunks):,} chunks")
        self._print_statistics()
    
    def _load_metadata(self) -> pd.DataFrame:
        """Load metadata from JSON or TSV."""
        if self.metadata_format == "json":
            json_path = os.path.join(self.dataset_root, "json", "metadata.json")
            with open(json_path, 'r') as f:
                data = json.load(f)
            return pd.DataFrame(data['dataset_metadata'])
        else:  # tsv
            tsv_path = os.path.join(self.dataset_root, "csv", "metadata.tsv")
            return pd.read_csv(tsv_path, sep='\t')
    
    def _load_and_chunk_metadata(self) -> List[Dict]:
        """Load metadata and create chunk entries (lazy - no file existence check)."""
        df = self._load_metadata()
        chunks = []
        
        audio_dir = os.path.join(self.dataset_root, "audio")
        
        for _, row in df.iterrows():
            filename = row['filename']
            onset = row['onset']
            offset = row['offset']
            siren_class = row['siren_class']
            
            audio_path = os.path.join(audio_dir, filename)
            # Removed existence check - verification done in __getitem__
            
            # Create 4 chunks per file
            for chunk_idx in range(4):
                chunk_start = chunk_idx * self.chunk_duration
                chunk_end = (chunk_idx + 1) * self.chunk_duration
                
                # Calculate overlap
                overlap = calculate_overlap(chunk_start, chunk_end, onset, offset)
                
                # Determine label
                if overlap >= self.min_overlap:
                    # Positive chunk
                    if self.label_type == "binary":
                        label = 1
                    else:  # multiclass
                        label = SIREN_CLASS_MAPPING[siren_class]
                else:
                    # Negative chunk
                    if self.label_type == "binary":
                        label = 0
                    else:
                        label = 7  # negative class
                
                chunks.append({
                    'audio_path': audio_path,
                    'chunk_start': chunk_start,
                    'chunk_end': chunk_end,
                    'label': label,
                    'siren_class': siren_class if overlap >= self.min_overlap else 'negative',
                    'overlap': overlap
                })
        
        return chunks
    
    def _print_statistics(self):
        """Print dataset statistics."""
        labels = [chunk['label'] for chunk in self.chunks]
        
        if self.label_type == "binary":
            pos_count = sum(1 for lbl in labels if lbl == 1)
            neg_count = sum(1 for lbl in labels if lbl == 0)
            print(f"    Positives: {pos_count:,} ({pos_count/len(labels)*100:.1f}%)")
            print(f"    Negatives: {neg_count:,} ({neg_count/len(labels)*100:.1f}%)")
        else:  # multiclass
            counter = Counter(labels)
            print(f"    Class distribution:")
            for i, class_name in enumerate(SIREN_CLASS_NAMES):
                count = counter.get(i, 0)
                print(f"      {i}: {class_name:<12} {count:>7,} ({count/len(labels)*100:.1f}%)")
    
    def _define_augmentations(self) -> dict:
        """Define augmentation functions."""
        return {
            "add_noise": self._add_random_noise,
            "time_roll": self._time_roll,
            "polarity_inversion": self._polarity_inversion,
            "rand_amp_scaling": self._random_amplification,
        }
    
    def _add_random_noise(self, waveform: torch.Tensor, scale: float = 0.1) -> torch.Tensor:
        noise_type = random.choice(["white", "gaussian"])
        noise = torch.randn_like(waveform) if noise_type == "white" else torch.normal(0, 1, size=waveform.shape)
        noisy = waveform + noise * scale
        max_val = torch.max(torch.abs(noisy))
        if max_val > 0:
            return noisy / max_val
        return noisy
    
    def _time_roll(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim == 1 and waveform.size(0) > 1:
            shift = random.randint(1, waveform.size(0))
            return torch.roll(waveform, shifts=shift, dims=0)
        elif waveform.ndim == 2 and waveform.size(1) > 1:
            shift = random.randint(1, waveform.size(1))
            return torch.roll(waveform, shifts=shift, dims=1)
        return waveform
    
    def _polarity_inversion(self, waveform: torch.Tensor) -> torch.Tensor:
        return waveform * -1
    
    def _random_amplification(self, waveform: torch.Tensor) -> torch.Tensor:
        if random.random() > 0.5:
            scalar = random.uniform(0.1, 1.0)
            return waveform * scalar
        else:
            if waveform.ndim == 1:
                vector = torch.rand(waveform.size(0))
                return waveform * vector
            else:
                vector = torch.rand(waveform.size(1))
                return waveform * vector.unsqueeze(0)
    
    def _apply_augmentations(self, waveform: torch.Tensor) -> torch.Tensor:
        augment_order = list(self.augmentations.keys())
        random.shuffle(augment_order)
        
        for aug_name in augment_order:
            if random.random() < self.aug_prob:
                waveform = self.augmentations[aug_name](waveform)
        
        return waveform
    
    def __len__(self):
        return len(self.chunks)
    
    def __getitem__(self, idx):
        chunk_info = self.chunks[idx]
        audio_path = chunk_info['audio_path']
        chunk_start = chunk_info['chunk_start']
        chunk_end = chunk_info['chunk_end']
        label = chunk_info['label']
        
        # Lazy file existence check
        if not os.path.exists(audio_path):
            # Return zero waveform with correct label
            return torch.zeros(self.target_size), label
        
        try:
            # Load full audio
            waveform_np, sr = sf.read(audio_path, dtype='float32')
            waveform = torch.from_numpy(waveform_np).unsqueeze(0)
            
            # Resample if needed
            if sr != self.target_sr:
                resampler = torchaudio.transforms.Resample(sr, self.target_sr)
                waveform = resampler(waveform)
            
            # Extract chunk
            start_sample = int(chunk_start * self.target_sr)
            end_sample = int(chunk_end * self.target_sr)
            chunk_waveform = waveform[:, start_sample:end_sample]
            
            # Pad or trim to target size
            if chunk_waveform.shape[1] < self.target_size:
                chunk_waveform = F.pad(chunk_waveform, (0, self.target_size - chunk_waveform.shape[1]))
            elif chunk_waveform.shape[1] > self.target_size:
                chunk_waveform = chunk_waveform[:, :self.target_size]
            
            # Apply augmentation if enabled
            if self.augmentation:
                chunk_waveform = self._apply_augmentations(chunk_waveform)
            
            return chunk_waveform.squeeze(0), label
        
        except Exception as e:
            # Return zeros on error
            return torch.zeros(self.target_size), label
            # Return zeros on error
            return torch.zeros(self.target_size), label


# =============================================================================
# KINESCAPER DETECTION DATASET (Detection mode)
# =============================================================================

class KineScaper_EV_DetectionDataset(Dataset):
    """
    KineScaper EV dataset with full 40s samples and temporal label tracks.
    
    Compatible with AudioSet_EV_Strong interface for sound event detection.
    """
    
    def __init__(self,
                 dataset_root: str,
                 metadata_format: str = "json",
                 window_size: float = 0.310,
                 target_sr: int = 32000,
                 target_duration: float = 40.0,
                 is_positive: bool = True,
                 label_value: int = 1,
                 augmentation: bool = False,
                 aug_prob: float = 0.7,
                 seed: int = 42):
        """
        Initialize KineScaper detection dataset.
        
        Args:
            dataset_root: Path to dataset root
            metadata_format: "json" or "tsv"
            window_size: Window duration for label tracks (seconds)
            target_sr: Target sample rate
            target_duration: Target audio duration (40s for full samples)
            is_positive: If True, load positives; if False, load negatives
            label_value: Value for positive windows in label track
            augmentation: Whether to apply augmentation
            aug_prob: Augmentation probability
            seed: Random seed
        """
        super().__init__()
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        self.dataset_root = dataset_root
        self.metadata_format = metadata_format
        self.window_size = window_size
        self.target_sr = target_sr
        self.target_duration = target_duration
        self.is_positive = is_positive
        self.label_value = label_value
        self.augmentation = augmentation
        self.aug_prob = aug_prob
        self.seed = seed
        
        self.num_windows = int(np.ceil(target_duration / window_size))
        self.target_size = int(target_sr * target_duration)
        
        if self.augmentation:
            self.augmentations = self._define_augmentations()
        
        # Load metadata
        self.samples = self._load_metadata()
        
        print(f"  Loaded {len(self.samples)} {'positive' if is_positive else 'negative'} samples")
    
    def _load_metadata(self) -> List[Dict]:
        """Load metadata and filter by is_positive flag."""
        if self.metadata_format == "json":
            json_path = os.path.join(self.dataset_root, "json", "metadata.json")
            with open(json_path, 'r') as f:
                data = json.load(f)
            df = pd.DataFrame(data['dataset_metadata'])
        else:
            tsv_path = os.path.join(self.dataset_root, "csv", "metadata.tsv")
            df = pd.read_csv(tsv_path, sep='\t')
        
        samples = []
        audio_dir = os.path.join(self.dataset_root, "audio")
        
        # Build sample list WITHOUT checking file existence (lazy verification)
        for _, row in df.iterrows():
            filename = row['filename']
            onset = row['onset']
            offset = row['offset']
            
            audio_path = os.path.join(audio_dir, filename)
            
            # For now, all KineScaper samples are positives (contain siren)
            # We can add negative samples from other datasets if needed
            if self.is_positive:
                samples.append({
                    'audio_path': audio_path,
                    'onset': onset,
                    'offset': offset,
                    'filename': filename
                })
        
        return samples
    
    def _create_label_track(self, onset: float, offset: float) -> torch.Tensor:
        """
        Create binary label track from onset/offset.
        
        Mark windows that overlap with [onset, offset] event.
        """
        label_track = torch.zeros(self.num_windows, dtype=torch.float32)
        
        onset_sample = int(onset * self.target_sr)
        offset_sample = int(offset * self.target_sr)
        window_samples = int(self.window_size * self.target_sr)
        
        for i in range(self.num_windows):
            window_start = i * window_samples
            window_end = (i + 1) * window_samples
            
            # Check if window overlaps with event
            if window_start < offset_sample and onset_sample < window_end:
                label_track[i] = self.label_value
        
        return label_track
    
    def _define_augmentations(self) -> dict:
        return {
            "add_noise": self._add_random_noise,
            "time_roll": self._time_roll,
            "polarity_inversion": self._polarity_inversion,
            "rand_amp_scaling": self._random_amplification,
        }
    
    def _add_random_noise(self, waveform: torch.Tensor, scale: float = 0.1) -> torch.Tensor:
        noise_type = random.choice(["white", "gaussian"])
        noise = torch.randn_like(waveform) if noise_type == "white" else torch.normal(0, 1, size=waveform.shape)
        noisy = waveform + noise * scale
        max_val = torch.max(torch.abs(noisy))
        if max_val > 0:
            return noisy / max_val
        return noisy
    
    def _time_roll(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim == 1 and waveform.size(0) > 1:
            shift = random.randint(1, waveform.size(0))
            return torch.roll(waveform, shifts=shift, dims=0)
        elif waveform.ndim == 2 and waveform.size(1) > 1:
            shift = random.randint(1, waveform.size(1))
            return torch.roll(waveform, shifts=shift, dims=1)
        return waveform
    
    def _polarity_inversion(self, waveform: torch.Tensor) -> torch.Tensor:
        return waveform * -1
    
    def _random_amplification(self, waveform: torch.Tensor) -> torch.Tensor:
        if random.random() > 0.5:
            scalar = random.uniform(0.1, 1.0)
            return waveform * scalar
        else:
            if waveform.ndim == 1:
                vector = torch.rand(waveform.size(0))
                return waveform * vector
            else:
                vector = torch.rand(waveform.size(1))
                return waveform * vector.unsqueeze(0)
    
    def _apply_augmentations(self, waveform: torch.Tensor) -> torch.Tensor:
        augment_order = list(self.augmentations.keys())
        random.shuffle(augment_order)
        
        for aug_name in augment_order:
            if random.random() < self.aug_prob:
                waveform = self.augmentations[aug_name](waveform)
        
        return waveform
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        audio_path = sample['audio_path']
        onset = sample['onset']
        offset = sample['offset']
        
        # Lazy file existence check
        if not os.path.exists(audio_path):
            return torch.zeros(self.target_size), torch.zeros(self.num_windows)
        
        try:
            # Load full audio
            waveform_np, sr = sf.read(audio_path, dtype='float32')
            waveform = torch.from_numpy(waveform_np).unsqueeze(0)
            
            # Resample if needed
            if sr != self.target_sr:
                resampler = torchaudio.transforms.Resample(sr, self.target_sr)
                waveform = resampler(waveform)
            
            # Pad or trim to target size
            if waveform.shape[1] < self.target_size:
                waveform = F.pad(waveform, (0, self.target_size - waveform.shape[1]))
            elif waveform.shape[1] > self.target_size:
                waveform = waveform[:, :self.target_size]
            
            # Apply augmentation if enabled
            if self.augmentation:
                waveform = self._apply_augmentations(waveform)
            
            # Create label track
            label_track = self._create_label_track(onset, offset)
            
            return waveform.squeeze(0), label_track
        
        except Exception as e:
            # Return zeros on error
            return torch.zeros(self.target_size), torch.zeros(self.num_windows)


# =============================================================================
# CUSTOM COLLATE FUNCTION
# =============================================================================

def kinescaper_collate_fn_with_augmentation(batch, negative_pool_manager, augmentation_ratio=0.69, label_type="binary"):
    """
    Custom collate function with batch-wise negative augmentation.
    
    Strategy:
    - Keep all positives as-is
    - For negatives: mix ~31% original + ~69% augmented
    
    Args:
        batch: List of (waveform, label) tuples
        negative_pool_manager: NegativePoolManager instance
        augmentation_ratio: Ratio of augmented negatives (default: 0.69)
        label_type: "binary" or "multiclass"
    """
    waveforms, labels = zip(*batch)
    
    # Identify positive and negative indices
    if label_type == "binary":
        pos_indices = [i for i, lbl in enumerate(labels) if lbl == 1]
        neg_indices = [i for i, lbl in enumerate(labels) if lbl == 0]
    else:  # multiclass
        pos_indices = [i for i, lbl in enumerate(labels) if lbl != 7]
        neg_indices = [i for i, lbl in enumerate(labels) if lbl == 7]
    
    # Collect all positives
    final_waveforms = [waveforms[i] for i in pos_indices]
    final_labels = [labels[i] for i in pos_indices]
    
    # Calculate negatives needed
    num_pos = len(pos_indices)
    num_neg_needed = num_pos  # 1:1 balance
    
    num_original = int(num_neg_needed * (1 - augmentation_ratio))
    num_augmented = num_neg_needed - num_original
    
    # Sample original negatives from batch
    if len(neg_indices) > 0:
        original_neg = random.sample(neg_indices, min(num_original, len(neg_indices)))
        final_waveforms.extend([waveforms[i] for i in original_neg])
        final_labels.extend([labels[i] for i in original_neg])
    
    # Fill remaining with augmented negatives from pool
    remaining = num_neg_needed - len(final_waveforms) + num_pos
    for _ in range(remaining):
        neg_waveform, neg_label = negative_pool_manager.sample_augmented()
        final_waveforms.append(neg_waveform)
        final_labels.append(neg_label if label_type == "binary" else 7)
    
    # Stack tensors
    waveforms_tensor = torch.stack(final_waveforms)
    labels_tensor = torch.tensor(final_labels, dtype=torch.long)
    
    # Shuffle batch
    perm = torch.randperm(len(labels_tensor))
    return waveforms_tensor[perm], labels_tensor[perm]


def simple_collate_fn(batch):
    """Simple collate function without augmentation."""
    waveforms, labels = zip(*batch)
    waveforms_tensor = torch.stack(waveforms)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    return waveforms_tensor, labels_tensor


def detection_collate_fn(batch):
    """Collate function for detection mode."""
    waveforms, label_tracks = zip(*batch)
    waveforms_tensor = torch.stack(waveforms)
    label_tracks_tensor = torch.stack(label_tracks)
    return waveforms_tensor, label_tracks_tensor


# =============================================================================
# LIGHTNING DATAMODULE
# =============================================================================

class KineScaper_EV_DataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for KineScaper EV dataset.
    
    Modes:
    - "train": train/val/test split (80/10/10) with balanced pos/neg
    - "benchmark": full dataset as test set with balanced pos/neg
    - "detection": full 40s samples with temporal label tracks
    """
    
    def __init__(self,
                 mode: str = "train",
                 label_type: str = "binary",
                 dataset_root: str = "/mnt/ssd/Kinescaper_EV/dataset/",
                 batch_size: int = 32,
                 num_workers: int = 4,
                 pin_memory: bool = True,
                 persistent_workers: bool = True,
                 split_ratios: List[float] = [0.8, 0.1, 0.1],
                 min_overlap: float = 0.5,
                 window_size: float = 0.310,
                 use_audioset_v2_negatives: bool = True,
                 use_other_negatives: bool = True,
                 augmentation: bool = False,
                 augmentation_ratio: float = 0.69,
                 seed: int = 42,
                 **kwargs):
        """
        Initialize KineScaper EV DataModule.
        
        Args:
            mode: "train", "benchmark", or "detection"
            label_type: "binary" or "multiclass"
            dataset_root: Path to dataset root
            batch_size: Batch size
            num_workers: Number of workers for data loading
            pin_memory: Whether to pin memory
            persistent_workers: Whether to keep workers alive
            split_ratios: [train, val, test] ratios (for "train" mode)
            min_overlap: Minimum overlap for positive label (seconds)
            window_size: Window duration for detection mode label tracks (seconds, default: 0.310)
            use_audioset_v2_negatives: Whether to use AudioSet_EV_v2 negatives
            use_other_negatives: Whether to use other dataset negatives
            augmentation: Whether to apply augmentation
            augmentation_ratio: Ratio of augmented negatives in batch
            seed: Random seed
        """
        super().__init__()
        
        self.mode = mode
        self.label_type = label_type
        self.dataset_root = dataset_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.split_ratios = split_ratios
        self.min_overlap = min_overlap
        self.window_size = window_size
        self.use_audioset_v2_negatives = use_audioset_v2_negatives
        self.use_other_negatives = use_other_negatives
        self.augmentation = augmentation
        self.augmentation_ratio = augmentation_ratio
        self.seed = seed
        
        seed_everything(seed)
        
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
        self.negative_pool_manager = None
    
    def setup(self, stage=None):
        """Setup datasets based on mode."""
        print(f"\n{'='*80}")
        print(f"Setting up KineScaper_EV DataModule (mode={self.mode}, label_type={self.label_type})")
        print(f"{'='*80}")
        
        if self.mode == "detection":
            self._setup_detection_mode()
        else:
            self._setup_classification_mode()
    
    def _setup_classification_mode(self):
        """Setup for train/benchmark modes with chunking."""
        print("\n1. Loading KineScaper chunks...")
        
        # Load full dataset
        full_dataset = KineScaper_EV_ChunkDataset(
            dataset_root=self.dataset_root,
            label_type=self.label_type,
            min_overlap=self.min_overlap,
            augmentation=False,  # Augmentation via collate_fn
            seed=self.seed
        )
        
        # Extract negative chunk info for pool manager
        kinescaper_negatives = []
        for chunk in full_dataset.chunks:
            if chunk['label'] == 0 or chunk['label'] == 7:  # negative
                kinescaper_negatives.append((chunk['audio_path'], 0))
        
        print(f"\n2. Initializing NegativePoolManager...")
        self.negative_pool_manager = NegativePoolManager(
            kinescaper_negatives=kinescaper_negatives,
            use_audioset_v2=self.use_audioset_v2_negatives,
            use_other_datasets=self.use_other_negatives,
            seed=self.seed
        )
        
        # Split dataset
        if self.mode == "train":
            print(f"\n3. Splitting dataset ({self.split_ratios[0]}/{self.split_ratios[1]}/{self.split_ratios[2]})...")
            
            train_size = int(self.split_ratios[0] * len(full_dataset))
            val_size = int(self.split_ratios[1] * len(full_dataset))
            test_size = len(full_dataset) - train_size - val_size
            
            generator = torch.Generator().manual_seed(self.seed)
            self.train_ds, self.val_ds, self.test_ds = random_split(
                full_dataset,
                [train_size, val_size, test_size],
                generator=generator
            )
            
            print(f"  Train: {len(self.train_ds):,} chunks")
            print(f"  Val:   {len(self.val_ds):,} chunks")
            print(f"  Test:  {len(self.test_ds):,} chunks")
        
        else:  # benchmark
            print(f"\n3. Using full dataset as test set...")
            self.test_ds = full_dataset
            print(f"  Test: {len(self.test_ds):,} chunks")
    
    def _setup_detection_mode(self):
        """Setup for detection mode with full samples."""
        print("\n1. Loading KineScaper full samples (detection mode)...")
        
        # Load positive samples
        self.test_ds = KineScaper_EV_DetectionDataset(
            dataset_root=self.dataset_root,
            window_size=self.window_size,
            target_duration=40.0,
            is_positive=True,
            augmentation=False,
            seed=self.seed
        )
        
        print(f"  Full samples: {len(self.test_ds):,}")
        print(f"  Window size: {self.window_size}s")
        print(f"  Num windows: {self.test_ds.num_windows}")
    
    def train_dataloader(self):
        """Return train dataloader."""
        if self.train_ds is None:
            return None
        
        # Create collate function with negative pool
        if self.negative_pool_manager is not None:
            collate_fn = lambda batch: kinescaper_collate_fn_with_augmentation(
                batch, self.negative_pool_manager, self.augmentation_ratio, self.label_type
            )
        else:
            collate_fn = simple_collate_fn
        
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
            collate_fn=collate_fn
        )
    
    def val_dataloader(self):
        """Return validation dataloader."""
        if self.val_ds is None:
            return None
        
        collate_fn = simple_collate_fn if self.mode != "detection" else detection_collate_fn
        
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
            collate_fn=collate_fn
        )
    
    def test_dataloader(self):
        """Return test dataloader."""
        if self.test_ds is None:
            return None
        
        collate_fn = simple_collate_fn if self.mode != "detection" else detection_collate_fn
        
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
            collate_fn=collate_fn
        )


# =============================================================================
# MAIN - Testing Script
# =============================================================================

if __name__ == "__main__":
    from collections import Counter
    
    print("=" * 80)
    print("DATALOADER TEST - KineScaper Emergency Vehicles Dataset")
    print("=" * 80)
    
    # Configuration
    DATASET_ROOT = "/mnt/ssd/Kinescaper_EV/dataset/"
    BATCH_SIZE = 16  # Reduced to avoid memory issues
    NUM_WORKERS = 0  # Start with 0 to avoid multiprocessing overhead
    
    # Test 1: Train mode - Binary classification
    print("\n" + "=" * 80)
    print("TEST 1: Train Mode - Binary Classification")
    print("=" * 80)
    
    dm_train_binary = KineScaper_EV_DataModule(
        mode="train",
        label_type="binary",
        dataset_root=DATASET_ROOT,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        split_ratios=[0.8, 0.1, 0.1],
        use_audioset_v2_negatives=True,
        use_other_negatives=True,
        augmentation_ratio=0.69,
        seed=42
    )
    
    dm_train_binary.setup()
    
    # Test train dataloader
    print("\nTesting train dataloader...")
    train_loader = dm_train_binary.train_dataloader()
    batch_waveforms, batch_labels = next(iter(train_loader))
    print(f"  Batch shape: {batch_waveforms.shape}")
    print(f"  Labels shape: {batch_labels.shape}")
    print(f"  Label distribution: {Counter(batch_labels.tolist())}")
    print(f"  Waveform range: [{batch_waveforms.min():.3f}, {batch_waveforms.max():.3f}]")
    
    # Test val dataloader
    print("\nTesting val dataloader...")
    val_loader = dm_train_binary.val_dataloader()
    batch_waveforms, batch_labels = next(iter(val_loader))
    print(f"  Batch shape: {batch_waveforms.shape}")
    print(f"  Labels shape: {batch_labels.shape}")
    print(f"  Label distribution: {Counter(batch_labels.tolist())}")
    
    # Test 2: Train mode - Multiclass classification
    print("\n" + "=" * 80)
    print("TEST 2: Train Mode - Multiclass Classification")
    print("=" * 80)
    
    dm_train_multiclass = KineScaper_EV_DataModule(
        mode="train",
        label_type="multiclass",
        dataset_root=DATASET_ROOT,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        split_ratios=[0.8, 0.1, 0.1],
        use_audioset_v2_negatives=True,
        use_other_negatives=True,
        augmentation_ratio=0.69,
        seed=42
    )
    
    dm_train_multiclass.setup()
    
    train_loader = dm_train_multiclass.train_dataloader()
    batch_waveforms, batch_labels = next(iter(train_loader))
    print(f"  Batch shape: {batch_waveforms.shape}")
    print(f"  Labels shape: {batch_labels.shape}")
    label_counts = Counter(batch_labels.tolist())
    print(f"  Label distribution:")
    for label_id in sorted(label_counts.keys()):
        class_name = SIREN_CLASS_NAMES[label_id]
        count = label_counts[label_id]
        print(f"    {label_id}: {class_name:<12} {count:>3}")
    
    # Test 3: Benchmark mode
    print("\n" + "=" * 80)
    print("TEST 3: Benchmark Mode - Binary Classification")
    print("=" * 80)
    
    dm_benchmark = KineScaper_EV_DataModule(
        mode="benchmark",
        label_type="binary",
        dataset_root=DATASET_ROOT,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        use_audioset_v2_negatives=True,
        use_other_negatives=True,
        seed=42
    )
    
    dm_benchmark.setup()
    
    test_loader = dm_benchmark.test_dataloader()
    batch_waveforms, batch_labels = next(iter(test_loader))
    print(f"  Batch shape: {batch_waveforms.shape}")
    print(f"  Labels shape: {batch_labels.shape}")
    print(f"  Label distribution: {Counter(batch_labels.tolist())}")
    
    # Test 4: Detection mode
    print("\n" + "=" * 80)
    print("TEST 4: Detection Mode")
    print("=" * 80)
    
    dm_detection = KineScaper_EV_DataModule(
        mode="detection",
        dataset_root=DATASET_ROOT,
        batch_size=8,  # Smaller batch for 40s samples
        num_workers=NUM_WORKERS,
        seed=42
    )
    
    dm_detection.setup()
    
    test_loader = dm_detection.test_dataloader()
    batch_waveforms, batch_label_tracks = next(iter(test_loader))
    print(f"  Batch waveforms shape: {batch_waveforms.shape} (should be [B, 1280000])")
    print(f"  Batch label tracks shape: {batch_label_tracks.shape}")
    print(f"  Num windows: {batch_label_tracks.shape[1]}")
    print(f"  Label track example (first sample):")
    print(f"    Positive windows: {torch.sum(batch_label_tracks[0] > 0).item()}")
    print(f"    Negative windows: {torch.sum(batch_label_tracks[0] == 0).item()}")
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    print("\nDataset composition (Train mode - Binary):")
    print(f"  Total train chunks: {len(dm_train_binary.train_ds):,}")
    print(f"  Total val chunks:   {len(dm_train_binary.val_ds):,}")
    print(f"  Total test chunks:  {len(dm_train_binary.test_ds):,}")
    
    print("\nNegative pool statistics:")
    if dm_train_binary.negative_pool_manager:
        print(f"  Total negatives available: {len(dm_train_binary.negative_pool_manager):,}")
        print(f"  Augmentation ratio: {dm_train_binary.augmentation_ratio:.1%}")
    
    print("\nDetection mode:")
    print(f"  Total samples: {len(dm_detection.test_ds):,}")
    print(f"  Sample duration: 40s")
    print(f"  Window size: 0.310s")
    print(f"  Windows per sample: {dm_detection.test_ds.num_windows}")
    
    print("\n" + "=" * 80)
    print("✓ ALL TESTS COMPLETED SUCCESSFULLY")
    print("=" * 80)
