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
from typing import List, Optional, Dict

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

class KineScaper_NegativeChunkGenerator:
    """
    Generate negative chunks from KineScaper_EV/Negatives/ urban traffic recordings.
    
    Features:
    - Load WAV files from Negatives/ directory (10 city recordings)
    - Extract overlapping 10s chunks
    - Apply augmentation to generate multiple versions
    - Lazy loading for memory efficiency
    
    Augmentation factor is calculated automatically to match the number
    of positives in the dataset while maintaining the ~5% negative ratio.
    """
    
    def __init__(self,
                 negatives_dir: str,
                 num_positives: int,
                 chunk_duration: float = 10.0,
                 overlap: float = 0.20,
                 target_sr: int = 32000,
                 augmentation_prob: float = 0.7,
                 seed: int = 42):
        """
        Initialize negative chunk generator.
        
        Args:
            negatives_dir: Path to Negatives/ directory with urban traffic WAV files
            num_positives: Number of positive samples (to calculate augmentation factor)
            chunk_duration: Duration of each chunk in seconds
            overlap: Overlap ratio between chunks (e.g., 0.20 = 20%)
            target_sr: Target sample rate for audio
            augmentation_prob: Probability of applying each augmentation
            seed: Random seed for reproducibility
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        self.negatives_dir = negatives_dir
        self.num_positives = num_positives
        self.chunk_duration = chunk_duration
        self.overlap = overlap
        self.target_sr = target_sr
        self.target_size = int(chunk_duration * target_sr)
        self.augmentation_prob = augmentation_prob
        self.seed = seed
        
        # Load negative files and extract chunk metadata
        self.negative_files = self._discover_negative_files()
        self.base_chunks = self._extract_chunk_metadata()
        
        # Calculate augmentation factor to match positive/negative ratio
        # Original ratio: 234,269 pos / 12,131 neg = ~19.3:1
        target_negatives = int(num_positives * 0.0518)  # 5.18% negatives
        self.augmentation_factor = max(1, int(np.ceil(target_negatives / len(self.base_chunks))))
        
        # Total chunks = base chunks × augmentation factor
        self.total_chunks = len(self.base_chunks) * self.augmentation_factor
        
        # Define augmentations
        self.augmentations = self._define_augmentations()
        
        print(f"  KineScaper Negative Generator initialized:")
        print(f"    Negative files: {len(self.negative_files)}")
        print(f"    Base chunks (overlap {overlap*100:.0f}%): {len(self.base_chunks):,}")
        print(f"    Augmentation factor: {self.augmentation_factor}x")
        print(f"    Total negative chunks: {self.total_chunks:,}")
        print(f"    Ratio pos/neg: {num_positives/self.total_chunks:.1f}:1")
    
    def _discover_negative_files(self) -> List[str]:
        """Discover all WAV files in Negatives/ directory."""
        wav_files = []
        if not os.path.exists(self.negatives_dir):
            print(f"Warning: Negatives directory not found: {self.negatives_dir}")
            return wav_files
        
        for filename in sorted(os.listdir(self.negatives_dir)):
            if filename.endswith('.wav'):
                filepath = os.path.join(self.negatives_dir, filename)
                wav_files.append(filepath)
        
        return wav_files
    
    def _extract_chunk_metadata(self) -> List[Dict]:
        """
        Extract chunk metadata from all negative files using sliding window.
        
        Returns:
            List of dicts with {'audio_path', 'start_time', 'end_time', 'chunk_idx'}
        """
        chunks = []
        stride = self.chunk_duration * (1 - self.overlap)
        
        for audio_path in self.negative_files:
            try:
                # Get file duration without loading entire file
                info = sf.info(audio_path)
                duration = info.duration
                
                # Extract overlapping chunks
                start_time = 0.0
                chunk_idx = 0
                
                while start_time + self.chunk_duration <= duration:
                    end_time = start_time + self.chunk_duration
                    
                    chunks.append({
                        'audio_path': audio_path,
                        'start_time': start_time,
                        'end_time': end_time,
                        'chunk_idx': chunk_idx,
                        'file_sr': info.samplerate
                    })
                    
                    start_time += stride
                    chunk_idx += 1
                    
            except Exception as e:
                print(f"Warning: Could not process {os.path.basename(audio_path)}: {e}")
        
        return chunks
    
    def _define_augmentations(self) -> dict:
        """Define augmentation functions (same as used for positives)."""
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
        if waveform.size(0) > 1:
            shift = random.randint(1, waveform.size(0))
            return torch.roll(waveform, shifts=shift, dims=0)
        return waveform
    
    def _polarity_inversion(self, waveform: torch.Tensor) -> torch.Tensor:
        return -waveform
    
    def _random_amplification(self, waveform: torch.Tensor, min_gain: float = 0.5, max_gain: float = 1.5) -> torch.Tensor:
        gain = random.uniform(min_gain, max_gain)
        amplified = waveform * gain
        max_val = torch.max(torch.abs(amplified))
        if max_val > 1.0:
            return amplified / max_val
        return amplified
    
    def _apply_augmentations(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply random augmentations to waveform."""
        augment_order = list(self.augmentations.keys())
        random.shuffle(augment_order)
        
        for aug_name in augment_order:
            if random.random() < self.augmentation_prob:
                waveform = self.augmentations[aug_name](waveform)
        
        return waveform
    
    def _load_chunk(self, chunk_metadata: Dict) -> torch.Tensor:
        """
        Load a single chunk from file with proper audio processing.
        
        Args:
            chunk_metadata: Dict with audio_path, start_time, end_time, file_sr
        
        Returns:
            1D tensor of shape [target_size]
        """
        audio_path = chunk_metadata['audio_path']
        start_time = chunk_metadata['start_time']
        end_time = chunk_metadata['end_time']
        file_sr = chunk_metadata['file_sr']
        
        try:
            # Calculate sample indices
            start_sample = int(start_time * file_sr)
            num_samples = int(self.chunk_duration * file_sr)
            
            # Load only the required chunk (memory efficient)
            waveform_np, sr = sf.read(audio_path, start=start_sample, frames=num_samples, dtype='float32')
           
            # Convert to torch
            if waveform_np.ndim == 1:
                waveform = torch.from_numpy(waveform_np).float()
            else:
                # Multi-channel: transpose and convert to mono
                waveform = torch.from_numpy(waveform_np.T).float()
                waveform = torch.mean(waveform, dim=0)
            
            # Resample if needed
            if sr != self.target_sr:
                resampler = torchaudio.transforms.Resample(sr, self.target_sr)
                waveform = resampler(waveform.unsqueeze(0)).squeeze(0)
            
            # Pad or trim to exact target size
            current_length = waveform.shape[0]
            if current_length < self.target_size:
                waveform = F.pad(waveform, (0, self.target_size - current_length))
            elif current_length > self.target_size:
                waveform = waveform[:self.target_size]
            
            # Final guarantee
            assert waveform.ndim == 1 and waveform.shape[0] == self.target_size
            
            return waveform
            
        except Exception as e:
            print(f"Error loading chunk from {os.path.basename(audio_path)}: {e}")
            return torch.zeros(self.target_size)
    
    def get_chunk(self, idx: int) -> torch.Tensor:
        """
        Get negative chunk by global index with augmentation.
        
        Args:
            idx: Global index (0 to total_chunks-1)
        
        Returns:
            1D tensor of shape [target_size]
        """
        # Map global index to base chunk and augmentation version
        base_idx = idx % len(self.base_chunks)
        aug_version = idx // len(self.base_chunks)
        
        # Load base chunk
        chunk_metadata = self.base_chunks[base_idx]
        waveform = self._load_chunk(chunk_metadata)
        
        # Apply augmentation for versions > 0
        if aug_version > 0:
            waveform = self._apply_augmentations(waveform)
        
        return waveform
    
    def __len__(self):
        return self.total_chunks




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
                 negative_overlap: float = 0.20,
                 use_negatives: bool = True,
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
            augmentation: Whether to apply augmentation on positives
            aug_prob: Augmentation probability
            negative_overlap: Overlap ratio for negative chunking (0.20 = 20%)
            use_negatives: Whether to include negatives from Negatives/ folder
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
        self.negative_overlap = negative_overlap
        self.use_negatives = use_negatives
        self.seed = seed
        
        if self.augmentation:
            self.augmentations = self._define_augmentations()
        
        # Load positive chunks from metadata
        positive_chunks = self._load_and_chunk_metadata()
        print(f"  Loaded {len(positive_chunks):,} positive chunks")
        
        # Initialize negative generator if enabled
        self.negative_generator = None
        negative_chunks = []
        
        if self.use_negatives:
            # Try to find Negatives directory:
            # 1. First in dataset_root (e.g., /mnt/ssd/KineScaper_EV/dataset/Negatives)
            # 2. If not found, try in repo (e.g., ./datasets/KineScaper_EV/Negatives)
            negatives_dir = os.path.join(self.dataset_root, "Negatives")
            
            if not os.path.exists(negatives_dir):
                # Fallback to repo location
                repo_negatives_dir = os.path.join(
                    os.path.dirname(os.path.dirname(__file__)),
                    "KineScaper_EV", "Negatives"
                )
                if os.path.exists(repo_negatives_dir):
                    negatives_dir = repo_negatives_dir
                    print(f"  Using Negatives from repository: {negatives_dir}")
            
            if os.path.exists(negatives_dir):
                self.negative_generator = KineScaper_NegativeChunkGenerator(
                    negatives_dir=negatives_dir,
                    num_positives=len(positive_chunks),
                    chunk_duration=chunk_duration,
                    overlap=negative_overlap,
                    target_sr=target_sr,
                    augmentation_prob=aug_prob,
                    seed=seed
                )
                
                # Add negative chunk metadata
                for neg_idx in range(len(self.negative_generator)):
                    negative_chunks.append({
                        'type': 'negative',
                        'generator_idx': neg_idx,
                        'label': 0 if label_type == "binary" else 7,
                        'siren_class': 'negative',
                        'overlap': 0.0
                    })
            else:
                print(f"  Warning: Negatives directory not found in:")
                print(f"    - Primary: {os.path.join(self.dataset_root, 'Negatives')}")
                print(f"    - Fallback: {repo_negatives_dir if 'repo_negatives_dir' in locals() else 'N/A'}")
        
        # Combine positive and negative chunks
        self.chunks = positive_chunks + negative_chunks
        
        print(f"  Total chunks: {len(self.chunks):,}")
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
        """
        Load metadata and create chunk entries.
        
        NOTE: This now loads ONLY POSITIVE chunks (with sirens).
        Negative chunks are handled by KineScaper_NegativeChunkGenerator.
        """
        df = self._load_metadata()
        positive_chunks = []
        
        audio_dir = os.path.join(self.dataset_root, "audio")
        
        for _, row in df.iterrows():
            filename = row['filename']
            onset = row['onset']
            offset = row['offset']
            siren_class = row['siren_class']
            
            audio_path = os.path.join(audio_dir, filename)
            
            # Create 4 chunks per file
            for chunk_idx in range(4):
                chunk_start = chunk_idx * self.chunk_duration
                chunk_end = (chunk_idx + 1) * self.chunk_duration
                
                # Calculate overlap
                overlap = calculate_overlap(chunk_start, chunk_end, onset, offset)
                
                # ONLY add chunks with sufficient overlap (positives)
                if overlap >= self.min_overlap:
                    if self.label_type == "binary":
                        label = 1
                    else:  # multiclass
                        label = SIREN_CLASS_MAPPING[siren_class]
                    
                    positive_chunks.append({
                        'type': 'positive',  # Mark as positive chunk
                        'audio_path': audio_path,
                        'chunk_start': chunk_start,
                        'chunk_end': chunk_end,
                        'label': label,
                        'siren_class': siren_class,
                        'overlap': overlap
                    })
        
        return positive_chunks
    
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
        chunk_type = chunk_info['type']
        label = chunk_info['label']
        
        # Handle negative chunks from generator
        if chunk_type == 'negative' and self.negative_generator is not None:
            generator_idx = chunk_info['generator_idx']
            waveform = self.negative_generator.get_chunk(generator_idx)
            return waveform, label
        
        # Handle positive chunks from audio files
        elif chunk_type == 'positive':
            audio_path = chunk_info['audio_path']
            chunk_start = chunk_info['chunk_start']
            chunk_end = chunk_info['chunk_end']
            
            # Lazy file existence check
            if not os.path.exists(audio_path):
                return torch.zeros(self.target_size), label
            
            try:
                # Load full audio
                waveform_np, sr = sf.read(audio_path, dtype='float32')
                
                # Convert numpy to torch - handle both mono and stereo correctly
                if waveform_np.ndim == 1:
                    waveform = torch.from_numpy(waveform_np).float()
                else:
                    waveform = torch.from_numpy(waveform_np.T).float()
                
                # Convert to mono if multi-channel
                if waveform.ndim == 2:
                    waveform = torch.mean(waveform, dim=0)
                
                # At this point, waveform is ALWAYS 1D: [samples]
                assert waveform.ndim == 1, f"Expected 1D tensor, got shape {waveform.shape}"
                
                # Resample if needed
                if sr != self.target_sr:
                    resampler = torchaudio.transforms.Resample(sr, self.target_sr)
                    waveform = resampler(waveform.unsqueeze(0)).squeeze(0)
                
                # Extract chunk - waveform is 1D [samples]
                start_sample = int(chunk_start * self.target_sr)
                end_sample = int(chunk_end * self.target_sr)
                chunk_waveform = waveform[start_sample:end_sample]
                
                # Pad or trim to target size
                current_length = chunk_waveform.shape[0]
                if current_length < self.target_size:
                    chunk_waveform = F.pad(chunk_waveform, (0, self.target_size - current_length))
                elif current_length > self.target_size:
                    chunk_waveform = chunk_waveform[:self.target_size]
                
                # Apply augmentation if enabled (only for positives)
                if self.augmentation:
                    chunk_waveform = self._apply_augmentations(chunk_waveform)
                
                # Final guarantee
                assert chunk_waveform.ndim == 1 and chunk_waveform.shape[0] == self.target_size
                
                return chunk_waveform, label
            
            except Exception as e:
                return torch.zeros(self.target_size), label
        
        # Fallback for unknown type
        else:
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
        """Load metadata and filter by is_positive flag.
        
        Returns samples in a format compatible with AudioSet_EV_Strong interface:
            {
                'segment_id': str,
                'file_path': str,
                'audio_path': str,   # same as file_path (for backward compat)
                'onset': float,
                'offset': float,
                'filename': str,
                'events': [{'mid': 'kinescaper_ev', 'start': onset, 'end': offset}]
            }
        """
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
            siren_class = row.get('siren_class', 'unknown')
            
            audio_path = os.path.join(audio_dir, filename)
            segment_id = os.path.splitext(filename)[0]  # filename without extension
            
            # For now, all KineScaper samples are positives (contain siren)
            # We can add negative samples from other datasets if needed
            if self.is_positive:
                samples.append({
                    # AudioSet Strong compatible interface
                    'segment_id': segment_id,
                    'file_path': audio_path,
                    'events': [{'mid': 'kinescaper_ev', 'start': onset, 'end': offset}],
                    # KineScaper-specific (kept for backward compatibility)
                    'audio_path': audio_path,
                    'onset': onset,
                    'offset': offset,
                    'filename': filename,
                    'siren_class': siren_class
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
            
            # Convert numpy to torch - handle both mono and stereo correctly
            # soundfile returns: mono=[samples], stereo=[samples, channels]
            if waveform_np.ndim == 1:
                # Mono: [samples]
                waveform = torch.from_numpy(waveform_np).float()
            else:
                # Multi-channel: [samples, channels] -> transpose to [channels, samples]
                waveform = torch.from_numpy(waveform_np.T).float()
            
            # Convert to mono if multi-channel
            if waveform.ndim == 2:
                # waveform is [channels, samples] -> average channels to get [samples]
                waveform = torch.mean(waveform, dim=0)  # NO keepdim! Result: [samples]
            
            # At this point, waveform is ALWAYS 1D: [samples]
            assert waveform.ndim == 1, f"Expected 1D tensor, got shape {waveform.shape}"
            
            # Resample if needed
            if sr != self.target_sr:
                resampler = torchaudio.transforms.Resample(sr, self.target_sr)
                waveform = resampler(waveform.unsqueeze(0)).squeeze(0)  # Add/remove channel dim
            
            # Pad or trim to target size - waveform is 1D [samples]
            current_length = waveform.shape[0]
            if current_length < self.target_size:
                waveform = F.pad(waveform, (0, self.target_size - current_length))
            elif current_length > self.target_size:
                waveform = waveform[:self.target_size]
            
            # Apply augmentation if enabled
            if self.augmentation:
                waveform = self._apply_augmentations(waveform)
            
            # Create label track
            label_track = self._create_label_track(onset, offset)
            
            # Final guarantee: ALWAYS return 1D tensor of exact target size
            assert waveform.ndim == 1 and waveform.shape[0] == self.target_size, \
                f"Expected shape [{self.target_size}], got {waveform.shape}"
            
            return waveform, label_track
        
        except Exception as e:
            # Return zeros on error
            return torch.zeros(self.target_size), torch.zeros(self.num_windows)


# =============================================================================
# CUSTOM COLLATE FUNCTION
# =============================================================================

def kinescaper_collate_fn_balanced(batch, negative_generator=None, label_type="binary"):
    """
    Collate function with guaranteed 50/50 balancing and fixed batch size.
    
    Strategy:
    - Target: EXACTLY batch_size samples
    - Balance: 50% positives / 50% negatives
    - If batch lacks negatives, sample from negative_generator
    
    Args:
        batch: List of (waveform, label) tuples from DataLoader
        negative_generator: KineScaper_NegativeChunkGenerator instance (optional)
        label_type: "binary" or "multiclass"
    
    Returns:
        (waveforms_tensor, labels_tensor) with shape (batch_size, samples)
    """
    waveforms, labels = zip(*batch)
    batch_size = len(batch)
    
    # Identify positive and negative indices
    if label_type == "binary":
        pos_indices = [i for i, lbl in enumerate(labels) if lbl == 1]
        neg_indices = [i for i, lbl in enumerate(labels) if lbl == 0]
    else:  # multiclass
        pos_indices = [i for i, lbl in enumerate(labels) if lbl != 7]
        neg_indices = [i for i, lbl in enumerate(labels) if lbl == 7]
    
    # Target: batch_size total samples, balanced 50/50
    target_pos = batch_size // 2
    target_neg = batch_size - target_pos
    
    # Sample positives
    if len(pos_indices) >= target_pos:
        selected_pos = random.sample(pos_indices, target_pos)
    else:
        # Take all available positives
        selected_pos = pos_indices
        # Adjust negative target to fill batch
        target_neg = batch_size - len(selected_pos)
    
    # Build positive part of batch
    final_waveforms = [waveforms[i] for i in selected_pos]
    final_labels = [labels[i] for i in selected_pos]
    
    # Sample negatives from batch
    if len(neg_indices) >= target_neg:
        selected_neg = random.sample(neg_indices, target_neg)
        final_waveforms.extend([waveforms[i] for i in selected_neg])
        final_labels.extend([labels[i] for i in selected_neg])
    else:
        # Take all available negatives from batch
        final_waveforms.extend([waveforms[i] for i in neg_indices])
        final_labels.extend([labels[i] for i in neg_indices])
        
        # Fill remaining with negatives from generator
        remaining_neg = target_neg - len(neg_indices)
        if remaining_neg > 0 and negative_generator is not None:
            for _ in range(remaining_neg):
                # Sample random negative from generator
                neg_idx = random.randint(0, len(negative_generator) - 1)
                neg_waveform = negative_generator.get_chunk(neg_idx)
                neg_label = 0 if label_type == "binary" else 7
                
                final_waveforms.append(neg_waveform)
                final_labels.append(neg_label)
    
    # Guarantee: batch_size maintained
    assert len(final_waveforms) == batch_size, \
        f"Expected {batch_size} samples, got {len(final_waveforms)}"
    
    # Stack tensors
    waveforms_tensor = torch.stack(final_waveforms)
    labels_tensor = torch.tensor(final_labels, dtype=torch.long)
    
    # Shuffle batch
    perm = torch.randperm(len(labels_tensor))
    return waveforms_tensor[perm], labels_tensor[perm]


# DEPRECATED: Old collate function with NegativePoolManager (kept for reference)
def kinescaper_collate_fn_with_augmentation_DEPRECATED(batch, negative_pool_manager, augmentation_ratio=0.69, label_type="binary"):
    """DEPRECATED: Use kinescaper_collate_fn_balanced instead."""
    raise NotImplementedError("This function is deprecated. Use kinescaper_collate_fn_balanced instead.")


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
                 augmentation: bool = False,
                 negative_overlap: float = 0.20,
                 use_negatives: bool = True,
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
            augmentation: Whether to apply augmentation on positives
            negative_overlap: Overlap ratio for negative chunking (0.20 = 20%)
            use_negatives: Whether to use negatives from Negatives/ folder
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
        self.augmentation = augmentation
        self.negative_overlap = negative_overlap
        self.use_negatives = use_negatives
        self.seed = seed
        
        seed_everything(seed)
        
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
        self.test_datasets = {}  # For benchmark mode CV (siren-type folds)
    
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
        
        # Load full dataset with integrated negative generation
        full_dataset = KineScaper_EV_ChunkDataset(
            dataset_root=self.dataset_root,
            label_type=self.label_type,
            min_overlap=self.min_overlap,
            augmentation=self.augmentation,
            negative_overlap=self.negative_overlap,
            use_negatives=self.use_negatives,
            seed=self.seed
        )
        
        # Split dataset
        if self.mode == "train":
            print(f"\n2. Splitting dataset ({self.split_ratios[0]}/{self.split_ratios[1]}/{self.split_ratios[2]})...")
            
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
            print(f"\n2. Creating CV folds by siren type...")
            
            # Group chunks by siren class
            siren_types = ['hi-lo', 'two-tone', 'wail', 'phaser', 'piercer', 'rumbler', 'yelp']
            
            # Collect indices for negatives (shared across all folds)
            negative_indices = [i for i, chunk in enumerate(full_dataset.chunks) 
                              if chunk['label'] == 0 or chunk['label'] == 7]
            
            # Create fold for each siren type
            for siren_type in siren_types:
                # Collect indices for this siren type (positives only)
                if self.label_type == 'binary':
                    positive_indices = [i for i, chunk in enumerate(full_dataset.chunks)
                                      if chunk['siren_class'] == siren_type and chunk['label'] == 1]
                else:  # multiclass
                    siren_label = SIREN_CLASS_MAPPING[siren_type]
                    positive_indices = [i for i, chunk in enumerate(full_dataset.chunks)
                                      if chunk['label'] == siren_label]
                
                # Combine positives of this type + all negatives
                fold_indices = positive_indices + negative_indices
                
                # Create Subset for this fold
                from torch.utils.data import Subset
                fold_dataset = Subset(full_dataset, fold_indices)
                
                self.test_datasets[siren_type] = fold_dataset
                
                pos_count = len(positive_indices)
                neg_count = len(negative_indices)
                total = pos_count + neg_count
                print(f"  {siren_type:12} fold: {pos_count:>7,} pos + {neg_count:>7,} neg = {total:>7,} chunks")
            
            print(f"\n  Total folds: {len(self.test_datasets)}")
            print(f"  Mode: Cross-Validation by siren type")
    
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
        
        # Create collate function with balancing
        if self.mode == "train":
            # Access negative_generator from underlying dataset
            # train_ds is a Subset, so we need to access .dataset
            negative_generator = None
            if hasattr(self.train_ds, 'dataset'):
                # train_ds is a Subset
                underlying_dataset = self.train_ds.dataset
                if hasattr(underlying_dataset, 'negative_generator'):
                    negative_generator = underlying_dataset.negative_generator
            elif hasattr(self.train_ds, 'negative_generator'):
                # train_ds is the dataset directly
                negative_generator = self.train_ds.negative_generator
            
            collate_fn = lambda batch: kinescaper_collate_fn_balanced(
                batch, negative_generator, self.label_type
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
    print("DATALOADER TEST - KineScaper Emergency Vehicles Dataset (REFACTORED)")
    print("=" * 80)
    
    # Configuration
    DATASET_ROOT = "/mnt/ssd/KineScaper_EV/dataset/"
    BATCH_SIZE = 16
    NUM_WORKERS = 0  # Single process for testing
    
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
        augmentation=True,          # Apply augmentation to positives
        negative_overlap=0.20,       # 20% overlap for negative chunking
        use_negatives=True,          # Use negatives from Negatives/ folder
        seed=42
    )
    
    dm_train_binary.setup()
    
    # Test train dataloader
    print("\nTesting train dataloader...")
    train_loader = dm_train_binary.train_dataloader()
    
    # Test multiple batches to verify balancing
    print("  Testing 5 batches to verify balancing:")
    for i, (batch_waveforms, batch_labels) in enumerate(train_loader):
        if i >= 5:
            break
        pos_count = (batch_labels == 1).sum().item()
        neg_count = (batch_labels == 0).sum().item()
        print(f"    Batch {i}: shape {list(batch_waveforms.shape)}, "
              f"{pos_count} pos + {neg_count} neg = {batch_waveforms.shape[0]} total, "
              f"range [{batch_waveforms.min():.3f}, {batch_waveforms.max():.3f}]")
        
        # Verify batch size is maintained
        assert batch_waveforms.shape[0] == BATCH_SIZE, \
            f"Batch size mismatch: expected {BATCH_SIZE}, got {batch_waveforms.shape[0]}"
    
    # Test val dataloader
    print("\nTesting val dataloader...")
    val_loader = dm_train_binary.val_dataloader()
    batch_waveforms, batch_labels = next(iter(val_loader))
    print(f"  Batch shape: {batch_waveforms.shape}")
    print(f"  Labels shape: {batch_labels.shape}")
    print(f"  Label distribution: {Counter(batch_labels.tolist())}")
    print(f"  (Val uses real distribution, not balanced)")
    
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
        augmentation=True,
        negative_overlap=0.20,
        use_negatives=True,
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
    
    # Test 3: Benchmark mode - CV by siren type
    print("\n" + "=" * 80)
    print("TEST 3: Benchmark Mode - Binary Classification (CV by siren type)")
    print("=" * 80)
    
    dm_benchmark = KineScaper_EV_DataModule(
        mode="benchmark",
        label_type="binary",
        dataset_root=DATASET_ROOT,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        use_negatives=True,  # Included in benchmark
        seed=42
    )
    
    dm_benchmark.setup()
    
    # Check CV structure
    print(f"\nCV Folds: {len(dm_benchmark.test_datasets)}")
    print(f"Fold names: {list(dm_benchmark.test_datasets.keys())}")
    
    # Test first fold
    if dm_benchmark.test_datasets:
        first_fold_name = list(dm_benchmark.test_datasets.keys())[0]
        first_fold_dataset = dm_benchmark.test_datasets[first_fold_name]
        
        print(f"\nTesting first fold: {first_fold_name}")
        print(f"  Fold size: {len(first_fold_dataset):,} chunks")
        
        # Create dataloader for this fold
        from torch.utils.data import DataLoader
        fold_loader = DataLoader(first_fold_dataset, 
                                batch_size=BATCH_SIZE, 
                                shuffle=False,
                                collate_fn=simple_collate_fn)
        
        batch_waveforms, batch_labels = next(iter(fold_loader))
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
    
    # Access underlying dataset to get statistics
    underlying_dataset = dm_train_binary.train_ds.dataset
    pos_chunks = sum(1 for c in underlying_dataset.chunks if c['type'] == 'positive')
    neg_chunks = sum(1 for c in underlying_dataset.chunks if c['type'] == 'negative')
    
    print(f"\nFull dataset composition:")
    print(f"  Positive chunks: {pos_chunks:,} ({pos_chunks/len(underlying_dataset.chunks)*100:.1f}%)")
    print(f"  Negative chunks: {neg_chunks:,} ({neg_chunks/len(underlying_dataset.chunks)*100:.1f}%)")
    print(f"  Ratio pos:neg = {pos_chunks/neg_chunks if neg_chunks > 0 else 'inf'}:1")
    
    if hasattr(underlying_dataset, 'negative_generator') and underlying_dataset.negative_generator:
        gen = underlying_dataset.negative_generator
        print(f"\nNegative Generator statistics:")
        print(f"  Negative files: {len(gen.negative_files)}")
        print(f"  Base chunks (overlap {gen.overlap*100:.0f}%): {len(gen.base_chunks):,}")
        print(f"  Augmentation factor: {gen.augmentation_factor}x (auto-calculated)")
        print(f"  Total negative chunks generated: {len(gen):,}")
    
    print(f"\nDetection mode:")
    print(f"  Total samples: {len(dm_detection.test_ds):,}")
    print(f"  Sample duration: 40s")
    print(f"  Window size: {dm_detection.window_size}s")
    print(f"  Windows per sample: {dm_detection.test_ds.num_windows}")
    
    print("\n" + "=" * 80)
    print("✓ ALL TESTS COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print("\nKEY IMPROVEMENTS IN REFACTORED VERSION:")
    print("  - Negatives standalone from Negatives/ folder (no external datasets)")
    print("  - Automatic augmentation factor calculation")
    print("  - Guaranteed balanced batches (50/50 pos/neg) in training")
    print("  - Simplified architecture without NegativePoolManager")
    print("  - Lazy loading for memory efficiency")
    print("=" * 80)

# =============================================================================
# STANDALONE DATASET FOR UNIFIED TRAINING
# =============================================================================

def get_stratified_sample_indices(metadata: List[Dict], 
                                   num_samples_per_class: int,
                                   seed: int = 42) -> List[int]:
    """
    Get stratified sample indices balanced across siren classes.
    
    Args:
        metadata: List of metadata entries (dataset_metadata from JSON)
        num_samples_per_class: Number of samples to get per siren class
        seed: Random seed for reproducibility
    
    Returns:
        List of sample indices (into metadata list)
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Group metadata indices by siren class
    class_indices = {class_name: [] for class_name in SIREN_CLASS_NAMES[:-1]}  # Exclude 'negative'
    
    for idx, entry in enumerate(metadata):
        siren_class = entry['siren_class']
        if siren_class in class_indices:
            class_indices[siren_class].append(idx)
    
    # Sample from each class
    sampled_indices = []
    for siren_class, indices in class_indices.items():
        if len(indices) < num_samples_per_class:
            # If not enough samples, use all and repeat
            sampled = random.choices(indices, k=num_samples_per_class)
        else:
            # Sample without replacement
            sampled = random.sample(indices, num_samples_per_class)
        
        sampled_indices.extend(sampled)
    
    # Shuffle to mix classes
    random.shuffle(sampled_indices)
    
    return sampled_indices


class KineScaper_PositiveChunkDataset(Dataset):
    """
    Standalone dataset for KineScaper positive chunks.
    Used for unified training with stratified  re-sampling.
    
    Features:
    - Loads only positive chunks (overlap >= 0.5s)
    - Stratified sampling by siren class
    - Re-sampling via set_epoch() for epoch-to-epoch diversity
    - No balancing (handled externally by unified training)
    """
    
    def __init__(self,
                 dataset_root: str,
                 num_samples: int,
                 augmentation: bool = False,
                 aug_prob: float = 0.5,
                 target_sr: int = 32000,
                 target_duration: float = 10.0,
                 seed: int = 42):
        """
        Args:
            dataset_root: Path to KineScaper dataset root
            num_samples: Number of positive samples to use (will be stratified)
            augmentation: Enable augmentation
            aug_prob: Augmentation probability per technique
            target_sr: Target sample rate
            target_duration: Target chunk duration in seconds
            seed: Random seed
        """
        super().__init__()
        
        self.dataset_root = dataset_root
        self.num_samples = num_samples
        self.augmentation = augmentation
        self.aug_prob = aug_prob
        self.target_sr = target_sr
        self.target_duration = target_duration
        self.target_length = int(target_duration * target_sr)
        self.seed = seed
        self.current_epoch = 0
        
        # Load metadata
        metadata_path = os.path.join(dataset_root, 'json', 'metadata.json')
        with open(metadata_path, 'r') as f:
            data = json.load(f)
            self.full_metadata = data['dataset_metadata']
        
        # Extract all positive chunks (overlap >= 0.5s)
        self.positive_chunks = []
        for entry in self.full_metadata:
            onset = entry['onset']
            offset = entry['offset']
            siren_class = entry['siren_class']
            filename = entry['filename']
            audio_path = os.path.join(dataset_root, 'audio', filename)
            
            # Check 4 chunks
            for chunk_idx in range(4):
                chunk_start = chunk_idx * 10.0
                chunk_end = (chunk_idx + 1) * 10.0
                
                overlap = calculate_overlap(chunk_start, chunk_end, onset, offset)
                
                if overlap >= 0.5:
                    self.positive_chunks.append({
                        'chunk_idx': chunk_idx,
                        'audio_path': audio_path,
                        'siren_class': siren_class,
                        'onset': onset,
                        'offset': offset
                    })
        
        # Initial sampling
        self._resample()
    
    def _resample(self):
        """Re-sample indices with stratification."""
        # Calculate samples per class
        num_classes = len(SIREN_CLASS_NAMES) - 1  # exclude negative
        samples_per_class = self.num_samples // num_classes
        
        # Group by siren class
        class_chunks = {class_name: [] for class_name in SIREN_CLASS_NAMES[:-1]}
        for idx, chunk in enumerate(self.positive_chunks):
            class_chunks[chunk['siren_class']].append(idx)
        
        # Sample from each class
        self.sampled_indices = []
        rng = random.Random(self.seed + self.current_epoch)
        
        for siren_class, indices in class_chunks.items():
            if len(indices) < samples_per_class:
                sampled = rng.choices(indices, k=samples_per_class)
            else:
                sampled = rng.sample(indices, samples_per_class)
            self.sampled_indices.extend(sampled)
        
        # Shuffle
        rng.shuffle(self.sampled_indices)
    
    def set_epoch(self, epoch: int):
        """Set current epoch for re-sampling."""
        self.current_epoch = epoch
        self._resample()
    
    def __len__(self):
        return len(self.sampled_indices)
    
    def __getitem__(self, idx):
        """Load and return chunk."""
        chunk_idx = self.sampled_indices[idx]
        chunk_info = self.positive_chunks[chunk_idx]
        
        # Load audio
        try:
            waveform, sr = sf.read(chunk_info['audio_path'])
            if waveform.ndim > 1:
                waveform = waveform[:, 0]
            waveform = torch.from_numpy(waveform).float()
            
            # Resample if needed
            if sr != self.target_sr:
                resampler = torchaudio.transforms.Resample(sr, self.target_sr)
                waveform = resampler(waveform)
            
            # Extract chunk
            chunk_start_sample = int(chunk_info['chunk_idx'] * 10.0 * self.target_sr)
            chunk_end_sample = chunk_start_sample + self.target_length
            waveform = waveform[chunk_start_sample:chunk_end_sample]
            
            # Pad if needed
            if len(waveform) < self.target_length:
                waveform = F.pad(waveform, (0, self.target_length - len(waveform)))
            
            waveform = waveform.unsqueeze(0)  # [1, samples]
            
            # Apply augmentation if enabled
            if self.augmentation and random.random() < self.aug_prob:
                waveform = self._apply_augmentations(waveform)
            
            # Binary label = 1 (positive)
            label = torch.tensor(1, dtype=torch.long)
            
            return waveform, label
            
        except Exception as e:
            print(f"Error loading {chunk_info['audio_path']}: {e}")
            # Return zero tensor as fallback
            return torch.zeros(1, self.target_length), torch.tensor(1, dtype=torch.long)
    
    def _apply_augmentations(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply random augmentations."""
        # Simple augmentations (same as other datasets)
        augmentations = []
        
        # Add noise
        if random.random() < self.aug_prob:
            noise = torch.randn_like(waveform) * 0.005
            waveform = waveform + noise
        
        # Time roll
        if random.random() < self.aug_prob:
            shift = random.randint(1, waveform.size(1) // 4)
            waveform = torch.roll(waveform, shifts=shift, dims=1)
        
        # Polarity inversion
        if random.random() < self.aug_prob:
            waveform = waveform * -1
        
        # Random amplification
        if random.random() < self.aug_prob:
            scalar = random.uniform(0.7, 1.3)
            waveform = waveform * scalar
        
        # Normalize
        max_val = torch.max(torch.abs(waveform))
        if max_val > 0:
            waveform = waveform / max_val
        
        return waveform


class KineScaper_NegativeChunkDataset(Dataset):
    """
    Standalone dataset for KineScaper negative chunks.
    Uses lazy loading from KineScaper_NegativeChunkGenerator.
    """
    
    def __init__(self,
                 dataset_root: str,
                 num_positives: int,  # Added: needed for augmentation factor calculation
                 augmentation: bool = False,
                 aug_prob: float = 0.5,
                 target_sr: int = 32000,
                 target_duration: float = 10.0,
                 negative_overlap: float = 0.20,
                 seed: int = 42):
        """
        Args:
            dataset_root: Path to KineScaper dataset root  
            num_positives: Number of positive samples (for auto augmentation factor)
            augmentation: Enable augmentation (NOTE: negatives already augmented by generator)
            aug_prob: Augmentation probability  
            target_sr: Target sample rate
            target_duration: Target chunk duration
            negative_overlap: Overlap for negative chunking
            seed: Random seed
        """
        super().__init__()
        
        # Initialize generator for negatives
        negatives_dir = os.path.join(os.path.dirname(__file__), 'Negatives')
        
        # Generator will calculate augmentation factor based on num_positives
        # BUT we want to use ALL negatives available (max augmentation)
        # Hack: use a very large num_positives to force max augmentation
        # This ensures we get all 12,180 negatives (1,218 base × 10 aug factor)
        self.generator = KineScaper_NegativeChunkGenerator(
            negatives_dir=negatives_dir,
            num_positives=250000,  # Large value → forces augmentation_factor=10x → 12,180 total
            chunk_duration=target_duration,
            overlap=negative_overlap,
            target_sr=target_sr,
            seed=seed
        )
        
        # Augmentation settings (optional, negatives already augmented)
        self.augmentation = augmentation
        self.aug_prob = aug_prob
        self.target_length = int(target_duration * target_sr)
    
    def __len__(self):
        return self.generator.total_chunks
    
    def __getitem__(self, idx):
        """Load negative chunk via generator."""
        waveform = self.generator.get_chunk(idx)
        
        # Additional augmentation if requested (on top of pre-augmented)
        if self.augmentation and random.random() < self.aug_prob:
            waveform = self._apply_augmentations(waveform)
        
        # Binary label = 0 (negative)
        label = torch.tensor(0, dtype=torch.long)
        
        return waveform, label
    
    def _apply_augmentations(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply additional augmentations (same as positives)."""
        if random.random() < self.aug_prob:
            noise = torch.randn_like(waveform) * 0.005
            waveform = waveform + noise
        
        if random.random() < self.aug_prob:
            shift = random.randint(1, waveform.size(1) // 4)
            waveform = torch.roll(waveform, shifts=shift, dims=1)
        
        if random.random() < self.aug_prob:
            waveform = waveform * -1
        
        if random.random() < self.aug_prob:
            scalar = random.uniform(0.7, 1.3)
            waveform = waveform * scalar
        
        max_val = torch.max(torch.abs(waveform))
        if max_val > 0:
            waveform = waveform / max_val
        
        return waveform


def create_kinescaper_dataset_for_unified(dataset_root: str,
                                           num_positive_samples: int,
                                           chunk_type: str,
                                           augmentation: bool = False,
                                           aug_prob: float = 0.5,
                                           target_sr: int = 32000,
                                           target_duration: float = 10.0,
                                           negative_overlap: float = 0.20,
                                           seed: int = 42) -> Dataset:
    """
    Factory function to create KineScaper dataset for unified training.
    
    Args:
        dataset_root: Path to KineScaper dataset
        num_positive_samples: Number of positive samples (for stratified sampling AND negative augmentation factor)
        chunk_type: 'positive' or 'negative'
        augmentation: Enable augmentation
        aug_prob: Augmentation probability
        target_sr: Target sample rate
        target_duration: Chunk duration in seconds
        negative_overlap: Overlap for negative chunking
        seed: Random seed
    
    Returns:
        Dataset instance (either positive or negative)
    """
    if chunk_type == 'positive':
        return KineScaper_PositiveChunkDataset(
            dataset_root=dataset_root,
            num_samples=num_positive_samples,
            augmentation=augmentation,
            aug_prob=aug_prob,
            target_sr=target_sr,
            target_duration=target_duration,
            seed=seed
        )
    elif chunk_type == 'negative':
        return KineScaper_NegativeChunkDataset(
            dataset_root=dataset_root,
            num_positives=num_positive_samples,  # Pass for augmentation factor calculation
            augmentation=augmentation,
            aug_prob=aug_prob,
            target_sr=target_sr,
            target_duration=target_duration,
            negative_overlap=negative_overlap,
            seed=seed
        )
    else:
        raise ValueError(f"chunk_type must be 'positive' or 'negative', got: {chunk_type}")