"""
AudioSet Emergency Vehicles Strong Dataset Dataloader
======================================================
PyTorch Dataset and Lightning DataModule for AudioSet EV Strong with temporal annotations.

This dataloader supports Sound Event Detection (SED) with strong temporal labels:
- Uses AudioSet Strong metadata (audioset_train_strong.tsv, audioset_eval_strong.tsv)
- Compatible with two audio file sources:
  * AudioSet_EV_v1_Strong: Uses metadata + audio files from AudioSet_EV_v1_2025
  * AudioSet_EV_v2_Strong: Uses metadata + audio files from AudioSet_EV_v2PANNs_2020

Features:
- Window-based label tracks for temporal event detection
- Configurable window size (default: 310ms)
- Train mode: balanced train/dev/test split
- Detection mode: balanced full dataset as test set
- Data augmentation support
- File tracking and logging

Output format:
- Waveform: (1, num_samples) - Full audio segment (default: 10s @ 32kHz)
- Label track: (num_windows,) - Binary labels per time window

Author: Stefano Giacomelli - Ph.D. candidate in ICT (DISIM dpt. - University of L'Aquila)
"""

import os
import sys
import json
import random
import csv
from collections import defaultdict, Counter
from typing import List, Optional, Tuple, Dict
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
import torchaudio
import pytorch_lightning as pl
from pytorch_lightning import seed_everything


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def load_label_mapping(dataset_name: str = "AUDIOSET", 
                       custom_map: Optional[dict] = None,
                       json_path: Optional[str] = None) -> dict:
    """
    Load label mapping from datasets_mapping.json.
    
    Args:
        dataset_name: Name of the dataset section to load from JSON
        custom_map: Optional custom label mapping dict
        json_path: Optional path to JSON file (defaults to ../datasets_mapping.json)
    
    Returns:
        Dictionary mapping label names to binary labels or MID codes
    """
    if custom_map is not None:
        return custom_map
    
    if json_path is None:
        # Default path: go up to datasets/
        script_dir = os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.join(os.path.dirname(script_dir), "datasets_mapping.json")
    
    try:
        with open(json_path, 'r') as f:
            all_mappings = json.load(f)
        
        if dataset_name not in all_mappings:
            raise ValueError(f"Dataset '{dataset_name}' not found in {json_path}")
        
        return all_mappings[dataset_name]
    
    except FileNotFoundError:
        raise FileNotFoundError(f"Label mapping file not found: {json_path}")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON format in {json_path}")


def load_mid_mapping(tsv_path: str) -> Dict[str, str]:
    """
    Load MID to display name mapping from mid_to_display_name.tsv.
    
    Args:
        tsv_path: Path to mid_to_display_name.tsv
    
    Returns:
        Dictionary mapping MID to display name
    """
    mid_map = {}
    with open(tsv_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                mid, display_name = parts
                mid_map[mid] = display_name
    return mid_map


# =============================================================================
# DATASET CLASS
# =============================================================================

class AudioSetEV_Strong_Dataset(Dataset):
    """
    AudioSet Emergency Vehicles Strong Dataset with temporal annotations.
    
    Supports Sound Event Detection with window-based label tracks.
    Each sample returns (waveform, label_track) where:
    - waveform: Full audio segment (e.g., 10s @ 32kHz)
    - label_track: Binary labels per time window (e.g., 32 windows of 310ms)
    """
    
    def __init__(self,
                 strong_metadata_paths: List[str],
                 audio_folders: List[str],
                 ev_mids: List[str],
                 window_size: float = 0.310,
                 target_sr: int = 32000,
                 target_duration: float = 10.0,
                 label_value: int = 1,
                 is_positive: bool = True,
                 seed: int = 42,
                 augmentation: bool = False,
                 aug_prob: float = 0.7,
                 file_tracker: Optional[Dict] = None):
        """
        Initialize AudioSet EV Strong Dataset.
        
        Args:
            strong_metadata_paths: List of paths to TSV metadata files
            audio_folders: List of folders to search for audio files
            ev_mids: List of Emergency Vehicle MIDs (e.g., ['/m/03j1ly', '/m/012n7d', ...])
            window_size: Window size in seconds for label tracks (default: 0.310)
            target_sr: Target sample rate (Hz)
            target_duration: Target audio duration in seconds
            label_value: Value to assign in label track when EV detected
            is_positive: If True, load only files with EV events; if False, load files without EV
            seed: Random seed for reproducibility
            augmentation: Whether to apply data augmentation
            aug_prob: Probability of applying each augmentation
            file_tracker: Optional dict to track file search results
        """
        super().__init__()
        
        # Set seeds
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        self.strong_metadata_paths = strong_metadata_paths
        self.audio_folders = [os.path.abspath(f) for f in audio_folders]
        self.ev_mids = set(ev_mids)
        self.window_size = window_size
        self.target_sr = target_sr
        self.target_duration = target_duration
        self.label_value = label_value
        self.is_positive = is_positive
        self.seed = seed
        self.file_tracker = file_tracker
        
        # Augmentation
        self.augmentation = augmentation
        self.aug_prob = aug_prob
        if self.augmentation:
            self.augmentations = self._define_augmentations()
        
        # Compute number of windows
        self.num_windows = int(np.ceil(target_duration / window_size))
        self.target_size = int(target_sr * target_duration)
        
        # Parse metadata and collect samples
        self.samples = self._parse_metadata_and_collect_samples()
        
        print(f"  Loaded {len(self.samples)} samples ({'positives' if is_positive else 'negatives'})")
    
    def _parse_metadata_and_collect_samples(self) -> List[Dict]:
        """
        Parse metadata TSV files and collect samples with their events.
        
        Returns:
            List of sample dicts with format:
            {
                'segment_id': str,
                'yt_id': str,
                'file_path': str,
                'events': List[Dict] with {'start': float, 'end': float, 'mid': str}
            }
        """
        # Load MID to display name mapping if available
        mid_to_display = {}
        script_dir = os.path.dirname(os.path.abspath(__file__))
        mid_mapping_path = os.path.join(script_dir, "audioset_strong_metadata", "mid_to_display_name.tsv")
        if os.path.exists(mid_mapping_path):
            with open(mid_mapping_path, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) == 2:
                        mid, display_name = parts
                        mid_to_display[mid] = display_name
        
        # Step 1: Parse all metadata files and organize by segment_id
        segment_events = defaultdict(list)
        
        for metadata_path in self.strong_metadata_paths:
            if not os.path.exists(metadata_path):
                print(f"  Warning: Metadata file not found: {metadata_path}")
                continue
            
            with open(metadata_path, 'r') as f:
                # Skip header
                next(f)
                
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) != 4:
                        continue
                    
                    segment_id, start_time, end_time, mid = parts
                    
                    try:
                        start_time = float(start_time)
                        end_time = float(end_time)
                    except ValueError:
                        continue
                    
                    segment_events[segment_id].append({
                        'start': start_time,
                        'end': end_time,
                        'mid': mid
                    })
        
        # Step 2: For each segment, determine if it contains EV events
        samples = []
        
        for segment_id, events in segment_events.items():
            # Extract yt_id (remove temporal suffix _XXXXX)
            yt_id = segment_id.rsplit('_', 1)[0] if '_' in segment_id else segment_id
            
            # Check if segment has EV events and collect EV labels
            ev_mids_in_sample = set()
            for event in events:
                if event['mid'] in self.ev_mids:
                    ev_mids_in_sample.add(event['mid'])
            
            has_ev = len(ev_mids_in_sample) > 0
            
            # Filter based on is_positive flag
            if has_ev != self.is_positive:
                continue
            
            # Find audio file
            file_path = self._find_audio_file(yt_id)
            
            if file_path is None:
                # Track not found
                if self.file_tracker is not None:
                    self.file_tracker['not_found'].append({
                        'yt_id': yt_id,
                        'segment_id': segment_id,
                        'labels': []
                    })
                continue
            
            # Get human-readable labels
            if has_ev:
                human_labels = [mid_to_display.get(mid, mid) for mid in sorted(ev_mids_in_sample)]
            else:
                human_labels = ["No EV"]
            
            # Track found - convert to relative path
            if self.file_tracker is not None:
                folder_path = os.path.dirname(file_path)
                # Convert to relative path from project root
                relative_path = self._make_relative_path(folder_path)
                
                self.file_tracker['found'].append({
                    'yt_id': yt_id,
                    'segment_id': segment_id,
                    'path': relative_path,
                    'labels': human_labels
                })
            
            # Add to samples
            samples.append({
                'segment_id': segment_id,
                'yt_id': yt_id,
                'file_path': file_path,
                'events': events
            })
        
        return samples
    
    def _make_relative_path(self, abs_path: str) -> str:
        """Convert absolute path to relative from project root (E2PANNs/)."""
        # Find the E2PANNs directory in the path
        parts = abs_path.split(os.sep)
        try:
            e2panns_idx = parts.index('E2PANNs')
            # Return path from E2PANNs onwards
            return os.path.join(*parts[e2panns_idx:])
        except ValueError:
            # If E2PANNs not found, return last 3 components
            return os.path.join(*parts[-3:]) if len(parts) >= 3 else abs_path
    
    def _find_audio_file(self, yt_id: str) -> Optional[str]:
        """
        Search for audio file across all configured folders.
        
        Args:
            yt_id: YouTube ID to search for
        
        Returns:
            Full path to audio file or None if not found
        """
        # Try different filename patterns
        patterns = [
            f"{yt_id}_Original.wav",
            f"{yt_id}_Reduced.wav",
            f"{yt_id}.wav"
        ]
        
        for folder in self.audio_folders:
            for pattern in patterns:
                file_path = os.path.join(folder, pattern)
                if os.path.exists(file_path):
                    return file_path
        
        return None
    
    def _create_label_track(self, events: List[Dict]) -> torch.Tensor:
        """
        Create binary label track from events.
        
        Args:
            events: List of event dicts with 'start', 'end', 'mid'
        
        Returns:
            Binary label track tensor of shape (num_windows,)
        """
        label_track = torch.zeros(self.num_windows, dtype=torch.float32)
        
        # For each EV event, mark affected windows
        for event in events:
            if event['mid'] not in self.ev_mids:
                continue
            
            start_time = event['start']
            end_time = event['end']
            
            # Convert to sample indices
            start_sample = int(start_time * self.target_sr)
            end_sample = int(end_time * self.target_sr)
            
            # Mark all windows that contain at least 1 sample of the event
            # Window i covers samples [i*window_samples, (i+1)*window_samples)
            window_samples = int(self.window_size * self.target_sr)
            
            for i in range(self.num_windows):
                window_start = i * window_samples
                window_end = (i + 1) * window_samples
                
                # Check if window overlaps with event
                # Overlap if: window_start < end_sample AND start_sample < window_end
                if window_start < end_sample and start_sample < window_end:
                    label_track[i] = self.label_value
        
        return label_track
    
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
        return noisy / torch.max(torch.abs(noisy))
    
    def _time_roll(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply circular time shift."""
        shift = random.randint(1, waveform.size(1))
        return torch.roll(waveform, shifts=shift, dims=1)
    
    def _polarity_inversion(self, waveform: torch.Tensor) -> torch.Tensor:
        """Invert waveform polarity."""
        return waveform * -1
    
    def _random_amplification(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply random amplitude scaling."""
        if random.random() > 0.5:
            scalar = random.uniform(0.1, 1.0)
            return waveform * scalar
        else:
            vector = torch.rand(waveform.size(1))
            return waveform * vector.unsqueeze(0)
    
    def _apply_augmentations(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply random augmentations to waveform."""
        augment_order = list(self.augmentations.keys())
        random.shuffle(augment_order)
        
        for aug_name in augment_order:
            if random.random() < self.aug_prob:
                waveform = self.augmentations[aug_name](waveform)
        
        return waveform
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Load and process audio file with label track.
        
        Args:
            idx: Index of sample
        
        Returns:
            Tuple of (waveform, label_track)
            - waveform: shape (1, target_size)
            - label_track: shape (num_windows,)
        """
        sample = self.samples[idx]
        file_path = sample['file_path']
        events = sample['events']
        
        try:
            # Load audio using soundfile
            waveform_np, sr = sf.read(file_path, dtype='float32')
            
            # Convert to torch tensor and ensure correct shape (channels, samples)
            if waveform_np.ndim == 1:
                # Mono audio: add channel dimension
                waveform = torch.from_numpy(waveform_np).unsqueeze(0)
            else:
                # Stereo/multi-channel: transpose to (channels, samples)
                waveform = torch.from_numpy(waveform_np.T)
            
            # Resample if necessary
            if sr != self.target_sr:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.target_sr)
                waveform = resampler(waveform)
            
            # Convert to mono if multi-channel
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Pad or truncate to target size
            current_size = waveform.shape[1]
            if current_size < self.target_size:
                # Pad with zeros
                pad_size = self.target_size - current_size
                waveform = F.pad(waveform, (0, pad_size), mode='constant', value=0)
            elif current_size > self.target_size:
                # Truncate
                waveform = waveform[:, :self.target_size]
            
            # Apply augmentations if enabled
            if self.augmentation:
                waveform = self._apply_augmentations(waveform)
            
            # Normalize to [-1, 1]
            max_val = torch.max(torch.abs(waveform))
            if max_val > 0:
                waveform = waveform / max_val
            
            # Create label track
            label_track = self._create_label_track(events)
            
            return waveform, label_track
        
        except Exception as e:
            print(f"  Error loading {file_path}: {e}")
            # Return zeros as fallback
            waveform = torch.zeros(1, self.target_size, dtype=torch.float32)
            label_track = torch.zeros(self.num_windows, dtype=torch.float32)
            return waveform, label_track


# =============================================================================
# DATAMODULE CLASS
# =============================================================================

class AudioSetEV_Strong_DataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for AudioSet EV Strong datasets.
    
    Supports two dataset versions:
    - 'v1': Uses AudioSet_EV_v1_2025 audio files with Strong metadata
    - 'v2': Uses AudioSet_EV_v2PANNs_2020 audio files with Strong metadata
    
    Modes:
    - 'train': Balanced dataset with train/dev/test split
    - 'detection': Balanced dataset, all as test set
    """
    
    def __init__(self,
                 dataset_version: str = 'v1',
                 mode: str = 'train',
                 window_size: float = 0.310,
                 batch_size: int = 32,
                 split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
                 shuffle: bool = True,
                 augmentation: bool = False,
                 aug_prob: float = 0.7,
                 target_sr: int = 32000,
                 target_duration: float = 10.0,
                 num_workers: Optional[int] = None,
                 seed: int = 42,
                 log_path: Optional[str] = None):
        """
        Initialize AudioSet EV Strong DataModule.
        
        Args:
            dataset_version: 'v1' or 'v2'
            mode: 'train' or 'detection'
            window_size: Window size in seconds for label tracks
            batch_size: Batch size for dataloaders
            split_ratios: (train, dev, test) ratios for train mode
            shuffle: Whether to shuffle training data
            augmentation: Whether to apply data augmentation
            aug_prob: Probability of applying augmentation
            target_sr: Target sample rate
            target_duration: Target audio duration in seconds
            num_workers: Number of dataloader workers
            seed: Random seed
            log_path: Optional path to save file search log CSV
        """
        super().__init__()
        
        # Set global seed
        seed_everything(seed, workers=True)
        
        self.dataset_version = dataset_version
        self.mode = mode
        self.window_size = window_size
        self.batch_size = batch_size
        self.split_ratios = split_ratios
        self.train_shuffle = shuffle
        self.augmentation = augmentation
        self.aug_prob = aug_prob
        self.target_sr = target_sr
        self.target_duration = target_duration
        self.seed = seed
        self.log_path = log_path
        
        # Auto-configure workers
        self.num_workers = num_workers if num_workers is not None else min(8, os.cpu_count() // 4)
        self.pin_memory = torch.cuda.is_available()
        self.persistent_workers = self.num_workers > 0
        
        # Configure paths based on dataset version
        self._configure_paths()
        
        # Emergency Vehicle MIDs from AudioSet
        self.ev_mids = [
            '/m/03j1ly',  # Emergency vehicle
            '/m/04qvtq',  # Police car (siren)
            '/m/012n7d',  # Ambulance (siren)
            '/m/012ndj'   # Fire engine, fire truck (siren)
        ]
        
        # File tracker for logging
        self.file_tracker = {
            'found': [],
            'not_found': []
        } if log_path is not None else None
        
        # Datasets
        self.train_dataset = None
        self.dev_dataset = None
        self.test_dataset = None
        
        # Generator for reproducible splits
        self.generator = torch.Generator().manual_seed(seed)
    
    def _configure_paths(self):
        """Configure paths based on dataset version."""
        # Get base datasets directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        datasets_dir = os.path.dirname(script_dir)
        
        # Strong metadata paths (same for both versions)
        strong_metadata_dir = os.path.join(script_dir, "audioset_strong_metadata")
        self.train_metadata_path = os.path.join(strong_metadata_dir, "audioset_train_strong.tsv")
        self.eval_metadata_path = os.path.join(strong_metadata_dir, "audioset_eval_strong.tsv")
        
        if self.dataset_version == 'v1':
            # AudioSet_EV_v1_2025 structure
            v1_dir = os.path.join(datasets_dir, "AudioSet_EV_v1_2025")
            self.audio_folders = [
                os.path.join(v1_dir, "Positive_files"),
                os.path.join(v1_dir, "Negative_files")
            ]
        elif self.dataset_version == 'v2':
            # AudioSet_EV_v2PANNs_2020 structure with subfolders
            v2_dir = os.path.join(datasets_dir, "AudioSet_EV_v2PANNs_2020")
            self.audio_folders = [
                os.path.join(v2_dir, "Positive_files", "balanced_train"),
                os.path.join(v2_dir, "Positive_files", "eval"),
                os.path.join(v2_dir, "Positive_files", "unbalanced"),
                os.path.join(v2_dir, "Negative_files", "balanced_train"),
                os.path.join(v2_dir, "Negative_files", "eval")
            ]
        else:
            raise ValueError(f"Invalid dataset_version: {self.dataset_version}. Must be 'v1' or 'v2'.")
    
    def setup(self, stage: Optional[str] = None):
        """Setup datasets based on mode."""
        print(f"\n{'='*80}")
        print(f"Setting up AudioSet_EV_{self.dataset_version}_Strong - {self.mode.upper()} mode")
        print(f"{'='*80}")
        print(f"Window size: {self.window_size}s")
        print(f"Target SR: {self.target_sr}Hz")
        print(f"Target duration: {self.target_duration}s")
        print(f"Num windows: {int(np.ceil(self.target_duration / self.window_size))}")
        
        # Load positive samples (files with EV events)
        print(f"\nLoading POSITIVE samples (with EV events)...")
        pos_dataset = AudioSetEV_Strong_Dataset(
            strong_metadata_paths=[self.train_metadata_path, self.eval_metadata_path],
            audio_folders=self.audio_folders,
            ev_mids=self.ev_mids,
            window_size=self.window_size,
            target_sr=self.target_sr,
            target_duration=self.target_duration,
            label_value=1,
            is_positive=True,
            seed=self.seed,
            augmentation=self.augmentation,
            aug_prob=self.aug_prob,
            file_tracker=self.file_tracker
        )
        
        # Load negative samples (files without EV events)
        print(f"\nLoading NEGATIVE samples (without EV events)...")
        neg_dataset_full = AudioSetEV_Strong_Dataset(
            strong_metadata_paths=[self.train_metadata_path, self.eval_metadata_path],
            audio_folders=self.audio_folders,
            ev_mids=self.ev_mids,
            window_size=self.window_size,
            target_sr=self.target_sr,
            target_duration=self.target_duration,
            label_value=0,
            is_positive=False,
            seed=self.seed,
            augmentation=self.augmentation,
            aug_prob=self.aug_prob,
            file_tracker=self.file_tracker
        )
        
        # Balance negatives to match positives
        num_positives = len(pos_dataset)
        num_negatives = len(neg_dataset_full)
        
        print(f"\nBalancing dataset:")
        print(f"  Positives: {num_positives}")
        print(f"  Negatives (available): {num_negatives}")
        
        if num_negatives > num_positives:
            # Random sample negatives
            random.seed(self.seed)
            neg_indices = random.sample(range(num_negatives), num_positives)
            neg_dataset = torch.utils.data.Subset(neg_dataset_full, neg_indices)
            print(f"  Negatives (sampled): {len(neg_dataset)}")
        else:
            neg_dataset = neg_dataset_full
            print(f"  Negatives (using all): {len(neg_dataset)}")
        
        # Concatenate
        full_dataset = ConcatDataset([pos_dataset, neg_dataset])
        print(f"  Total balanced dataset: {len(full_dataset)}")
        
        # Mode-specific setup
        if self.mode == 'train':
            # Split into train/dev/test
            total_len = len(full_dataset)
            train_len = int(self.split_ratios[0] * total_len)
            dev_len = int(self.split_ratios[1] * total_len)
            test_len = total_len - train_len - dev_len
            
            self.train_dataset, self.dev_dataset, self.test_dataset = random_split(
                full_dataset,
                [train_len, dev_len, test_len],
                generator=self.generator
            )
            
            print(f"\nDataset splits:")
            print(f"  Train: {len(self.train_dataset)}")
            print(f"  Dev: {len(self.dev_dataset)}")
            print(f"  Test: {len(self.test_dataset)}")
        
        elif self.mode == 'detection':
            # All as test set
            self.test_dataset = full_dataset
            print(f"\nDetection mode: All {len(self.test_dataset)} samples as test set")
        
        else:
            raise ValueError(f"Invalid mode: {self.mode}. Must be 'train' or 'detection'.")
        
        # Save file tracking log if requested
        if self.log_path is not None and self.file_tracker is not None:
            self._save_file_log()
    
    def _save_file_log(self):
        """Save file search log to CSV."""
        log_dir = os.path.dirname(self.log_path)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        with open(self.log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['yt_id', 'segment_id', 'dataset_version', 'found', 'path', 'label'])
            
            # Write found files
            for item in self.file_tracker['found']:
                # Join multiple labels with '; '
                label_str = '; '.join(item['labels']) if item['labels'] else 'Unknown'
                writer.writerow([
                    item['yt_id'],
                    item['segment_id'],
                    self.dataset_version,
                    True,
                    item['path'],
                    label_str
                ])
            
            # Write not found files
            for item in self.file_tracker['not_found']:
                writer.writerow([
                    item['yt_id'],
                    item['segment_id'],
                    self.dataset_version,
                    False,
                    'N/A',
                    'N/A'
                ])
        
        print(f"\nFile tracking log saved to: {self.log_path}")
        print(f"  Found: {len(self.file_tracker['found'])}")
        print(f"  Not found: {len(self.file_tracker['not_found'])}")
    
    def train_dataloader(self):
        """Return training dataloader."""
        if self.train_dataset is None:
            raise RuntimeError("train_dataset not initialized. Call setup() first.")
        
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.train_shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers
        )
    
    def val_dataloader(self):
        """Return validation dataloader."""
        if self.dev_dataset is None:
            raise RuntimeError("dev_dataset not initialized. Call setup() first.")
        
        return DataLoader(
            self.dev_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers
        )
    
    def test_dataloader(self):
        """Return test dataloader."""
        if self.test_dataset is None:
            raise RuntimeError("test_dataset not initialized. Call setup() first.")
        
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers
        )


# =============================================================================
# MAIN TEST SCRIPT
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("DATALOADER TEST - AudioSet Emergency Vehicles Strong")
    print("=" * 80)
    
    # Helper function to analyze label tracks
    def analyze_label_tracks(dataloader, max_samples=100):
        """Analyze label track statistics."""
        total_windows = 0
        positive_windows = 0
        samples_with_events = 0
        
        for i, (waveforms, label_tracks) in enumerate(dataloader):
            if i * dataloader.batch_size >= max_samples:
                break
            
            batch_size = label_tracks.shape[0]
            num_windows = label_tracks.shape[1]
            
            total_windows += batch_size * num_windows
            positive_windows += (label_tracks > 0).sum().item()
            samples_with_events += (label_tracks.sum(dim=1) > 0).sum().item()
        
        return {
            'total_windows': total_windows,
            'positive_windows': positive_windows,
            'positive_ratio': positive_windows / total_windows if total_windows > 0 else 0,
            'samples_with_events': samples_with_events
        }
    
    # =========================================================================
    # TEST 1: AudioSet_EV_v1_Strong - TRAIN MODE
    # =========================================================================
    print("\n\n" + "=" * 80)
    print("TEST 1: AUDIOSET_EV_V1_STRONG - TRAIN MODE")
    print("=" * 80)
    
    dm_v1_train = AudioSetEV_Strong_DataModule(
        dataset_version='v1',
        mode='train',
        window_size=0.310,
        batch_size=32,
        split_ratios=(0.8, 0.1, 0.1),
        shuffle=True,
        augmentation=False,
        target_sr=32000,
        target_duration=10.0,
        num_workers=0,
        seed=42,
        log_path='./logs/audioset_strong_v1_train_file_log.csv'
    )
    
    dm_v1_train.setup()
    
    # Train loader
    train_loader = dm_v1_train.train_dataloader()
    print("\n" + "─" * 80)
    print("TRAIN LOADER")
    print("─" * 80)
    print(f"Total batches: {len(train_loader)}")
    print(f"Total samples: {len(train_loader.dataset)}")
    
    print("\nFirst batch analysis:")
    for waveforms, label_tracks in train_loader:
        print(f"  Waveform shape: {waveforms.shape}")
        print(f"  Label track shape: {label_tracks.shape}")
        print(f"  Batch size: {waveforms.shape[0]}")
        print(f"  Audio duration: {waveforms.shape[2] / 32000:.2f}s")
        print(f"  Num windows: {label_tracks.shape[1]}")
        
        # Analyze label tracks in batch
        samples_with_ev = (label_tracks.sum(dim=1) > 0).sum().item()
        total_positive_windows = (label_tracks > 0).sum().item()
        print(f"  Samples with EV events: {samples_with_ev}/{waveforms.shape[0]}")
        print(f"  Total positive windows: {total_positive_windows}")
        
        # Show example label track
        print("\nExample label track (first sample):")
        track = label_tracks[0].numpy()
        print(f"  {track}")
        print(f"  Has events: {track.sum() > 0}")
        break
    
    # Validation loader
    val_loader = dm_v1_train.val_dataloader()
    print("\n" + "─" * 80)
    print("VALIDATION LOADER")
    print("─" * 80)
    print(f"Total batches: {len(val_loader)}")
    print(f"Total samples: {len(val_loader.dataset)}")
    
    # Test loader
    test_loader = dm_v1_train.test_dataloader()
    print("\n" + "─" * 80)
    print("TEST LOADER")
    print("─" * 80)
    print(f"Total batches: {len(test_loader)}")
    print(f"Total samples: {len(test_loader.dataset)}")
    
    # =========================================================================
    # TEST 2: AudioSet_EV_v1_Strong - DETECTION MODE
    # =========================================================================
    print("\n\n" + "=" * 80)
    print("TEST 2: AUDIOSET_EV_V1_STRONG - DETECTION MODE")
    print("=" * 80)
    
    dm_v1_detect = AudioSetEV_Strong_DataModule(
        dataset_version='v1',
        mode='detection',
        window_size=0.310,
        batch_size=32,
        augmentation=False,
        target_sr=32000,
        target_duration=10.0,
        num_workers=0,
        seed=42,
        log_path=None  # No logging for this test
    )
    
    dm_v1_detect.setup()
    
    test_loader_detect = dm_v1_detect.test_dataloader()
    print("\n" + "─" * 80)
    print("DETECTION TEST LOADER")
    print("─" * 80)
    print(f"Total batches: {len(test_loader_detect)}")
    print(f"Total samples: {len(test_loader_detect.dataset)}")
    
    # =========================================================================
    # TEST 3: AudioSet_EV_v2_Strong - TRAIN MODE
    # =========================================================================
    print("\n\n" + "=" * 80)
    print("TEST 3: AUDIOSET_EV_V2_STRONG - TRAIN MODE")
    print("=" * 80)
    
    dm_v2_train = AudioSetEV_Strong_DataModule(
        dataset_version='v2',
        mode='train',
        window_size=0.310,
        batch_size=32,
        split_ratios=(0.8, 0.1, 0.1),
        shuffle=True,
        augmentation=False,
        target_sr=32000,
        target_duration=10.0,
        num_workers=0,
        seed=42,
        log_path='./logs/audioset_strong_v2_train_file_log.csv'
    )
    
    dm_v2_train.setup()
    
    train_loader_v2 = dm_v2_train.train_dataloader()
    print("\n" + "─" * 80)
    print("TRAIN LOADER")
    print("─" * 80)
    print(f"Total batches: {len(train_loader_v2)}")
    print(f"Total samples: {len(train_loader_v2.dataset)}")
    
    print("\nFirst batch analysis:")
    for waveforms, label_tracks in train_loader_v2:
        print(f"  Waveform shape: {waveforms.shape}")
        print(f"  Label track shape: {label_tracks.shape}")
        
        # Show example
        print("\nExample label track (first sample):")
        track = label_tracks[0].numpy()
        print(f"  {track}")
        break
    
    # =========================================================================
    # TEST 4: AudioSet_EV_v2_Strong - DETECTION MODE
    # =========================================================================
    print("\n\n" + "=" * 80)
    print("TEST 4: AUDIOSET_EV_V2_STRONG - DETECTION MODE")
    print("=" * 80)
    
    dm_v2_detect = AudioSetEV_Strong_DataModule(
        dataset_version='v2',
        mode='detection',
        window_size=0.310,
        batch_size=32,
        augmentation=False,
        target_sr=32000,
        target_duration=10.0,
        num_workers=0,
        seed=42
    )
    
    dm_v2_detect.setup()
    
    test_loader_v2_detect = dm_v2_detect.test_dataloader()
    print("\n" + "─" * 80)
    print("DETECTION TEST LOADER")
    print("─" * 80)
    print(f"Total batches: {len(test_loader_v2_detect)}")
    print(f"Total samples: {len(test_loader_v2_detect.dataset)}")
    
    print("\n" + "=" * 80)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 80)
