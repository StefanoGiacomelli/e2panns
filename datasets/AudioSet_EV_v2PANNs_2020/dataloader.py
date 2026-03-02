"""
AudioSet Emergency Vehicles v2 (PANNs 2020) Dataset Dataloader
==============================================================
PyTorch Dataset and Lightning DataModule for AudioSet EV v2 binary classification.

Dataset structure:
- Positive_files/ (3 subfolders):
  - balanced_train/  (~135 files)
  - eval/            (~135 files)
  - unbalanced/      (~7630 files)
  Total: ~7900 positive samples

- Negative_files/ (2 subfolders):
  - balanced_train/  (~10963 files)
  - eval/            (~9953 files)
  Total: ~20916 negative samples (to be balanced to ~7900)

Features:
- Multi-subfolder support for both Positives and Negatives
- Stratified balancing on negative labels (respecting multi-label distribution)
- Seed-controlled for reproducibility
- PyTorch Lightning DataModule

Author: Stefano Giacomelli - Ph.D. candidate in ICT (DISIM dpt. - University of L'Aquila)
"""

import os
import sys
import json
import ast
import random
from collections import defaultdict, Counter
from typing import List, Optional

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
import torchaudio
import pytorch_lightning as pl


class StratifiedNegativeBalancer:
    """
    Stratified balancer for negative samples with multi-label support.
    
    Balances negative samples by:
    1. Loading label mapping (MID → human-readable)
    2. Organizing samples by their labels
    3. Sampling proportionally from each label category
    4. Avoiding duplicate samples
    """
    
    def __init__(self, csv_path: str, label_mapping_csv: str, 
                 negative_focus_labels: List[str], seed: int = 42):
        """
        Args:
            csv_path: Path to negative samples CSV
            label_mapping_csv: Path to class_labels_indices.csv
            negative_focus_labels: List of negative label names (human-readable)
            seed: Random seed for reproducibility
        """
        self.csv_path = csv_path
        self.seed = seed
        
        # Load label mapping (MID → human-readable)
        self.label_map = self._load_label_mapping(label_mapping_csv)
        
        # Convert focus labels to MIDs
        reverse_map = {v: k for k, v in self.label_map.items()}
        self.focus_mids = {
            reverse_map[label] for label in negative_focus_labels 
            if label in reverse_map
        }
        
        # Set seed
        random.seed(seed)
        np.random.seed(seed)
    
    def _load_label_mapping(self, csv_path: str) -> dict:
        """Load class_labels_indices.csv and create MID → display_name mapping"""
        df = pd.read_csv(csv_path)
        return {row['mid']: row['display_name'] for _, row in df.iterrows()}
    
    def balance(self, target_count: int) -> pd.DataFrame:
        """
        Balance negative samples using stratified sampling.
        
        Args:
            target_count: Target number of samples (typically matches positive count)
        
        Returns:
            Balanced DataFrame with ~target_count samples
        """
        # Load CSV and filter downloaded samples
        df = pd.read_csv(self.csv_path)
        df = df[df['downloaded'] == True].reset_index(drop=True)
        
        print(f"\nBalancing negatives from {len(df)} downloaded samples...")
        
        # Organize samples by label
        label_to_indices = defaultdict(list)
        for idx, row in df.iterrows():
            try:
                sample_labels = set(ast.literal_eval(row['positive_labels']))
                common_labels = sample_labels & self.focus_mids
                
                # Associate sample with all its labels
                for label in common_labels:
                    label_to_indices[label].append(idx)
            except:
                continue
        
        if not label_to_indices:
            print("WARNING: No samples matched focus labels. Using random sampling.")
            selected = df.sample(n=min(target_count, len(df)), random_state=self.seed)
            return selected.reset_index(drop=True)
        
        # Calculate samples per label
        min_count = min(len(indices) for indices in label_to_indices.values())
        samples_per_label = max(1, target_count // len(label_to_indices))
        
        print(f"Stratifying across {len(label_to_indices)} label categories")
        print(f"Target: ~{samples_per_label} samples per label")
        
        # Stratified sampling (avoiding duplicates)
        selected_indices = set()
        for label, indices in label_to_indices.items():
            random.shuffle(indices)  # Seed-controlled
            count = 0
            for idx in indices:
                if idx not in selected_indices:
                    selected_indices.add(idx)
                    count += 1
                    if count >= samples_per_label:
                        break
        
        # If we haven't reached target_count, add random samples
        if len(selected_indices) < target_count:
            remaining_indices = set(range(len(df))) - selected_indices
            if remaining_indices:
                additional_count = min(target_count - len(selected_indices), 
                                      len(remaining_indices))
                additional = random.sample(list(remaining_indices), additional_count)
                selected_indices.update(additional)
        
        result_df = df.loc[list(selected_indices)].reset_index(drop=True)
        print(f"Selected {len(result_df)} balanced negative samples")
        
        return result_df


class AudioSetEV_v2_Dataset(Dataset):
    """
    AudioSet EV v2 Dataset with multi-subfolder and multi-label support.
    
    Supports loading from multiple subfolders (e.g., balanced_train, eval, unbalanced).
    Each sample is processed to 10-second mono audio at 32kHz.
    Supports optional data augmentation.
    """
    
    def __init__(self, csv_df: pd.DataFrame, audio_folder: str, 
                 subfolders: List[str], target_size: int = 320000,
                 binary_label: int = 1,
                 augmentation: bool = False,
                 aug_prob: float = 0.7,
                 seed: int = 42):
        """
        Args:
            csv_df: DataFrame with sample metadata
            audio_folder: Root folder path (e.g., "Positive_files")
            subfolders: List of subfolders to load from 
                       (e.g., ["balanced_train", "eval", "unbalanced"])
            target_size: Target audio length in samples (default: 10s @ 32kHz)
            binary_label: Binary label (1 for positive, 0 for negative)
            augmentation: Whether to apply data augmentation
            aug_prob: Probability of applying each augmentation
            seed: Random seed for reproducibility
        """
        self.df = csv_df
        self.audio_folder = audio_folder
        self.subfolders = subfolders
        self.target_size = target_size
        self.binary_label = binary_label
        
        # Set seeds for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Augmentation settings
        self.augmentation = augmentation
        self.aug_prob = aug_prob
        if self.augmentation:
            self.augmentations = self._define_augmentations()
        
        # Build file list with full paths
        self.file_list = []
        self.skipped_files = []
        
        # Mapping for segment_type names (CSV vs actual folder names)
        segment_mapping = {
            'unbalanced_train': 'unbalanced',  # CSV uses "unbalanced_train", folder is "unbalanced"
            'balanced_train': 'balanced_train',
            'eval': 'eval'
        }
        
        for idx, row in self.df.iterrows():
            yt_id = row['yt_id']
            segment_type = row['segment_type']
            
            # Map segment_type to actual folder name
            mapped_segment = segment_mapping.get(segment_type, segment_type)
            
            # Check if mapped segment is in allowed subfolders
            if mapped_segment in self.subfolders:
                audio_path = os.path.join(
                    self.audio_folder,
                    mapped_segment,
                    f"{yt_id}.wav"
                )
                
                if os.path.exists(audio_path):
                    self.file_list.append(audio_path)
                else:
                    self.skipped_files.append(audio_path)
        
        print(f"  → Found {len(self.file_list)} audio files")
        if self.skipped_files:
            print(f"  → Warning: {len(self.skipped_files)} files not found (skipped)")

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
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        audio_path = self.file_list[idx]
        
        try:
            # Load audio with soundfile
            waveform, sr = sf.read(audio_path, dtype='float32')
            
            # Convert to PyTorch tensor and add channel dimension
            waveform = torch.from_numpy(waveform).unsqueeze(0)
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Resample to 32kHz if needed
            if sr != 32000:
                resampler = torchaudio.transforms.Resample(sr, 32000)
                waveform = resampler(waveform)
            
            # Pad or truncate to target_size
            if waveform.shape[1] < self.target_size:
                pad_size = self.target_size - waveform.shape[1]
                waveform = F.pad(waveform, (0, pad_size))
            else:
                waveform = waveform[:, :self.target_size]
            
            # Apply augmentation if enabled
            if self.augmentation:
                waveform = self._apply_augmentations(waveform)
            
            return waveform, self.binary_label
            
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            return None


def custom_collate_fn(batch):
    """Custom collate function to handle None values (failed loads)"""
    batch = [item for item in batch if item is not None]
    if not batch:
        return None, None
    return torch.utils.data.default_collate(batch)


class AudioSetEV_v2_DataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for AudioSet EV v2.
    
    Features:
    - Loads all positive samples from 3 subfolders (no additional balancing)
    - Balances negative samples using stratified sampling
    - Seed-controlled for reproducibility
    - Standard train/val/test split
    
    Supports two label modes:
    - 'binary': Binary classification (0=negative, 1=positive)
    - 'multi_class': 4-class classification (0=negative, 1=police, 2=ambulance, 3=fire)
    """
    
    def __init__(self,
                 # Positives
                 TP_csv: str,
                 TP_folder: str,
                 
                 # Negatives
                 TN_csv: str,
                 TN_folder: str,
                 
                 # Label mapping
                 label_mapping_csv: str,
                 negative_focus_labels: List[str],
                 
                 # Subfolders
                 TP_subfolders: List[str] = None,
                 TN_subfolders: List[str] = None,
                 
                 # Training params
                 label_mode: str = 'binary',
                 batch_size: int = 32,
                 split_ratios: tuple = (0.8, 0.1, 0.1),
                 balance_negatives: bool = True,
                 seed: int = 42,
                 num_workers: Optional[int] = None):
        """
        Args:
            TP_csv: Path to positives CSV
            TP_folder: Path to positives audio folder
            TP_subfolders: List of positive subfolders to use
            TN_csv: Path to negatives CSV
            TN_folder: Path to negatives audio folder
            TN_subfolders: List of negative subfolders to use
            label_mapping_csv: Path to class_labels_indices.csv
            negative_focus_labels: List of negative label names for stratification
            label_mode: 'binary' (0/1) or 'multi_class' (0/1/2/3)
            batch_size: Batch size for dataloaders
            split_ratios: Train/val/test split ratios
            balance_negatives: Whether to balance negative samples
            seed: Random seed for reproducibility
        """
        super().__init__()
        self.TP_csv = TP_csv
        self.TP_folder = TP_folder
        self.TP_subfolders = TP_subfolders if TP_subfolders is not None else ["balanced_train", "eval", "unbalanced"]
        self.TN_csv = TN_csv
        self.TN_folder = TN_folder
        self.TN_subfolders = TN_subfolders if TN_subfolders is not None else ["balanced_train", "eval"]
        self.label_mapping_csv = label_mapping_csv
        self.negative_focus_labels = negative_focus_labels
        self.label_mode = label_mode
        self.batch_size = batch_size
        self.split_ratios = split_ratios
        self.balance_negatives = balance_negatives
        self.seed = seed
        
        # Auto-configure workers and memory settings
        self.num_workers = num_workers if num_workers is not None else min(8, os.cpu_count() // 4)
        self.pin_memory = torch.cuda.is_available()
        self.persistent_workers = self.num_workers > 0
        
        # Set global seed
        pl.seed_everything(seed)
    
    def setup(self, stage: Optional[str] = None):
        """Setup datasets with integrated balancing"""
        
        if self.label_mode == 'multi_class':
            # Multi-class mode: 4-way balancing
            self._setup_multiclass()
        else:
            # Binary mode: original behavior
            self._setup_binary()
    
    def _setup_binary(self):
        """Setup for binary classification (original behavior)."""
        # 1. Load Positives (use all files, no additional balancing)
        pos_df = pd.read_csv(self.TP_csv)
        pos_df = pos_df[pos_df['downloaded'] == True].reset_index(drop=True)
        
        print(f"\nPositives: {len(pos_df)} samples from subfolders {self.TP_subfolders}")
        
        # 2. Load/Balance Negatives
        if self.balance_negatives:
            balancer = StratifiedNegativeBalancer(
                csv_path=self.TN_csv,
                label_mapping_csv=self.label_mapping_csv,
                negative_focus_labels=self.negative_focus_labels,
                seed=self.seed
            )
            neg_df = balancer.balance(target_count=len(pos_df))
        else:
            neg_df = pd.read_csv(self.TN_csv)
            neg_df = neg_df[neg_df['downloaded'] == True].reset_index(drop=True)
            print(f"\nNegatives: {len(neg_df)} samples (no balancing)")
        
        # 3. Create Datasets
        print(f"\nCreating positive dataset...")
        pos_dataset = AudioSetEV_v2_Dataset(
            csv_df=pos_df,
            audio_folder=self.TP_folder,
            subfolders=self.TP_subfolders,
            binary_label=1
        )
        
        print(f"Creating negative dataset...")
        neg_dataset = AudioSetEV_v2_Dataset(
            csv_df=neg_df,
            audio_folder=self.TN_folder,
            subfolders=self.TN_subfolders,
            binary_label=0
        )
        
        # 4. Concatenate and split
        full_dataset = ConcatDataset([pos_dataset, neg_dataset])
        
        train_size = int(self.split_ratios[0] * len(full_dataset))
        val_size = int(self.split_ratios[1] * len(full_dataset))
        test_size = len(full_dataset) - train_size - val_size
        
        generator = torch.Generator().manual_seed(self.seed)
        self.train_ds, self.val_ds, self.test_ds = random_split(
            full_dataset,
            [train_size, val_size, test_size],
            generator=generator
        )
        
        # Calculate split statistics
        print(f"\n{'─'*80}")
        print("DATASET SPLIT STATISTICS")
        print('─'*80)
        
        # Count labels in each split
        def count_labels(dataset):
            pos_count = sum(1 for i in range(len(dataset)) if dataset[i][1] == 1)
            neg_count = len(dataset) - pos_count
            return pos_count, neg_count
        
        print("Counting training labels...")
        train_pos, train_neg = count_labels(self.train_ds)
        print("Counting validation labels...")
        val_pos, val_neg = count_labels(self.val_ds)
        print("Counting test labels...")
        test_pos, test_neg = count_labels(self.test_ds)
        
        print(f"Train set: {len(self.train_ds)} samples (Pos: {train_pos}, Neg: {train_neg})")
        print(f"Validation set: {len(self.val_ds)} samples (Pos: {val_pos}, Neg: {val_neg})")
        print(f"Test set: {len(self.test_ds)} samples (Pos: {test_pos}, Neg: {test_neg})")
        print('─'*80)
    
    def _setup_multiclass(self):
        """Setup for multi-class classification with 4-way balancing."""
        import sys
        import ast
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from multi_class_utils import FourWayBalancer, print_balance_summary, parse_audioset_multi_labels
        
        # Load AudioSet multi-class mapping
        mid_mapping_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets_mapping.json")
        with open(mid_mapping_path, 'r') as f:
            all_mappings = json.load(f)
        mid_mapping = all_mappings["AUDIOSET_MULTICLASS"]
        
        # Load positives
        pos_df = pd.read_csv(self.TP_csv)
        pos_df = pos_df[pos_df['downloaded'] == True].reset_index(drop=True)
        
        # Parse positive labels into classes
        pos_info = parse_audioset_multi_labels(pos_df, mid_mapping)
        
        # Load negatives
        neg_df = pd.read_csv(self.TN_csv)
        neg_df = neg_df[neg_df['downloaded'] == True].reset_index(drop=True)
        
        # Balance negatives using stratified balancer (same logic as binary)
        if self.balance_negatives:
            # Use stratified balancer to get balanced negatives
            balancer = StratifiedNegativeBalancer(
                csv_path=self.TN_csv,
                label_mapping_csv=self.label_mapping_csv,
                negative_focus_labels=self.negative_focus_labels,
                seed=self.seed
            )
            # Calculate target: min of positive classes
            min_pos_count = min(len(pos_info['pure'][1]), len(pos_info['pure'][2]), len(pos_info['pure'][3]))
            neg_df_balanced = balancer.balance(target_count=min_pos_count)
            neg_indices = list(range(len(neg_df_balanced)))
        else:
            neg_indices = list(range(len(neg_df)))
            neg_df_balanced = neg_df
        
        # Balance 4 classes
        four_way_balancer = FourWayBalancer(target_mode='auto', min_samples_per_class=50)
        
        result = four_way_balancer.balance(
            pure_samples={
                0: neg_indices,
                1: pos_info['pure'][1],
                2: pos_info['pure'][2],
                3: pos_info['pure'][3]
            },
            multi_samples=pos_info['multi'],
            seed=self.seed
        )
        
        print_balance_summary(result, title="AudioSet EV v2 - 4-Way Balance")
        
        # Build file paths
        file_label_pairs = []
        
        # Negatives (class 0)
        for idx in result['balanced_indices'][0]:
            file_info = neg_df_balanced.iloc[idx]
            yt_id = file_info['yt_id']
            segment_type = file_info['segment_type']
            
            # Map segment_type to folder
            segment_mapping = {
                'unbalanced_train': 'unbalanced',
                'balanced_train': 'balanced_train',
                'eval': 'eval'
            }
            mapped_segment = segment_mapping.get(segment_type, segment_type)
            
            if mapped_segment in self.TN_subfolders:
                file_path = os.path.join(self.TN_folder, mapped_segment, f"{yt_id}.wav")
                if os.path.exists(file_path):
                    file_label_pairs.append((file_path, 0))
        
        # Positives (classes 1, 2, 3)
        for cls in [1, 2, 3]:
            for idx in result['balanced_indices'][cls]:
                file_info = pos_df.iloc[idx]
                yt_id = file_info['yt_id']
                segment_type = file_info['segment_type']
                
                # Map segment_type to folder
                segment_mapping = {
                    'unbalanced_train': 'unbalanced',
                    'balanced_train': 'balanced_train',
                    'eval': 'eval'
                }
                mapped_segment = segment_mapping.get(segment_type, segment_type)
                
                if mapped_segment in self.TP_subfolders:
                    file_path = os.path.join(self.TP_folder, mapped_segment, f"{yt_id}.wav")
                    if os.path.exists(file_path):
                        file_label_pairs.append((file_path, cls))
        
        print(f"\n  Total files found: {len(file_label_pairs)} / {sum(len(result['balanced_indices'][c]) for c in [0,1,2,3])}")
        
        # Create custom dataset
        from torch.utils.data import Dataset as TorchDataset
        
        class IndexedAudioSetDataset(TorchDataset):
            def __init__(self, file_label_pairs, target_size=320000, target_sr=32000):
                self.file_label_pairs = file_label_pairs
                self.target_size = target_size
                self.target_sr = target_sr
            
            def __len__(self):
                return len(self.file_label_pairs)
            
            def __getitem__(self, idx):
                file_path, label = self.file_label_pairs[idx]
                try:
                    waveform, sr = torchaudio.load(file_path)
                    if waveform.shape[0] > 1:
                        waveform = torch.mean(waveform, dim=0, keepdim=True)
                    if sr != self.target_sr:
                        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.target_sr)
                        waveform = resampler(waveform)
                    if waveform.shape[1] < self.target_size:
                        waveform = torch.nn.functional.pad(waveform, (0, self.target_size - waveform.shape[1]))
                    elif waveform.shape[1] > self.target_size:
                        waveform = waveform[:, :self.target_size]
                    return waveform.squeeze(0), label
                except:
                    return torch.zeros(self.target_size), label
        
        full_dataset = IndexedAudioSetDataset(file_label_pairs)
        
        # Split dataset
        train_size = int(self.split_ratios[0] * len(full_dataset))
        val_size = int(self.split_ratios[1] * len(full_dataset))
        test_size = len(full_dataset) - train_size - val_size
        
        generator = torch.Generator().manual_seed(self.seed)
        self.train_ds, self.val_ds, self.test_ds = random_split(
            full_dataset,
            [train_size, val_size, test_size],
            generator=generator
        )
        
        print(f"\nDataset splits: Train={len(self.train_ds)}, Val={len(self.val_ds)}, Test={len(self.test_ds)}")
    
    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            collate_fn=custom_collate_fn
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            collate_fn=custom_collate_fn
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            collate_fn=custom_collate_fn
        )


# ============================================================================
# PLOT GENERATION FUNCTION
# ============================================================================

def plot_negative_label_distribution(balanced_df, label_mapping_csv, output_path):
    """
    Generate horizontal bar plot of TOP 10 negative label distribution after stratified balancing.
    Style matches analyze_datasets.py plots.
    
    Args:
        balanced_df: DataFrame of balanced negative samples
        label_mapping_csv: Path to class_labels_indices.csv
        output_path: Where to save the figure (.svg format)
    """
    import matplotlib.pyplot as plt
    from collections import Counter
    
    # Load MID → display_name mapping
    mid_to_name = {}
    with open(label_mapping_csv, 'r') as f:
        import csv
        reader = csv.DictReader(f)
        for row in reader:
            mid_to_name[row['mid']] = row['display_name']
    
    # Group by segment and count labels
    segments = ['balanced_train', 'eval']
    label_counts_by_segment = {seg: Counter() for seg in segments}
    total_counts = Counter()
    
    for _, row in balanced_df.iterrows():
        segment = row.get('segment_type', 'unknown')
        if segment in segments:
            try:
                labels = set(ast.literal_eval(row['positive_labels']))
                for mid in labels:
                    label_counts_by_segment[segment][mid] += 1
                    total_counts[mid] += 1
            except:
                continue
    
    # Get TOP 10 MIDs by total count
    top_10_mids = [mid for mid, count in total_counts.most_common(10)]
    
    # Convert to display names (REVERSED for top to bottom)
    top_10_labels = [mid_to_name.get(mid, mid) for mid in top_10_mids]
    top_10_labels = list(reversed(top_10_labels))  # Most common at top
    top_10_mids = list(reversed(top_10_mids))
    
    # Prepare data for plotting
    data_by_segment = {seg: [] for seg in segments}
    for mid in top_10_mids:
        for seg in segments:
            data_by_segment[seg].append(label_counts_by_segment[seg].get(mid, 0))
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, max(8, len(top_10_labels) * 0.5)))
    
    y_pos = np.arange(len(top_10_labels))
    bar_height = 0.25
    colors = ['#2196F3', '#FF9800']  # Blue, Orange
    
    for i, (seg, color) in enumerate(zip(segments, colors)):
        offset = (i - 0.5) * bar_height
        bars = ax.barh(y_pos + offset, data_by_segment[seg], bar_height,
                      label=seg.replace('_', ' ').title(), color=color, alpha=0.8)
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            if width > 0:
                ax.text(width, bar.get_y() + bar.get_height()/2.,
                       f' {int(width)}',
                       ha='left', va='center', fontsize=11)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_10_labels, fontsize=11)
    ax.tick_params(axis='both', which='major', labelsize=11)
    ax.set_xlabel('Number of Samples', fontsize=13, fontweight='bold')
    ax.set_title('Top 10 Negatives Labels by Segment\n(AudioSet EV v2 - After Stratified Balancing)',
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, format='svg', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Figure saved: {output_path}")


# ============================================================================
# MAIN - Testing Script
# ============================================================================

if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    
    print("=" * 80)
    print("DATALOADER TEST - AudioSet Emergency Vehicles v2 (2020 PANNs)")
    print("=" * 80)
    
    # Get current directory (datasets/AudioSet_EV_v2PANNs_2020/)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load negative focus labels from datasets_mapping.json
    mapping_path = os.path.join(os.path.dirname(current_dir), "datasets_mapping.json")
    
    if not os.path.exists(mapping_path):
        print(f"ERROR: datasets_mapping.json not found at {mapping_path}")
        sys.exit(1)
    
    with open(mapping_path, 'r') as f:
        mapping = json.load(f)
    
    # Extract negative labels (value == 0)
    negative_labels = [
        label for label, val in mapping["AUDIOSET"].items() 
        if val == 0
    ]
    
    print(f"\nUsing {len(negative_labels)} negative focus labels for stratification")
    
    # Helper function to count labels
    def count_labels_in_dataset(dataset):
        pos_count = 0
        neg_count = 0
        for idx in range(len(dataset)):
            try:
                _, label = dataset[idx]
                if label == 1:
                    pos_count += 1
                else:
                    neg_count += 1
            except:
                pass
        return pos_count, neg_count
    
    # ========================================================================
    # TEST 1: TRAINING MODE
    # ========================================================================
    print("\n" + "="*80)
    print("TEST 1: TRAINING MODE")
    print("="*80)
    
    # Use absolute paths
    dm_train = AudioSetEV_v2_DataModule(
        TP_csv=os.path.join(current_dir, "EV_Positives.csv"),
        TP_folder=os.path.join(current_dir, "Positive_files"),
        
        TN_csv=os.path.join(current_dir, "EV_Negatives.csv"),
        TN_folder=os.path.join(current_dir, "Negative_files"),
        
        label_mapping_csv=os.path.join(current_dir, "audioset_metadata/class_labels_indices.csv"),
        negative_focus_labels=negative_labels,
        
        TP_subfolders=["balanced_train", "eval", "unbalanced"],
        TN_subfolders=["balanced_train", "eval"],
        
        batch_size=32,
        split_ratios=(0.8, 0.1, 0.1),
        balance_negatives=True,
        seed=42
    )
    
    dm_train.setup()
    
    # Generate label distribution plot
    print("\n" + "─"*80)
    print("GENERATING NEGATIVE LABELS DISTRIBUTION PLOT")
    print("─"*80)
    
    # Create balancer to get balanced negative samples
    balancer = StratifiedNegativeBalancer(
        csv_path=os.path.join(current_dir, "EV_Negatives.csv"),
        label_mapping_csv=os.path.join(current_dir, "audioset_metadata/class_labels_indices.csv"),
        negative_focus_labels=negative_labels,
        seed=42
    )
    pos_df = pd.read_csv(os.path.join(current_dir, "EV_Positives.csv"))
    pos_df = pos_df[pos_df['downloaded'] == True]
    balanced_neg_df = balancer.balance(target_count=len(pos_df))
    
    # Generate plot
    plot_negative_label_distribution(
        balanced_df=balanced_neg_df,
        label_mapping_csv=os.path.join(current_dir, "audioset_metadata/class_labels_indices.csv"),
        output_path=os.path.join(current_dir, "label_distribution_negatives_audioset_ev_v2_balanced.svg")
    )
    
    # Get dataloaders
    train_loader = dm_train.train_dataloader()
    val_loader = dm_train.val_dataloader()
    test_loader = dm_train.test_dataloader()
   
    # ─────────────────────────────────────────────────────────────────────────
    # Train Loader Statistics
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 80)
    print("TRAIN LOADER STATISTICS")
    print("─" * 80)
    
    train_samples = len(train_loader.dataset)
    train_batches = len(train_loader)
    
    print("Counting labels...")
    train_pos, train_neg = count_labels_in_dataset(train_loader.dataset)
    
    print(f"Total samples: {train_samples} ({train_batches} batches)")
    print(f"  - Positives: {train_pos}")
    print(f"  - Negatives: {train_neg}")
    
    # First batch analysis
    print("\nFirst batch analysis:")
    for waveforms, labels in train_loader:
        pos = (labels == 1).sum().item()
        neg = (labels ==  0).sum().item()
        duration = waveforms.shape[2] / 32000
        print(f"  - Samples: {waveforms.shape[0]}")
        print(f"  - Positives: {pos}, Negatives: {neg}")
        print(f"  - Duration: {duration:.2f}s, Sample rate: 32000Hz")
        break
    
    # ─────────────────────────────────────────────────────────────────────────
    # Validation Loader Statistics
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 80)
    print("VALIDATION LOADER STATISTICS")
    print("─" * 80)
    
    val_samples = len(val_loader.dataset)
    val_batches = len(val_loader)
    
    print("Counting labels...")
    val_pos, val_neg = count_labels_in_dataset(val_loader.dataset)
    
    print(f"Total samples: {val_samples} ({val_batches} batches)")
    print(f"  - Positives: {val_pos}")
    print(f"  - Negatives: {val_neg}")
    
    # First batch analysis
    print("\nFirst batch analysis:")
    for waveforms, labels in val_loader:
        pos = (labels == 1).sum().item()
        neg = (labels == 0).sum().item()
        duration = waveforms.shape[2] / 32000
        print(f"  - Samples: {waveforms.shape[0]}")
        print(f"  - Positives: {pos}, Negatives: {neg}")
        print(f"  - Duration: {duration:.2f}s, Sample rate: 32000Hz")
        break
    
    # ─────────────────────────────────────────────────────────────────────────
    # Test Loader Statistics
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 80)
    print("TEST LOADER STATISTICS")
    print("─" * 80)
    
    test_samples = len(test_loader.dataset)
    test_batches = len(test_loader)
    
    print("Counting labels...")
    test_pos, test_neg = count_labels_in_dataset(test_loader.dataset)
    
    print(f"Total samples: {test_samples} ({test_batches} batches)")
    print(f"  - Positives: {test_pos}")
    print(f"  - Negatives: {test_neg}")
    
    # First batch analysis
    print("\nFirst batch analysis:")
    for waveforms, labels in test_loader:
        pos = (labels == 1).sum().item()
        neg = (labels == 0).sum().item()
        duration = waveforms.shape[2] / 32000
        print(f"  - Samples: {waveforms.shape[0]}")
        print(f"  - Positives: {pos}, Negatives: {neg}")
        print(f"  - Duration: {duration:.2f}s, Sample rate: 32000Hz")
        break
    
    # ========================================================================
    # TEST 2: BENCHMARK MODE
    # ========================================================================
    print("\n\n" + "="*80)
    print("TEST 2: BENCHMARK MODE")
    print("="*80)
    
    # Create benchmark dataloader (no shuffle, full test set)
    dm_bench = AudioSetEV_v2_DataModule(
        TP_csv=os.path.join(current_dir, "EV_Positives.csv"),
        TP_folder=os.path.join(current_dir, "Positive_files"),
        
        TN_csv=os.path.join(current_dir, "EV_Negatives.csv"),
        TN_folder=os.path.join(current_dir, "Negative_files"),
        
        label_mapping_csv=os.path.join(current_dir, "audioset_metadata/class_labels_indices.csv"),
        negative_focus_labels=negative_labels,
        
        TP_subfolders=["balanced_train", "eval", "unbalanced"],
        TN_subfolders=["balanced_train", "eval"],
        
        batch_size=32,
        split_ratios=(0.0, 0.0, 1.0),  # All data in test set
        balance_negatives=True,
        seed=42
    )
    
    dm_bench.setup()
    
    # ─────────────────────────────────────────────────────────────────────────
    # Benchmark Statistics
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─"*80)
    print("BENCHMARK STATISTICS")
    print("─"*80)
    
    bench_loader = dm_bench.test_dataloader()
    bench_samples = len(bench_loader.dataset)
    bench_batches = len(bench_loader)
    
    print("Counting labels...")
    bench_pos, bench_neg = count_labels_in_dataset(bench_loader.dataset)
    
    print(f"Total samples: {bench_samples} ({bench_batches} batches)")
    print(f"  - Positives: {bench_pos}")
    print(f"  - Negatives: {bench_neg}")
    
    # First batch analysis
    print("\nFirst batch analysis:")
    for waveforms, labels in bench_loader:
        pos = (labels == 1).sum().item()
        neg = (labels == 0).sum().item()
        duration = waveforms.shape[2] / 32000
        print(f"  - Samples: {waveforms.shape[0]}")
        print(f"  - Positives: {pos}, Negatives: {neg}")
        print(f"  - Duration: {duration:.2f}s, Sample rate: 32000Hz")
        break
    
    # =========================================================================
    # TEST 3: MULTI-CLASS MODE (4-WAY CLASSIFICATION)
    # =========================================================================
    print("\n\n" + "="*80)
    print("TEST 3: MULTI-CLASS MODE (4-WAY CLASSIFICATION)")
    print("="*80)
    
    # Helper function to count multi-class labels
    def count_multiclass_labels(dataset):
        """Count samples per class (0, 1, 2, 3)."""
        counts = {0: 0, 1: 0, 2: 0, 3: 0}
        for idx in range(len(dataset)):
            try:
                _, label = dataset[idx]
                if label in counts:
                    counts[label] += 1
            except:
                pass
        return counts
    
    # Initialize DataModule in multi-class mode
    dm_mc = AudioSetEV_v2_DataModule(
        TP_csv=os.path.join(current_dir, "EV_Positives.csv"),
        TP_folder=os.path.join(current_dir, "Positive_files"),
        TN_csv=os.path.join(current_dir, "EV_Negatives.csv"),
        TN_folder=os.path.join(current_dir, "Negative_files"),
        label_mapping_csv=os.path.join(current_dir, "audioset_metadata/class_labels_indices.csv"),
        negative_focus_labels=negative_labels,
        TP_subfolders=["balanced_train", "eval", "unbalanced"],
        TN_subfolders=["balanced_train", "eval"],
        label_mode='multi_class',
        batch_size=32,
        split_ratios=(0.8, 0.1, 0.1),
        balance_negatives=True,
        seed=42
    )
    
    dm_mc.setup()
    
    # Get dataloaders
    train_loader_mc = dm_mc.train_dataloader()
    val_loader_mc = dm_mc.val_dataloader()
    test_loader_mc = dm_mc.test_dataloader()
    
    # ─────────────────────────────────────────────────────────────────────────
    # Multi-Class Train Loader
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─"*80)
    print("MULTI-CLASS TRAIN LOADER STATISTICS")
    print("─"*80)
    
    train_samples_mc = len(train_loader_mc.dataset)
    train_batches_mc = len(train_loader_mc)
    
    print("Counting labels per class...")
    train_counts = count_multiclass_labels(train_loader_mc.dataset)
    
    print(f"Total samples: {train_samples_mc} ({train_batches_mc} batches)")
    print(f"  - Class 0 (Negative): {train_counts[0]}")
    print(f"  - Class 1 (Police): {train_counts[1]}")
    print(f"  - Class 2 (Ambulance): {train_counts[2]}")
    print(f"  - Class 3 (Fire): {train_counts[3]}")
    
    # First batch analysis
    print("\nFirst batch analysis:")
    for waveforms, labels in train_loader_mc:
        class_dist = Counter(labels.tolist())
        if len(waveforms.shape) == 2:
            duration = waveforms.shape[1] / 32000
        else:
            duration = waveforms.shape[2] / 32000
        print(f"  - Samples: {waveforms.shape[0]}")
        print(f"  - Class distribution: 0:{class_dist.get(0,0)}, 1:{class_dist.get(1,0)}, "
              f"2:{class_dist.get(2,0)}, 3:{class_dist.get(3,0)}")
        print(f"  - Duration: {duration:.2f}s, Sample rate: 32000Hz")
        break
    
    # ─────────────────────────────────────────────────────────────────────────
    # Multi-Class Validation Loader
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─"*80)
    print("MULTI-CLASS VALIDATION LOADER STATISTICS")
    print("─"*80)
    
    val_samples_mc = len(val_loader_mc.dataset)
    val_batches_mc = len(val_loader_mc)
    
    print("Counting labels per class...")
    val_counts = count_multiclass_labels(val_loader_mc.dataset)
    
    print(f"Total samples: {val_samples_mc} ({val_batches_mc} batches)")
    print(f"  - Class 0 (Negative): {val_counts[0]}")
    print(f"  - Class 1 (Police): {val_counts[1]}")
    print(f"  - Class 2 (Ambulance): {val_counts[2]}")
    print(f"  - Class 3 (Fire): {val_counts[3]}")
    
    # First batch analysis
    print("\nFirst batch analysis:")
    for waveforms, labels in val_loader_mc:
        class_dist = Counter(labels.tolist())
        if len(waveforms.shape) == 2:
            duration = waveforms.shape[1] / 32000
        else:
            duration = waveforms.shape[2] / 32000
        print(f"  - Samples: {waveforms.shape[0]}")
        print(f"  - Class distribution: 0:{class_dist.get(0,0)}, 1:{class_dist.get(1,0)}, "
              f"2:{class_dist.get(2,0)}, 3:{class_dist.get(3,0)}")
        print(f"  - Duration: {duration:.2f}s, Sample rate: 32000Hz")
        break
    
    # ─────────────────────────────────────────────────────────────────────────
    # Multi-Class Test Loader
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─"*80)
    print("MULTI-CLASS TEST LOADER STATISTICS")
    print("─"*80)
    
    test_samples_mc = len(test_loader_mc.dataset)
    test_batches_mc = len(test_loader_mc)
    
    print("Counting labels per class...")
    test_counts = count_multiclass_labels(test_loader_mc.dataset)
    
    print(f"Total samples: {test_samples_mc} ({test_batches_mc} batches)")
    print(f"  - Class 0 (Negative): {test_counts[0]}")
    print(f"  - Class 1 (Police): {test_counts[1]}")
    print(f"  - Class 2 (Ambulance): {test_counts[2]}")
    print(f"  - Class 3 (Fire): {test_counts[3]}")
    
    # First batch analysis
    print("\nFirst batch analysis:")
    for waveforms, labels in test_loader_mc:
        class_dist = Counter(labels.tolist())
        if len(waveforms.shape) == 2:
            duration = waveforms.shape[1] / 32000
        else:
            duration = waveforms.shape[2] / 32000
        print(f"  - Samples: {waveforms.shape[0]}")
        print(f"  - Class distribution: 0:{class_dist.get(0,0)}, 1:{class_dist.get(1,0)}, "
              f"2:{class_dist.get(2,0)}, 3:{class_dist.get(3,0)}")
        print(f"  - Duration: {duration:.2f}s, Sample rate: 32000Hz")
        break
    
    # ─────────────────────────────────────────────────────────────────────────
    # Multi-Class Benchmark Mode
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─"*80)
    print("MULTI-CLASS BENCHMARK MODE")
    print("─"*80)
    
    # Initialize in benchmark multi-class mode (all data as test set)
    dm_mc_bench = AudioSetEV_v2_DataModule(
        TP_csv=os.path.join(current_dir, "EV_Positives.csv"),
        TP_folder=os.path.join(current_dir, "Positive_files"),
        
        TN_csv=os.path.join(current_dir, "EV_Negatives.csv"),
        TN_folder=os.path.join(current_dir, "Negative_files"),
        
        label_mapping_csv=os.path.join(current_dir, "audioset_metadata/class_labels_indices.csv"),
        negative_focus_labels=negative_labels,
        
        TP_subfolders=["balanced_train", "eval", "unbalanced"],
        TN_subfolders=["balanced_train", "eval"],
        
        label_mode='multi_class',
        batch_size=32,
        split_ratios=(0.0, 0.0, 1.0),  # All data as test
        balance_negatives=True,
        seed=42
    )
    
    dm_mc_bench.setup()
    
    # Get test loader
    test_loader_mc_bench = dm_mc_bench.test_dataloader()
    
    bench_samples_mc = len(test_loader_mc_bench.dataset)
    bench_batches_mc = len(test_loader_mc_bench)
    
    print("Counting labels per class...")
    bench_counts = count_multiclass_labels(test_loader_mc_bench.dataset)
    
    print(f"\nTotal samples: {bench_samples_mc} ({bench_batches_mc} batches)")
    print(f"  - Class 0 (Negative): {bench_counts[0]}")
    print(f"  - Class 1 (Police): {bench_counts[1]}")
    print(f"  - Class 2 (Ambulance): {bench_counts[2]}")
    print(f"  - Class 3 (Fire): {bench_counts[3]}")
    
    # First batch analysis
    print("\nFirst batch analysis:")
    for waveforms, labels in test_loader_mc_bench:
        class_dist = Counter(labels.tolist())
        if len(waveforms.shape) == 2:
            duration = waveforms.shape[1] / 32000
        else:
            duration = waveforms.shape[2] / 32000
        print(f"  - Samples: {waveforms.shape[0]}")
        print(f"  - Class distribution: 0:{class_dist.get(0,0)}, 1:{class_dist.get(1,0)}, "
              f"2:{class_dist.get(2,0)}, 3:{class_dist.get(3,0)}")
        print(f"  - Duration: {duration:.2f}s, Sample rate: 32000Hz")
        break
    
    print("\n" + "="*80)
    print("TEST COMPLETED SUCCESSFULLY!")
    print("="*80)
