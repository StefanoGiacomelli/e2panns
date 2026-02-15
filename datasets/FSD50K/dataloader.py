"""
FSD50K Dataset Dataloader
=========================
PyTorch Dataset and Lightning DataModule for FSD50K audio classification.

Supports two modes:
- 'train': Merges dev+eval data, random splits into train/dev/test
- 'benchmark': Uses dev/eval natural splits

Author: Stefano Giacomelli - Ph.D. candidate in ICT (DISIM dpt. - University of L'Aquila)
"""

import os
import json
import random
import numpy as np
import pandas as pd
import soundfile as sf
from collections import Counter
from typing import Optional, Tuple, List

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
import torchaudio
import pytorch_lightning as pl
from pytorch_lightning import seed_everything


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def load_label_mapping(dataset_name: str = "FSD50K", 
                       custom_map: Optional[dict] = None,
                       json_path: Optional[str] = None) -> dict:
    """
    Load label mapping from datasets_mapping.json or use custom mapping.
    
    Args:
        dataset_name: Name of the dataset section to load from JSON
        custom_map: Optional custom label mapping dict
        json_path: Optional path to JSON file (defaults to ../datasets_mapping.json)
    
    Returns:
        Dictionary mapping label names to binary labels (0 or 1)
    """
    if custom_map is not None:
        return custom_map
    
    if json_path is None:
        # Default path: go up one level from this file to datasets/
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


# =============================================================================
# DATASET CLASS
# =============================================================================

class FSD50KDataset(Dataset):
    """
    FSD50K Dataset for audio classification.
    
    Loads audio files from CSV files and assigns labels based on label_map.
    Supports optional data augmentation.
    """
    
    def __init__(self,
                 csv_files: List[str],
                 audio_folders: List[str],
                 label_map: Optional[dict] = None,
                 seed: int = 42,
                 augmentation: bool = False,
                 aug_prob: float = 0.7,
                 target_sr: int = 16000):
        """
        Initialize FSD50K Dataset.
        
        Args:
            csv_files: List of CSV file paths to load
            audio_folders: List of corresponding audio folder paths
            label_map: Dict mapping label names to labels (0 or 1)
            seed: Random seed for reproducibility
            augmentation: Whether to apply data augmentation
            aug_prob: Probability of applying each augmentation
            target_sr: Target sample rate (Hz)
        """
        super().__init__()
        
        # Set seeds for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        self.target_sr = target_sr
        self.seed = seed
        
        # Load label mapping
        self.label_map = label_map if label_map is not None else load_label_mapping("FSD50K")
        
        # Augmentation settings
        self.augmentation = augmentation
        self.aug_prob = aug_prob
        if self.augmentation:
            self.augmentations = self._define_augmentations()
        
        # Load file paths and labels from CSVs
        self.file_paths, self.labels = self._load_files(csv_files, audio_folders)
        
        # Track skipped files
        self.skipped_files = []
    
    def _load_files(self, csv_files: List[str], audio_folders: List[str]) -> Tuple[List[str], List[int]]:
        """
        Read CSV files and collect file paths with labels.
        
        Args:
            csv_files: List of CSV paths
            audio_folders: List of corresponding audio folder paths
        
        Returns:
            Tuple of (file_paths, labels)
        """
        file_paths = []
        labels = []
        
        for csv_file, audio_folder in zip(csv_files, audio_folders):
            if not os.path.exists(csv_file):
                print(f"Warning: CSV file not found: {csv_file}")
                continue
            
            if not os.path.exists(audio_folder):
                print(f"Warning: Audio folder not found: {audio_folder}")
                continue
            
            # Read CSV
            try:
                df = pd.read_csv(csv_file)
                
                for idx, row in df.iterrows():
                    # Get file ID (fname column)
                    if 'fname' in df.columns:
                        file_id = str(row['fname']).strip()
                    else:
                        file_id = str(df.iloc[idx, 0]).strip()
                    
                    # Build file path (add .wav extension)
                    file_path = os.path.join(audio_folder, f"{file_id}.wav")
                    
                    # Check if file exists
                    if not os.path.exists(file_path):
                        continue
                    
                    # Get labels from row (can be multi-label)
                    if 'labels' in df.columns:
                        label_str = str(row['labels'])
                        sample_labels = [l.strip() for l in label_str.split(',')]
                    else:
                        # Default: use label based on CSV type (positive/negative)
                        # This is determined by which CSV file we're reading
                        if 'positive' in os.path.basename(csv_file):
                            sample_labels = ['Siren']  # Assume positive
                        else:
                            sample_labels = []  # Negative
                    
                    # Determine binary label: if ANY label in self.label_map has value 1, assign 1
                    binary_label = 0
                    for label in sample_labels:
                        if label in self.label_map and self.label_map[label] == 1:
                            binary_label = 1
                            break
                    
                    # If no positive label found, check if ANY label is in negative list
                    if binary_label == 0:
                        for label in sample_labels:
                            if label in self.label_map and self.label_map[label] == 0:
                                binary_label = 0
                                break
                    
                    file_paths.append(file_path)
                    labels.append(binary_label)
            
            except Exception as e:
                print(f"Error reading CSV {csv_file}: {e}")
                continue
        
        return file_paths, labels
    
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
        return len(self.file_paths)
    
    def __getitem__(self, idx: int) -> Optional[Tuple[torch.Tensor, int]]:
        """
        Load and process audio file.
        
        Args:
            idx: Index of sample
        
        Returns:
            Tuple of (waveform, label) or None if loading fails
        """
        file_path = self.file_paths[idx]
        label = self.labels[idx]
        
        try:
            # Load audio using soundfile
            waveform_np, sr = sf.read(file_path, dtype='float32')
            
            # Convert to torch tensor and ensure correct shape (channels, samples)
            if waveform_np.ndim == 1:
                # Mono audio: add channel dimension
                waveform = torch.from_numpy(waveform_np).unsqueeze(0)
            else:
                # Stereo/multi-channel: convert to mono by averaging
                waveform = torch.from_numpy(waveform_np.mean(axis=1)).unsqueeze(0)
            
            # Resample if necessary
            if sr != self.target_sr:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.target_sr)
                waveform = resampler(waveform)
            
            # Apply augmentation if enabled
            if self.augmentation:
                waveform = self._apply_augmentations(waveform)
        
        except Exception as e:
            self.skipped_files.append((idx, file_path))
            print(f"Error loading {file_path}: {e}")
            return None
        
        return waveform, label


# =============================================================================
# COLLATE FUNCTION
# =============================================================================

def fsd50k_collate_fn(batch):
    """
    Custom collate function with dynamic padding.
    
    Args:
        batch: List of samples from Dataset
    
    Returns:
        Tuple of (batched_waveforms, batched_labels)
    """
    # Filter out None samples
    batch = [item for item in batch if item is not None]
    
    if not batch:
        raise ValueError("All samples in the batch are invalid.")
    
    waveforms, labels = zip(*batch)
    
    # Find the maximum length in the batch
    max_length = max(waveform.size(1) for waveform in waveforms)
    
    # Pad all waveforms to the maximum length
    padded_waveforms = torch.stack([F.pad(waveform, (0, max_length - waveform.size(1)), "constant", 0) for waveform in waveforms])
    
    # Convert labels to a tensor
    labels = torch.tensor(labels, dtype=torch.long)
    
    return padded_waveforms, labels


# =============================================================================
# DATAMODULE CLASS
# =============================================================================

class FSD50KDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for FSD50K.
    
    Supports two modes:
    - 'train': Merges dev+eval data and randomly splits into train/dev/test
    - 'benchmark': Uses dev dataset for test (natural split)
    """
    
    def __init__(self,
                 fsd_root: str,
                 mode: str = 'train',
                 seed: int = 42,
                 batch_size: int = 32,
                 split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
                 shuffle: bool = True,
                 augmentation: bool = False,
                 aug_prob: float = 0.7,
                 label_map: Optional[dict] = None,
                 target_sr: int = 16000,
                 num_workers: int = 2):
        """
        Initialize FSD50K DataModule.
        
        Args:
            fsd_root: Path to FSD50K root folder
            mode: 'train' or 'benchmark'
            seed: Random seed for reproducibility
            batch_size: Batch size for dataloaders
            split_ratios: (train, dev, test) ratios for train mode
            shuffle: Whether to shuffle training data
            augmentation: Whether to apply data augmentation
            aug_prob: Probability of applying augmentation
            label_map: Optional custom label mapping
            target_sr: Target sample rate
            num_workers: Number of dataloader workers
        """
        super().__init__()
        
        # Set global seed
        seed_everything(seed, workers=True)
        
        self.fsd_root = fsd_root
        self.mode = mode
        self.seed = seed
        self.batch_size = batch_size
        self.split_ratios = split_ratios
        self.train_shuffle = shuffle
        self.augmentation = augmentation
        self.aug_prob = aug_prob
        self.label_map = label_map
        self.target_sr = target_sr
        self.num_workers = num_workers
        
        # Define CSV and audio folder paths
        self.pos_dev_csv = os.path.join(fsd_root, "FSD-dev_positives.csv")
        self.neg_dev_csv = os.path.join(fsd_root, "FSD-dev_negatives.csv")
        self.pos_eval_csv = os.path.join(fsd_root, "FSD-eval_positives.csv")
        self.neg_eval_csv = os.path.join(fsd_root, "FSD-eval_negatives.csv")
        self.dev_audio = os.path.join(fsd_root, "FSD50K.dev_audio")
        self.eval_audio = os.path.join(fsd_root, "FSD50K.eval_audio")
        
        # Datasets (will be initialized in setup())
        self.train_dataset = None
        self.dev_dataset = None
        self.test_dataset = None
        
        # Generator for reproducible random_split
        self.generator = torch.Generator().manual_seed(seed)
    
    def setup(self, stage: Optional[str] = None):
        """
        Setup datasets based on mode.
        
        Args:
            stage: Optional stage ('fit', 'test', etc.)
        """
        if self.mode == 'train':
            self._setup_train_mode()
        elif self.mode == 'benchmark':
            self._setup_benchmark_mode()
        else:
            raise ValueError(f"Invalid mode: {self.mode}. Must be 'train' or 'benchmark'.")
    
    def _setup_train_mode(self):
        """Setup for training mode: dev folder→train, eval folder→dev/test split."""
        # Dev data for TRAINING
        self.train_dataset = FSD50KDataset(csv_files=[self.pos_dev_csv, self.neg_dev_csv],
                                          audio_folders=[self.dev_audio, self.dev_audio],
                                          label_map=self.label_map,
                                          seed=self.seed,
                                          augmentation=self.augmentation,
                                          aug_prob=self.aug_prob,
                                          target_sr=self.target_sr)
        
        # Eval data for DEV/TEST split
        eval_dataset = FSD50KDataset(csv_files=[self.pos_eval_csv, self.neg_eval_csv],
                                     audio_folders=[self.eval_audio, self.eval_audio],
                                     label_map=self.label_map,
                                     seed=self.seed,
                                     augmentation=False,  # No augmentation for dev/test
                                     target_sr=self.target_sr)
        
        # Split eval into dev and test (50/50)
        eval_len = len(eval_dataset)
        dev_len = eval_len // 2
        test_len = eval_len - dev_len
        
        # Random split with reproducible generator
        self.dev_dataset, self.test_dataset = random_split(eval_dataset,
                                                           [dev_len, test_len],
                                                           generator=self.generator)
    
    def _setup_benchmark_mode(self):
        """Setup for benchmark mode: use eval folder only."""
        # Load eval data as test set (positive + negative)
        self.test_dataset = FSD50KDataset(csv_files=[self.pos_eval_csv, self.neg_eval_csv],
                                          audio_folders=[self.eval_audio, self.eval_audio],
                                          label_map=self.label_map,
                                          seed=self.seed,
                                          augmentation=False,  # No augmentation in benchmark
                                          target_sr=self.target_sr)
    
    def _seed_worker(self, worker_id):
        """Seed worker for reproducible DataLoader."""
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    
    def train_dataloader(self) -> DataLoader:
        """Return training dataloader."""
        if self.mode != 'train':
            raise ValueError("train_dataloader() only available in 'train' mode")
        
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=self.train_shuffle,
                          num_workers=self.num_workers,
                          collate_fn=fsd50k_collate_fn,
                          worker_init_fn=self._seed_worker,
                          generator=self.generator,
                          persistent_workers=True if self.num_workers > 0 else False)
    
    def val_dataloader(self) -> DataLoader:
        """Return validation dataloader."""
        if self.mode != 'train':
            raise ValueError("val_dataloader() only available in 'train' mode")
        
        return DataLoader(self.dev_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          collate_fn=fsd50k_collate_fn,
                          persistent_workers=True if self.num_workers > 0 else False)
    
    def test_dataloader(self) -> DataLoader:
        """Return test dataloader."""
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          collate_fn=fsd50k_collate_fn,
                          persistent_workers=True if self.num_workers > 0 else False)


# =============================================================================
# HELPER FUNCTION
# =============================================================================

def count_labels_in_dataset(dataset):
    """
    Count positive and negative labels in a dataset.
    
    Args:
        dataset: PyTorch Dataset or Subset
    
    Returns:
        Tuple of (positive_count, negative_count)
    """
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


# =============================================================================
# TESTING CODE
# =============================================================================

if __name__ == "__main__":
    import sys
    
    print("=" * 80)
    print("DATALOADER TEST - FSD50K")
    print("=" * 80)
    
    # Get current directory (datasets/FSD50K/)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # =========================================================================
    # TEST 1: TRAINING MODE
    # =========================================================================
    print("\n" + "=" * 80)
    print("TEST 1: TRAINING MODE")
    print("=" * 80)
    
    # Initialize DataModule in training mode
    dm_train = FSD50KDataModule(fsd_root=current_dir,
                                mode='train',
                                seed=42,
                                batch_size=32,
                                shuffle=True,
                                augmentation=False,
                                target_sr=16000,
                                num_workers=0)  # For testing
    
    # Setup
    dm_train.setup()
    
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
    
    print(f"Counting labels...")
    train_pos, train_neg = count_labels_in_dataset(train_loader.dataset)
    
    print(f"Total samples: {train_samples} ({train_batches} batches)")
    print(f"  - Positives: {train_pos}")
    print(f"  - Negatives: {train_neg}")
    
    # First batch analysis
    print(f"\nFirst batch analysis:")
    for waveforms, labels in train_loader:
        pos = (labels == 1).sum().item()
        neg = (labels == 0).sum().item()
        duration = waveforms.shape[2] / 16000
        print(f"  - Samples: {waveforms.shape[0]}")
        print(f"  - Positives: {pos}, Negatives: {neg}")
        print(f"  - Duration: {duration:.2f}s, Sample rate: 16000Hz")
        break
    
    # ─────────────────────────────────────────────────────────────────────────
    # Validation Loader Statistics
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 80)
    print("VALIDATION LOADER STATISTICS")
    print("─" * 80)
    
    val_samples = len(val_loader.dataset)
    val_batches = len(val_loader)
    
    print(f"Counting labels...")
    val_pos, val_neg = count_labels_in_dataset(val_loader.dataset)
    
    print(f"Total samples: {val_samples} ({val_batches} batches)")
    print(f"  - Positives: {val_pos}")
    print(f"  - Negatives: {val_neg}")
    
    # First batch analysis
    print(f"\nFirst batch analysis:")
    for waveforms, labels in val_loader:
        pos = (labels == 1).sum().item()
        neg = (labels == 0).sum().item()
        duration = waveforms.shape[2] / 16000
        print(f"  - Samples: {waveforms.shape[0]}")
        print(f"  - Positives: {pos}, Negatives: {neg}")
        print(f"  - Duration: {duration:.2f}s, Sample rate: 16000Hz")
        break
    
    # ─────────────────────────────────────────────────────────────────────────
    # Test Loader Statistics
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 80)
    print("TEST LOADER STATISTICS")
    print("─" * 80)
    
    test_samples = len(test_loader.dataset)
    test_batches = len(test_loader)
    
    print(f"Counting labels...")
    test_pos, test_neg = count_labels_in_dataset(test_loader.dataset)
    
    print(f"Total samples: {test_samples} ({test_batches} batches)")
    print(f"  - Positives: {test_pos}")
    print(f"  - Negatives: {test_neg}")
    
    # First batch analysis
    print(f"\nFirst batch analysis:")
    for waveforms, labels in test_loader:
        pos = (labels == 1).sum().item()
        neg = (labels == 0).sum().item()
        duration = waveforms.shape[2] / 16000
        print(f"  - Samples: {waveforms.shape[0]}")
        print(f"  - Positives: {pos}, Negatives: {neg}")
        print(f"  - Duration: {duration:.2f}s, Sample rate: 16000Hz")
        break
    
    # =========================================================================
    # TEST 2: BENCHMARK MODE
    # =========================================================================
    print("\n\n" + "=" * 80)
    print("TEST 2: BENCHMARK MODE")
    print("=" * 80)
    
    # Initialize DataModule in benchmark mode
    dm_bench = FSD50KDataModule(fsd_root=current_dir,
                                mode='benchmark',
                                seed=42,
                                batch_size=32,
                                target_sr=16000,
                                num_workers=0)  # For testing
    
    # Setup
    dm_bench.setup()
    
    # Get test loader
    test_loader_bench = dm_bench.test_dataloader()
    
    # ─────────────────────────────────────────────────────────────────────────
    # Benchmark Statistics
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 80)
    print("BENCHMARK STATISTICS")
    print("─" * 80)
    
    bench_samples = len(test_loader_bench.dataset)
    bench_batches = len(test_loader_bench)
    
    print(f"Counting labels...")
    bench_pos, bench_neg = count_labels_in_dataset(test_loader_bench.dataset)
    
    print(f"Total samples: {bench_samples} ({bench_batches} batches)")
    print(f"  - Positives: {bench_pos}")
    print(f"  - Negatives: {bench_neg}")
    
    # First batch analysis
    print(f"\nFirst batch analysis:")
    for waveforms, labels in test_loader_bench:
        pos = (labels == 1).sum().item()
        neg = (labels == 0).sum().item()
        duration = waveforms.shape[2] / 16000
        print(f"  - Samples: {waveforms.shape[0]}")
        print(f"  - Positives: {pos}, Negatives: {neg}")
        print(f"  - Duration: {duration:.2f}s, Sample rate: 16000Hz")
        break
    
    print("\n" + "=" * 80)
    print("TEST COMPLETED SUCCESSFULLY!")
    print("=" * 80)
