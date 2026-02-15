"""
AudioSet Emergency Vehicles v1 (2025) Dataset Dataloader
=========================================================
PyTorch Dataset and Lightning DataModule for AudioSet EV v1 audio classification.

Supports two modes:
- 'train': Loads all downloaded samples, random splits into train/dev/test
- 'benchmark': Loads all downloaded samples as test set

Dataset Structure:
- EV_Positives.csv: Emergency vehicle sounds (label=1)
- EV_Negatives.csv: Non-EV Urban/Challenging sounds (label=0)
- Positive_files/: Audio files for positives
- Negative_files/: Audio files for negatives

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

def load_label_mapping(dataset_name: str = "AudioSet_EV_v1", 
                       custom_map: Optional[dict] = None,
                       json_path: Optional[str] = None) -> dict:
    """
    Load label mapping from datasets_mapping.json or use custom mapping.
    
    Args:
        dataset_name: Name of the dataset section to load from JSON
        custom_map: Optional custom label mapping dict
        json_path: Optional path to JSON file (defaults to ../datasets_mapping.json)
    
    Returns:
        Dictionary mapping 'Positive' and 'Negative' to binary labels (1 or 0)
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

class AudioSetEV_v1_Dataset(Dataset):
    """
    AudioSet Emergency Vehicles v1 Dataset for audio classification.
    
    Loads audio files from CSV files and assigns labels based on label_map.
    Supports optional data augmentation.
    """
    
    def __init__(self,
                 csv_path: str,
                 audio_folder: str,
                 binary_label: int,
                 seed: int = 42,
                 augmentation: bool = False,
                 aug_prob: float = 0.7,
                 target_size: int = 320000,
                 target_sr: int = 32000):
        """
        Initialize AudioSet EV v1 Dataset.
        
        Args:
            csv_path: Path to CSV file (EV_Positives.csv or EV_Negatives.csv)
            audio_folder: Path to audio folder (Positive_files/ or Negative_files/)
            binary_label: Label to assign (1 for positives, 0 for negatives)
            seed: Random seed for reproducibility
            augmentation: Whether to apply data augmentation
            aug_prob: Probability of applying each augmentation
            target_size: Target number of audio samples (default: 320000 = 10s @ 32kHz)
            target_sr: Target sample rate (Hz)
        """
        super().__init__()
        
        # Set seeds for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        self.csv_path = os.path.abspath(csv_path)
        self.audio_folder = os.path.abspath(audio_folder)
        self.binary_label = binary_label
        self.target_size = target_size
        self.target_sr = target_sr
        self.seed = seed
        
        # Augmentation settings
        self.augmentation = augmentation
        self.aug_prob = aug_prob
        if self.augmentation:
            self.augmentations = self._define_augmentations()
        
        # Load file paths from CSV
        self.file_paths = self._load_files()
        
        # Track skipped files
        self.skipped_files = []
    
    def _load_files(self) -> List[str]:
        """
        Read CSV file and collect file paths for downloaded samples.
        
        Returns:
            List of file paths
        """
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")
        
        if not os.path.exists(self.audio_folder):
            raise FileNotFoundError(f"Audio folder not found: {self.audio_folder}")
        
        # Read CSV
        df = pd.read_csv(self.csv_path)
        
        # Filter only downloaded samples
        if 'downloaded' in df.columns:
            df = df[df['downloaded'] == True]
        
        file_paths = []
        
        # Iterate through rows
        for _, row in df.iterrows():
            yt_id = row['yt_id']
            
            # Try both Original and Reduced versions
            for suffix in ['_Original.wav', '_Reduced.wav']:
                filename = f"{yt_id}{suffix}"
                file_path = os.path.join(self.audio_folder, filename)
                
                if os.path.exists(file_path):
                    file_paths.append(file_path)
                    break  # Use first match
        
        return file_paths
    
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
        label = self.binary_label
        
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
            
            # Pad or truncate to target_size
            current_size = waveform.size(1)
            if current_size < self.target_size:
                padding = self.target_size - current_size
                waveform = F.pad(waveform, (0, padding), "constant", 0)
            elif current_size > self.target_size:
                waveform = waveform[:, :self.target_size]
            
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

def audioset_ev_v1_collate_fn(batch):
    """
    Custom collate function to handle None values from failed loads.
    
    Args:
        batch: List of samples from Dataset
    
    Returns:
        Tuple of (batched_waveforms, batched_labels) or (None, None)
    """
    # Filter out None samples
    batch = [item for item in batch if item is not None]
    
    if not batch:
        return None, None
    
    return torch.utils.data.default_collate(batch)


# =============================================================================
# DATAMODULE CLASS
# =============================================================================

class AudioSetEV_v1_DataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for AudioSet Emergency Vehicles v1.
    
    Supports two modes:
    - 'train': Merges positives+negatives and randomly splits into train/dev/test
    - 'benchmark': Loads all data as test set
    """
    
    def __init__(self,
                 pos_csv_path: str,
                 pos_audio_folder: str,
                 neg_csv_path: str,
                 neg_audio_folder: str,
                 mode: str = 'train',
                 seed: int = 42,
                 batch_size: int = 32,
                 split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
                 shuffle: bool = True,
                 augmentation: bool = False,
                 aug_prob: float = 0.7,
                 label_map: Optional[dict] = None,
                 target_size: int = 320000,
                 target_sr: int = 32000,
                 num_workers: int = 2):
        """
        Initialize AudioSet EV v1 DataModule.
        
        Args:
            pos_csv_path: Path to EV_Positives.csv
            pos_audio_folder: Path to Positive_files/
            neg_csv_path: Path to EV_Negatives.csv
            neg_audio_folder: Path to Negative_files/
            mode: 'train' or 'benchmark'
            seed: Random seed for reproducibility
            batch_size: Batch size for dataloaders
            split_ratios: (train, dev, test) ratios for train mode
            shuffle: Whether to shuffle training data
            augmentation: Whether to apply data augmentation
            aug_prob: Probability of applying augmentation
            label_map: Optional custom label mapping
            target_size: Target audio length in samples
            target_sr: Target sample rate
            num_workers: Number of dataloader workers
        """
        super().__init__()
        
        # Set global seed
        seed_everything(seed, workers=True)
        
        self.pos_csv_path = pos_csv_path
        self.pos_audio_folder = pos_audio_folder
        self.neg_csv_path = neg_csv_path
        self.neg_audio_folder = neg_audio_folder
        self.mode = mode
        self.seed = seed
        self.batch_size = batch_size
        self.split_ratios = split_ratios
        self.train_shuffle = shuffle
        self.augmentation = augmentation
        self.aug_prob = aug_prob
        self.label_map = label_map
        self.target_size = target_size
        self.target_sr = target_sr
        self.num_workers = num_workers
        
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
        """Setup for training mode: merge positives+negatives and split randomly."""
        # Create positive dataset
        pos_dataset = AudioSetEV_v1_Dataset(csv_path=self.pos_csv_path,
                                            audio_folder=self.pos_audio_folder,
                                            binary_label=1,
                                            seed=self.seed,
                                            augmentation=self.augmentation,
                                            aug_prob=self.aug_prob,
                                            target_size=self.target_size,
                                            target_sr=self.target_sr)
        
        # Create negative dataset
        neg_dataset = AudioSetEV_v1_Dataset(csv_path=self.neg_csv_path,
                                            audio_folder=self.neg_audio_folder,
                                            binary_label=0,
                                            seed=self.seed,
                                            augmentation=self.augmentation,
                                            aug_prob=self.aug_prob,
                                            target_size=self.target_size,
                                            target_sr=self.target_sr)
        
        # Concatenate datasets
        full_dataset = ConcatDataset([pos_dataset, neg_dataset])
        
        # Compute split sizes
        total_len = len(full_dataset)
        train_len = int(self.split_ratios[0] * total_len)
        dev_len = int(self.split_ratios[1] * total_len)
        test_len = total_len - train_len - dev_len
        
        # Random split with reproducible generator
        self.train_dataset, self.dev_dataset, self.test_dataset = random_split(full_dataset,
                                                                               [train_len, dev_len, test_len],
                                                                               generator=self.generator)
    
    def _setup_benchmark_mode(self):
        """Setup for benchmark mode: load all data as test set."""
        # Create positive dataset (no augmentation in benchmark)
        pos_dataset = AudioSetEV_v1_Dataset(csv_path=self.pos_csv_path,
                                            audio_folder=self.pos_audio_folder,
                                            binary_label=1,
                                            seed=self.seed,
                                            augmentation=False,
                                            target_size=self.target_size,
                                            target_sr=self.target_sr)
        
        # Create negative dataset (no augmentation in benchmark)
        neg_dataset = AudioSetEV_v1_Dataset(csv_path=self.neg_csv_path,
                                            audio_folder=self.neg_audio_folder,
                                            binary_label=0,
                                            seed=self.seed,
                                            augmentation=False,
                                            target_size=self.target_size,
                                            target_sr=self.target_sr)
        
        # Concatenate as test set
        self.test_dataset = ConcatDataset([pos_dataset, neg_dataset])
    
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
                          collate_fn=audioset_ev_v1_collate_fn,
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
                          collate_fn=audioset_ev_v1_collate_fn,
                          persistent_workers=True if self.num_workers > 0 else False)
    
    def test_dataloader(self) -> DataLoader:
        """Return test dataloader."""
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          collate_fn=audioset_ev_v1_collate_fn,
                          persistent_workers=True if self.num_workers > 0 else False)


# =============================================================================
# TESTING CODE
# =============================================================================

if __name__ == "__main__":  
    print("=" * 80)
    print("DATALOADER TEST - AudioSet Emergency Vehicles v1 (2025)")
    print("=" * 80)
    
    # Get current directory (datasets/AudioSet_EV_v1_2025/)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define paths
    pos_csv = os.path.join(current_dir, "EV_Positives.csv")
    pos_folder = os.path.join(current_dir, "Positive_files")
    neg_csv = os.path.join(current_dir, "EV_Negatives.csv")
    neg_folder = os.path.join(current_dir, "Negative_files")
    
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
    
    # =========================================================================
    # TEST 1: TRAINING MODE
    # =========================================================================
    print("\n" + "=" * 80)
    print("TEST 1: TRAINING MODE")
    print("=" * 80)
    
    # Initialize DataModule in training mode
    dm_train = AudioSetEV_v1_DataModule(pos_csv_path=pos_csv,
                                        pos_audio_folder=pos_folder,
                                        neg_csv_path=neg_csv,
                                        neg_audio_folder=neg_folder,
                                        mode='train',
                                        seed=42,
                                        batch_size=32,
                                        split_ratios=(0.8, 0.1, 0.1),
                                        shuffle=True,
                                        augmentation=False,
                                        target_size=320000,  # 10s @ 32kHz
                                        target_sr=32000,
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
    
    print("Counting labels...")
    train_pos, train_neg = count_labels_in_dataset(train_loader.dataset)
    
    print(f"Total samples: {train_samples} ({train_batches} batches)")
    print(f"  - Positives: {train_pos}")
    print(f"  - Negatives: {train_neg}")
    
    # First batch analysis
    print("\nFirst batch analysis:")
    for waveforms, labels in train_loader:
        pos = (labels == 1).sum().item()
        neg = (labels == 0).sum().item()
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
    
    # =========================================================================
    # TEST 2: BENCHMARK MODE
    # =========================================================================
    print("\n\n" + "=" * 80)
    print("TEST 2: BENCHMARK MODE")
    print("=" * 80)
    
    # Initialize DataModule in benchmark mode
    dm_bench = AudioSetEV_v1_DataModule(pos_csv_path=pos_csv,
                                        pos_audio_folder=pos_folder,
                                        neg_csv_path=neg_csv,
                                        neg_audio_folder=neg_folder,
                                        mode='benchmark',
                                        seed=42,
                                        batch_size=32,
                                        augmentation=False,
                                        target_size=320000,
                                        target_sr=32000,
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
    
    print("Counting labels...")
    bench_pos, bench_neg = count_labels_in_dataset(test_loader_bench.dataset)
    
    print(f"Total samples: {bench_samples} ({bench_batches} batches)")
    print(f"  - Positives: {bench_pos}")
    print(f"  - Negatives: {bench_neg}")
    
    # First batch analysis
    print("\nFirst batch analysis:")
    for waveforms, labels in test_loader_bench:
        pos = (labels == 1).sum().item()
        neg = (labels == 0).sum().item()
        duration = waveforms.shape[2] / 32000
        print(f"  - Samples: {waveforms.shape[0]}")
        print(f"  - Positives: {pos}, Negatives: {neg}")
        print(f"  - Duration: {duration:.2f}s, Sample rate: 32000Hz")
        break
    
    print("\n" + "=" * 80)
    print("TEST COMPLETED SUCCESSFULLY!")
    print("=" * 80)
