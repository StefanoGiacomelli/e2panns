"""
sireNNet Dataset Dataloader
============================
PyTorch Dataset and Lightning DataModule for sireNNet audio classification.

Supports two modes:
- 'train': Loads all data, random splits into train/dev/test
- 'benchmark': Loads all data as test set

Author: Stefano Giacomelli - Ph.D. candidate in ICT (DISIM dpt. - University of L'Aquila)
"""

import os
import json
import random
import numpy as np
import soundfile as sf
from collections import Counter
from typing import Optional, Tuple, List

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import torchaudio
import pytorch_lightning as pl
from pytorch_lightning import seed_everything


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def load_label_mapping(dataset_name: str = "SIRENNET", 
                       custom_map: Optional[dict] = None,
                       json_path: Optional[str] = None) -> dict:
    """
    Load label mapping from datasets_mapping.json or use custom mapping.
    
    Args:
        dataset_name: Name of the dataset section to load from JSON
        custom_map: Optional custom label mapping dict
        json_path: Optional path to JSON file (defaults to ../datasets_mapping.json)
    
    Returns:
        Dictionary mapping folder names to binary labels (0 or 1)
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

class sireNNetDataset(Dataset):
    """
    sireNNet Dataset for audio classification.
    
    Loads audio files from category folders and assigns labels based on label_map.
    Supports optional data augmentation.
    """
    
    def __init__(self,
                 folder_path: str,
                 label_map: Optional[dict] = None,
                 seed: int = 42,
                 augmentation: bool = False,
                 aug_prob: float = 0.7,
                 target_size: int = 96000,
                 target_sr: int = 32000):
        """
        Initialize sireNNet Dataset.
        
        Args:
            folder_path: Path to root folder containing category subfolders
            label_map: Dict mapping folder names to labels (0 or 1)
            seed: Random seed for reproducibility
            augmentation: Whether to apply data augmentation
            aug_prob: Probability of applying each augmentation
            target_size: Target number of audio samples
            target_sr: Target sample rate (Hz)
        """
        super().__init__()
        
        # Set seeds for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        self.folder_path = os.path.abspath(folder_path)
        self.target_size = target_size
        self.target_sr = target_sr
        self.seed = seed
        
        # Load label mapping
        self.label_map = label_map if label_map is not None else load_label_mapping("SIRENNET")
        
        # Augmentation settings
        self.augmentation = augmentation
        self.aug_prob = aug_prob
        if self.augmentation:
            self.augmentations = self._define_augmentations()
        
        # Load file paths and labels
        self.file_paths, self.labels = self._load_files()
        
        # Track skipped files
        self.skipped_files = []
    
    def _load_files(self) -> Tuple[List[str], List[int]]:
        """
        Scan category folders and collect file paths with labels.
        
        Returns:
            Tuple of (file_paths, labels)
        """
        file_paths = []
        labels = []
        
        for category, label in self.label_map.items():
            category_path = os.path.join(self.folder_path, category)
            
            if not os.path.exists(category_path):
                print(f"Warning: Category folder not found: {category_path}")
                continue
            
            # Get all wav files in this category folder
            for file_name in os.listdir(category_path):
                if file_name.endswith('.wav'):
                    file_path = os.path.join(category_path, file_name)
                    file_paths.append(file_path)
                    labels.append(label)
        
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

def sirennet_collate_fn(batch):
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

class sireNNetDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for sireNNet.
    
    Supports two modes:
    - 'train': Randomly splits data into train/dev/test
    - 'benchmark': Uses all data as test set
    
    Supports two label modes:
    - 'binary': Binary classification (0=negative, 1=positive)
    - 'multi_class': 4-class classification (0=traffic, 1=police, 2=ambulance, 3=fire)
    """
    
    def __init__(self,
                 folder_path: str,
                 mode: str = 'train',
                 label_mode: str = 'binary',
                 seed: int = 42,
                 batch_size: int = 32,
                 split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
                 shuffle: bool = True,
                 augmentation: bool = False,
                 aug_prob: float = 0.7,
                 label_map: Optional[dict] = None,
                 target_size: int = 96000,
                 target_sr: int = 32000,
                 num_workers: int = 2):
        """
        Initialize sireNNet DataModule.
        
        Args:
            folder_path: Path to root folder with category subfolders
            mode: 'train' or 'benchmark'
            label_mode: 'binary' (0/1) or 'multi_class' (0/1/2/3)
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
        
        self.folder_path = folder_path
        self.mode = mode
        self.label_mode = label_mode
        self.seed = seed
        self.batch_size = batch_size
        self.split_ratios = split_ratios
        self.train_shuffle = shuffle
        self.augmentation = augmentation
        self.aug_prob = aug_prob
        self.target_size = target_size
        self.target_sr = target_sr
        self.num_workers = num_workers
        
        # Set label_map based on label_mode (if not provided)
        if label_map is None:
            if label_mode == 'multi_class':
                self.label_map = load_label_mapping("SIRENNET_MULTICLASS")
            else:
                self.label_map = load_label_mapping("SIRENNET")
        else:
            self.label_map = label_map
        
        # Datasets (will be initialized in setup())
        self.train_dataset = None
        self.dev_dataset = None
        self.test_dataset = None
        self.test_datasets = {}  # For benchmark mode (partition-based)
        
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
        """Setup for training mode: load all data and split randomly."""
        
        if self.label_mode == 'multi_class':
            # Multi-class mode: balance 4 classes first, then split
            self._setup_train_mode_multiclass()
        else:
            # Binary mode: original behavior
            self._setup_train_mode_binary()
    
    def _setup_train_mode_binary(self):
        """Setup for binary training mode (original behavior)."""
        # Create full dataset
        full_dataset = sireNNetDataset(folder_path=self.folder_path,
                                       label_map=self.label_map,
                                       seed=self.seed,
                                       augmentation=self.augmentation,
                                       aug_prob=self.aug_prob,
                                       target_size=self.target_size,
                                       target_sr=self.target_sr)
        
        # Compute split sizes
        total_len = len(full_dataset)
        train_len = int(self.split_ratios[0] * total_len)
        dev_len = int(self.split_ratios[1] * total_len)
        test_len = total_len - train_len - dev_len
        
        # Random split with reproducible generator
        self.train_dataset, self.dev_dataset, self.test_dataset = random_split(full_dataset,
                                                                               [train_len, dev_len, test_len],
                                                                               generator=self.generator)
    
    def _setup_train_mode_multiclass(self):
        """Setup for multi-class training mode with 4-way balancing."""
        # Add parent directory to path for imports
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from multi_class_utils import FourWayBalancer, print_balance_summary
        
        # Collect all files by class
        class_files = {0: [], 1: [], 2: [], 3: []}  # traffic, police, ambulance, fire
        
        for folder, label in self.label_map.items():
            folder_path = os.path.join(self.folder_path, folder)
            if not os.path.exists(folder_path):
                continue
            
            files = [os.path.join(folder_path, f) 
                    for f in os.listdir(folder_path) if f.endswith('.wav')]
            class_files[label] = files
        
        # Balance 4 classes
        balancer = FourWayBalancer(target_mode='auto', min_samples_per_class=10)
        
        # Convert file lists to indices for balancer
        pure_samples = {cls: list(range(len(files))) for cls, files in class_files.items()}
        
        result = balancer.balance(pure_samples=pure_samples, seed=self.seed)
        print_balance_summary(result, title="sireNNet 4-Way Balance (Train Mode)")
        
        # Create balanced dataset
        balanced_file_paths = []
        balanced_labels = []
        
        for cls in [0, 1, 2, 3]:
            for idx in result['balanced_indices'][cls]:
                balanced_file_paths.append(class_files[cls][idx])
                balanced_labels.append(cls)
        
        # Create dataset with balanced files
        full_dataset = sireNNetDataset(folder_path=self.folder_path,
                                       label_map=self.label_map,
                                       seed=self.seed,
                                       augmentation=self.augmentation,
                                       aug_prob=self.aug_prob,
                                       target_size=self.target_size,
                                       target_sr=self.target_sr)
        
        # Override file_paths and labels with balanced ones
        full_dataset.file_paths = balanced_file_paths
        full_dataset.labels = balanced_labels
        
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
        """Setup for benchmark mode: progressive partitions CV with balanced classes."""
        
        if self.label_mode == 'multi_class':
            # Multi-class mode: balance 4 classes in each partition
            self._setup_benchmark_mode_multiclass()
        else:
            # Binary mode: original behavior (2-way balance)
            self._setup_benchmark_mode_binary()
    
    def _setup_benchmark_mode_binary(self):
        """Setup benchmark mode for binary classification (original behavior)."""
        # Create full dataset
        full_dataset = sireNNetDataset(folder_path=self.folder_path,
                                      label_map=self.label_map,
                                      seed=self.seed,
                                      augmentation=False,  # No augmentation in benchmark
                                      target_size=self.target_size,
                                      target_sr=self.target_sr)
        
        total_samples = len(full_dataset)
        
        # Separate indices by label (positive vs negative)
        pos_indices = []
        neg_indices = []
        
        for idx in range(total_samples):
            try:
                _, label = full_dataset[idx]
                if label == 1:
                    pos_indices.append(idx)
                else:
                    neg_indices.append(idx)
            except:
                pass
        
        # Progressive partitions as specified
        partitions = [0.0025, 0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.0]
        
        # Create balanced subset datasets for each partition
        for partition in partitions:
            n_samples = int(total_samples * partition)
            
            # Calculate balanced split (50/50 pos/neg)
            n_pos = n_samples // 2
            n_neg = n_samples - n_pos
            
            # Ensure we don't exceed available samples
            n_pos = min(n_pos, len(pos_indices))
            n_neg = min(n_neg, len(neg_indices))
            
            # Take first n_pos positives and first n_neg negatives (deterministic)
            selected_indices = pos_indices[:n_pos] + neg_indices[:n_neg]
            
            # Create subset
            subset = Subset(full_dataset, selected_indices)
            self.test_datasets[partition] = subset
    
    def _setup_benchmark_mode_multiclass(self):
        """Setup benchmark mode for multi-class with 4-way balance (best effort)."""
        # Collect all files by class
        class_files = {0: [], 1: [], 2: [], 3: []}  # traffic, police, ambulance, fire
        
        for folder, label in self.label_map.items():
            folder_path = os.path.join(self.folder_path, folder)
            if not os.path.exists(folder_path):
                continue
            
            files = [os.path.join(folder_path, f) 
                    for f in os.listdir(folder_path) if f.endswith('.wav')]
            
            # Shuffle files for randomness within class (seed-controlled)
            random.seed(self.seed)
            random.shuffle(files)
            
            class_files[label] = files
        
        # Get counts per class
        class_counts = {cls: len(files) for cls, files in class_files.items()}
        total_min = min(class_counts.values())
        
        print(f"\nBenchmark Multi-Class Setup:")
        print(f"  Class counts: {class_counts}")
        print(f"  Min class: {total_min} samples")
        
        # Progressive partitions
        partitions = [0.0025, 0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.0]
        
        for partition in partitions:
            # Target per class in this partition
            target_per_class = int(total_min * partition)
            
            if target_per_class < 1:
                print(f"⚠️  Partition {partition}: too small for 4-way balance (< 1 per class), using 1")
                target_per_class = 1
            
            # Sample equally from each class
            partition_file_paths = []
            partition_labels = []
            
            for cls in [0, 1, 2, 3]:
                n_available = len(class_files[cls])
                n_select = min(target_per_class, n_available)
                selected_files = class_files[cls][:n_select]
                
                partition_file_paths.extend(selected_files)
                partition_labels.extend([cls] * n_select)
            
            # Create dataset for this partition
            partition_dataset = sireNNetDataset(folder_path=self.folder_path,
                                                label_map=self.label_map,
                                                seed=self.seed,
                                                augmentation=False,
                                                target_size=self.target_size,
                                                target_sr=self.target_sr)
            
            # Override with partition files
            partition_dataset.file_paths = partition_file_paths
            partition_dataset.labels = partition_labels
            
            self.test_datasets[partition] = partition_dataset
            
            # Count per class for verification
            class_dist = Counter(partition_labels)
            print(f"  Partition {partition:5.4f}: {len(partition_labels):4d} samples " +
                  f"(0:{class_dist[0]}, 1:{class_dist[1]}, 2:{class_dist[2]}, 3:{class_dist[3]})")
    
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
                          collate_fn=sirennet_collate_fn,
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
                          collate_fn=sirennet_collate_fn,
                          persistent_workers=True if self.num_workers > 0 else False)
    
    def test_dataloader(self) -> DataLoader:
        """Return test dataloader (train mode only)."""
        if self.mode != 'train':
            raise ValueError("test_dataloader() only available in 'train' mode. Use test_dataloaders() for benchmark.")
        
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          collate_fn=sirennet_collate_fn,
                          persistent_workers=True if self.num_workers > 0 else False)
    
    def test_dataloaders(self):
        """Return list of test dataloaders (benchmark mode: one per partition)."""
        if self.mode != 'benchmark':
            raise ValueError("test_dataloaders() only available in 'benchmark' mode")
        
        loaders = []
        for partition in sorted(self.test_datasets.keys()):
            loader = DataLoader(self.test_datasets[partition],
                               batch_size=self.batch_size,
                               shuffle=False,
                               num_workers=self.num_workers,
                               collate_fn=sirennet_collate_fn,
                               persistent_workers=True if self.num_workers > 0 else False)
            loaders.append(loader)
        
        return loaders


# =============================================================================
# TESTING CODE
# =============================================================================

if __name__ == "__main__":
    import sys
    
    print("=" * 80)
    print("DATALOADER TEST - sireNNet")
    print("=" * 80)
    
    # Get current directory (datasets/sireNNet/)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
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
    dm_train = sireNNetDataModule(folder_path=current_dir,
                                  mode='train',
                                  seed=42,
                                  batch_size=32,
                                  split_ratios=(0.8, 0.1, 0.1),
                                  shuffle=True,
                                  augmentation=False,
                                  target_size=96000,
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
    dm_bench = sireNNetDataModule(folder_path=current_dir,
                                  mode='benchmark',
                                  seed=42,
                                  batch_size=32,
                                  target_size=96000,
                                  target_sr=32000,
                                  num_workers=0)  # For testing
    
    # Setup
    dm_bench.setup()
    
    # Get test loaders (one per partition)
    test_loaders = dm_bench.test_dataloaders()
    partitions = sorted(dm_bench.test_datasets.keys())
    
    print("\n" + "─" * 80)
    print("BENCHMARK STATISTICS")
    print("─" * 80)
    
    print(f"\nNumber of partitions: {len(test_loaders)}")
    print(f"Partitions: {partitions}")
    
    # Statistics for each partition
    for idx, (loader, partition) in enumerate(zip(test_loaders, partitions), 1):
        loader_samples = len(loader.dataset)
        loader_batches = len(loader)
        
        print(f"\n--- Partition {partition} ({loader_samples} samples, {loader_batches} batches) ---")
        
        # Count labels
        pos, neg = count_labels_in_dataset(loader.dataset)
        print(f"  - Positives: {pos}, Negatives: {neg}")
        
        # First batch analysis
        print(f"  First batch:")
        for waveforms, labels in loader:
            pos_b = (labels == 1).sum().item()
            neg_b = (labels == 0).sum().item()
            duration = waveforms.shape[2] / 32000
            print(f"    - Samples: {waveforms.shape[0]}")
            print(f"    - Positives: {pos_b}, Negatives: {neg_b}")
            print(f"    - Duration: {duration:.2f}s, SR: 32000Hz")
            break
    
    # =========================================================================
    # TEST 3: MULTI-CLASS MODE (4-WAY CLASSIFICATION)
    # =========================================================================
    print("\n\n" + "=" * 80)
    print("TEST 3: MULTI-CLASS MODE (4-WAY CLASSIFICATION)")
    print("=" * 80)
    
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
    
    # Initialize DataModule in multi-class train mode
    dm_mc_train = sireNNetDataModule(folder_path=current_dir,
                                     mode='train',
                                     label_mode='multi_class',
                                     seed=42,
                                     batch_size=32,
                                     split_ratios=(0.8, 0.1, 0.1),
                                     shuffle=True,
                                     augmentation=False,
                                     target_size=96000,
                                     target_sr=32000,
                                     num_workers=0)
    
    # Setup
    dm_mc_train.setup()
    
    # Get dataloaders
    train_loader_mc = dm_mc_train.train_dataloader()
    val_loader_mc = dm_mc_train.val_dataloader()
    test_loader_mc = dm_mc_train.test_dataloader()
    
    # ─────────────────────────────────────────────────────────────────────────
    # Multi-Class Train Loader
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 80)
    print("MULTI-CLASS TRAIN LOADER STATISTICS")
    print("─" * 80)
    
    train_samples_mc = len(train_loader_mc.dataset)
    train_batches_mc = len(train_loader_mc)
    
    print("Counting labels per class...")
    train_counts = count_multiclass_labels(train_loader_mc.dataset)
    
    print(f"Total samples: {train_samples_mc} ({train_batches_mc} batches)")
    print(f"  - Class 0 (Traffic): {train_counts[0]}")
    print(f"  - Class 1 (Police): {train_counts[1]}")
    print(f"  - Class 2 (Ambulance): {train_counts[2]}")
    print(f"  - Class 3 (Fire): {train_counts[3]}")
    
    # First batch analysis
    print("\nFirst batch analysis:")
    for waveforms, labels in train_loader_mc:
        class_dist = Counter(labels.tolist())
        duration = waveforms.shape[2] / 32000
        print(f"  - Samples: {waveforms.shape[0]}")
        print(f"  - Class distribution: 0:{class_dist.get(0,0)}, 1:{class_dist.get(1,0)}, "
              f"2:{class_dist.get(2,0)}, 3:{class_dist.get(3,0)}")
        print(f"  - Duration: {duration:.2f}s, Sample rate: 32000Hz")
        break
    
    # ─────────────────────────────────────────────────────────────────────────
    # Multi-Class Benchmark Mode
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 80)
    print("MULTI-CLASS BENCHMARK MODE")
    print("─" * 80)
    
    # Initialize in benchmark multi-class mode
    dm_mc_bench = sireNNetDataModule(folder_path=current_dir,
                                     mode='benchmark',
                                     label_mode='multi_class',
                                     seed=42,
                                     batch_size=32,
                                     target_size=96000,
                                     target_sr=32000,
                                     num_workers=0)
    
    # Setup
    dm_mc_bench.setup()
    
    # Get test loaders
    test_loaders_mc = dm_mc_bench.test_dataloaders()
    partitions_mc = sorted(dm_mc_bench.test_datasets.keys())
    
    print(f"\nNumber of partitions: {len(test_loaders_mc)}")
    print(f"Partitions: {partitions_mc}")
    
    # Statistics for each partition (showing ALL partitions)
    for idx, (loader, partition) in enumerate(zip(test_loaders_mc, partitions_mc), 1):
        loader_samples = len(loader.dataset)
        loader_batches = len(loader)
        
        print(f"\n--- Partition {partition} ({loader_samples} samples, {loader_batches} batches) ---")
        
        # Count per class
        counts = count_multiclass_labels(loader.dataset)
        print(f"  - Class 0 (Traffic): {counts[0]}")
        print(f"  - Class 1 (Police): {counts[1]}")
        print(f"  - Class 2 (Ambulance): {counts[2]}")
        print(f"  - Class 3 (Fire): {counts[3]}")
        
        # First batch analysis
        print(f"  First batch:")
        for waveforms, labels in loader:
            class_dist = Counter(labels.tolist())
            duration = waveforms.shape[2] / 32000
            print(f"    - Samples: {waveforms.shape[0]}")
            print(f"    - Class 0: {class_dist.get(0,0)}, Class 1: {class_dist.get(1,0)}, "
                  f"Class 2: {class_dist.get(2,0)}, Class 3: {class_dist.get(3,0)}")
            print(f"    - Duration: {duration:.2f}s, SR: 32000Hz")
            break
    
    print("\n" + "=" * 80)
    print("TEST COMPLETED SUCCESSFULLY!")
    print("=" * 80)
