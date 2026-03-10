"""
Unified Multi-Dataset Training for Emergency Vehicle Recognition
================================================================
Trains EPANNs on a unified dataset combining all 7 EV datasets.

This script:
1. Loads all 7 datasets (AudioSet_EV v1/v2, sireNNet, LSSiren, ESC50, FSD50K, UrbanSound8K)
2. Concatenates them into a single unified dataset
3. Performs global train/val/test split
4. Trains EPANNs with binary classification

Usage:
    python main_unified_EV.py --config configs/epanns_unified-EV_binary.yaml

Author: Stefano Giacomelli - Ph.D. candidate in ICT (DISIM dpt. - University of L'Aquila)
"""

import os
import argparse
import json
from typing import Dict, Any

import yaml
import torch
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, ConcatDataset, random_split
from tqdm import tqdm

# Import Lightning modules
from models.lightning_models import BinaryEVClassifier
from models.callbacks import ModelCheckpoint, EpochResamplingCallback

# Import dataset classes (NOT DataModules)
from datasets.AudioSet_EV_v1_2025.dataloader import AudioSetEV_v1_Dataset
from datasets.AudioSet_EV_v2PANNs_2020.dataloader import AudioSetEV_v2_Dataset
from datasets.sireNNet.dataloader import sireNNetDataset
from datasets.LSSiren.dataloader import LSSirenDataset
from datasets.ESC50.dataloader import ESC50Dataset
from datasets.FSD50K.dataloader import FSD50KDataset
from datasets.UrbanSound8K.dataloader import UrbanSound8KDataset
from datasets.KineScaper_EV.dataloader import create_kinescaper_dataset_for_unified


# =============================================================================
# CONFIGURATION LOADING
# =============================================================================

def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"✓ Loaded configuration from: {config_path}")
    return config


def validate_config(config: Dict[str, Any]):
    """Validate configuration parameters."""
    required_sections = ['experiment', 'paths', 'data', 'training', 'model']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required section '{section}' in config")
    
    # Check task
    if config['experiment']['task'] != 'binary':
        raise ValueError(f"This script only supports binary classification. Got: {config['experiment']['task']}")
    
    # Check model
    valid_models = ['epanns', 'ced', 'clap']
    if config['experiment']['model_name'] not in valid_models:
        raise ValueError(f"Invalid model_name. Must be one of: {valid_models}. Got: {config['experiment']['model_name']}")
    
    print("✓ Configuration validated successfully")


# =============================================================================
# DATASET CREATION
# =============================================================================

def load_datasets_mapping(data_root: str) -> Dict:
    """Load datasets_mapping.json for label mappings."""
    mapping_path = os.path.join(data_root, "datasets_mapping.json")
    with open(mapping_path, 'r') as f:
        return json.load(f)


def count_labels_from_metadata(dataset_name: str, config: Dict) -> tuple:
    """
    Count positives and negatives directly from metadata/CSV files.
    NO audio loading — reads only filenames, CSV rows, or folder contents.

    Returns:
        (num_positive, num_negative) as exact integers
    """
    import pandas as pd
    data_root = config['paths']['data_root']

    if dataset_name == 'sireNNet':
        # Folder structure: ambulance/firetruck/police = pos, traffic = neg
        base = os.path.join(data_root, 'sireNNet')
        pos = sum(len([f for f in os.listdir(os.path.join(base, d)) if f.endswith('.wav')])
                  for d in ['ambulance', 'firetruck', 'police'])
        neg = len([f for f in os.listdir(os.path.join(base, 'traffic')) if f.endswith('.wav')])
        return pos, neg

    elif dataset_name == 'LSSiren':
        # Two CSV files: Ambulance_final.csv (pos), Road_final.csv (neg)
        base = os.path.join(data_root, 'LSSiren')
        pos_df = pd.read_csv(os.path.join(base, 'Ambulance_final.csv'))
        neg_df = pd.read_csv(os.path.join(base, 'Road_final.csv'))
        # Each CSV row = one sample; Road_final has first row as data (no real header)
        pos = len(pos_df)
        neg = len(neg_df)
        return pos, neg

    elif dataset_name == 'ESC50':
        # esc50.csv with 'category' column; only 'siren' is positive
        # Filter to only categories present in mapping (not all 50 ESC50 classes)
        base = os.path.join(data_root, 'ESC50')
        df = pd.read_csv(os.path.join(base, 'esc50.csv'))
        mapping = load_datasets_mapping(data_root)['ESC50']
        df_used = df[df['category'].isin(mapping.keys())]  # Only rows used by the dataset
        pos = int((df_used['category'].map(mapping) == 1).sum())
        neg = len(df_used) - pos
        return pos, neg

    elif dataset_name == 'FSD50K':
        # Four pre-split CSVs: dev/eval × positives/negatives
        base = os.path.join(data_root, 'FSD50K')
        pos = (len(pd.read_csv(os.path.join(base, 'FSD-dev_positives.csv')))
             + len(pd.read_csv(os.path.join(base, 'FSD-eval_positives.csv'))))
        neg = (len(pd.read_csv(os.path.join(base, 'FSD-dev_negatives.csv')))
             + len(pd.read_csv(os.path.join(base, 'FSD-eval_negatives.csv'))))
        return pos, neg

    elif dataset_name == 'UrbanSound8K':
        # metadata/UrbanSound8K.csv with 'class' column; only 'siren' is positive
        base = os.path.join(data_root, 'UrbanSound8K')
        df = pd.read_csv(os.path.join(base, 'metadata', 'UrbanSound8K.csv'))
        pos = int((df['class'] == 'siren').sum())
        neg = len(df) - pos
        return pos, neg

    elif dataset_name == 'AudioSet_EV_v1_2025':
        # Count actual downloaded files in Positive_files/ and Negative_files/
        base = os.path.join(data_root, 'AudioSet_EV_v1_2025')
        pos = len([f for f in os.listdir(os.path.join(base, 'Positive_files')) if f.endswith('.wav')])
        neg = len([f for f in os.listdir(os.path.join(base, 'Negative_files')) if f.endswith('.wav')])
        return pos, neg

    elif dataset_name == 'AudioSet_EV_v2PANNs_2020':
        # Count positives from CSV (downloaded==True); negatives balanced 1:1
        base = os.path.join(data_root, 'AudioSet_EV_v2PANNs_2020')
        pos_csv = pd.read_csv(os.path.join(base, 'EV_Positives.csv'))
        pos = int((pos_csv['downloaded'] == True).sum())
        neg = pos  # Negatives are balanced to match positives at runtime
        return pos, neg

    elif dataset_name == 'KineScaper_EV':
        # Already known from dataloader: returned as tuple (dataset, pos, neg)
        # This function won't be called for KineScaper (handled separately)
        return None, None

    else:
        return None, None


def create_audioset_ev_v1_dataset(config: Dict, augmentation: bool, aug_prob: float, seed: int, verbose: bool = True) -> tuple:
    """Create AudioSet_EV_v1_2025 dataset."""
    data_root = config['paths']['data_root']
    dataset_path = os.path.join(data_root, 'AudioSet_EV_v1_2025')
    target_size = int(config['data']['target_duration'] * config['data']['target_sr'])
    
    # Create positive dataset
    pos_dataset = AudioSetEV_v1_Dataset(
        csv_path=os.path.join(dataset_path, "EV_Positives.csv"),
        audio_folder=os.path.join(dataset_path, "Positive_files"),
        binary_label=1,
        seed=seed,
        augmentation=augmentation,
        aug_prob=aug_prob,
        target_size=target_size,
        target_sr=config['data']['target_sr']
    )
    
    # Create negative dataset
    neg_dataset = AudioSetEV_v1_Dataset(
        csv_path=os.path.join(dataset_path, "EV_Negatives.csv"),
        audio_folder=os.path.join(dataset_path, "Negative_files"),
        binary_label=0,
        seed=seed,
        augmentation=augmentation,
        aug_prob=aug_prob,
        target_size=target_size,
        target_sr=config['data']['target_sr']
    )
    
    # Concatenate
    combined = ConcatDataset([pos_dataset, neg_dataset])
    pos_count = len(pos_dataset)
    neg_count = len(neg_dataset)
    if verbose:
        aug_str = " [AUG]" if augmentation else ""
        print(f"    Loaded: {len(combined):,} samples ({pos_count:,} pos + {neg_count:,} neg){aug_str}")
    return combined, pos_count, neg_count


def create_audioset_ev_v2_dataset(config: Dict, augmentation: bool, aug_prob: float, seed: int, verbose: bool = True) -> tuple:
    """Create AudioSet_EV_v2PANNs_2020 dataset."""
    data_root = config['paths']['data_root']
    dataset_path = os.path.join(data_root, 'AudioSet_EV_v2PANNs_2020')
    
    # Load mapping for negative focus labels
    mapping = load_datasets_mapping(data_root)
    negative_focus_labels = [label for label, val in mapping["AUDIOSET"].items() if val == 0]
    
    # Load label mapping CSV
    label_mapping_csv = os.path.join(dataset_path, "audioset_metadata", "class_labels_indices.csv")
    
    # Load positives
    pos_df = pd.read_csv(os.path.join(dataset_path, "EV_Positives.csv"))
    pos_df = pos_df[pos_df['downloaded'] == True].reset_index(drop=True)
    
    pos_dataset = AudioSetEV_v2_Dataset(csv_df=pos_df,
                                        audio_folder=os.path.join(dataset_path, "Positive_files"),
                                        subfolders=["balanced_train", "eval", "unbalanced"],
                                        target_size=int(config['data']['target_duration'] * config['data']['target_sr']),
                                        binary_label=1,
                                        augmentation=augmentation,
                                        aug_prob=aug_prob,
                                        seed=seed)
    
    # Load negatives (balanced to match positives)
    from datasets.AudioSet_EV_v2PANNs_2020.dataloader import StratifiedNegativeBalancer
    
    balancer = StratifiedNegativeBalancer(csv_path=os.path.join(dataset_path, "EV_Negatives.csv"),
                                          label_mapping_csv=label_mapping_csv,
                                          negative_focus_labels=negative_focus_labels,
                                          seed=seed)
    neg_df = balancer.balance(target_count=len(pos_df))
    
    neg_dataset = AudioSetEV_v2_Dataset(csv_df=neg_df,
                                        audio_folder=os.path.join(dataset_path, "Negative_files"),
                                        subfolders=["balanced_train", "eval"],
                                        target_size=int(config['data']['target_duration'] * config['data']['target_sr']),
                                        binary_label=0,
                                        augmentation=augmentation,
                                        aug_prob=aug_prob,
                                        seed=seed)
    
    # Concatenate pos and neg
    combined = ConcatDataset([pos_dataset, neg_dataset])
    if verbose:
        aug_str = " [AUG]" if augmentation else ""
        print(f"    Loaded: {len(combined):,} samples ({len(pos_dataset):,} pos + {len(neg_dataset):,} neg){aug_str}")
    return combined


def create_sirennet_dataset(config: Dict, augmentation: bool, aug_prob: float, seed: int, verbose: bool = True) -> sireNNetDataset:
    """Create sireNNet dataset."""
    data_root = config['paths']['data_root']
    dataset_path = os.path.join(data_root, 'sireNNet')
    target_size = int(config['data']['target_duration'] * config['data']['target_sr'])
    
    # Load label mapping
    mapping = load_datasets_mapping(data_root)
    label_map = mapping["SIRENNET"]
    
    dataset = sireNNetDataset(folder_path=dataset_path,
                              label_map=label_map,
                              seed=seed,
                              augmentation=augmentation,
                              aug_prob=aug_prob,
                              target_size=target_size,
                              target_sr=config['data']['target_sr'])
    
    if verbose:
        aug_str = " [AUG]" if augmentation else ""
        print(f"    Loaded: {len(dataset):,} samples{aug_str}")
    return dataset


def create_lssiren_dataset(config: Dict, augmentation: bool, aug_prob: float, seed: int, verbose: bool = True) -> LSSirenDataset:
    """Create LSSiren dataset."""
    data_root = config['paths']['data_root']
    dataset_path = os.path.join(data_root, 'LSSiren')
    min_length = int(config['data']['target_duration'] * config['data']['target_sr'])
    
    # Load label mapping
    mapping = load_datasets_mapping(data_root)
    label_map = mapping["LSSIREN"]
    
    dataset = LSSirenDataset(folder_path=dataset_path,
                             label_map=label_map,
                             seed=seed,
                             augmentation=augmentation,
                             aug_prob=aug_prob,
                             target_sr=config['data']['target_sr'],
                             min_length=min_length)
    
    if verbose:
        aug_str = " [AUG]" if augmentation else ""
        print(f"    Loaded: {len(dataset):,} samples{aug_str}")
    return dataset


def create_esc50_dataset(config: Dict, augmentation: bool, aug_prob: float, seed: int, verbose: bool = True) -> tuple:
    """Create ESC50 dataset."""
    data_root = config['paths']['data_root']
    dataset_path = os.path.join(data_root, 'ESC50')
    target_size = int(config['data']['target_duration'] * config['data']['target_sr'])
    
    # Load label mapping
    mapping = load_datasets_mapping(data_root)
    label_map = mapping["ESC50"]
    
    dataset = ESC50Dataset(csv_path=os.path.join(dataset_path, "esc50.csv"),
                           audio_folder=os.path.join(dataset_path, "original_audio"),
                           label_map=label_map,
                           seed=seed,
                           augmentation=augmentation,
                           aug_prob=aug_prob,
                           target_size=target_size,
                           target_sr=config['data']['target_sr'],
                           mode='train')
    
    if verbose:
        aug_str = " [AUG]" if augmentation else ""
        print(f"    Loaded: {len(dataset):,} samples{aug_str}")
    return dataset


def create_fsd50k_dataset(config: Dict, augmentation: bool, aug_prob: float, seed: int, verbose: bool = True) -> tuple:
    """Create FSD50K dataset."""
    data_root = config['paths']['data_root']
    dataset_path = os.path.join(data_root, 'FSD50K')
    
    # Load label mapping
    mapping = load_datasets_mapping(data_root)
    label_map = mapping["FSD50K"]
    
    # Load dev positives/negatives CSVs
    pos_dev_csv = os.path.join(dataset_path, "FSD-dev_positives.csv")
    neg_dev_csv = os.path.join(dataset_path, "FSD-dev_negatives.csv")
    dev_folder = os.path.join(dataset_path, "FSD50K.dev_audio")
    
    # Load eval positives/negatives CSVs
    pos_eval_csv = os.path.join(dataset_path, "FSD-eval_positives.csv")
    neg_eval_csv = os.path.join(dataset_path, "FSD-eval_negatives.csv")
    eval_folder = os.path.join(dataset_path, "FSD50K.eval_audio")
    
    csv_files = [pos_dev_csv, neg_dev_csv, pos_eval_csv, neg_eval_csv]
    audio_folders = [dev_folder, dev_folder, eval_folder, eval_folder]
    
    dataset = FSD50KDataset(csv_files=csv_files,
                            audio_folders=audio_folders,
                            label_map=label_map,
                            seed=seed,
                            augmentation=augmentation,
                            aug_prob=aug_prob,
                            target_sr=config['data']['target_sr'])
    
    if verbose:
        aug_str = " [AUG]" if augmentation else ""
        print(f"    Loaded: {len(dataset):,} samples{aug_str}")
    return dataset


def create_urbansound8k_dataset(config: Dict, augmentation: bool, aug_prob: float, seed: int, verbose: bool = True) -> UrbanSound8KDataset:
    """Create UrbanSound8K dataset."""
    data_root = config['paths']['data_root']
    dataset_path = os.path.join(data_root, 'UrbanSound8K')
    min_length = int(config['data']['target_duration'] * config['data']['target_sr'])
    
    # Load label mapping
    mapping = load_datasets_mapping(data_root)
    label_map = mapping["US8K"]
    
    dataset = UrbanSound8KDataset(metadata_path=os.path.join(dataset_path, "metadata", "UrbanSound8K.csv"),
                                  audio_folder=os.path.join(dataset_path, "audio"),
                                  label_map=label_map,
                                  seed=seed,
                                  augmentation=augmentation,
                                  aug_prob=aug_prob,
                                  target_sr=config['data']['target_sr'],
                                  min_length=min_length,
                                  mode='train')
    
    if verbose:
        aug_str = " [AUG]" if augmentation else ""
        print(f"    Loaded: {len(dataset):,} samples{aug_str}")
    return dataset


def create_kinescaper_dataset(config: Dict, augmentation: bool, aug_prob: float, seed: int, verbose: bool = True) -> tuple:
    """
    Create KineScaper-EV dataset for unified training.
    
    NOTE: This is different from the DataModule used for cross-validation.
    Here we create a standalone dataset that will be split globally.
    
    Uses stratified re-sampling for positives to increase diversity.
    """
    # KineScaper dataset is on external SSD mount
    dataset_root = '/mnt/ssd/Kinescaper_EV/dataset/'
    
    # From analysis (datasets/dataset_distribution_analysis.json):
    # - Other 7 datasets contribute: 18,582 pos + 88,057 neg
    # - Target: 50/50 balance → 100,237 pos + 100,237 neg
    # - Needed from KineScaper: 81,655 pos (= 100,237 - 18,582) + ALL negatives (12,180)
    # - KineScaper positives use stratified re-sampling each epoch
    num_positive_samples = 81655  # Stratified, re-sampled each epoch
    
    # Create positive chunks dataset
    pos_dataset = create_kinescaper_dataset_for_unified(
        dataset_root=dataset_root,
        num_positive_samples=num_positive_samples,
        chunk_type='positive',
        augmentation=augmentation,
        aug_prob=aug_prob,
        target_sr=config['data']['target_sr'],
        target_duration=config['data']['target_duration'],
        seed=seed
    )
    
    # Create negative chunks dataset
    neg_dataset = create_kinescaper_dataset_for_unified(
        dataset_root=dataset_root,
        num_positive_samples=num_positive_samples,
        chunk_type='negative',
        augmentation=False,  # Negatives already augmented by generator
        aug_prob=aug_prob,
        target_sr=config['data']['target_sr'],
        target_duration=config['data']['target_duration'],
        seed=seed
    )
    
    # Concatenate positives and negatives
    combined = ConcatDataset([pos_dataset, neg_dataset])
    
    if verbose:
        aug_str = " [AUG]" if augmentation else ""
        print(f"    Loaded: {len(combined):,} samples ({len(pos_dataset):,} pos + {len(neg_dataset):,} neg){aug_str}")
        if augmentation:
            print(f"    Note: Positives use stratified re-sampling across 7 siren classes")
    
    # Return dataset with counts
    return combined, len(pos_dataset), len(neg_dataset)


def create_unified_datasets(config: Dict) -> tuple:
    """
    Create unified dataset by concatenating all individual datasets.
    
    Strategy:
    1. Create each dataset twice (with and without augmentation)
    2. Split each dataset individually into train/val/test
    3. Concatenate all train splits (with aug) -> unified_train
    4. Concatenate all val splits (no aug) -> unified_val
    5. Concatenate all test splits (no aug) -> unified_test
    
    Returns:
        (train_dataset, val_dataset, test_dataset)
    """
    seed = config['experiment']['seed']
    pl.seed_everything(seed)
    
    # Augmentation settings
    aug_enabled = config['data']['augmentation']['enabled']
    aug_prob = config['data']['augmentation']['probability']
    split_ratios = config['data']['split_ratios']
    
    print(f"\n{'='*80}")
    print(f"CREATING UNIFIED DATASET WITH PER-DATASET SPLITTING")
    print(f"{'='*80}")
    print(f"Augmentation: {'ENABLED (training only)' if aug_enabled else 'DISABLED'}")
    if aug_enabled:
        print(f"Aug probability: {aug_prob}")
    print(f"Target duration: {config['data']['target_duration']}s @ {config['data']['target_sr']}Hz")
    print(f"Split ratios: {split_ratios[0]*100:.0f}% train / {split_ratios[1]*100:.0f}% val / {split_ratios[2]*100:.0f}% test")
    
    # Lists to collect splits AND counts
    all_train_splits = []
    all_val_splits = []
    all_test_splits = []
    
    # Track pos/neg counts per dataset per split
    dataset_counts = {
        'train': [],  # [(dataset_name, total, pos, neg), ...]
        'val': [],
        'test': []
    }
    
    # Process each dataset
    dataset_names = config['data']['datasets']
    
    for idx, dataset_name in enumerate(dataset_names, 1):
        print(f"\n[{idx}/{len(dataset_names)}] {dataset_name}")
        
        # Create dataset WITHOUT augmentation first
        if dataset_name == 'AudioSet_EV_v1_2025':
            result = create_audioset_ev_v1_dataset(config, augmentation=False, aug_prob=aug_prob, seed=seed, verbose=False)
        elif dataset_name == 'AudioSet_EV_v2PANNs_2020':
            result = create_audioset_ev_v2_dataset(config, augmentation=False, aug_prob=aug_prob, seed=seed, verbose=False)
        elif dataset_name == 'sireNNet':
            result = create_sirennet_dataset(config, augmentation=False, aug_prob=aug_prob, seed=seed, verbose=False)
        elif dataset_name == 'LSSiren':
            result = create_lssiren_dataset(config, augmentation=False, aug_prob=aug_prob, seed=seed, verbose=False)
        elif dataset_name == 'ESC50':
            result = create_esc50_dataset(config, augmentation=False, aug_prob=aug_prob, seed=seed, verbose=False)
        elif dataset_name == 'FSD50K':
            result = create_fsd50k_dataset(config, augmentation=False, aug_prob=aug_prob, seed=seed, verbose=False)
        elif dataset_name == 'UrbanSound8K':
            result = create_urbansound8k_dataset(config, augmentation=False, aug_prob=aug_prob, seed=seed, verbose=False)
        elif dataset_name == 'KineScaper_EV':
            result = create_kinescaper_dataset(config, augmentation=False, aug_prob=aug_prob, seed=seed, verbose=False)
        else:
            continue
        
        # Handle return value (dataset or tuple with counts)
        if isinstance(result, tuple):
            dataset_no_aug, ds_pos, ds_neg = result
        else:
            dataset_no_aug = result
            # Count pos/neg directly from metadata — no audio loading!
            ds_pos, ds_neg = count_labels_from_metadata(dataset_name, config)
        
        # Split this dataset
        total_size = len(dataset_no_aug)
        train_size = int(split_ratios[0] * total_size)
        val_size = int(split_ratios[1] * total_size)
        test_size = total_size - train_size - val_size
        
        generator = torch.Generator().manual_seed(seed)
        train_subset, val_subset, test_subset = random_split(
            dataset_no_aug,
            [train_size, val_size, test_size],
            generator=generator
        )
        
        print(f"    Total: {total_size:,} samples | Split: {train_size:,} train / {val_size:,} val / {test_size:,} test")
        
        # Store counts per split (calculated from total counts + split ratios)
        if ds_pos is not None and ds_neg is not None:
            train_pos = int(ds_pos * split_ratios[0])
            train_neg = int(ds_neg * split_ratios[0])
            val_pos   = int(ds_pos * split_ratios[1])
            val_neg   = int(ds_neg * split_ratios[1])
            test_pos  = ds_pos - train_pos - val_pos
            test_neg  = ds_neg - train_neg - val_neg
        else:
            train_pos = train_neg = val_pos = val_neg = test_pos = test_neg = None

        dataset_counts['train'].append((dataset_name, train_size, train_pos, train_neg))
        dataset_counts['val'].append((dataset_name, val_size,   val_pos,   val_neg))
        dataset_counts['test'].append((dataset_name, test_size,  test_pos,  test_neg))
        
        # For TRAINING split: create dataset WITH augmentation if enabled
        if aug_enabled:
            print(f"    Augmentation: ENABLED for training split")
            if dataset_name == 'AudioSet_EV_v1_2025':
                result_aug = create_audioset_ev_v1_dataset(config, augmentation=True, aug_prob=aug_prob, seed=seed, verbose=True)
            elif dataset_name == 'AudioSet_EV_v2PANNs_2020':
                result_aug = create_audioset_ev_v2_dataset(config, augmentation=True, aug_prob=aug_prob, seed=seed, verbose=True)
            elif dataset_name == 'sireNNet':
                result_aug = create_sirennet_dataset(config, augmentation=True, aug_prob=aug_prob, seed=seed, verbose=True)
            elif dataset_name == 'LSSiren':
                result_aug = create_lssiren_dataset(config, augmentation=True, aug_prob=aug_prob, seed=seed, verbose=True)
            elif dataset_name == 'ESC50':
                result_aug = create_esc50_dataset(config, augmentation=True, aug_prob=aug_prob, seed=seed, verbose=True)
            elif dataset_name == 'FSD50K':
                result_aug = create_fsd50k_dataset(config, augmentation=True, aug_prob=aug_prob, seed=seed, verbose=True)
            elif dataset_name == 'UrbanSound8K':
                result_aug = create_urbansound8k_dataset(config, augmentation=True, aug_prob=aug_prob, seed=seed, verbose=True)
            elif dataset_name == 'KineScaper_EV':
                result_aug = create_kinescaper_dataset(config, augmentation=True, aug_prob=aug_prob, seed=seed, verbose=True)
            
            # Handle return value (dataset or tuple)
            if isinstance(result_aug, tuple):
                dataset_with_aug = result_aug[0]  # Extract dataset from tuple
            else:
                dataset_with_aug = result_aug
            
            # Create Subset with same indices but using augmented dataset
            from torch.utils.data import Subset
            train_subset_aug = Subset(dataset_with_aug, train_subset.indices)
            all_train_splits.append(train_subset_aug)
        else:
            print(f"    Augmentation: DISABLED")
            all_train_splits.append(train_subset)
        
        # Val and test use non-augmented dataset
        all_val_splits.append(val_subset)
        all_test_splits.append(test_subset)
    
    # Concatenate all splits
    print(f"\n{'='*80}")
    print(f"Concatenating splits from {len(dataset_names)} datasets...")
    print(f"{'='*80}")
    
    unified_train = ConcatDataset(all_train_splits)
    unified_val = ConcatDataset(all_val_splits)
    unified_test = ConcatDataset(all_test_splits)
    
    print(f"✓ Unified training set:   {len(unified_train):,} samples (augmentation: {'ON' if aug_enabled else 'OFF'})")
    print(f"✓ Unified validation set: {len(unified_val):,} samples (augmentation: OFF)")
    print(f"✓ Unified test set:        {len(unified_test):,} samples (augmentation: OFF)")
    print(f"✓ Total samples:           {len(unified_train) + len(unified_val) + len(unified_test):,}")
    
    # Print label distribution — computed from metadata, no audio loading
    print(f"\n{'='*80}")
    print("LABEL DISTRIBUTION (Positive vs Negative):")
    print(f"{'='*80}")
    print()

    # Aggregate totals across all datasets per split
    train_pos_total = sum(p for _, _, p, _ in dataset_counts['train'] if p is not None)
    train_neg_total = sum(n for _, _, _, n in dataset_counts['train'] if n is not None)
    val_pos_total   = sum(p for _, _, p, _ in dataset_counts['val']   if p is not None)
    val_neg_total   = sum(n for _, _, _, n in dataset_counts['val']   if n is not None)
    test_pos_total  = sum(p for _, _, p, _ in dataset_counts['test']  if p is not None)
    test_neg_total  = sum(n for _, _, _, n in dataset_counts['test']  if n is not None)

    overall_pos   = train_pos_total + val_pos_total + test_pos_total
    overall_neg   = train_neg_total + val_neg_total + test_neg_total
    overall_total = overall_pos + overall_neg

    def fmt_row(name, total_ds, pos, neg):
        p_pct = pos / (pos + neg) * 100 if (pos + neg) > 0 else 0
        n_pct = neg / (pos + neg) * 100 if (pos + neg) > 0 else 0
        print(f"{name:<12} {total_ds:>10,}  {pos:>12,} {p_pct:>5.1f}%  {neg:>12,} {n_pct:>5.1f}%")

    print(f"{'Split':<12} {'Total':>10}  {'Positives':>12} {'%':>6}  {'Negatives':>12} {'%':>6}")
    print(f"{'─'*80}")
    fmt_row("Train",      len(unified_train), train_pos_total, train_neg_total)
    fmt_row("Validation", len(unified_val),   val_pos_total,   val_neg_total)
    fmt_row("Test",       len(unified_test),  test_pos_total,  test_neg_total)
    print(f"{'─'*80}")
    fmt_row("TOTAL",      overall_total,      overall_pos,     overall_neg)

    # Detailed breakdown by dataset
    print(f"\n{'='*80}")
    print("DETAILED BREAKDOWN BY DATASET:")
    print(f"{'='*80}")

    for split_name, split_data in [('TRAIN',      dataset_counts['train']),
                                    ('VALIDATION', dataset_counts['val']),
                                    ('TEST',       dataset_counts['test'])]:
        print(f"\n{split_name}:")
        print(f"{'  Dataset':<30} {'Total':>10}  {'Pos':>10}  {'Neg':>10}  {'Balance':>12}")
        print(f"  {'─'*78}")
        for ds_name, total, pos, neg in split_data:
            if pos is not None and neg is not None and (pos + neg) > 0:
                balance = f"{pos/(pos+neg)*100:.1f}% / {neg/(pos+neg)*100:.1f}%"
            else:
                balance = "N/A"
                pos = pos or 0
                neg = neg or 0
            print(f"  {ds_name:<28} {total:>10,}  {pos:>10,}  {neg:>10,}  {balance:>12}")
        split_t = sum(t for _, t, _, _ in split_data)
        split_p = sum(p for _, _, p, _ in split_data if p is not None)
        split_n = sum(n for _, _, _, n in split_data if n is not None)
        bal = f"{split_p/(split_p+split_n)*100:.1f}% / {split_n/(split_p+split_n)*100:.1f}%" if (split_p+split_n) > 0 else "N/A"
        print(f"  {'─'*78}")
        print(f"  {'TOTAL':<28} {split_t:>10,}  {split_p:>10,}  {split_n:>10,}  {bal:>12}")

    return unified_train, unified_val, unified_test


# =============================================================================
# DATALOADER CREATION
# =============================================================================

def unified_collate_fn(batch):
    """
    Collate function for unified dataset.
    Filters out None values (failed loads) and ensures all waveforms have the same length.
    """
    # Filter out None values
    batch = [item for item in batch if item is not None]
    if not batch:
        return None, None
    
    # Separate waveforms and labels
    waveforms, labels = zip(*batch)
    
    # Normalize all waveforms to 2D [1, samples]
    normalized = []
    for w in waveforms:
        if w.dim() == 1:
            w = w.unsqueeze(0)  # [samples] → [1, samples]
        normalized.append(w)
    waveforms = normalized
    
    # Check if all waveforms have the same length
    lengths = [w.shape[1] for w in waveforms]
    if len(set(lengths)) > 1:
        # Different lengths - pad to max
        max_length = max(lengths)
        padded = []
        for w in waveforms:
            if w.shape[1] < max_length:
                w = torch.nn.functional.pad(w, (0, max_length - w.shape[1]), value=0.0)
            padded.append(w)
        waveforms = padded
    
    waveforms = torch.stack(waveforms)
    labels = torch.stack([l if isinstance(l, torch.Tensor) else torch.tensor(l, dtype=torch.long) for l in labels])
    
    return waveforms, labels


def create_dataloaders(train_ds, val_ds, test_ds, config: Dict) -> tuple:
    """
    Create DataLoaders for train/val/test.
    
    Returns:
        (train_loader, val_loader, test_loader)
    """
    batch_size = config['data']['batch_size']
    num_workers = min(4, os.cpu_count() // 4)  # reduced to save memory
    pin_memory = torch.cuda.is_available()
    
    print(f"\n{'='*80}")
    print(f"Creating DataLoaders...")
    print(f"{'='*80}")
    print(f"Batch size: {batch_size}")
    print(f"Num workers: {num_workers}")
    print(f"Pin memory: {pin_memory}")
    
    train_loader = DataLoader(train_ds,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=pin_memory,
                              collate_fn=unified_collate_fn,
                              persistent_workers=num_workers > 0)
    
    val_loader = DataLoader(val_ds,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers,
                            pin_memory=pin_memory,
                            collate_fn=unified_collate_fn,
                            persistent_workers=num_workers > 0)
    
    test_loader = DataLoader(test_ds,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers,
                             pin_memory=pin_memory,
                             collate_fn=unified_collate_fn,
                             persistent_workers=num_workers > 0)
    
    print(f"✓ DataLoaders created")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches:   {len(val_loader)}")
    print(f"  Test batches:  {len(test_loader)}")
    
    return train_loader, val_loader, test_loader


# =============================================================================
# MODEL AND TRAINING
# =============================================================================

def get_model(config: Dict) -> BinaryEVClassifier:
    """Initialize Binary EV Classifier model."""
    exp_config = config['experiment']
    model_config = config['model']
    training_config = config['training']
    paths_config = config['paths']
    
    print(f"\n{'='*80}")
    print(f"Initializing model...")
    print(f"{'='*80}")
    
    # Optimizer kwargs
    optimizer_kwargs = {'lr': training_config['optimizer']['lr'],
                        'weight_decay': training_config['optimizer']['weight_decay'],
                        'betas': tuple(training_config['optimizer']['betas']),
                        'eps': training_config['optimizer']['eps']}
    
    # Scheduler kwargs
    scheduler_type = training_config['scheduler']['type']
    scheduler_kwargs = {k: v for k, v in training_config['scheduler'].items() if k != 'type'}
    
    model = BinaryEVClassifier(model_name=exp_config['model_name'],
                               pretrained=exp_config['pretrained'],
                               threshold=model_config['threshold'],
                               optimizer_kwargs=optimizer_kwargs,
                               scheduler_type=scheduler_type,
                               scheduler_kwargs=scheduler_kwargs,
                               results_path=paths_config['results'],
                               f_beta=model_config['f_beta'])
    
    print(f"✓ Initialized BinaryEVClassifier")
    print(f"  Model: {exp_config['model_name'].upper()}")
    print(f"  Pretrained: {exp_config['pretrained']}")
    print(f"  Learning rate: {optimizer_kwargs['lr']}")
    print(f"  Threshold: {model_config['threshold']}")
    print(f"  F-beta: {model_config['f_beta']}")
    
    return model


def get_callbacks(config: Dict):
    """Create callbacks for training."""
    checkpoint_dir = config['paths']['checkpoints']
    patience = config['training']['patience']
    
    # Epoch re-sampling callback (for KineScaper stratified sampling)
    resampling_callback = EpochResamplingCallback()
    
    # Model checkpoint
    checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_dir,
                                          filename="{epoch:03d}_{val_f1:.4f}",
                                          monitor="val_f1",
                                          mode="max",
                                          save_top_k=1,
                                          save_weights_only=False,
                                          verbose=True)
    
    # Early stopping
    early_stopping = EarlyStopping(monitor="val_f1",
                                   mode="max",
                                   patience=patience,
                                   verbose=True)
    
    return [resampling_callback, checkpoint_callback, early_stopping]


def get_logger(config: Dict):
    """Create TensorBoard logger."""
    log_dir = config['paths']['logs']
    
    logger = TensorBoardLogger(save_dir=log_dir,
                               name=config['experiment']['name'])
    
    print(f"\n✓ TensorBoard logs will be saved to: {logger.log_dir}")
    
    return logger


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function."""
    # Set CUDA memory allocation config to reduce fragmentation
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Unified Multi-Dataset Training for EV Recognition')
    parser.add_argument('--config', type=str, default='configs/epanns_unified-EV_binary.yaml',
                        help='Path to config file')
    args = parser.parse_args()
    
    # Load and validate config
    print("\n" + "="*80)
    print("UNIFIED MULTI-DATASET TRAINING")
    print("="*80 + "\n")
    
    config = load_config(args.config)
    validate_config(config)
    
    # Create output directories
    os.makedirs(config['paths']['results'], exist_ok=True)
    os.makedirs(config['paths']['checkpoints'], exist_ok=True)
    os.makedirs(config['paths']['logs'], exist_ok=True)
    
    # Create unified datasets
    train_ds, val_ds, test_ds = create_unified_datasets(config)
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(train_ds, val_ds, test_ds, config)
    
    # Initialize model
    model = get_model(config)
    
    # Setup callbacks and logger
    callbacks = get_callbacks(config)
    logger = get_logger(config)
    
    # Create trainer
    print(f"\n{'='*80}")
    print(f"Creating Trainer...")
    print(f"{'='*80}")
    
    # Check for batch limits (for quick testing)
    trainer_kwargs = {'max_epochs': config['training']['max_epochs'],
                      'accelerator': 'auto',
                      'devices': 1,
                      'precision': 32,
                      'callbacks': callbacks,
                      'logger': logger,
                      'log_every_n_steps': 10,
                      'deterministic': True,
                      'default_root_dir': config['paths']['results']}
    
    # Gradient accumulation to reduce memory usage
    if 'accumulate_grad_batches' in config['data']:
        trainer_kwargs['accumulate_grad_batches'] = config['data']['accumulate_grad_batches']
        effective_batch = config['data']['batch_size'] * config['data']['accumulate_grad_batches']
        print(f"⚡ Gradient Accumulation ENABLED:")
        print(f"   Batch size: {config['data']['batch_size']}")
        print(f"   Accumulate: {config['data']['accumulate_grad_batches']} batches")
        print(f"   Effective batch size: {effective_batch}")
    
    # Add batch limits if specified in config
    if 'train_batch_limit' in config['data']:
        trainer_kwargs['limit_train_batches'] = config['data']['train_batch_limit']
        print(f"⚠️  TRAINING LIMITED TO {config['data']['train_batch_limit']} BATCHES PER EPOCH")
    
    if 'val_batch_limit' in config['data']:
        trainer_kwargs['limit_val_batches'] = config['data']['val_batch_limit']
        print(f"⚠️  VALIDATION LIMITED TO {config['data']['val_batch_limit']} BATCHES")
    
    if 'test_batch_limit' in config['data']:
        trainer_kwargs['limit_test_batches'] = config['data']['test_batch_limit']
        print(f"⚠️  TESTING LIMITED TO {config['data']['test_batch_limit']} BATCHES")
    
    trainer = pl.Trainer(**trainer_kwargs)
    
    print(f"✓ Trainer created")
    print(f"  Max epochs: {config['training']['max_epochs']}")
    print(f"  Early stopping patience: {config['training']['patience']}")
    
    # Train
    print(f"\n{'='*80}")
    print(f"STARTING TRAINING")
    print(f"{'='*80}\n")
    
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    
    # Test
    print(f"\n{'='*80}")
    print(f"TESTING")
    print(f"{'='*80}\n")
    
    trainer.test(model, dataloaders=test_loader)
    
    print(f"\n{'='*80}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"Results saved to: {config['paths']['results']}")
    print(f"Checkpoints saved to: {config['paths']['checkpoints']}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
