"""
Main Experiment Script for Emergency Vehicle Recognition
========================================================
Train and test GP-AT models (E-PANNs, CED, CLAP) for EV recognition.

Usage:
    python main_experiment.py --config configs/example.yaml

Author: Stefano Giacomelli - Ph.D. candidate in ICT (DISIM dpt. - University of L'Aquila)
"""

import os
import sys
import argparse
import subprocess
import time
from typing import Dict, Any

import numpy as np
import json
import yaml
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import Lightning modules
from models.lightning_models import BinaryEVClassifier, MultiClassSirenClassifier
from models.callbacks import ModelCheckpoint

# Import dataloaders
from datasets.AudioSet_EV_v1_2025.dataloader import AudioSetEV_v1_DataModule, audioset_ev_v1_collate_fn
from datasets.AudioSet_EV_v2PANNs_2020.dataloader import AudioSetEV_v2_DataModule, custom_collate_fn as audioset_ev_v2_collate_fn
from datasets.sireNNet.dataloader import sireNNetDataModule, sirennet_collate_fn
from datasets.LSSiren.dataloader import LSSirenDataModule, lssiren_collate_fn
from datasets.ESC50.dataloader import ESC50DataModule, esc50_collate_fn
from datasets.FSD50K.dataloader import FSD50KDataModule, fsd50k_collate_fn
from datasets.UrbanSound8K.dataloader import UrbanSound8KDataModule, urbansound8k_collate_fn


# =============================================================================
# CONFIGURATION LOADING
# =============================================================================

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load YAML configuration file.
    
    Args:
        config_path: Path to YAML config file
    
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"✓ Loaded configuration from: {config_path}")
    return config


def validate_config(config: Dict[str, Any]):
    """
    Validate configuration parameters.
    
    Args:
        config: Configuration dictionary
    
    Raises:
        ValueError: If configuration is invalid
    """
    # Check required sections
    required_sections = ['experiment', 'paths', 'data', 'training', 'model']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required section '{section}' in config")
    
    # Check task
    if config['experiment']['task'] not in ['binary', 'multiclass']:
        raise ValueError(f"Invalid task: {config['experiment']['task']}. Must be 'binary' or 'multiclass'.")
    
    # Check model name
    if config['experiment']['model_name'] not in ['epanns', 'ced', 'clap']:
        raise ValueError(f"Invalid model_name: {config['experiment']['model_name']}. Must be 'epanns', 'ced', or 'clap'.")
    
    # Check sample rate matches model
    expected_sr = {'epanns': 32000,
                   'ced': 16000,
                   'clap': 48000}
    model_name = config['experiment']['model_name']
    target_sr = config['data']['target_sr']
    if target_sr != expected_sr[model_name]:
        raise ValueError(f"Sample rate mismatch: {model_name} requires {expected_sr[model_name]} Hz, "
                         f"but config specifies {target_sr} Hz")
    
    print("✓ Configuration validated successfully")


# =============================================================================
# CROSS-VALIDATION RESULTS HANDLING
# =============================================================================

def aggregate_cv_results(results: list) -> dict:
    """
    Aggregate cross-validation results across multiple folds.
    
    Args:
        results: List of dicts with metrics for each fold
    
    Returns:
        Dict with mean and std of each metric
    """    
    # Extract all metrics
    all_metrics = {}
    for result in results:
        for key, value in result.items():
            if key not in all_metrics:
                all_metrics[key] = []
            all_metrics[key].append(value)
    
    # Calculate mean and std
    aggregated = {}
    for key, values in all_metrics.items():
        aggregated[f'{key}_mean'] = float(np.mean(values))
        aggregated[f'{key}_std'] = float(np.std(values))
        aggregated[f'{key}_all_folds'] = [float(v) for v in values]
    
    return aggregated


def save_cv_results(results: list, aggregated: dict, results_path: str):
    """
    Save cross-validation results (individual folds + aggregated).
    
    Args:
        results: List of dicts with metrics for each fold
        aggregated: Dict with aggregated metrics
        results_path: Base path for results
    """    
    test_dir = os.path.join(results_path, 'test')
    os.makedirs(test_dir, exist_ok=True)
    
    # Save individual fold results
    for fold_idx, fold_results in enumerate(results):
        fold_path = os.path.join(test_dir, f'fold_{fold_idx}_metrics.json')
        with open(fold_path, 'w') as f:
            json.dump(fold_results, f, indent=2)
    
    # Save aggregated results
    cv_path = os.path.join(test_dir, 'cross_val_metrics.json')
    with open(cv_path, 'w') as f:
        json.dump(aggregated, f, indent=2)
    
    print(f"✓ Saved {len(results)} fold results to: {test_dir}")
    print(f"✓ Saved aggregated CV metrics to: {cv_path}")


def print_cv_results(aggregated: dict, num_folds: int):
    """
    Print aggregated cross-validation results.
    
    Args:
        aggregated: Dict with aggregated metrics
        num_folds: Number of folds
    """
    print("\n" + "="*80)
    print(f"CROSS-VALIDATION RESULTS ({num_folds} folds)")
    print("="*80)
    
    # Extract base metric names (without _mean/_std/_all_folds suffix)
    base_metrics = set()
    for key in aggregated.keys():
        if key.endswith('_mean'):
            base_metrics.add(key.replace('_mean', ''))
    
    # Print mean ± std for each metric
    for metric in sorted(base_metrics):
        mean_key = f'{metric}_mean'
        std_key = f'{metric}_std'
        if mean_key in aggregated and std_key in aggregated:
            mean_val = aggregated[mean_key]
            std_val = aggregated[std_key]
            print(f"{metric:<20} {mean_val:.4f} ± {std_val:.4f}")
    
    print("="*80)


# =============================================================================
# CV-AWARE TESTING
# =============================================================================

# Collate function mapping
COLLATE_FN_MAP = {'AudioSet_EV_v1_2025': audioset_ev_v1_collate_fn,
                  'AudioSet_EV_v2PANNs_2020': audioset_ev_v2_collate_fn,
                  'sireNNet': sirennet_collate_fn,
                  'LSSiren': lssiren_collate_fn,
                  'ESC50': esc50_collate_fn,
                  'FSD50K': fsd50k_collate_fn,
                  'UrbanSound8K': urbansound8k_collate_fn}


def test_with_cv_support(trainer, model, datamodule, dataset_name: str):
    """
    Test model with proper CV fold separation.
    
    When a datamodule has multiple test_datasets (CV mode), this function
    tests each fold separately to get accurate per-fold metrics instead of
    aggregated metrics replicated across folds.
    
    Args:
        trainer: PyTorch Lightning Trainer
        model: Lightning module
        datamodule: DataModule (already setup)
        dataset_name: Name of dataset for collate_fn lookup
    
    Returns:
        List of result dictionaries (one per fold for CV, single for non-CV)
    """
    # Check if this is a CV dataset (has multiple test_datasets)
    is_cv = hasattr(datamodule, 'test_datasets') and len(getattr(datamodule, 'test_datasets', {})) > 1
    
    if is_cv:
        # Test each fold separately for CV datasets
        results = []
        test_datasets = datamodule.test_datasets
        collate_fn = COLLATE_FN_MAP.get(dataset_name)
        
        print(f"\n  → Testing {len(test_datasets)} folds separately for accurate CV metrics...")
        
        for fold_idx, (fold_name, fold_dataset) in enumerate(sorted(test_datasets.items())):
            # Create single dataloader for this fold
            fold_loader = DataLoader(fold_dataset,
                                     batch_size=datamodule.batch_size,
                                     shuffle=False,
                                     num_workers=datamodule.num_workers,
                                     collate_fn=collate_fn)
            
            # Test this fold
            fold_results = trainer.test(model, dataloaders=fold_loader, verbose=False)
            results.extend(fold_results)
            
            # Clear predictions for next fold
            model.test_predictions.clear()
            model.test_targets.clear()
    else:
        # Single test for non-CV datasets
        results = trainer.test(model, datamodule=datamodule)
    
    return results


# =============================================================================
# DATAMODULE LOADING
# =============================================================================

def get_datamodule(config: Dict[str, Any], dataset_name: str = None, mode_override: str = None) -> pl.LightningDataModule:
    """
    Initialize and return the appropriate DataModule based on config.
    
    Args:
        config: Configuration dictionary
        dataset_name: Override dataset name (for mixed datasets)
        mode_override: Override mode ('train' or 'benchmark')
    
    Returns:
        PyTorch Lightning DataModule
    """
    data_config = config['data']
    exp_config = config['experiment']
    
    # Determine dataset name
    if dataset_name is None:
        dataset_name = data_config['dataset']
    
    # Determine mode
    if mode_override is not None:
        mode = mode_override
    else:
        mode = data_config.get('mode', 'train')
    
    # Determine label_mode based on task
    label_mode = 'multi_class' if exp_config['task'] == 'multiclass' else 'binary'
    
    # Build dataset root path
    data_root = config['paths']['data_root']
    
    print(f"✓ Initializing {dataset_name} DataModule (label_mode='{label_mode}', mode='{mode}')")
    
    # Common parameters (num_workers auto-configured in DataModule)
    common_params = {'batch_size': data_config['batch_size'],
                     'target_sr': data_config['target_sr'],
                     'seed': exp_config['seed'],
                     'mode': mode}
    
    # Dataset-specific initialization with hardcoded paths
    if dataset_name == 'AudioSet_EV_v2PANNs_2020':
        dataset_path = os.path.join(data_root, dataset_name)
        
        # Load negative focus labels from datasets_mapping.json
        mapping_path = os.path.join(data_root, "datasets_mapping.json")
        with open(mapping_path, 'r') as f:
            mapping = json.load(f)
        negative_focus_labels = [label for label, val in mapping["AUDIOSET"].items() if val == 0]
        
        datamodule = AudioSetEV_v2_DataModule(TP_csv=os.path.join(dataset_path, "EV_Positives.csv"),
                                              TP_folder=os.path.join(dataset_path, "Positive_files"),
                                              TN_csv=os.path.join(dataset_path, "EV_Negatives.csv"),
                                              TN_folder=os.path.join(dataset_path, "Negative_files"),
                                              label_mapping_csv=os.path.join(dataset_path, "audioset_metadata", "class_labels_indices.csv"),
                                              negative_focus_labels=negative_focus_labels,
                                              label_mode=label_mode,
                                              batch_size=data_config['batch_size'],
                                              seed=exp_config['seed'])
    
    elif dataset_name == 'AudioSet_EV_v1_2025':
        dataset_path = os.path.join(data_root, dataset_name)
        datamodule = AudioSetEV_v1_DataModule(pos_csv_path=os.path.join(dataset_path, "EV_Positives.csv"),
                                              pos_audio_folder=os.path.join(dataset_path, "Positive_files"),
                                              neg_csv_path=os.path.join(dataset_path, "EV_Negatives.csv"),
                                              neg_audio_folder=os.path.join(dataset_path, "Negative_files"),
                                              label_mode=label_mode,
                                              target_size=int(data_config.get('target_duration', 10.0) * data_config['target_sr']),
                                              **common_params)
    
    elif dataset_name == 'sireNNet':
        dataset_path = os.path.join(data_root, dataset_name)
        datamodule = sireNNetDataModule(folder_path=dataset_path,
                                        label_mode=label_mode,
                                        target_size=int(data_config.get('target_duration', 3.0) * data_config['target_sr']),
                                        **common_params)
    
    elif dataset_name == 'LSSiren':
        dataset_path = os.path.join(data_root, dataset_name)
        datamodule = LSSirenDataModule(folder_path=dataset_path,
                                       min_length=int(data_config.get('target_duration', 10.0) * data_config['target_sr']),
                                       **common_params)
    
    elif dataset_name == 'ESC50':
        dataset_path = os.path.join(data_root, dataset_name)
        datamodule = ESC50DataModule(csv_path=os.path.join(dataset_path, "esc50.csv"),
                                     audio_folder=os.path.join(dataset_path, "original_audio"),
                                     target_size=int(data_config.get('target_duration', 5.0) * data_config['target_sr']),
                                     **common_params)
    
    elif dataset_name == 'FSD50K':
        dataset_path = os.path.join(data_root, dataset_name)
        datamodule = FSD50KDataModule(fsd_root=dataset_path,
                                      **common_params)
    
    elif dataset_name == 'UrbanSound8K':
        dataset_path = os.path.join(data_root, dataset_name)
        datamodule = UrbanSound8KDataModule(metadata_path=os.path.join(dataset_path, "metadata", "UrbanSound8K.csv"),
                                            audio_folder=os.path.join(dataset_path, "audio"),
                                            min_length=int(data_config.get('target_duration', 4.0) * data_config['target_sr']),
                                            **common_params)
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. "
                         f"Available: AudioSet_EV_v1_2025, AudioSet_EV_v2PANNs_2020, sireNNet, "
                         f"LSSiren, ESC50, FSD50K, UrbanSound8K")
    
    return datamodule


# =============================================================================
# MODEL INITIALIZATION
# =============================================================================

def get_lightning_module(config: Dict[str, Any]) -> pl.LightningModule:
    """
    Initialize and return the appropriate Lightning Module based on config.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        PyTorch Lightning Module
    """
    exp_config = config['experiment']
    model_config = config['model']
    training_config = config['training']
    paths_config = config['paths']
    
    # Check if loading from Lightning checkpoint
    lightning_ckpt_path = exp_config.get('lightning_ckpt_path', None)
    
    # Handle YAML None/null variants
    if lightning_ckpt_path in [None, 'None', 'null', 'none', '']:
        lightning_ckpt_path = None
    
    if lightning_ckpt_path:
        # Load from Lightning checkpoint
        print(f"Loading from Lightning checkpoint: {lightning_ckpt_path}")
        
        if exp_config['task'] == 'binary':
            model = BinaryEVClassifier.load_from_checkpoint(lightning_ckpt_path,
                                                            results_path=paths_config['results'],
                                                            f_beta=model_config['f_beta'])
        elif exp_config['task'] == 'multiclass':
            model = MultiClassSirenClassifier.load_from_checkpoint(lightning_ckpt_path,
                                                                   results_path=paths_config['results'],
                                                                   f_beta=model_config['f_beta'])
        else:
            raise ValueError(f"Invalid task: {exp_config['task']}")
        
        print(f"✓ Loaded model from Lightning checkpoint")
        return model
    
    # Common parameters (pretrained model initialization)
    common_params = {'model_name': exp_config['model_name'],
                     'pretrained': exp_config['pretrained'],
                     'optimizer_kwargs': training_config['optimizer'],
                     'scheduler_type': training_config['scheduler']['type'],
                     'scheduler_kwargs': {k: v for k, v in training_config['scheduler'].items() if k != 'type'},
                     'results_path': paths_config['results'],
                     'f_beta': model_config['f_beta']}
    
    # Task-specific initialization
    if exp_config['task'] == 'binary':
        model = BinaryEVClassifier(threshold=model_config['threshold'],
                                   **common_params)
        print(f"✓ Initialized BinaryEVClassifier with {exp_config['model_name'].upper()}")
    
    elif exp_config['task'] == 'multiclass':
        model = MultiClassSirenClassifier(num_classes=4,
                                          **common_params)
        print(f"✓ Initialized MultiClassSirenClassifier with {exp_config['model_name'].upper()}")
    
    else:
        raise ValueError(f"Invalid task: {exp_config['task']}")
    
    return model


# =============================================================================
# CALLBACKS AND LOGGER
# =============================================================================

def get_callbacks(config: Dict[str, Any]):
    """
    Create callbacks for training.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        List of callbacks
    """
    callbacks = []
    callback_config = config['callbacks']
    paths_config = config['paths']
    
    # Early Stopping
    if 'early_stopping' in callback_config:
        es_config = callback_config['early_stopping']
        early_stop = EarlyStopping(monitor=es_config['monitor'],
                                   patience=es_config['patience'],
                                   mode=es_config['mode'],
                                   min_delta=es_config.get('min_delta', 0.0))
        callbacks.append(early_stop)
        print(f"✓ Added EarlyStopping (monitor={es_config['monitor']}, patience={es_config['patience']})")
    
    # Model Checkpoint
    if 'model_checkpoint' in callback_config:
        ckpt_config = callback_config['model_checkpoint']
        checkpoint = ModelCheckpoint(dirpath=paths_config['checkpoints'],
                                     monitor=ckpt_config['monitor'],
                                     save_top_k=ckpt_config['save_top_k'],
                                     mode=ckpt_config['mode'],
                                     save_last=ckpt_config.get('save_last', True),
                                     filename=ckpt_config.get('filename', 'epoch={epoch:03d}'))
        callbacks.append(checkpoint)
        print(f"✓ Added ModelCheckpoint (monitor={ckpt_config['monitor']}, save_top_k={ckpt_config['save_top_k']})")
    
    return callbacks


def get_logger(config: Dict[str, Any]):
    """
    Create TensorBoard logger.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        TensorBoard logger
    """
    paths_config = config['paths']
    exp_config = config['experiment']
    
    logger = TensorBoardLogger(save_dir=paths_config['logs'],
                               name=exp_config['name'],
                               default_hp_metric=False)
    
    print(f"✓ TensorBoard logs will be saved to: {paths_config['logs']}/{exp_config['name']}")
    
    return logger


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def main():
    """Main experiment function."""
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train/test EV detection models')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to YAML configuration file')
    parser.add_argument('--mode', type=str, default='fit+test',
                        choices=['fit', 'test', 'fit+test', 'benchmark'],
                        help='Execution mode: fit (train+val), test, fit+test, or benchmark (test only with benchmark splits)')
    args = parser.parse_args()
    
    # Load and validate configuration
    config = load_config(args.config)
    validate_config(config)
    
    # Set random seed
    pl.seed_everything(config['experiment']['seed'], workers=True)
    print(f"✓ Set random seed to {config['experiment']['seed']}")
    
    # Create directories
    for _, path_value in config['paths'].items():
        os.makedirs(path_value, exist_ok=True)
    print("✓ Created all required directories")
    
    # Initialize components
    print("\n" + "="*80)
    print("INITIALIZING COMPONENTS")
    print("="*80)
    
    datamodule = get_datamodule(config)
    model = get_lightning_module(config)
    callbacks = get_callbacks(config)
    logger = get_logger(config)
    
    # Launch TensorBoard in background if training (uncomment to enable)
    tensorboard_process = None
    
    if args.mode in ['fit', 'fit+test']:
        try:
            # Get tensorboard executable path
            venv_tensorboard = os.path.join(os.path.dirname(sys.executable), 'tensorboard')
            
            # Check if tensorboard exists
            if not os.path.exists(venv_tensorboard):
                print(f"\n⚠ TensorBoard not found. Install with: pip install tensorboard")
                print(f"   You can manually run later: tensorboard --logdir {config['paths']['logs']}")
            else:
                # Start TensorBoard in background
                log_dir = config['paths']['logs']
                port = 6006
                tensorboard_process = subprocess.Popen([venv_tensorboard, '--logdir', log_dir, '--port', str(port), '--bind_all'],
                                                       stdout=subprocess.PIPE,
                                                       stderr=subprocess.PIPE)
                
                time.sleep(2)  # Give TensorBoard time to start
                
                # Verify it's still running
                if tensorboard_process.poll() is None:
                    print(f"\n🚀 TensorBoard launched successfully!")
                    print(f"   Open in browser: http://localhost:{port}")
                    print(f"   PID: {tensorboard_process.pid}")
                else:
                    stderr = tensorboard_process.stderr.read().decode('utf-8') if tensorboard_process.stderr else ''
                    print(f"\n⚠ TensorBoard failed to start")
                    if stderr:
                        print(f"   Error: {stderr[:200]}")
                    tensorboard_process = None
                    
        except Exception as e:
            print(f"\n⚠ Failed to launch TensorBoard: {e}")
            print(f"   You can manually run: tensorboard --logdir {config['paths']['logs']}")
            tensorboard_process = None
    
    # Initialize Trainer
    print("\n" + "="*80)
    print("INITIALIZING TRAINER")
    print("="*80)
    
    hardware_config = config.get('hardware', {})
    tensorboard_config = config.get('tensorboard', {})
    
    # Handle strategy parameter (None is not valid, use 'auto' instead)
    strategy = hardware_config.get('strategy')
    if strategy is None:
        strategy = 'auto'
    
    trainer = pl.Trainer(max_epochs=config['training']['max_epochs'],
                         accelerator=hardware_config.get('accelerator', 'auto'),
                         devices=hardware_config.get('devices', 1),
                         strategy=strategy,
                         precision=config['training'].get('precision', 32),
                         gradient_clip_val=config['training'].get('gradient_clip_val', None),
                         val_check_interval=config['training'].get('val_check_interval', 1.0),
                         limit_train_batches=config['training'].get('limit_train_batches', None),
                         limit_val_batches=config['training'].get('limit_val_batches', None),
                         limit_test_batches=config['training'].get('limit_test_batches', None),
                         log_every_n_steps=tensorboard_config.get('log_every_n_steps', 50),
                         callbacks=callbacks,
                         logger=logger,
                         deterministic="warn")  # Allows non-deterministic ops (e.g., CLAP upsampling) with warnings
    
    print(f"✓ Trainer initialized with max_epochs={config['training']['max_epochs']}")
    
    # Execute based on mode
    print("\n" + "="*80)
    print(f"STARTING EXPERIMENT: {config['experiment']['name']}")
    print(f"Task: {config['experiment']['task']}")
    print(f"Model: {config['experiment']['model_name'].upper()}")
    print(f"Mode: {args.mode}")
    print("="*80 + "\n")
    
    # Handle benchmark mode
    if args.mode == 'benchmark':
        print("\n BENCHMARK MODE: Test only with benchmark splits")
        # Force benchmark mode and use test dataset if specified
        config['data']['mode'] = 'benchmark'
        test_dataset = config['data'].get('test_dataset', config['data']['dataset'])
        test_mode = config['data'].get('test_mode', 'benchmark')
        test_datamodule = get_datamodule(config, dataset_name=test_dataset, mode_override=test_mode)
        test_datamodule.setup('test')
        
        # Test with CV-aware fold separation
        test_results = test_with_cv_support(trainer, model, test_datamodule, test_dataset)
        
        # Handle cross-validation results if multiple folds
        if len(test_results) > 1:
            aggregated = aggregate_cv_results(test_results)
            print_cv_results(aggregated, len(test_results))
            save_cv_results(test_results, aggregated, config['paths']['results'])
        
        print("\n✓ Benchmark testing completed!")
    
    # Handle mixed datasets (train/dev from one, test from another)
    elif 'train_dev_dataset' in config['data'] and 'test_dataset' in config['data']:
        print("\n MIXED DATASETS MODE")
        train_dev_dataset = config['data']['train_dev_dataset']
        test_dataset = config['data']['test_dataset']
        train_dev_mode = config['data'].get('train_dev_mode', 'train')
        test_mode = config['data'].get('test_mode', 'train')
        
        print(f"   Train/Dev: {train_dev_dataset} (mode={train_dev_mode})")
        print(f"   Test: {test_dataset} (mode={test_mode})")
        
        train_dev_datamodule = get_datamodule(config, dataset_name=train_dev_dataset, mode_override=train_dev_mode)
        test_datamodule = get_datamodule(config, dataset_name=test_dataset, mode_override=test_mode)
        
        if args.mode in ['fit', 'fit+test']:
            trainer.fit(model, datamodule=train_dev_datamodule)
            print("\n✓ Training completed!")
        
        if args.mode in ['test', 'fit+test']:
            if args.mode == 'fit+test':
                print(f"\n✓ Loading best checkpoint for testing...")
            
            # Setup test datamodule and test with CV-aware fold separation
            test_datamodule.setup('test')
            test_results = test_with_cv_support(trainer, model, test_datamodule, test_dataset)
            
            # Handle cross-validation results if multiple folds
            if len(test_results) > 1:
                aggregated = aggregate_cv_results(test_results)
                print_cv_results(aggregated, len(test_results))
                save_cv_results(test_results, aggregated, config['paths']['results'])
            
            print("\n✓ Testing completed!")
    
    # Handle single dataset mode
    else:
        if args.mode in ['fit', 'fit+test']:
            trainer.fit(model, datamodule=datamodule)
            print("\n✓ Training completed!")
        
        if args.mode in ['test', 'fit+test']:
            if args.mode == 'fit+test':
                print(f"\n✓ Loading best checkpoint for testing...")
            
            # Setup and test with CV-aware fold separation
            dataset_name = config['data']['dataset']
            datamodule.setup('test')
            test_results = test_with_cv_support(trainer, model, datamodule, dataset_name)
            
            # Handle cross-validation results if multiple folds
            if len(test_results) > 1:
                aggregated = aggregate_cv_results(test_results)
                print_cv_results(aggregated, len(test_results))
                save_cv_results(test_results, aggregated, config['paths']['results'])
            
            print("\n✓ Testing completed!")
    
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETED SUCCESSFULLY")
    print("="*80)
    print(f"\nResults saved to: {config['paths']['results']}")
    print(f"Checkpoints saved to: {config['paths']['checkpoints']}")
    print(f"TensorBoard logs: {config['paths']['logs']}")
    
    # TensorBoard cleanup - interactive prompt (uncomment to enable)
    if tensorboard_process is not None and tensorboard_process.poll() is None:
        print(f"\n📊 TensorBoard is still running (PID: {tensorboard_process.pid})")
        print(f"   View at: http://localhost:6006")
        print("\n" + "="*80)
        
        # Interactive prompt to close TensorBoard
        try:
            response = input("🔴 Stop TensorBoard? (y/n): ").strip().lower()
            if response == 'y':
                print("Stopping TensorBoard...")
                tensorboard_process.terminate()
                try:
                    tensorboard_process.wait(timeout=5)
                    print("✓ TensorBoard stopped successfully")
                except subprocess.TimeoutExpired:
                    tensorboard_process.kill()
                    print("✓ TensorBoard forcefully stopped")
            else:
                print(f"✓ TensorBoard left running (PID: {tensorboard_process.pid})")
                print(f"   To stop later: kill {tensorboard_process.pid}")
        except KeyboardInterrupt:
            print("\n\n✓ Keeping TensorBoard running")
            print(f"   To stop: kill {tensorboard_process.pid}")
    elif tensorboard_process is None:
        print("\nTo view TensorBoard logs, run:")
        print(f"  tensorboard --logdir {config['paths']['logs']}")
    
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
