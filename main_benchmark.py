"""
Benchmark Evaluation Script for E2PANNs
========================================
Test pretrained or finetuned models on all available datasets.

Usage:
    python main_benchmark.py

Configure parameters in the CONFIGURATION section below.

Author: Stefano Giacomelli - Ph.D. candidate in ICT (DISIM dpt. - University of L'Aquila)
"""

import os
import sys
import csv
import json
import shutil
import tempfile
from datetime import datetime
from typing import Dict, List

import random
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import Lightning modules
from models.lightning_models import BinaryEVClassifier, MultiClassSirenClassifier

# Import dataloaders
from datasets.AudioSet_EV_v1_2025.dataloader import AudioSetEV_v1_DataModule, audioset_ev_v1_collate_fn
from datasets.AudioSet_EV_v2PANNs_2020.dataloader import AudioSetEV_v2_DataModule, custom_collate_fn as audioset_ev_v2_collate_fn
from datasets.sireNNet.dataloader import sireNNetDataModule, sirennet_collate_fn
from datasets.LSSiren.dataloader import LSSirenDataModule, lssiren_collate_fn
from datasets.ESC50.dataloader import ESC50DataModule, esc50_collate_fn
from datasets.FSD50K.dataloader import FSD50KDataModule, fsd50k_collate_fn
from datasets.UrbanSound8K.dataloader import UrbanSound8KDataModule, urbansound8k_collate_fn


# ============================================================================
# CONFIGURATION
# ============================================================================

# Model Configuration
MODEL_NAME = 'epanns'                                           # 'epanns', 'ced', 'clap'
CHECKPOINT_TYPE = 'pretrained'                                  # 'pretrained' (.pt) or 'finetuned' (.ckpt)
CHECKPOINT_PATH = './models/epanns/checkpoint_closeto_.44.pt'   # Path to AudioSet checkpoint

# For finetuned (EV-framework) checkpoint example:
# CHECKPOINT_TYPE = 'finetuned'
# CHECKPOINT_PATH = './checkpoints/best_model.ckpt'

# Datasets to test
DATASETS_TO_TEST = []                                           # Empty = all, or ['ESC50', 'AudioSet_EV_v1_2025', ...]

# Test Configuration
BATCH_SIZE = 32
NUM_WORKERS = 0
LIMIT_TEST_BATCHES = None                                       # Set to a float (e.g., 0.1) to limit to a fraction of the test set

# Hardware
DEVICE = 'auto'

# Reproducibility
SEED = 42

# Output Configuration
OUTPUT_DIR = './benchmark_results'

# Sample Rate per model (auto-determined)
SAMPLE_RATES = {'epanns': 32000, 'ced': 16000, 'clap': 48000}

# Dataset compatibility with tasks
DATASET_TASK_SUPPORT = {'AudioSet_EV_v1_2025': ['binary', 'multiclass'],
                        'AudioSet_EV_v2PANNs_2020': ['binary', 'multiclass'],
                        'sireNNet': ['binary', 'multiclass'],
                        'LSSiren': ['binary'],
                        'ESC50': ['binary'],
                        'FSD50K': ['binary'],
                        'UrbanSound8K': ['binary']}

# Datasets with cross-validation support
CV_DATASETS = ['ESC50', 'UrbanSound8K', 'sireNNet']

# Collate function mapping
COLLATE_FN_MAP = {'AudioSet_EV_v1_2025': audioset_ev_v1_collate_fn,
                  'AudioSet_EV_v2PANNs_2020': audioset_ev_v2_collate_fn,
                  'sireNNet': sirennet_collate_fn,
                  'LSSiren': lssiren_collate_fn,
                  'ESC50': esc50_collate_fn,
                  'FSD50K': fsd50k_collate_fn,
                  'UrbanSound8K': urbansound8k_collate_fn}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed, workers=True)


def load_model(checkpoint_type: str, checkpoint_path: str, model_name: str, task: str, results_path: str):
    """
    Load model from checkpoint.
    
    Args:
        checkpoint_type: 'pretrained' (.pt) or 'finetuned' (.ckpt)
        checkpoint_path: Path to checkpoint file
        model_name: Model name ('epanns', 'ced', 'clap')
        task: Task type ('binary', 'multiclass')
        results_path: Path for saving results
    
    Returns:
        Lightning module
    """
    print(f"\n{'='*80}")
    print(f"LOADING MODEL")
    print(f"{'='*80}")
    print(f"Model: {model_name.upper()}")
    print(f"Task: {task}")
    print(f"Checkpoint type: {checkpoint_type}")
    print(f"Checkpoint path: {checkpoint_path}")
    
    if checkpoint_type == 'finetuned':
        # Load from Lightning checkpoint
        if task == 'binary':
            model = BinaryEVClassifier.load_from_checkpoint(checkpoint_path,
                                                            results_path=results_path)
        else:
            model = MultiClassSirenClassifier.load_from_checkpoint(checkpoint_path,
                                                                   results_path=results_path)
        print(f"✓ Loaded finetuned model from Lightning checkpoint")
    
    else:  # pretrained
        # Load with pretrained weights
        if task == 'binary':
            model = BinaryEVClassifier(model_name=model_name,
                                       pretrained=True,
                                       threshold=0.5,
                                       f_beta=0.5,
                                       optimizer_kwargs={'lr': 0.001},
                                       scheduler_type='exponential',
                                       scheduler_kwargs={'gamma': 0.95},
                                       results_path=results_path)
        else:
            model = MultiClassSirenClassifier(model_name=model_name,
                                              pretrained=True,
                                              num_classes=4,
                                              f_beta=0.5,
                                              optimizer_kwargs={'lr': 0.001},
                                              scheduler_type='exponential',
                                              scheduler_kwargs={'gamma': 0.95},
                                              results_path=results_path)
        print(f"✓ Loaded AudioSet pretrained {model_name.upper()} model")
    
    return model


def initialize_datamodule(dataset_name: str, task: str, model_name: str):
    """
    Initialize datamodule for benchmark testing.
    
    Args:
        dataset_name: Dataset name
        task: Task type ('binary', 'multiclass')
        model_name: Model name (for sample rate)
    
    Returns:
        PyTorch Lightning DataModule
    """
    data_root = './datasets'
    target_sr = SAMPLE_RATES[model_name]
    
    # Determine mode: 'benchmark' for CV datasets, 'train' for others
    mode = 'benchmark' if dataset_name in CV_DATASETS else 'train'
    label_mode = 'binary' if task == 'binary' else 'multi_class'
    
    common_params = {'batch_size': BATCH_SIZE,
                     'num_workers': NUM_WORKERS,
                     'seed': SEED,
                     'mode': mode,
                     'target_sr': target_sr}
    
    if dataset_name == 'AudioSet_EV_v1_2025':
        dataset_path = os.path.join(data_root, dataset_name)
        datamodule = AudioSetEV_v1_DataModule(pos_csv_path=os.path.join(dataset_path, 'EV_Positives.csv'),
                                              pos_audio_folder=os.path.join(dataset_path, 'Positive_files'),
                                              neg_csv_path=os.path.join(dataset_path, 'EV_Negatives.csv'),
                                              neg_audio_folder=os.path.join(dataset_path, 'Negative_files'),
                                              label_mode=label_mode,
                                              target_size=int(10.0 * target_sr),
                                              **common_params)
    
    elif dataset_name == 'AudioSet_EV_v2PANNs_2020':
        dataset_path = os.path.join(data_root, dataset_name)
        
        # Load negative focus labels from datasets_mapping.json
        mapping_path = os.path.join(data_root, 'datasets_mapping.json')
        
        with open(mapping_path, 'r') as f:
            mapping = json.load(f)
        
        negative_focus_labels = [label for label, val in mapping['AUDIOSET'].items() if val == 0]
        
        datamodule = AudioSetEV_v2_DataModule(TP_csv=os.path.join(dataset_path, 'EV_Positives.csv'),
                                              TP_folder=os.path.join(dataset_path, 'Positive_files'),
                                              TN_csv=os.path.join(dataset_path, 'EV_Negatives.csv'),
                                              TN_folder=os.path.join(dataset_path, 'Negative_files'),
                                              label_mapping_csv=os.path.join(dataset_path, 'audioset_metadata', 'class_labels_indices.csv'),
                                              negative_focus_labels=negative_focus_labels,
                                              label_mode=label_mode,
                                              batch_size=BATCH_SIZE,
                                              seed=SEED)
    
    elif dataset_name == 'sireNNet':
        dataset_path = os.path.join(data_root, dataset_name)
        datamodule = sireNNetDataModule(folder_path=dataset_path,
                                        label_mode=label_mode,
                                        target_size=int(3.0 * target_sr),
                                        **common_params)
    
    elif dataset_name == 'LSSiren':
        dataset_path = os.path.join(data_root, dataset_name)
        datamodule = LSSirenDataModule(folder_path=dataset_path,
                                       min_length=int(10.0 * target_sr),
                                       **common_params)
    
    elif dataset_name == 'ESC50':
        dataset_path = os.path.join(data_root, dataset_name)
        datamodule = ESC50DataModule(csv_path=os.path.join(dataset_path, 'esc50.csv'),
                                     audio_folder=os.path.join(dataset_path, 'original_audio'),
                                     target_size=int(5.0 * target_sr),
                                     **common_params)
    
    elif dataset_name == 'FSD50K':
        dataset_path = os.path.join(data_root, dataset_name)
        datamodule = FSD50KDataModule(fsd_root=dataset_path,
                                      **common_params)
    
    elif dataset_name == 'UrbanSound8K':
        dataset_path = os.path.join(data_root, dataset_name)
        datamodule = UrbanSound8KDataModule(metadata_path=os.path.join(dataset_path, 'metadata', 'UrbanSound8K.csv'),
                                            audio_folder=os.path.join(dataset_path, 'audio'),
                                            min_length=int(4.0 * target_sr),
                                            **common_params)
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return datamodule


def test_dataset(model, datamodule, dataset_name: str, task: str, output_dir: str) -> List[Dict]:
    """
    Test model on a dataset.
    
    Args:
        model: Lightning module
        datamodule: DataModule
        dataset_name: Dataset name
        task: Task type
        output_dir: Directory for saving predictions
    
    Returns:
        List of result dictionaries (one per fold for CV datasets)
    """
    # Setup datamodule
    datamodule.setup('test')
    
    # Check if this is a CV dataset (has multiple test_datasets)
    is_cv = hasattr(datamodule, 'test_datasets') and len(getattr(datamodule, 'test_datasets', {})) > 1
    
    # Get collate function for this dataset
    collate_fn = COLLATE_FN_MAP.get(dataset_name, None)
    
    if is_cv:
        # Test each fold separately for CV datasets
        results = []
        test_datasets = datamodule.test_datasets
        
        for fold_idx, (fold_name, fold_dataset) in enumerate(sorted(test_datasets.items())):
            # Create single dataloader for this fold
            fold_loader = DataLoader(fold_dataset,
                                     batch_size=datamodule.batch_size,
                                     shuffle=False,
                                     num_workers=datamodule.num_workers,
                                     collate_fn=collate_fn)
            
            # Create trainer
            trainer = Trainer(accelerator=DEVICE,
                              devices=1,
                              logger=False,
                              enable_checkpointing=False,
                              enable_progress_bar=True,
                              limit_test_batches=LIMIT_TEST_BATCHES,
                              deterministic=True)
            
            # Test this fold
            fold_results = trainer.test(model, dataloaders=fold_loader)
            results.extend(fold_results)
            
            # Clear predictions for next fold
            model.test_predictions.clear()
            model.test_targets.clear()
    else:
        # Single test for non-CV datasets
        trainer = Trainer(accelerator=DEVICE,
                          devices=1,
                          logger=False,
                          enable_checkpointing=False,
                          enable_progress_bar=True,
                          limit_test_batches=LIMIT_TEST_BATCHES,
                          deterministic=True)
        
        results = trainer.test(model, datamodule=datamodule)
    
    return results


def format_results_for_csv(dataset_name: str, results: List[Dict], task: str) -> List[Dict]:
    """
    Format results for CSV output.
    
    Args:
        dataset_name: Dataset name
        results: List of result dicts from Lightning
        task: Task type
    
    Returns:
        List of formatted row dictionaries for CSV
    """
    rows = []
    
    if len(results) > 1:  # Cross-validation
        # Individual fold rows
        for fold_idx, fold_result in enumerate(results):
            row = {'Dataset': dataset_name,
                   'Fold': f'fold_{fold_idx + 1}',
                   'Task': task,
                   'Accuracy': f"{fold_result.get('test_accuracy', 0.0):.4f}",
                   'Precision': f"{fold_result.get('test_precision', 0.0):.4f}",
                   'Recall': f"{fold_result.get('test_recall', 0.0):.4f}",
                   'Specificity': f"{fold_result.get('test_specificity', 0.0):.4f}",
                   'F1': f"{fold_result.get('test_f1_score', 0.0):.4f}",
                   'AUROC': f"{fold_result.get('test_auroc', 0.0):.4f}",
                   'FBeta': f"{fold_result.get('test_fbeta_score', 0.0):.4f}"}
            rows.append(row)
        
        # Aggregate row
        metrics = {}
        for metric_key in ['test_accuracy', 'test_precision', 'test_recall', 'test_specificity', 
                           'test_f1_score', 'test_auroc', 'test_fbeta_score']:
            values = [r.get(metric_key, 0.0) for r in results]
            metrics[metric_key] = {'mean': np.mean(values),
                                   'std': np.std(values)}
        
        agg_row = {'Dataset': dataset_name,
                   'Fold': 'CV_Aggregate',
                   'Task': task,
                   'Accuracy': f"{metrics['test_accuracy']['mean']:.4f} ± {metrics['test_accuracy']['std']:.4f}",
                   'Precision': f"{metrics['test_precision']['mean']:.4f} ± {metrics['test_precision']['std']:.4f}",
                   'Recall': f"{metrics['test_recall']['mean']:.4f} ± {metrics['test_recall']['std']:.4f}",
                   'Specificity': f"{metrics['test_specificity']['mean']:.4f} ± {metrics['test_specificity']['std']:.4f}",
                   'F1': f"{metrics['test_f1_score']['mean']:.4f} ± {metrics['test_f1_score']['std']:.4f}",
                   'AUROC': f"{metrics['test_auroc']['mean']:.4f} ± {metrics['test_auroc']['std']:.4f}",
                   'FBeta': f"{metrics['test_fbeta_score']['mean']:.4f} ± {metrics['test_fbeta_score']['std']:.4f}"}
        rows.append(agg_row)
    
    else:  # Single result
        result = results[0]
        row = {'Dataset': dataset_name,
               'Fold': '-',
               'Task': task,
               'Accuracy': f"{result.get('test_accuracy', 0.0):.4f}",
               'Precision': f"{result.get('test_precision', 0.0):.4f}",
               'Recall': f"{result.get('test_recall', 0.0):.4f}",
               'Specificity': f"{result.get('test_specificity', 0.0):.4f}",
               'F1': f"{result.get('test_f1_score', 0.0):.4f}",
               'AUROC': f"{result.get('test_auroc', 0.0):.4f}",
               'FBeta': f"{result.get('test_fbeta_score', 0.0):.4f}"}
        rows.append(row)
    
    return rows


def save_to_csv(rows: List[Dict], output_path: str):
    """Save results to CSV file."""
    fieldnames = ['Dataset', 'Fold', 'Task', 'Accuracy', 'Precision', 'Recall', 
                  'Specificity', 'F1', 'AUROC', 'FBeta']
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"\n✓ Results saved to: {output_path}")


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main benchmark evaluation function."""
    # Set seed for reproducibility
    set_seed(SEED)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Create temporary directory for Lightning model outputs (will be deleted)
    temp_results_dir = tempfile.mkdtemp(prefix='benchmark_temp_')
    
    # Generate timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Print header
    print("\n" + "="*80)
    print("BENCHMARK EVALUATION")
    print("="*80)
    print(f"Model: {MODEL_NAME.upper()} ({CHECKPOINT_TYPE})")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Seed: {SEED}")
    print(f"Testing both tasks: binary + multiclass")
    print("="*80)
    
    # Test both tasks
    tasks_to_test = ['binary', 'multiclass']
    
    for task in tasks_to_test:
        print(f"\n{'#'*80}")
        print(f"# TASK: {task.upper()}")
        print(f"{'#'*80}")
        
        # Generate output filename
        csv_filename = f"{MODEL_NAME}_{task}_benchmark_{timestamp}.csv"
        csv_path = os.path.join(OUTPUT_DIR, csv_filename)
        print(f"\nOutput CSV: {csv_path}")
        
        # Load model for this task (use temp directory for Lightning outputs)
        model = load_model(CHECKPOINT_TYPE, CHECKPOINT_PATH, MODEL_NAME, task, temp_results_dir)
        
        # Determine datasets to test for this task
        if DATASETS_TO_TEST:
            datasets = [ds for ds in DATASETS_TO_TEST if task in DATASET_TASK_SUPPORT.get(ds, [])]
        else:
            datasets = [ds for ds, tasks in DATASET_TASK_SUPPORT.items() if task in tasks]
        
        print(f"\nDatasets to test: {len(datasets)}")
        for ds in datasets:
            cv_marker = " (CV)" if ds in CV_DATASETS else ""
            print(f"  - {ds}{cv_marker}")
        
        if not datasets:
            print(f"\n⚠️  No datasets available for {task} task, skipping...")
            continue
        
        # Test all datasets
        all_rows = []
        for idx, dataset_name in enumerate(datasets, 1):
            print(f"\n{'='*80}")
            print(f"[{idx}/{len(datasets)}] Testing {dataset_name}")
            print(f"{'='*80}")
            
            try:
                # Initialize datamodule
                datamodule = initialize_datamodule(dataset_name, task, MODEL_NAME)
                print(f"✓ Initialized {dataset_name} DataModule")
                
                # Test
                results = test_dataset(model, datamodule, dataset_name, task, OUTPUT_DIR)
                
                # Format results
                rows = format_results_for_csv(dataset_name, results, task)
                all_rows.extend(rows)
                
                # Print summary
                if len(results) > 1:
                    print(f"\n✓ CV Results ({len(results)} folds):")
                    # Extract mean metrics from aggregate row
                    agg_row = rows[-1]
                    print(f"  Accuracy: {agg_row['Accuracy']}")
                    print(f"  F1: {agg_row['F1']}")
                else:
                    print(f"\n✓ Results:")
                    print(f"  Accuracy: {rows[0]['Accuracy']}")
                    print(f"  F1: {rows[0]['F1']}")
            
            except Exception as e:
                print(f"\n✗ FAILED: {dataset_name}")
                print(f"  Error: {str(e)}")
                
                import traceback
                traceback.print_exc()
                raise
        
        # Save results to CSV
        save_to_csv(all_rows, csv_path)
        
        print(f"\n✓ {task.upper()} task completed - {len(datasets)} datasets tested")
    
    # Final summary
    print("\n" + "="*80)
    print("BENCHMARK COMPLETED")
    print("="*80)
    print(f"Model: {MODEL_NAME.upper()}")
    print(f"Tasks tested: binary + multiclass")
    print(f"\nResults saved to:")
    print(f"  - {MODEL_NAME}_binary_benchmark_{timestamp}.csv")
    print(f"  - {MODEL_NAME}_multiclass_benchmark_{timestamp}.csv")
    print("="*80 + "\n")
    
    # Clean up temporary directory
    if os.path.exists(temp_results_dir):
        shutil.rmtree(temp_results_dir)
        print(f"✓ Cleaned up temporary files")


if __name__ == '__main__':
    main()
