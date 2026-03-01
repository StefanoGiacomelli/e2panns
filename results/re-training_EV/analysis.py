"""
Re-training from Scratch - Emergency Vehicle Classification Analysis
=====================================================================
Analyzes re-training results for models trained from scratch on AudioSet-EV datasets.

This script:
1. Loads validation metrics across epochs for 8 models
2. Loads test confusion matrices for 8 models
3. Generates 6 comprehensive figures (SVG 600 DPI):
   - 3 for binary classification
   - 3 for multiclass classification

Models analyzed:
Binary Classification (2 classes: EV / non-EV):
- CED trained from scratch on AudioSet-EV v1 and v2
- EPANNs trained from scratch on AudioSet-EV v1 and v2

Multiclass Classification (4 classes: Traffic, Police, Ambulance, Fire):
- CED trained from scratch on AudioSet-EV v1 and v2
- EPANNs trained from scratch on AudioSet-EV v1 and v2

Author: Stefano Giacomelli - Ph.D. candidate in ICT (DISIM dpt. - University of L'Aquila)
"""

import json
from pathlib import Path
from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt

# Set high-quality plot parameters (matching preliminary_profiling style)
plt.rcParams['figure.dpi'] = 600
plt.rcParams['savefig.dpi'] = 600
plt.rcParams['savefig.format'] = 'svg'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10


# =============================================================================
# Configuration
# =============================================================================

# Binary classification models (2 classes)
BINARY_MODELS = {'CED_AS-EV_v1': 'ced_scratch_binary_AS-EV_v1',
                 'CED_AS-EV_v2': 'ced_scratch_binary_AS-EV_v2',
                 'EPANNs_AS-EV_v1': 'epanns_scratch_binary_AS-EV_v1',
                 'EPANNs_AS-EV_v2': 'epanns_scratch_binary_AS-EV_v2'}

# Multiclass classification models (4 classes)
MULTICLASS_MODELS = {'CED_AS-EV_v1': 'ced_scratch_multiclass_AS-EV_v1',
                     'CED_AS-EV_v2': 'ced_scratch_multiclass_AS-EV_v2',
                     'EPANNs_AS-EV_v1': 'epanns_scratch_multiclass_AS-EV_v1',
                     'EPANNs_AS-EV_v2': 'epanns_scratch_multiclass_AS-EV_v2'}

# Color scheme: EPANNs (blue), CED (red)
# v1 = lighter, v2 = darker
COLORS = {'EPANNs_AS-EV_v1': '#87CEEB',  # skyblue
          'EPANNs_AS-EV_v2': '#1E3A8A',  # darkblue
          'CED_AS-EV_v1': '#F08080',     # lightcoral
          'CED_AS-EV_v2': '#8B0000'}     # darkred

# Class labels for 4-way classification
CLASS_LABELS = ['Traffic', 'Police', 'Ambulance', 'Fire']

# Output directory
FIGURES_DIR = Path(__file__).parent.parent.parent / 'figures'


# =============================================================================
# Data Loading Functions
# =============================================================================

def load_validation_metrics(model_dir: Path) -> Dict[str, List]:
    """Load all validation metrics from epoch JSON files."""
    validation_dir = model_dir / 'validation'
    
    metrics = {'epochs': [],
               'accuracy': [],
               'precision': [],
               'recall': [],
               'f1_score': []}
    
    # Sort epoch files numerically
    epoch_files = sorted(validation_dir.glob('epoch_*_metrics.json'),
                        key=lambda x: int(x.stem.split('_')[1]))
    
    for epoch_file in epoch_files:
        with open(epoch_file, 'r') as f:
            data = json.load(f)
        
        metrics['epochs'].append(data['epoch'])
        metrics['accuracy'].append(data['accuracy'] * 100)  # Convert to percentage
        metrics['precision'].append(data['precision'] * 100)
        metrics['recall'].append(data['recall'] * 100)
        metrics['f1_score'].append(data['f1_score'] * 100)
    
    return metrics


def load_test_metrics_binary(model_dir: Path) -> Dict:
    """Load test metrics including 2x2 confusion matrix for binary classification."""
    test_file = model_dir / 'test' / 'test_metrics.json'
    
    with open(test_file, 'r') as f:
        data = json.load(f)
    
    # Extract confusion matrix
    cm = data['confusion_matrix']
    confusion_matrix = np.array([[cm['tn'], cm['fp']], [cm['fn'], cm['tp']]])
    
    return {'accuracy': data['accuracy'] * 100,
            'f1_score': data['f1_score'] * 100,
            'confusion_matrix': confusion_matrix,
            'confusion_matrix_normalized': confusion_matrix / confusion_matrix.sum() * 100}


def load_test_metrics_multiclass(model_dir: Path) -> Dict:
    """Load test metrics including 4x4 confusion matrix for multiclass classification."""
    test_file = model_dir / 'test' / 'test_metrics.json'
    
    with open(test_file, 'r') as f:
        data = json.load(f)
    
    # Extract 4x4 confusion matrix
    confusion_matrix = np.array(data['confusion_matrix'])
    
    return {'accuracy': data['accuracy'] * 100,
            'f1_score': data['f1_score'] * 100,
            'confusion_matrix': confusion_matrix,
            'confusion_matrix_normalized': confusion_matrix / confusion_matrix.sum() * 100}


def load_all_data():
    """Load all validation and test data for all models."""
    results_dir = Path(__file__).parent
    
    binary_validation_data = {}
    binary_test_data = {}
    multiclass_validation_data = {}
    multiclass_test_data = {}
    
    # Load binary models
    for model_name, model_dir_name in BINARY_MODELS.items():
        model_dir = results_dir / model_dir_name
        binary_validation_data[model_name] = load_validation_metrics(model_dir)
        binary_test_data[model_name] = load_test_metrics_binary(model_dir)
    
    # Load multiclass models
    for model_name, model_dir_name in MULTICLASS_MODELS.items():
        model_dir = results_dir / model_dir_name
        multiclass_validation_data[model_name] = load_validation_metrics(model_dir)
        multiclass_test_data[model_name] = load_test_metrics_multiclass(model_dir)
    
    return binary_validation_data, binary_test_data, multiclass_validation_data, multiclass_test_data


# =============================================================================
# Plotting Functions - Binary Classification
# =============================================================================

def plot_binary_validation_metrics_fig1(validation_data: Dict):
    """Figure 1: Binary - Accuracy and F1-Score during validation."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Accuracy
    ax = axes[0]
    for model_name in BINARY_MODELS.keys():
        data = validation_data[model_name]
        ax.plot(data['epochs'], data['accuracy'], 
                label=model_name, color=COLORS[model_name], 
                linewidth=2, marker='o', markersize=4)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Binary Classification - Validation Accuracy (Scratch Training)')
    ax.set_ylim(50, 100)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', framealpha=0.9)
    
    # Plot 2: F1-Score
    ax = axes[1]
    for model_name in BINARY_MODELS.keys():
        data = validation_data[model_name]
        ax.plot(data['epochs'], data['f1_score'], 
                label=model_name, color=COLORS[model_name], 
                linewidth=2, marker='o', markersize=4)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('F1-Score (%)')
    ax.set_title('Binary Classification - Validation F1-Score (Scratch Training)')
    ax.set_ylim(50, 100)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', framealpha=0.9)
    
    plt.tight_layout()
    
    # Save figure
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FIGURES_DIR / 'retraining_binary_EV_validation_accuracy_f1score.svg'
    plt.savefig(output_path, dpi=600, format='svg', bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_binary_validation_metrics_fig2(validation_data: Dict):
    """Figure 2: Binary - Precision and Recall during validation."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Precision
    ax = axes[0]
    for model_name in BINARY_MODELS.keys():
        data = validation_data[model_name]
        ax.plot(data['epochs'], data['precision'], 
                label=model_name, color=COLORS[model_name], 
                linewidth=2, marker='o', markersize=4)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Precision (%)')
    ax.set_title('Binary Classification - Validation Precision (Scratch Training)')
    ax.set_ylim(50, 100)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', framealpha=0.9)
    
    # Plot 2: Recall
    ax = axes[1]
    for model_name in BINARY_MODELS.keys():
        data = validation_data[model_name]
        ax.plot(data['epochs'], data['recall'], 
                label=model_name, color=COLORS[model_name], 
                linewidth=2, marker='o', markersize=4)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Recall (%)')
    ax.set_title('Binary Classification - Validation Recall (Scratch Training)')
    ax.set_ylim(50, 100)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', framealpha=0.9)
    
    plt.tight_layout()
    
    # Save figure
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FIGURES_DIR / 'retraining_binary_EV_validation_precision_recall.svg'
    plt.savefig(output_path, dpi=600, format='svg', bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_binary_confusion_matrices(test_data: Dict):
    """Figure 3: Binary - Confusion matrices for all 4 models (2 rows x 2 columns)."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Row 1: AS-EV_v1 models (CED, EPANNs)
    # Row 2: AS-EV_v2 models (CED, EPANNs)
    
    model_order = [['CED_AS-EV_v1', 'EPANNs_AS-EV_v1'],
                   ['CED_AS-EV_v2', 'EPANNs_AS-EV_v2']]
    
    for row_idx, row_models in enumerate(model_order):
        for col_idx, model_name in enumerate(row_models):
            ax = axes[row_idx, col_idx]
            data = test_data[model_name]
            
            # Plot normalized confusion matrix (percentages)
            cm_norm = data['confusion_matrix_normalized']
            cm_abs = data['confusion_matrix']
            
            im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=100, aspect='auto')
            
            # Add text annotations (absolute values only)
            for i in range(2):
                for j in range(2):
                    text_color = 'white' if cm_norm[i, j] > 50 else 'black'
                    ax.text(j, i, f'{int(cm_abs[i, j])}',
                           ha='center', va='center', color=text_color, 
                           fontsize=11, fontweight='bold')
            
            # Labels and title
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(['Negative', 'Positive'])
            ax.set_yticklabels(['Negative', 'Positive'])
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
            
            # Title with model name and metrics
            title = f"{model_name} (Scratch)\nAcc: {data['accuracy']:.2f}% | F1: {data['f1_score']:.2f}%"
            ax.set_title(title, fontsize=11)
            
            # Colorbar for each subplot
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Percentage (%)', fontsize=9)
    
    plt.suptitle('Binary Classification - Test Confusion Matrices (Scratch Training)', 
                 fontsize=14, y=0.995)
    plt.tight_layout()
    
    # Save figure
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FIGURES_DIR / 'retraining_binary_EV_confusion_matrices.svg'
    plt.savefig(output_path, dpi=600, format='svg', bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


# =============================================================================
# Plotting Functions - Multiclass Classification
# =============================================================================

def plot_multiclass_validation_metrics_fig1(validation_data: Dict):
    """Figure 4: Multiclass - Accuracy and F1-Score during validation."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Accuracy
    ax = axes[0]
    for model_name in MULTICLASS_MODELS.keys():
        data = validation_data[model_name]
        ax.plot(data['epochs'], data['accuracy'], 
                label=model_name, color=COLORS[model_name], 
                linewidth=2, marker='o', markersize=4)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Multiclass Classification - Validation Accuracy (Scratch Training)')
    ax.set_ylim(20, 80)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', framealpha=0.9)
    
    # Plot 2: F1-Score
    ax = axes[1]
    for model_name in MULTICLASS_MODELS.keys():
        data = validation_data[model_name]
        ax.plot(data['epochs'], data['f1_score'], 
                label=model_name, color=COLORS[model_name], 
                linewidth=2, marker='o', markersize=4)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('F1-Score (%)')
    ax.set_title('Multiclass Classification - Validation F1-Score (Scratch Training)')
    ax.set_ylim(20, 80)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', framealpha=0.9)
    
    plt.tight_layout()
    
    # Save figure
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FIGURES_DIR / 'retraining_multiclass_EV_validation_accuracy_f1score.svg'
    plt.savefig(output_path, dpi=600, format='svg', bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_multiclass_validation_metrics_fig2(validation_data: Dict):
    """Figure 5: Multiclass - Precision and Recall during validation."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Precision
    ax = axes[0]
    for model_name in MULTICLASS_MODELS.keys():
        data = validation_data[model_name]
        ax.plot(data['epochs'], data['precision'], 
                label=model_name, color=COLORS[model_name], 
                linewidth=2, marker='o', markersize=4)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Precision (%)')
    ax.set_title('Multiclass Classification - Validation Precision (Scratch Training)')
    ax.set_ylim(20, 80)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', framealpha=0.9)
    
    # Plot 2: Recall
    ax = axes[1]
    for model_name in MULTICLASS_MODELS.keys():
        data = validation_data[model_name]
        ax.plot(data['epochs'], data['recall'], 
                label=model_name, color=COLORS[model_name], 
                linewidth=2, marker='o', markersize=4)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Recall (%)')
    ax.set_title('Multiclass Classification - Validation Recall (Scratch Training)')
    ax.set_ylim(20, 80)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', framealpha=0.9)
    
    plt.tight_layout()
    
    # Save figure
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FIGURES_DIR / 'retraining_multiclass_EV_validation_precision_recall.svg'
    plt.savefig(output_path, dpi=600, format='svg', bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_multiclass_confusion_matrices(test_data: Dict):
    """Figure 6: Multiclass - 4x4 Confusion matrices for all 4 models (2 rows x 2 columns)."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Row 1: AS-EV_v1 models (CED, EPANNs)
    # Row 2: AS-EV_v2 models (CED, EPANNs)
    
    model_order = [['CED_AS-EV_v1', 'EPANNs_AS-EV_v1'],
                   ['CED_AS-EV_v2', 'EPANNs_AS-EV_v2']]
    
    for row_idx, row_models in enumerate(model_order):
        for col_idx, model_name in enumerate(row_models):
            ax = axes[row_idx, col_idx]
            data = test_data[model_name]
            
            # Plot normalized confusion matrix (for coloring)
            cm_norm = data['confusion_matrix_normalized']
            cm_abs = data['confusion_matrix']
            
            im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=100, aspect='auto')
            
            # Add text annotations (absolute values only)
            for i in range(4):
                for j in range(4):
                    # Determine text color based on background intensity
                    text_color = 'white' if cm_norm[i, j] > 50 else 'black'
                    ax.text(j, i, f'{int(cm_abs[i, j])}',
                           ha='center', va='center', color=text_color, 
                           fontsize=12, fontweight='bold')
            
            # Labels and title
            ax.set_xticks(range(4))
            ax.set_yticks(range(4))
            ax.set_xticklabels(CLASS_LABELS, rotation=45, ha='right')
            ax.set_yticklabels(CLASS_LABELS)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
            
            # Title with model name and metrics
            title = f"{model_name} (Scratch)\nAcc: {data['accuracy']:.2f}% | F1: {data['f1_score']:.2f}%"
            ax.set_title(title, fontsize=11)
            
            # Colorbar for each subplot
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Percentage (%)', fontsize=9)
    
    plt.suptitle('Multiclass Classification - Test Confusion Matrices (Scratch Training)', 
                 fontsize=14, y=0.995)
    plt.tight_layout()
    
    # Save figure
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FIGURES_DIR / 'retraining_multiclass_EV_confusion_matrices.svg'
    plt.savefig(output_path, dpi=600, format='svg', bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


# =============================================================================
# Main Execution
# =============================================================================

def main():
    """Main execution function."""
    print("\n" + "="*80)
    print("Re-training from Scratch - Emergency Vehicle Classification Analysis")
    print("="*80 + "\n")
    
    print("Loading data...")
    binary_val, binary_test, multiclass_val, multiclass_test = load_all_data()
    print(f"✓ Loaded binary validation data for {len(binary_val)} models")
    print(f"✓ Loaded binary test data for {len(binary_test)} models")
    print(f"✓ Loaded multiclass validation data for {len(multiclass_val)} models")
    print(f"✓ Loaded multiclass test data for {len(multiclass_test)} models\n")
    
    print("Generating figures for BINARY classification...")
    plot_binary_validation_metrics_fig1(binary_val)
    plot_binary_validation_metrics_fig2(binary_val)
    plot_binary_confusion_matrices(binary_test)
    
    print("\nGenerating figures for MULTICLASS classification...")
    plot_multiclass_validation_metrics_fig1(multiclass_val)
    plot_multiclass_validation_metrics_fig2(multiclass_val)
    plot_multiclass_confusion_matrices(multiclass_test)
    
    print("\n" + "="*80)
    print("Analysis complete! All figures saved to:", FIGURES_DIR)
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
