"""
Multiclass Emergency Vehicle Classification - Training Analysis
================================================================
Analyzes finetuning results for 6 models on AudioSet-EV datasets.

This script:
1. Loads validation metrics across epochs for 6 models
2. Loads test confusion matrices for 6 models
3. Generates 3 comprehensive figures (SVG 600 DPI)

Models analyzed:
- CED finetuned on AudioSet-EV v1 and v2
- CLAP finetuned on AudioSet-EV v1 and v2
- EPANNs finetuned on AudioSet-EV v1 and v2

Classification: 4-class (Traffic, Police, Ambulance, Fire)

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

# Model directories (relative to current script location)
MODELS = {'CED_AS-EV_v1': 'ced_finetune_fixedLR_AS-EV_v1',
          'CED_AS-EV_v2': 'ced_finetune_fixedLR_AS-EV_v2',
          'CLAP_AS-EV_v1': 'clap_finetune_fixedLR_AS-EV_v1',
          'CLAP_AS-EV_v2': 'clap_finetune_fixedLR_AS-EV_v2',
          'EPANNs_AS-EV_v1': 'epanns_finetune_fixedLR_AS-EV_v1',
          'EPANNs_AS-EV_v2': 'epanns_finetune_fixedLR_AS-EV_v2'}

# Color scheme: EPANNs (blue), CED (red), CLAP (orange/yellow)
# v1 = lighter, v2 = darker
COLORS = {'EPANNs_AS-EV_v1': '#87CEEB',  # skyblue
          'EPANNs_AS-EV_v2': '#1E3A8A',  # darkblue
          'CED_AS-EV_v1': '#F08080',     # lightcoral
          'CED_AS-EV_v2': '#8B0000',     # darkred
          'CLAP_AS-EV_v1': '#FFD700',    # gold
          'CLAP_AS-EV_v2': '#FF8C00'}    # darkorange

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


def load_test_metrics(model_dir: Path) -> Dict:
    """Load test metrics including 4x4 confusion matrix."""
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
    
    validation_data = {}
    test_data = {}
    
    for model_name, model_dir_name in MODELS.items():
        model_dir = results_dir / model_dir_name
        validation_data[model_name] = load_validation_metrics(model_dir)
        test_data[model_name] = load_test_metrics(model_dir)
    
    return validation_data, test_data


# =============================================================================
# Plotting Functions
# =============================================================================

def plot_validation_metrics_fig1(validation_data: Dict):
    """Figure 1: Accuracy and F1-Score during validation."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Accuracy
    ax = axes[0]
    for model_name in MODELS.keys():
        data = validation_data[model_name]
        ax.plot(data['epochs'], data['accuracy'], 
                label=model_name, color=COLORS[model_name], 
                linewidth=2, marker='o', markersize=4)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Validation Accuracy')
    ax.set_ylim(40, 80)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', framealpha=0.9)
    
    # Plot 2: F1-Score
    ax = axes[1]
    for model_name in MODELS.keys():
        data = validation_data[model_name]
        ax.plot(data['epochs'], data['f1_score'], 
                label=model_name, color=COLORS[model_name], 
                linewidth=2, marker='o', markersize=4)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('F1-Score (%)')
    ax.set_title('Validation F1-Score')
    ax.set_ylim(40, 80)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', framealpha=0.9)
    
    plt.tight_layout()
    
    # Save figure
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FIGURES_DIR / 'multiclass_EV_validation_accuracy_f1score.svg'
    plt.savefig(output_path, dpi=600, format='svg', bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_validation_metrics_fig2(validation_data: Dict):
    """Figure 2: Precision and Recall during validation."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Precision
    ax = axes[0]
    for model_name in MODELS.keys():
        data = validation_data[model_name]
        ax.plot(data['epochs'], data['precision'], 
                label=model_name, color=COLORS[model_name], 
                linewidth=2, marker='o', markersize=4)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Precision (%)')
    ax.set_title('Validation Precision')
    ax.set_ylim(40, 80)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', framealpha=0.9)
    
    # Plot 2: Recall
    ax = axes[1]
    for model_name in MODELS.keys():
        data = validation_data[model_name]
        ax.plot(data['epochs'], data['recall'], 
                label=model_name, color=COLORS[model_name], 
                linewidth=2, marker='o', markersize=4)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Recall (%)')
    ax.set_title('Validation Recall')
    ax.set_ylim(40, 80)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', framealpha=0.9)
    
    plt.tight_layout()
    
    # Save figure
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FIGURES_DIR / 'multiclass_EV_validation_precision_recall.svg'
    plt.savefig(output_path, dpi=600, format='svg', bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_confusion_matrices(test_data: Dict):
    """Figure 3: 4x4 Confusion matrices for all 6 models (2 rows x 3 columns)."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Row 1: AS-EV_v1 models (CED, CLAP, EPANNs)
    # Row 2: AS-EV_v2 models (CED, CLAP, EPANNs)
    
    model_order = [['CED_AS-EV_v1', 'CLAP_AS-EV_v1', 'EPANNs_AS-EV_v1'],
                   ['CED_AS-EV_v2', 'CLAP_AS-EV_v2', 'EPANNs_AS-EV_v2']]
    
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
            title = f"{model_name}\nAcc: {data['accuracy']:.2f}% | F1: {data['f1_score']:.2f}%"
            ax.set_title(title, fontsize=11)
            
            # Colorbar for each subplot
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Percentage (%)', fontsize=9)
    
    plt.tight_layout()
    
    # Save figure
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FIGURES_DIR / 'multiclass_EV_confusion_matrices.svg'
    plt.savefig(output_path, dpi=600, format='svg', bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


# =============================================================================
# Main Execution
# =============================================================================

def main():
    """Main execution function."""
    print("\n" + "="*70)
    print("Multiclass EV Classification - Training Analysis")
    print("="*70 + "\n")
    
    print("Loading data...")
    validation_data, test_data = load_all_data()
    print(f"✓ Loaded validation data for {len(validation_data)} models")
    print(f"✓ Loaded test data for {len(test_data)} models\n")
    
    print("Generating figures...")
    plot_validation_metrics_fig1(validation_data)
    plot_validation_metrics_fig2(validation_data)
    plot_confusion_matrices(test_data)
    
    print("\n" + "="*70)
    print("Analysis complete! All figures saved to:", FIGURES_DIR)
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
