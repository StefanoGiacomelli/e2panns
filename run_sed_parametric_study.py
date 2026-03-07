#!/usr/bin/env python3
"""Parametric Study for SED System
===================================
Test different combinations of collar size, adaptive factor, and detection threshold 
on 50 random samples from AudioSet_EV_Strong v2 dataset.

Parameters tested:
- Detection threshold: 0.5, 0.7
- Collar size (event tolerance): 0.1s, 0.5s, 1.0s, 2.0s
- Adaptive factor (adapt_coeff): 0.5, 0.75, 1.0

Total: 24 combinations (2x4x3)

Author: Stefano Giacomelli - Ph.D. candidate in ICT (DISIM dpt. - University of L'Aquila)
"""

import sys
import json
import yaml
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
PROJECT_ROOT = Path(__file__).parent
OUTPUT_DIR = PROJECT_ROOT / "results" / "sed_parametric_study"
CONFIG_DIR = OUTPUT_DIR / "configs" 
RESULTS_DIR = OUTPUT_DIR / "results"
PLOTS_DIR = OUTPUT_DIR / "plots"

# Parameters to test
COLLAR_SIZES = [0.1, 0.5, 1.0, 2.0]  # seconds
ADAPTIVE_FACTORS = [0.5, 0.75, 1.0]  # coefficient
THRESHOLDS = [0.5, 0.7]  # detection threshold
MODEL_NAME = "epanns"
CHECKPOINT_PATH = 'checkpoints/binary_EV/epanns_finetune_fixedLR_AS-EV_v2/epoch=002_val_f1=0.9625.pt'

# Number of samples
NUM_SAMPLES = 50

# Base configuration template
BASE_CONFIG = {
    'dataset': {
        'name': 'AudioSet_EV_v2',
        'mode': 'detection',
        'max_samples': NUM_SAMPLES
    },
    'model': {
        'name': MODEL_NAME,
        'checkpoint': CHECKPOINT_PATH,
        'device': 'cpu'
    },
    'inference': {
        'threshold': 0.5,
        'chunk_duration': 0.310,
        'buffer_duration': 20.0,
        'adaptive_window': {
            'enabled': True,
            'frame_duration_max': 1.0,
            'adapt_coeff': 0.5  # Will be varied
        }
    },
    'sed_metrics': {
        'segment_time_resolution': 0.310,
        'event_tolerance': 0.5  # Will be varied
    },
    'multiprocessing': {
        'enabled': False,
        'num_workers': 0
    },
    'output': {
        'dir': 'results/sed_parametric_study/results',
        'save_per_sample': True,
        'save_events': False
    },
    'performance': {
        'progress_bar': True
    },
    'logging': {
        'level': 'INFO',
        'log_file': 'run.log'
    }
}


def create_config(collar: float, adaptive: float, threshold: float) -> Path:
    """Create a config file for a specific combination."""
    config = BASE_CONFIG.copy()
    
    # Update parameters
    config['sed_metrics']['event_tolerance'] = collar
    config['inference']['adaptive_window']['adapt_coeff'] = adaptive
    config['inference']['threshold'] = threshold
    
    # Create unique identifier
    config_name = f"thr_{threshold:.1f}_collar_{collar:.1f}s_adaptive_{adaptive:.2f}.yaml"
    config_path = CONFIG_DIR / config_name
    
    # Update output directory to include parameters
    result_subdir = f"thr_{threshold:.1f}_collar_{collar:.1f}s_adaptive_{adaptive:.2f}"
    config['output']['dir'] = str(RESULTS_DIR / result_subdir)
    
    # Save config
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    return config_path


def run_sed_evaluation(config_path: Path) -> dict:
    """Run main_sed_dataset.py with the given config."""
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "main_sed_dataset.py"),
        str(config_path)
    ]
    
    print(f"\n{'='*80}")
    print(f"Running: {config_path.name}")
    print(f"{'='*80}")
    
    try:
        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=False,
            text=True
        )
        
        # Load results from summary.json
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        
        summary_path = Path(cfg['output']['dir']) / 'summary.json'
        
        if summary_path.exists():
            with open(summary_path, 'r') as f:
                summary = json.load(f)
            
            return {
                'success': True,
                'summary': summary,
                'config_path': str(config_path)
            }
        else:
            print(f"⚠️  Warning: summary.json not found at {summary_path}")
            return {'success': False, 'error': 'Summary file not found'}
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running {config_path.name}: {e}")
        return {'success': False, 'error': str(e)}


def collect_results(all_results: list) -> pd.DataFrame:
    """Collect all results into a DataFrame."""
    data = []
    
    for result in all_results:
        if not result['success']:
            continue
        
        summary = result['summary']
        metrics = summary.get('aggregated_metrics', {})
        
        # Extract config parameters from path
        config_name = Path(result['config_path']).stem
        parts = config_name.split('_')
        threshold = float(parts[1])
        collar = float(parts[3].replace('s', ''))
        adaptive = float(parts[5])
        
        # Segment-based metrics
        seg = metrics.get('segment_based', {})
        
        # Event-based metrics
        evt = metrics.get('event_based', {})
        
        # Performance
        perf = metrics.get('performance', {})
        
        data.append({
            'threshold': threshold,
            'collar_size_s': collar,
            'adaptive_factor': adaptive,
            # Segment-based
            'seg_precision': seg.get('precision', np.nan),
            'seg_recall': seg.get('recall', np.nan),
            'seg_f1': seg.get('f1', np.nan),
            'seg_accuracy': seg.get('accuracy', np.nan),
            'seg_balanced_accuracy': seg.get('balanced_accuracy', np.nan),
            'seg_error_rate': seg.get('error_rate', np.nan),
            'seg_std_precision': seg.get('std_precision', np.nan),
            'seg_std_recall': seg.get('std_recall', np.nan),
            'seg_std_f1': seg.get('std_f1', np.nan),
            # Event-based
            'evt_precision': evt.get('precision', np.nan),
            'evt_recall': evt.get('recall', np.nan),
            'evt_f1': evt.get('f1', np.nan),
            'evt_error_rate': evt.get('error_rate', np.nan),
            'evt_std_precision': evt.get('std_precision', np.nan),
            'evt_std_recall': evt.get('std_recall', np.nan),
            'evt_std_f1': evt.get('std_f1', np.nan),
            # Performance
            'throughput': perf.get('mean_throughput', np.nan),
            'cpu_percent': perf.get('mean_cpu', np.nan),
            'ram_mb': perf.get('mean_ram_mb', np.nan),
        })
    
    return pd.DataFrame(data)


def plot_results(df: pd.DataFrame):
    """Generate comprehensive plots of the results."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 150
    
    # Metrics to plot
    seg_metrics = ['seg_precision', 'seg_recall', 'seg_f1', 'seg_accuracy']
    evt_metrics = ['evt_precision', 'evt_recall', 'evt_f1']
    
    # ===========================================================================
    # 1. Heatmaps: Metrics vs Collar Size and Adaptive Factor (for each Threshold)
    # ===========================================================================
    
    n_thresholds = len(THRESHOLDS)
    n_cols = min(2, n_thresholds)
    n_rows = (n_thresholds + n_cols - 1) // n_cols  # Ceiling division
    
    for metric in seg_metrics + evt_metrics:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(9*n_cols, 7*n_rows))
        if n_thresholds == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for idx, thr in enumerate(THRESHOLDS):
            ax = axes[idx]
            
            # Filter data for this threshold
            df_thr = df[df['threshold'] == thr]
            
            # Pivot for heatmap
            pivot = df_thr.pivot(
                index='collar_size_s',
                columns='adaptive_factor',
                values=metric
            )
            
            # Plot heatmap
            sns.heatmap(
                pivot,
                annot=True,
                fmt='.3f',
                cmap='RdYlGn',
                vmin=0,
                vmax=1,
                cbar_kws={'label': metric.replace('_', ' ').title()},
                ax=ax
            )
            
            ax.set_title(f'Threshold = {thr:.1f}', fontsize=12, pad=10)
            ax.set_xlabel('Adaptive Factor', fontsize=10)
            ax.set_ylabel('Collar Size (s)', fontsize=10)
        
        # Hide unused subplots
        for idx in range(n_thresholds, len(axes)):
            axes[idx].axis('off')
        
        fig.suptitle(f'{metric.replace("_", " ").title()} vs Parameters', fontsize=16, y=0.995)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f'heatmap_{metric}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"✅ Saved heatmaps to {PLOTS_DIR}")
    
    # ===========================================================================
    # 2. Line Plots: Metrics vs Collar Size (for each Adaptive Factor and Threshold)
    # ===========================================================================
    
    # Segment-based metrics - one figure per threshold
    for thr in THRESHOLDS:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        df_thr = df[df['threshold'] == thr]
        
        for idx, metric in enumerate(seg_metrics):
            ax = axes[idx]
            
            for adapt in ADAPTIVE_FACTORS:
                subset = df_thr[df_thr['adaptive_factor'] == adapt].sort_values('collar_size_s')
                
                ax.plot(
                    subset['collar_size_s'],
                    subset[metric],
                    marker='o',
                    label=f'Adaptive={adapt:.2f}',
                    linewidth=2,
                    markersize=8
                )
            
            ax.set_xlabel('Collar Size (s)', fontsize=11)
            ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=11)
            ax.set_title(f'{metric.replace("_", " ").title()} vs Collar Size', fontsize=12, pad=10)
            ax.legend(loc='best', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1.05])
        
        fig.suptitle(f'Segment-Based Metrics vs Collar Size (Threshold={thr:.1f})', fontsize=16, y=0.995)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f'segment_metrics_vs_collar_thr{thr:.1f}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"✅ Saved segment metrics vs collar plots")
    
    # Event-based metrics - one figure per threshold
    for thr in THRESHOLDS:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        df_thr = df[df['threshold'] == thr]
        
        for idx, metric in enumerate(evt_metrics):
            ax = axes[idx]
            
            for adapt in ADAPTIVE_FACTORS:
                subset = df_thr[df_thr['adaptive_factor'] == adapt].sort_values('collar_size_s')
            
                ax.plot(
                    subset['collar_size_s'],
                    subset[metric],
                    marker='o',
                    label=f'Adaptive={adapt:.2f}',
                    linewidth=2,
                    markersize=8
                )
            
            ax.set_xlabel('Collar Size (s)', fontsize=11)
            ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=11)
            ax.set_title(f'{metric.replace("_", " ").title()} vs Collar Size', fontsize=12, pad=10)
            ax.legend(loc='best', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1.05])
        
        fig.suptitle(f'Event-Based Metrics vs Collar Size (Threshold={thr:.1f})', fontsize=16, y=1.02)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f'event_metrics_vs_collar_thr{thr:.1f}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"✅ Saved event metrics vs collar plots")
    
    # ===========================================================================
    # 3. Line Plots: Metrics vs Adaptive Factor (for each Collar Size and Threshold)
    # ===========================================================================
    
    for thr in THRESHOLDS:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        df_thr = df[df['threshold'] == thr]
        
        for idx, metric in enumerate(seg_metrics):
            ax = axes[idx]
            
            for collar in COLLAR_SIZES:
                subset = df_thr[df_thr['collar_size_s'] == collar].sort_values('adaptive_factor')
            
            ax.plot(
                subset['adaptive_factor'],
                subset[metric],
                marker='s',
                label=f'Collar={collar:.1f}s',
                linewidth=2,
                markersize=8
            )
        
            ax.set_xlabel('Adaptive Factor', fontsize=11)
            ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=11)
            ax.set_title(f'{metric.replace("_", " ").title()} vs Adaptive Factor', fontsize=12, pad=10)
            ax.legend(loc='best', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1.05])
        
        fig.suptitle(f'Segment-Based Metrics vs Adaptive Factor (Threshold={thr:.1f})', fontsize=16, y=0.995)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f'segment_metrics_vs_adaptive_thr{thr:.1f}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"✅ Saved segment metrics vs adaptive plots")
    
    # ===========================================================================
    # 4. Performance Metrics (using threshold=0.5 as baseline)
    # ===========================================================================
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    df_perf = df[df['threshold'] == 0.5]  # Use baseline threshold
    
    # Throughput vs Adaptive Factor
    ax = axes[0]
    for collar in COLLAR_SIZES:
        subset = df_perf[df_perf['collar_size_s'] == collar].sort_values('adaptive_factor')
        ax.plot(
            subset['adaptive_factor'],
            subset['throughput'],
            marker='o',
            label=f'Collar={collar:.1f}s',
            linewidth=2
        )
    ax.set_xlabel('Adaptive Factor', fontsize=11)
    ax.set_ylabel('Throughput (x real-time)', fontsize=11)
    ax.set_title('Throughput vs Adaptive Factor (Threshold=0.5)', fontsize=12, pad=10)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # CPU usage vs Adaptive Factor
    ax = axes[1]
    for collar in COLLAR_SIZES:
        subset = df_perf[df_perf['collar_size_s'] == collar].sort_values('adaptive_factor')
        ax.plot(
            subset['adaptive_factor'],
            subset['cpu_percent'],
            marker='s',
            label=f'Collar={collar:.1f}s',
            linewidth=2
        )
    ax.set_xlabel('Adaptive Factor', fontsize=11)
    ax.set_ylabel('CPU Usage (%)', fontsize=11)
    ax.set_title('CPU Usage vs Adaptive Factor (Threshold=0.5)', fontsize=12, pad=10)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'performance_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved performance metrics plot")
    
    # ===========================================================================
    # 5. Summary Statistics Table
    # ===========================================================================
    
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.axis('tight')
    ax.axis('off')
    
    # Create summary table
    table_data = []
    table_data.append(['Threshold', 'Collar (s)', 'Adaptive', 'Seg F1', 'Seg Acc', 'Evt F1', 'Throughput'])
    
    for _, row in df.iterrows():
        table_data.append([
            f"{row['threshold']:.1f}",
            f"{row['collar_size_s']:.1f}",
            f"{row['adaptive_factor']:.2f}",
            f"{row['seg_f1']:.3f}",
            f"{row['seg_accuracy']:.3f}",
            f"{row['evt_f1']:.3f}",
            f"{row['throughput']:.2f}x"
        ])
    
    table = ax.table(
        cellText=table_data,
        cellLoc='center',
        loc='center',
        colWidths=[0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.12]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.8)
    
    # Style header
    for i in range(7):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.title('Parametric Study Results Summary', fontsize=14, pad=20, weight='bold')
    plt.savefig(PLOTS_DIR / 'summary_table.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved summary table")


def main():
    """Main execution."""
    print("="*80)
    print("PARAMETRIC STUDY: SED System")
    print("="*80)
    print(f"Dataset: AudioSet_EV_Strong v2 (50 random samples)")
    print(f"Thresholds: {THRESHOLDS}")
    print(f"Collar sizes: {COLLAR_SIZES}")
    print(f"Adaptive factors: {ADAPTIVE_FACTORS}")
    print(f"Total combinations: {len(THRESHOLDS) * len(COLLAR_SIZES) * len(ADAPTIVE_FACTORS)}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("="*80)
    
    # Create output directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Generate all configurations
    print("\n📝 Generating configuration files...")
    configs = []
    for thr in THRESHOLDS:
        for collar in COLLAR_SIZES:
            for adaptive in ADAPTIVE_FACTORS:
                config_path = create_config(collar, adaptive, thr)
                configs.append((thr, collar, adaptive, config_path))
                print(f"   ✅ {config_path.name}")
    
    print(f"\n✅ Generated {len(configs)} configuration files")
    
    # Run all evaluations
    print("\n🚀 Running SED evaluations...")
    all_results = []
    
    for i, (thr, collar, adaptive, config_path) in enumerate(configs, 1):
        print(f"\n[{i}/{len(configs)}] Threshold={thr:.1f}, Collar={collar:.1f}s, Adaptive={adaptive:.2f}")
        result = run_sed_evaluation(config_path)
        result['threshold'] = thr
        result['collar'] = collar
        result['adaptive'] = adaptive
        all_results.append(result)
    
    # Collect results
    print("\n📊 Collecting results...")
    df = collect_results(all_results)
    
    # Save results DataFrame
    results_csv = OUTPUT_DIR / 'parametric_study_results.csv'
    df.to_csv(results_csv, index=False)
    print(f"✅ Saved results to {results_csv}")
    
    # Generate plots
    print("\n📈 Generating plots...")
    plot_results(df)
    
    # Print summary
    print("\n" + "="*80)
    print("PARAMETRIC STUDY COMPLETED")
    print("="*80)
    print(f"Results CSV: {results_csv}")
    print(f"Plots directory: {PLOTS_DIR}")
    print(f"Total runs: {len(all_results)}")
    print(f"Successful runs: {sum(1 for r in all_results if r['success'])}")
    print(f"Failed runs: {sum(1 for r in all_results if not r['success'])}")
    
    # Best configurations
    print("\n🏆 Best Configurations:")
    print(f"\nBest Segment F1: {df.loc[df['seg_f1'].idxmax(), ['collar_size_s', 'adaptive_factor', 'seg_f1']].to_dict()}")
    print(f"Best Event F1: {df.loc[df['evt_f1'].idxmax(), ['collar_size_s', 'adaptive_factor', 'evt_f1']].to_dict()}")
    print(f"Best Accuracy: {df.loc[df['seg_accuracy'].idxmax(), ['collar_size_s', 'adaptive_factor', 'seg_accuracy']].to_dict()}")
    
    print("\n✅ Done!\n")


if __name__ == "__main__":
    main()
