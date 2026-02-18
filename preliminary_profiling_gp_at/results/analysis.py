"""
Preliminary GP-AT Models Profiling Analysis
============================================
Analyzes profiling results from 18 General-Purpose Audio Tagging models.

This script:
1. Loads all JSON profiling results
2. Extracts top 3 models for each category (CPU-focused)
3. Prints formatted results to terminal
4. Generates 6 comprehensive figures (SVG 600 DPI)

Categories analyzed:
- CPU Forward Time (fastest)
- Min Input Length (shortest)
- GFLOPs (lowest computational cost)
- Parameters (smallest model size)

Author: Stefano Giacomelli - Ph.D. candidate in ICT (DISIM dpt. - University of L'Aquila)
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Set high-quality plot parameters
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
# Architecture Classification
# =============================================================================

ARCHITECTURE_MAP = {
    'ast': 'Transformer',
    'audioclip': 'Hybrid',
    'audiomae': 'Transformer',
    'beats': 'Transformer',
    'ced': 'Transformer',
    'clap': 'Hybrid',
    'convnext': 'CNN',
    'efficientat_dymn': 'CNN',
    'efficientat_mn': 'CNN',
    'epanns': 'CNN',
    'htsat': 'Transformer',
    'm2d': 'Hybrid',
    'panns_resnet38': 'CNN',
    'panns_wavegram_logmel_cnn14': 'CNN',
    'passt': 'Transformer',
    'psla': 'Hybrid',
    'vggish': 'CNN',
    'yamnet': 'CNN'
}


# =============================================================================
# Data Loading
# =============================================================================

def load_all_results() -> Dict[str, Dict]:
    """Load all JSON profiling results from current directory."""
    results_dir = Path(__file__).parent
    results = {}
    
    for json_file in sorted(results_dir.glob('*_stats.json')):
        model_name = json_file.stem.replace('_stats', '')
        with open(json_file, 'r') as f:
            results[model_name] = json.load(f)
    
    return results


# =============================================================================
# Top 3 Extraction
# =============================================================================

def get_top3_cpu_forward(results: Dict) -> List[Tuple[str, Dict]]:
    """Get top 3 models with fastest CPU forward time."""
    sorted_models = sorted(results.items(),
                           key=lambda x: x[1]['cpu']['fwd_times_stats']['mean'])
    return sorted_models[:3]


def get_top3_min_input(results: Dict) -> List[Tuple[str, Dict]]:
    """Get top 3 models with shortest minimum input length."""
    # Filter out models with None min_length_seconds
    valid_models = {
        name: data for name, data in results.items()
        if data['input']['min_length_seconds'] is not None
    }
    sorted_models = sorted(valid_models.items(),
                           key=lambda x: x[1]['input']['min_length_seconds'])
    return sorted_models[:3]


def get_top3_gflops(results: Dict) -> List[Tuple[str, Dict]]:
    """Get top 3 models with lowest GFLOPs."""
    sorted_models = sorted(results.items(),
                           key=lambda x: x[1]['gflops'])
    return sorted_models[:3]


def get_top3_parameters(results: Dict) -> List[Tuple[str, Dict]]:
    """Get top 3 models with smallest parameter count."""
    sorted_models = sorted(results.items(),
                           key=lambda x: x[1]['parameters']['total_mb'])
    return sorted_models[:3]


# =============================================================================
# Terminal Output Formatting
# =============================================================================

def print_category_header(title: str):
    """Print formatted category header."""
    print("\n" + "═" * 80)
    print(f"TOP 3: {title.upper()}")
    print("═" * 80 + "\n")


def print_model_rank(rank: int, model_name: str, data: Dict, metric_name: str, metric_value: float, unit: str):
    """Print formatted model ranking."""
    medals = {1: "🥇", 2: "🥈", 3: "🥉"}
    
    cpu_stats = data['cpu']['fwd_times_stats']
    params_mb = data['parameters']['total_mb']
    gflops = data['gflops']
    throughput = data['cpu']['throughput_samples_per_sec']
    
    print(f"{medals[rank]} Rank {rank}: {model_name}")
    print(f"├─ {metric_name}: {metric_value:.4f}{unit}")
    print(f"├─ CPU Forward Time: mean={cpu_stats['mean']:.4f}s | min={cpu_stats['min']:.4f}s | max={cpu_stats['max']:.4f}s | std={cpu_stats['std']:.4f}s")
    print(f"├─ Throughput: {throughput:,.0f} samples/sec")
    print(f"└─ Parameters: {params_mb:.2f} MB | GFLOPs: {gflops:.2f}")
    print()


def print_terminal_results(results: Dict):
    """Print all top 3 results to terminal."""
    print("\n" + "█" * 80)
    print("PRELIMINARY GP-AT MODELS PROFILING ANALYSIS")
    print("█" * 80)
    print(f"\nTotal models analyzed: {len(results)}")
    print("Focus: CPU performance metrics")
    
    # Category 1: CPU Forward Time
    print_category_header("FASTEST CPU FORWARD TIME")
    for rank, (model_name, data) in enumerate(get_top3_cpu_forward(results), 1):
        metric_val = data['cpu']['fwd_times_stats']['mean']
        print_model_rank(rank, model_name, data, "CPU Forward Time (mean)", metric_val, "s")
    
    # Category 2: Min Input Length
    print_category_header("SHORTEST MINIMUM INPUT LENGTH")
    for rank, (model_name, data) in enumerate(get_top3_min_input(results), 1):
        metric_val = data['input']['min_length_seconds']
        print_model_rank(rank, model_name, data, "Min Input Length", metric_val, "s")
    
    # Category 3: GFLOPs
    print_category_header("LOWEST GFLOPS (COMPUTATIONAL COST)")
    for rank, (model_name, data) in enumerate(get_top3_gflops(results), 1):
        metric_val = data['gflops']
        print_model_rank(rank, model_name, data, "GFLOPs", metric_val, "")
    
    # Category 4: Parameters
    print_category_header("SMALLEST PARAMETERS (MODEL SIZE)")
    for rank, (model_name, data) in enumerate(get_top3_parameters(results), 1):
        metric_val = data['parameters']['total_mb']
        print_model_rank(rank, model_name, data, "Parameters", metric_val, " MB")
    
    print("═" * 80)
    print("Analysis complete! Generating figures...")
    print("═" * 80 + "\n")


# =============================================================================
# Figure 1: Radar Comparison (Single radar with all unique top 3 models)
# =============================================================================

def plot_radar_comparison(results: Dict, output_dir: Path):
    """Generate single radar chart with all unique models from top 3 categories."""
    categories = {
        'CPU Time': get_top3_cpu_forward(results),
        'Min Input': get_top3_min_input(results),
        'GFLOPs': get_top3_gflops(results),
        'Parameters': get_top3_parameters(results)
    }
    
    # Collect all unique models and track their appearances
    model_appearances = {}  # {model_name: [(category, rank), ...]}
    
    for cat_name, top3 in categories.items():
        for rank, (model_name, data) in enumerate(top3, 1):
            if model_name not in model_appearances:
                model_appearances[model_name] = []
            model_appearances[model_name].append((cat_name, rank))
    
    # Create single radar plot
    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(projection='polar'))
    
    # Metrics for radar (all need to be inverted for "higher is better")
    metric_names = ['CPU Speed', 'Efficiency', 'Compactness', 'Input Flexibility']
    
    angles = np.linspace(0, 2 * np.pi, len(metric_names), endpoint=False).tolist()
    angles += angles[:1]  # Close the plot
    
    # Color palette (distinct colors for each model)
    colors = plt.cm.tab20(np.linspace(0, 1, len(model_appearances)))
    
    # Get max values from all results for normalization
    max_cpu = max(r['cpu']['fwd_times_stats']['mean'] for r in results.values())
    max_gflops = max(r['gflops'] for r in results.values())
    max_params = max(r['parameters']['total_mb'] for r in results.values())
    valid_inputs = [r['input']['min_length_seconds'] for r in results.values() 
                   if r['input']['min_length_seconds'] is not None]
    max_input = max(valid_inputs) if valid_inputs else 1.0
    
    # Plot each unique model
    legend_labels = []
    for idx, (model_name, appearances) in enumerate(sorted(model_appearances.items())):
        data = results[model_name]
        
        # Get metrics
        cpu_time = data['cpu']['fwd_times_stats']['mean']
        gflops = data['gflops']
        params_mb = data['parameters']['total_mb']
        min_input = data['input']['min_length_seconds']
        
        if min_input is None:
            min_input = max_input  # Worst case for normalization
        
        # Normalize (invert so higher is better)
        values = [
            1 - (cpu_time / max_cpu),  # CPU Speed
            1 - (gflops / max_gflops),  # Efficiency
            1 - (params_mb / max_params),  # Compactness
            1 - (min_input / max_input)  # Input Flexibility
        ]
        values += values[:1]
        
        # Plot line
        ax.plot(angles, values, 'o-', linewidth=2.5, color=colors[idx], label=model_name)
        ax.fill(angles, values, alpha=0.15, color=colors[idx])
        
        # Build legend label with category info
        rank_map = {1: '1st', 2: '2nd', 3: '3rd'}
        appearances_str = ', '.join([f"Top3 {cat}: {rank_map[rank]}" 
                                     for cat, rank in sorted(appearances)])
        legend_labels.append(f"{model_name} - {appearances_str}")
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_names, size=11, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.set_title('Radar Comparison - All Top 3 Models\\n(Metrics normalized: higher is better)', 
                 size=14, fontweight='bold', pad=25)
    ax.grid(True, alpha=0.2)
    
    # Legend with detailed info (outside plot area)
    ax.legend(legend_labels, loc='upper left', bbox_to_anchor=(1.15, 1.1), 
             fontsize=9, frameon=True, fancybox=True, shadow=True)
    
    plt.tight_layout()
    
    output_path = output_dir / 'profiling_radar_comparison.svg'
    plt.savefig(output_path, format='svg', dpi=600, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path.name}")


# =============================================================================
# Figure 2: Performance Matrix Heatmap (18 models × 4 metrics)
# =============================================================================

def plot_performance_heatmap(results: Dict, output_dir: Path):
    """Generate heatmap showing all models vs 4 metrics."""
    models = sorted(results.keys())
    metrics = ['CPU Time\n(mean, s)', 'Parameters\n(MB)', 'GFLOPs', 'Min Input\n(s)']
    
    # Build matrix
    data_matrix = []
    for model in models:
        data = results[model]
        min_input = data['input']['min_length_seconds']
        if min_input is None:
            min_input = 10.0  # Default to 10s for models with no min input length
        row = [
            data['cpu']['fwd_times_stats']['mean'],
            data['parameters']['total_mb'],
            data['gflops'],
            min_input
        ]
        data_matrix.append(row)
    
    data_matrix = np.array(data_matrix)
    
    # Normalize each column to 0-1 for color mapping
    data_normalized = np.zeros_like(data_matrix)
    for col in range(data_matrix.shape[1]):
        col_data = data_matrix[:, col]
        data_normalized[:, col] = (col_data - col_data.min()) / (col_data.max() - col_data.min())
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 14))
    
    im = ax.imshow(data_normalized, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=1)
    
    # Set ticks
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_yticks(np.arange(len(models)))
    ax.set_xticklabels(metrics, fontsize=11, fontweight='bold')
    ax.set_yticklabels(models, fontsize=9)
    
    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations with actual values
    for i in range(len(models)):
        for j in range(len(metrics)):
            val = data_matrix[i, j]
            text = ax.text(j, i, f'{val:.3f}' if val < 10 else f'{val:.1f}',
                          ha="center", va="center", color="black", fontsize=9, fontweight='bold')
    
    ax.set_title('Performance Matrix - All Models vs Metrics\n(Green=Best, Red=Worst)', 
                 fontsize=14, fontweight='bold', pad=15)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Normalized Score', rotation=270, labelpad=20, fontsize=10)
    
    plt.tight_layout()
    
    output_path = output_dir / 'profiling_performance_heatmap.svg'
    plt.savefig(output_path, format='svg', dpi=600, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path.name}")


# =============================================================================
# Figure 3: Pareto Front - CPU Speed vs Size
# =============================================================================

def plot_pareto_front(results: Dict, output_dir: Path):
    """Generate Pareto front scatter plot (CPU time vs Parameters)."""
    models = list(results.keys())
    cpu_times = [results[m]['cpu']['fwd_times_stats']['mean'] for m in models]
    params_mb = [results[m]['parameters']['total_mb'] for m in models]
    gflops = [results[m]['gflops'] for m in models]
    
    # Classify by architecture
    colors_map = {'CNN': '#3498db', 'Transformer': '#e74c3c', 'Hybrid': '#f39c12'}
    colors = [colors_map[ARCHITECTURE_MAP[m]] for m in models]
    
    # Calculate Pareto front
    points = list(zip(params_mb, cpu_times))
    pareto_indices = []
    for i, (x1, y1) in enumerate(points):
        is_pareto = True
        for j, (x2, y2) in enumerate(points):
            if i != j and x2 <= x1 and y2 <= y1 and (x2 < x1 or y2 < y1):
                is_pareto = False
                break
        if is_pareto:
            pareto_indices.append(i)
    
    # Sort Pareto points by x for line plotting
    pareto_points = sorted([(params_mb[i], cpu_times[i]) for i in pareto_indices])
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Scatter plot
    scatter = ax.scatter(params_mb, cpu_times, s=[g*2 for g in gflops], 
                        c=colors, alpha=0.6, edgecolors='black', linewidth=1)
    
    # Pareto front line
    if len(pareto_points) > 1:
        pareto_x, pareto_y = zip(*pareto_points)
        ax.plot(pareto_x, pareto_y, 'k--', alpha=0.5, linewidth=2, label='Pareto Front')
    
    # Label only Pareto front models
    pareto_models = [models[i] for i in pareto_indices]
    for model_name in pareto_models:
        idx = models.index(model_name)
        ax.annotate(model_name, (params_mb[idx], cpu_times[idx]), 
                   xytext=(5, 5), textcoords='offset points', 
                   fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Parameters (MB)', fontsize=12, fontweight='bold')
    ax.set_ylabel('CPU Forward Time (s)', fontsize=12, fontweight='bold')
    ax.set_title('Pareto Front: CPU Speed vs Model Size\n(Point size = GFLOPs)', 
                 fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, which='both')
    
    # Legend
    legend_elements = [mpatches.Patch(facecolor=c, label=a, edgecolor='black') 
                      for a, c in colors_map.items()]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
    
    plt.tight_layout()
    
    output_path = output_dir / 'profiling_pareto_speed_size.svg'
    plt.savefig(output_path, format='svg', dpi=600, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path.name}")


# =============================================================================
# Figure 4: Top 3 Categories Bar Charts (4 subplots)
# =============================================================================

def plot_top3_categories(results: Dict, output_dir: Path):
    """Generate bar charts showing top 3 for each of 4 categories."""
    categories = {
        'CPU Forward Time (s)': (get_top3_cpu_forward(results), lambda d: d['cpu']['fwd_times_stats']['mean']),
        'Min Input Length (s)': (get_top3_min_input(results), lambda d: d['input']['min_length_seconds']),
        'GFLOPs': (get_top3_gflops(results), lambda d: d['gflops']),
        'Parameters (MB)': (get_top3_parameters(results), lambda d: d['parameters']['total_mb'])
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()
    
    colors = ['#FFD700', '#C0C0C0', '#CD7F32']  # Gold, Silver, Bronze
    
    for idx, (cat_name, (top3, value_fn)) in enumerate(categories.items()):
        ax = axes[idx]
        
        models = [name for name, _ in top3]
        values = [value_fn(data) for _, data in top3]
        
        bars = ax.barh(range(len(models)), values, color=colors, edgecolor='black', linewidth=1.5)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, values)):
            ax.text(val, i, f'  {val:.3f}', va='center', fontsize=10, fontweight='bold')
        
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(models, fontsize=11)
        ax.set_xlabel(cat_name, fontsize=11, fontweight='bold')
        ax.set_title(f'Top 3: {cat_name}', fontsize=12, fontweight='bold', pad=10)
        ax.grid(axis='x', alpha=0.3)
        ax.invert_yaxis()
    
    fig.suptitle('Top 3 Models per Category', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    output_path = output_dir / 'profiling_top3_categories.svg'
    plt.savefig(output_path, format='svg', dpi=600, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path.name}")


# =============================================================================
# Figure 5: Architecture Distribution Violin Plot
# =============================================================================

def plot_architecture_distribution(results: Dict, output_dir: Path):
    """Generate violin plots showing metric distributions by architecture."""
    # Organize data by architecture
    arch_data = defaultdict(lambda: {'cpu_time': [], 'params': [], 'gflops': [], 'models': []})
    
    for model_name, data in results.items():
        arch = ARCHITECTURE_MAP[model_name]
        arch_data[arch]['cpu_time'].append(data['cpu']['fwd_times_stats']['mean'])
        arch_data[arch]['params'].append(data['parameters']['total_mb'])
        arch_data[arch]['gflops'].append(data['gflops'])
        arch_data[arch]['models'].append(model_name)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    metrics = [
        ('cpu_time', 'CPU Forward Time (s)', axes[0]),
        ('params', 'Parameters (MB)', axes[1]),
        ('gflops', 'GFLOPs', axes[2])
    ]
    
    colors_map = {'CNN': '#3498db', 'Transformer': '#e74c3c', 'Hybrid': '#f39c12'}
    
    for metric_key, metric_label, ax in metrics:
        # Prepare data for violin plot
        data_for_plot = []
        labels = []
        colors = []
        
        for arch in ['CNN', 'Transformer', 'Hybrid']:
            if arch in arch_data:
                data_for_plot.append(arch_data[arch][metric_key])
                labels.append(arch)
                colors.append(colors_map[arch])
        
        # Violin plot
        parts = ax.violinplot(data_for_plot, positions=range(len(labels)), 
                             showmeans=True, showmedians=True)
        
        # Color the violins
        for pc, color in zip(parts['bodies'], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.6)
            pc.set_edgecolor('black')
            pc.set_linewidth(1.5)
        
        # Add individual points with model names
        for i, arch in enumerate(labels):
            values = arch_data[arch][metric_key]
            models = arch_data[arch]['models']
            
            # Scatter points
            y_positions = values
            x_positions = [i + np.random.uniform(-0.1, 0.1) for _ in values]
            ax.scatter(x_positions, y_positions, alpha=0.7, s=50, 
                      color=colors[i], edgecolors='black', linewidths=1, zorder=3)
            
            # Add model names as text (bold font)
            for x, y, m in zip(x_positions, y_positions, models):
                ax.text(x, y, m, fontsize=7, fontweight='bold', ha='center', va='bottom', rotation=0, alpha=0.8)
        
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=12, fontweight='bold')
        ax.set_ylabel(metric_label, fontsize=12, fontweight='bold')
        ax.set_title(f'{metric_label} by Architecture', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
    
    fig.suptitle('Metric Distributions by Architecture Type', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    output_path = output_dir / 'profiling_architecture_distribution.svg'
    plt.savefig(output_path, format='svg', dpi=600, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path.name}")


# =============================================================================
# Figure 6: CPU Throughput Comparison
# =============================================================================

def plot_throughput_comparison(results: Dict, output_dir: Path):
    """Generate bar chart comparing CPU throughput across all models."""
    models = sorted(results.keys(), key=lambda m: results[m]['cpu']['throughput_samples_per_sec'], reverse=True)
    throughputs = [results[m]['cpu']['throughput_samples_per_sec'] for m in models]
    
    # Color by architecture
    colors_map = {'CNN': '#3498db', 'Transformer': '#e74c3c', 'Hybrid': '#f39c12'}
    colors = [colors_map[ARCHITECTURE_MAP[m]] for m in models]
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    bars = ax.barh(range(len(models)), throughputs, color=colors, edgecolor='black', linewidth=1)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, throughputs)):
        ax.text(val, i, f'  {val:,.0f}', va='center', fontsize=8, fontweight='bold')
    
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=9)
    ax.set_xlabel('Throughput (samples/sec)', fontsize=12, fontweight='bold')
    ax.set_title('CPU Throughput Comparison - All Models\n(Higher is Better)', 
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xscale('log')
    ax.grid(axis='x', alpha=0.3, which='both')
    ax.invert_yaxis()
    
    # Legend
    legend_elements = [mpatches.Patch(facecolor=c, label=a, edgecolor='black') 
                      for a, c in colors_map.items()]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    plt.tight_layout()
    
    output_path = output_dir / 'profiling_throughput_comparison.svg'
    plt.savefig(output_path, format='svg', dpi=600, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path.name}")


# =============================================================================
# Main Execution
# =============================================================================

def main():
    """Main execution function."""
    print("\n" + "═" * 80)
    print("PRELIMINARY GP-AT MODELS PROFILING ANALYSIS")
    print("\n" + "═" * 80)
    print("\nLoading profiling results...")
    
    # Load all results
    results = load_all_results()
    print(f"✓ Loaded {len(results)} model profiles\n")
    
    # Print terminal results
    print_terminal_results(results)
    
    # Generate figures
    output_dir = Path(__file__).parent
    
    print("\nGenerating figures (SVG 600 DPI)...")
    print("─" * 80)
    
    plot_radar_comparison(results, output_dir)
    plot_performance_heatmap(results, output_dir)
    plot_pareto_front(results, output_dir)
    # plot_top3_categories(results, output_dir)  # Removed - replaced by unified radar
    plot_architecture_distribution(results, output_dir)
    plot_throughput_comparison(results, output_dir)
    
    print("─" * 80)
    print(f"\n✓ All figures saved to: {output_dir}")
    print("\n" + "═" * 80)
    print("ANALYSIS COMPLETE!")
    print("═" * 80 + "\n")


if __name__ == '__main__':
    main()
