"""
KineScaper_EV Dataset Analysis
===============================
Comprehensive analysis of the KineScaper Emergency Vehicles synthetic dataset.

Author: Stefano Giacomelli - Ph.D. candidate in ICT (DISIM dpt. - University of L'Aquila)
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
from pathlib import Path
from scipy.interpolate import griddata, interp1d
from scipy.ndimage import gaussian_filter1d

# Add parent directory for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Import new dataloader for dynamic negative pool calculation
from datasets.KineScaper_EV.dataloader import KineScaper_NegativeChunkGenerator

# Configuration
SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "analysis_results"
OUTPUT_DIR.mkdir(exist_ok=True)

DATASET_ROOT = "/mnt/ssd/Kinescaper_EV/dataset/"
METADATA_JSON = os.path.join(DATASET_ROOT, "json", "metadata.json")

SEED = 42
np.random.seed(SEED)

# Constants
CHUNK_DURATION = 10.0  # seconds
MIN_OVERLAP = 0.5  # seconds
WINDOW_SIZE = 0.310  # seconds (detection mode)
TARGET_DURATION_DETECTION = 40.0  # seconds


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def calculate_overlap(chunk_start, chunk_end, event_start, event_end):
    """Calculate overlap duration between chunk and event."""
    overlap_start = max(chunk_start, event_start)
    overlap_end = min(chunk_end, event_end)
    overlap_duration = max(0, overlap_end - overlap_start)
    return overlap_duration


def load_metadata():
    """Load metadata from JSON file."""
    print(f"Loading metadata from: {METADATA_JSON}")
    with open(METADATA_JSON, 'r') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data['dataset_metadata'])
    print(f"  Loaded {len(df):,} samples")
    return df


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def analyze_dataset_overview(df):
    """Analyze overall dataset composition."""
    stats = {
        'total_files': len(df),
        'total_chunks': len(df) * 4,
        'class_distribution': {},
        'files_per_class': {}
    }
    
    class_counts = df['siren_class'].value_counts().to_dict()
    stats['class_distribution'] = class_counts
    stats['files_per_class'] = class_counts
    
    return stats


def analyze_temporal_statistics(df):
    """Analyze temporal characteristics (onset, offset, duration)."""
    stats = {
        'onset': {
            'min': df['onset'].min(),
            'max': df['onset'].max(),
            'mean': df['onset'].mean(),
            'median': df['onset'].median(),
            'std': df['onset'].std()
        },
        'offset': {
            'min': df['offset'].min(),
            'max': df['offset'].max(),
            'mean': df['offset'].mean(),
            'median': df['offset'].median(),
            'std': df['offset'].std()
        }
    }
    
    df['duration'] = df['offset'] - df['onset']
    stats['duration'] = {
        'min': df['duration'].min(),
        'max': df['duration'].max(),
        'mean': df['duration'].mean(),
        'median': df['duration'].median(),
        'std': df['duration'].std()
    }
    
    return stats, df


def analyze_acoustic_features(df):
    """Analyze SNR and SPL statistics."""
    stats = {
        'snr_avg': {
            'overall': {
                'min': df['snr_avg'].min(),
                'max': df['snr_avg'].max(),
                'mean': df['snr_avg'].mean(),
                'median': df['snr_avg'].median(),
                'std': df['snr_avg'].std()
            },
            'per_class': {}
        },
        'spl_targets': {
            'bg_levels': sorted(df['bg_spl_target'].unique().tolist()),
            'fg_levels': sorted(df['fg_spl_target'].unique().tolist()),
            'bg_count': len(df['bg_spl_target'].unique()),
            'fg_count': len(df['fg_spl_target'].unique())
        }
    }
    
    for siren_class in sorted(df['siren_class'].unique()):
        class_df = df[df['siren_class'] == siren_class]
        stats['snr_avg']['per_class'][siren_class] = {
            'mean': class_df['snr_avg'].mean(),
            'std': class_df['snr_avg'].std()
        }
    
    return stats


def parse_filename_metadata(filename):
    """Parse metadata from filename."""
    import re
    pattern = r'^([^_]+)_([^_]+)_([^_]+)_(\d+)_([\d.]+)_([\d.]+)_i0\.wav$'
    match = re.match(pattern, filename)
    
    if match:
        return {
            'siren_type': match.group(2),
            'waveform': match.group(3),
            'iteration': int(match.group(4))
        }
    return None


def analyze_siren_taxonomy(df):
    """Analyze siren type, waveform, and iteration distribution."""
    df['parsed'] = df['filename'].apply(parse_filename_metadata)
    df['siren_type'] = df['parsed'].apply(lambda x: x['siren_type'] if x else None)
    df['waveform'] = df['parsed'].apply(lambda x: x['waveform'] if x else None)
    df['iteration'] = df['parsed'].apply(lambda x: x['iteration'] if x else None)
    
    stats = {
        'siren_types': df['siren_type'].value_counts().to_dict(),
        'waveforms': df['waveform'].value_counts().to_dict(),
        'iterations': {
            'min': df['iteration'].min(),
            'max': df['iteration'].max(),
            'unique': len(df['iteration'].unique())
        }
    }
    
    return stats, df


def analyze_chunking(df):
    """Analyze chunking characteristics (positives only in new system)."""
    positive_chunks = 0
    positive_chunks_per_class = defaultdict(int)
    
    for _, row in df.iterrows():
        onset = row['onset']
        offset = row['offset']
        siren_class = row['siren_class']
        
        for chunk_idx in range(4):
            chunk_start = chunk_idx * CHUNK_DURATION
            chunk_end = (chunk_idx + 1) * CHUNK_DURATION
            
            overlap = calculate_overlap(chunk_start, chunk_end, onset, offset)
            
            if overlap >= MIN_OVERLAP:
                positive_chunks += 1
                positive_chunks_per_class[siren_class] += 1
    
    stats = {
        'total_positive_chunks': positive_chunks,
        'positive_per_class': dict(positive_chunks_per_class)
    }
    
    return stats


def analyze_detection_mode(df):
    """Analyze detection mode (40s full samples) statistics."""
    num_windows = int(np.ceil(TARGET_DURATION_DETECTION / WINDOW_SIZE))
    
    positive_windows_per_sample = []
    
    for _, row in df.iterrows():
        onset = row['onset']
        offset = row['offset']
        
        label_track = np.zeros(num_windows)
        
        for i in range(num_windows):
            window_start = i * WINDOW_SIZE
            window_end = (i + 1) * WINDOW_SIZE
            
            if window_start < offset and onset < window_end:
                label_track[i] = 1
        
        num_positive_windows = int(label_track.sum())
        positive_windows_per_sample.append(num_positive_windows)
    
    stats = {
        'num_windows_per_sample': num_windows,
        'window_size': WINDOW_SIZE,
        'positive_windows': {
            'mean': np.mean(positive_windows_per_sample),
            'median': np.median(positive_windows_per_sample),
            'min': np.min(positive_windows_per_sample),
            'max': np.max(positive_windows_per_sample),
            'std': np.std(positive_windows_per_sample)
        }
    }
    
    return stats, positive_windows_per_sample


def analyze_negative_pool(num_positives: int):
    """Analyze negative pool composition dynamically from Negatives/ folder."""
    print("  Computing negative pool statistics from Negatives/ folder...")
    
    # Path to Negatives folder
    negatives_dir = os.path.join(DATASET_ROOT, 'Negatives')
    
    # Fallback to repo location if not in dataset root
    if not os.path.exists(negatives_dir):
        repo_negatives_dir = os.path.join(SCRIPT_DIR, 'Negatives')
        if os.path.exists(repo_negatives_dir):
            negatives_dir = repo_negatives_dir
    
    if not os.path.exists(negatives_dir):
        print(f"    Warning: Negatives directory not found!")
        return {'error': 'Negatives directory not found'}
    
    # Create negative generator to compute statistics
    neg_generator = KineScaper_NegativeChunkGenerator(
        negatives_dir=negatives_dir,
        num_positives=num_positives,
        chunk_duration=CHUNK_DURATION,
        overlap=0.20,  # 20% overlap as in dataloader
        target_sr=32000,
        seed=SEED
    )
    
    # Collect statistics per city
    city_stats = {}
    city_base_chunks = {}
    
    for chunk_metadata in neg_generator.base_chunks:
        audio_path = chunk_metadata['audio_path']
        city_name = os.path.splitext(os.path.basename(audio_path))[0]
        
        if city_name not in city_base_chunks:
            city_base_chunks[city_name] = 0
        city_base_chunks[city_name] += 1
    
    # Calculate augmented chunks per city
    for city_name, base_count in city_base_chunks.items():
        city_stats[city_name] = {
            'base_chunks': base_count,
            'augmented_chunks': base_count * neg_generator.augmentation_factor,
            'augmentation_factor': neg_generator.augmentation_factor
        }
    
    stats = {
        'cities': city_stats,
        'total_base_chunks': len(neg_generator.base_chunks),
        'augmentation_factor': neg_generator.augmentation_factor,
        'total_augmented_chunks': len(neg_generator),
        'overlap_ratio': neg_generator.overlap,
        'num_files': len(neg_generator.negative_files)
    }
    
    print(f"    Found {stats['num_files']} negative files")
    print(f"    Base chunks: {stats['total_base_chunks']:,}")
    print(f"    Augmentation factor: {stats['augmentation_factor']}x")
    print(f"    Total augmented: {stats['total_augmented_chunks']:,}")
    
    return stats


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_fig1_overview(negative_pool_stats, chunk_stats):
    """Figure 1: Dataset Overview - Bar Chart (Negatives) + Pie Chart (Positives)."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # =========================================================================
    # Plot 1: Negative Pool - Bar Chart per City (Base + Augmented)
    # =========================================================================
    ax = axes[0]
    
    if 'cities' in negative_pool_stats and negative_pool_stats['cities']:
        cities = sorted(negative_pool_stats['cities'].keys())
        base_counts = [negative_pool_stats['cities'][city]['base_chunks'] for city in cities]
        aug_counts = [negative_pool_stats['cities'][city]['augmented_chunks'] for city in cities]
        
        x = np.arange(len(cities))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, base_counts, width, label='Base Chunks', 
                      color='#8DD3C7', edgecolor='black', linewidth=1)
        bars2 = ax.bar(x + width/2, aug_counts, width, label='Augmented Chunks',
                      color='#FB8072', edgecolor='black', linewidth=1)
        
        ax.set_xlabel('City (Urban Traffic Source)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Number of Chunks (log scale)', fontsize=13, fontweight='bold')
        ax.set_title(f'Negative Pool Composition\n'
                    f'({negative_pool_stats["total_augmented_chunks"]:,} total chunks, '
                    f'{negative_pool_stats["augmentation_factor"]}x augmentation, '
                    f'{negative_pool_stats["overlap_ratio"]*100:.0f}% overlap)',
                    fontsize=14, fontweight='bold', pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(cities, rotation=45, ha='right', fontsize=10)
        ax.legend(loc='upper right', fontsize=11)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_yscale('log')  # Scala logaritmica sull'asse Y
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{int(height):,}',
                           ha='center', va='bottom', fontsize=8)
    else:
        ax.text(0.5, 0.5, 'No negative pool data available',
               ha='center', va='center', transform=ax.transAxes,
               fontsize=14, fontweight='bold')
        ax.set_title('Negative Pool Composition', fontsize=14, fontweight='bold')
    
    # =========================================================================
    # Plot 2: Positive Chunks - Pie Chart per Siren Class
    # =========================================================================
    ax = axes[1]
    
    if 'positive_per_class' in chunk_stats:
        pos_classes = sorted(chunk_stats['positive_per_class'].keys())
        pos_counts = [chunk_stats['positive_per_class'][c] for c in pos_classes]
        
        colors_pos = plt.cm.Set3(np.linspace(0, 1, len(pos_classes)))
        
        def make_autopct(values):
            def my_autopct(pct):
                total = sum(values)
                val = int(round(pct*total/100.0))
                return f'{pct:.1f}%\n({val:,})'
            return my_autopct
        
        wedges, texts, autotexts = ax.pie(pos_counts, labels=pos_classes,
                                            autopct=make_autopct(pos_counts),
                                            colors=colors_pos,
                                            startangle=45,
                                            textprops={'fontsize': 10, 'fontweight': 'bold'})
        
        ax.set_title(f'Positive Chunks by Siren Class\n'
                    f'({chunk_stats["total_positive_chunks"]:,} total positive chunks)',
                    fontsize=14, fontweight='bold', pad=15)
    else:
        ax.text(0.5, 0.5, 'No positive chunk data available',
               ha='center', va='center', transform=ax.transAxes,
               fontsize=14, fontweight='bold')
        ax.set_title('Positive Chunks Distribution', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "overview.svg"
    plt.savefig(output_path, format='svg', dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def plot_fig3_acoustic_features(df, acoustic_stats, temporal_stats):
    """Figure 3: Acoustic Features (2x2)."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # =========================================================================
    # Plot 1: Overall SNR distribution
    # =========================================================================
    ax = axes[0, 0]
    ax.hist(df['snr_avg'], bins=100, color='#e67e22', alpha=0.7, edgecolor='black', linewidth=1.2)
    ax.axvline(acoustic_stats['snr_avg']['overall']['mean'], color='red', linestyle='--', linewidth=2,
               label=f"Mean: {acoustic_stats['snr_avg']['overall']['mean']:.2f} dB")
    ax.set_xlabel('Average SNR (dB)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=13, fontweight='bold')
    ax.set_title('SNR Distribution (Overall)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # =========================================================================
    # Plot 2: SNR per class (boxplot)
    # =========================================================================
    ax = axes[0, 1]
    classes = sorted(df['siren_class'].unique())
    snr_per_class = [df[df['siren_class'] == c]['snr_avg'].values for c in classes]
    
    bp = ax.boxplot(snr_per_class, tick_labels=classes, patch_artist=True,
                    boxprops=dict(facecolor='#3498db', alpha=0.7),
                    medianprops=dict(color='red', linewidth=2))
    ax.set_xlabel('Siren Class', fontsize=13, fontweight='bold')
    ax.set_ylabel('Average SNR (dB)', fontsize=13, fontweight='bold')
    ax.set_title('SNR Distribution per Class', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # =========================================================================
    # Plot 3: BG vs FG SPL heatmap (interpolated)
    # =========================================================================
    ax = axes[1, 0]
    
    # Estrai dati
    bg_vals = df['bg_spl_target'].values
    fg_vals = df['fg_spl_target'].values
    snr_vals = df['snr_avg'].values
    
    # Crea griglia per heatmap
    bg_unique = np.sort(df['bg_spl_target'].unique())
    fg_unique = np.sort(df['fg_spl_target'].unique())
    
    # Crea griglia interpolata
    bg_grid, fg_grid = np.meshgrid(
        np.linspace(bg_unique.min(), bg_unique.max(), 100),
        np.linspace(fg_unique.min(), fg_unique.max(), 100)
    )
    
    # Interpola SNR values sulla griglia
    snr_grid = griddata(
        (bg_vals, fg_vals), 
        snr_vals, 
        (bg_grid, fg_grid), 
        method='cubic'
    )
    
    # Plot heatmap interpolata
    im = ax.imshow(snr_grid, cmap='viridis', aspect='auto', interpolation='bilinear',
                   origin='lower', extent=[bg_unique.min(), bg_unique.max(), 
                                          fg_unique.min(), fg_unique.max()])
    ax.set_xlabel('Background SPL (dB)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Foreground SPL (dB)', fontsize=13, fontweight='bold')
    ax.set_title('BG vs FG SPL Targets\n(color = mean SNR, interpolated)', fontsize=14, fontweight='bold')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Mean SNR (dB)', fontsize=12, fontweight='bold')
    
    # =========================================================================
    # Plot 4: Siren Duration Distribution (continuous line)
    # =========================================================================
    ax = axes[1, 1]
    
    # Crea histogram data
    counts, bin_edges = np.histogram(df['duration'], bins=50, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Plot come linea continua
    ax.plot(bin_centers, counts, color='#27ae60', linewidth=2.5, label='Distribution')
    ax.fill_between(bin_centers, counts, alpha=0.3, color='#27ae60')
    
    # Aggiungi mean e median
    ax.axvline(temporal_stats['duration']['mean'], color='red', linestyle='--', linewidth=2,
               label=f"Mean: {temporal_stats['duration']['mean']:.2f}s")
    ax.axvline(temporal_stats['duration']['median'], color='orange', linestyle='--', linewidth=2,
               label=f"Median: {temporal_stats['duration']['median']:.2f}s")
    
    ax.set_xlabel('Event Duration (seconds)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Density', fontsize=13, fontweight='bold')
    ax.set_title('Siren Duration Distribution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "acoustic_features.svg"
    plt.savefig(output_path, format='svg', dpi=600, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_fig4_siren_taxonomy(df):
    """Figure 4: Siren Taxonomy Heatmaps (1x3)."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # Plot 1: Class × Type
    ax = axes[0]
    pivot = pd.crosstab(df['siren_class'], df['siren_type'])
    sns.heatmap(pivot, annot=True, fmt='d', cmap='YlOrRd', ax=ax, 
                cbar_kws={'label': 'Count'}, linewidths=0.5, linecolor='black')
    ax.set_xlabel('Siren Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Siren Class', fontsize=12, fontweight='bold')
    ax.set_title('Class × Type Distribution', fontsize=13, fontweight='bold')
    
    # Plot 2: Class × Waveform
    ax = axes[1]
    pivot = pd.crosstab(df['siren_class'], df['waveform'])
    sns.heatmap(pivot, annot=True, fmt='d', cmap='YlGnBu', ax=ax,
                cbar_kws={'label': 'Count'}, linewidths=0.5, linecolor='black')
    ax.set_xlabel('Waveform', fontsize=12, fontweight='bold')
    ax.set_ylabel('Siren Class', fontsize=12, fontweight='bold')
    ax.set_title('Class × Waveform Distribution', fontsize=13, fontweight='bold')
    
    # Plot 3: Type × Waveform
    ax = axes[2]
    pivot = pd.crosstab(df['siren_type'], df['waveform'])
    sns.heatmap(pivot, annot=True, fmt='d', cmap='RdPu', ax=ax,
                cbar_kws={'label': 'Count'}, linewidths=0.5, linecolor='black')
    ax.set_xlabel('Waveform', fontsize=12, fontweight='bold')
    ax.set_ylabel('Siren Type', fontsize=12, fontweight='bold')
    ax.set_title('Type × Waveform Distribution', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "siren_taxonomy.svg"
    plt.savefig(output_path, format='svg', dpi=600, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_fig5_detection_mode(positive_windows_per_sample, detection_stats):
    """Figure 5: Detection Mode - Positive Windows Distribution (smooth continuous line)."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    # Crea histogram data
    counts, bin_edges = np.histogram(positive_windows_per_sample, bins=50, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Smooth la curva con gaussian filter
    counts_smooth = gaussian_filter1d(counts, sigma=2)
    
    # Plot come linea continua smoothata
    ax.plot(bin_centers, counts_smooth, color='#3498db', linewidth=2.5, label='Distribution')
    ax.fill_between(bin_centers, counts_smooth, alpha=0.3, color='#3498db')
    
    # Aggiungi mean e median
    mean_val = detection_stats['positive_windows']['mean']
    median_val = detection_stats['positive_windows']['median']
    
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2,
               label=f"Mean: {mean_val:.1f}")
    ax.axvline(median_val, color='orange', linestyle='--', linewidth=2,
               label=f"Median: {median_val:.1f}")
    
    ax.set_xlabel('Number of Positive Windows', fontsize=13, fontweight='bold')
    ax.set_ylabel('Density', fontsize=13, fontweight='bold')
    ax.set_title(f'Positive Windows per Sample Distribution\n(Total windows per sample: {detection_stats["num_windows_per_sample"]})',
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "detection_mode.svg"
    plt.savefig(output_path, format='svg', dpi=600, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def save_statistics_json(all_stats):
    """Save all statistics to JSON file."""
    output_path = OUTPUT_DIR / "statistics.json"
    
    def convert_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(item) for item in obj]
        else:
            return obj
    
    all_stats_converted = convert_types(all_stats)
    
    with open(output_path, 'w') as f:
        json.dump(all_stats_converted, f, indent=2)
    
    print(f"✓ Saved statistics: {output_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main analysis function."""
    print("=" * 80)
    print("KINESCAPER_EV DATASET ANALYSIS")
    print("=" * 80)
    print(f"Seed: {SEED}")
    
    # Load metadata
    print("\n[1/8] Loading metadata...")
    df = load_metadata()
    
    # Dataset overview
    print("\n[2/8] Analyzing dataset overview...")
    overview_stats = analyze_dataset_overview(df)
    
    # Temporal statistics
    print("\n[3/8] Analyzing temporal statistics...")
    temporal_stats, df = analyze_temporal_statistics(df)
    
    # Acoustic features
    print("\n[4/8] Analyzing acoustic features...")
    acoustic_stats = analyze_acoustic_features(df)
    
    # Siren taxonomy
    print("\n[5/8] Analyzing siren taxonomy...")
    taxonomy_stats, df = analyze_siren_taxonomy(df)
    
    # Chunking analysis
    print("\n[6/8] Analyzing chunking statistics...")
    chunk_stats = analyze_chunking(df)
    
    # Detection mode
    print("\n[7/8] Analyzing detection mode...")
    detection_stats, positive_windows = analyze_detection_mode(df)
    
    # Negative pool (dynamic calculation using KineScaper_NegativeChunkGenerator)
    print("\n[8/8] Analyzing negative pool...")
    negative_pool_stats = analyze_negative_pool(num_positives=chunk_stats['total_positive_chunks'])
    
    # Combine all statistics
    all_stats = {
        'overview': overview_stats,
        'temporal': temporal_stats,
        'acoustic': acoustic_stats,
        'taxonomy': taxonomy_stats,
        'chunking': chunk_stats,
        'detection_mode': detection_stats,
        'negative_pool': negative_pool_stats,
        'parameters': {
            'seed': SEED,
            'chunk_duration': CHUNK_DURATION,
            'min_overlap': MIN_OVERLAP,
            'window_size': WINDOW_SIZE,
            'target_duration_detection': TARGET_DURATION_DETECTION
        }
    }
    
    # Save statistics
    print("\nSaving statistics and generating plots...")
    save_statistics_json(all_stats)
    
    # Generate all plots
    print("\nGenerating plots...")
    plot_fig1_overview(negative_pool_stats, chunk_stats)
    plot_fig3_acoustic_features(df, acoustic_stats, temporal_stats)
    plot_fig4_siren_taxonomy(df)
    plot_fig5_detection_mode(positive_windows, detection_stats)
    
    print("\n" + "=" * 80)
    print(f"ANALYSIS COMPLETE!")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("=" * 80)
    
    # Print summary
    print("\nSUMMARY:")
    print(f"  Total files: {overview_stats['total_files']:,}")
    print(f"  Total chunks: {overview_stats['total_chunks']:,}")
    print(f"  Positive chunks: {chunk_stats['total_positive_chunks']:,}")
    print(f"  7 siren classes (perfectly balanced)")
    
    if 'total_augmented_chunks' in negative_pool_stats:
        print(f"\n  Negative pool (NEW SYSTEM - Standalone):")
        print(f"    Files: {negative_pool_stats['num_files']} urban traffic recordings")
        print(f"    Base chunks: {negative_pool_stats['total_base_chunks']:,}")
        print(f"    Augmentation: {negative_pool_stats['augmentation_factor']}x")
        print(f"    Total negatives: {negative_pool_stats['total_augmented_chunks']:,}")
        print(f"    Overlap: {negative_pool_stats['overlap_ratio']*100:.0f}%")
    
    print(f"\n  Generated figures:")
    print(f"    - overview.svg (Negative Bar Chart + Positive Pie Chart)")
    print(f"    - acoustic_features.svg (SNR + SPL heatmap + Duration)")
    print(f"    - siren_taxonomy.svg (3 heatmaps)")
    print(f"    - detection_mode.svg (positive windows distribution, smoothed)")


if __name__ == "__main__":
    main()
