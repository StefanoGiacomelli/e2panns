"""
AudioSet EV v1 vs v2 Comparative Analysis
==========================================
Compare AudioSet_EV_v1_2025 and AudioSet_EV_v2PANNs_2020 datasets.

Author: Stefano Giacomelli - Ph.D. candidate in ICT (DISIM dpt. - University of L'Aquila)
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import json
import ast
import os

# Configuration
V1_POS_CSV = "./datasets/AudioSet_EV_v1_2025/EV_Positives.csv"
V1_NEG_CSV = "./datasets/AudioSet_EV_v1_2025/EV_Negatives.csv"
V2_POS_CSV = "./datasets/AudioSet_EV_v2PANNs_2020/EV_Positives.csv"
V2_NEG_CSV = "./datasets/AudioSet_EV_v2PANNs_2020/EV_Negatives.csv"
MAPPING_JSON = "./datasets/datasets_mapping.json"
CLASS_LABELS_CSV = "./datasets/AudioSet_EV_v2PANNs_2020/audioset_metadata/class_labels_indices.csv"
OUTPUT_DIR = "."

# Load datasets_mapping.json for label names
with open(MAPPING_JSON, 'r') as f:
    label_mapping = json.load(f)['AUDIOSET']

# Load class_labels_indices.csv for complete MID to display name mapping
mid_to_name = {}
with open(CLASS_LABELS_CSV, 'r') as f:
    import csv
    reader = csv.DictReader(f)
    for row in reader:
        mid = row['mid']
        display_name = row['display_name'].strip('"')
        mid_to_name[mid] = display_name


def load_and_analyze_csv(csv_path, name):
    """Load CSV and compute basic statistics."""
    df = pd.read_csv(csv_path)
    
    stats = {
        'name': name,
        'total': len(df),
        'downloaded': df['downloaded'].sum(),
        'not_downloaded': (~df['downloaded']).sum(),
    }
    
    # Check if segment_type exists (v2 only initially)
    if 'segment_type' in df.columns:
        by_segment = df.groupby('segment_type').agg({
            'yt_id': 'count',
            'downloaded': 'sum'
        }).rename(columns={'yt_id': 'total', 'downloaded': 'downloaded'})
        stats['by_segment'] = by_segment.to_dict('index')
    
    return df, stats


def recalculate_stats_with_segments(df, name):
    """Recalculate statistics after segment_type has been inferred."""
    stats = {
        'name': name,
        'total': len(df),
        'downloaded': df['downloaded'].sum(),
        'not_downloaded': (~df['downloaded']).sum(),
    }
    
    # Calculate by segment
    by_segment = df.groupby('segment_type').agg({
        'yt_id': 'count',
        'downloaded': 'sum'
    }).rename(columns={'yt_id': 'total', 'downloaded': 'downloaded'})
    stats['by_segment'] = by_segment.to_dict('index')
    
    return stats


def extract_label_counts(df, is_positive=True):
    """Extract label counts from positive_labels column."""
    label_counts = Counter()
    
    for labels_str in df['positive_labels']:
        try:
            labels = ast.literal_eval(labels_str)
            for mid in labels:
                label_counts[mid] += 1
        except:
            continue
    
    return label_counts


def infer_segment_type_v1(df_v1, v2_pos, v2_neg):
    """
    Infer segment_type for v1 by matching yt_id with v2.
    """
    # Create yt_id to segment_type mapping from v2
    v2_combined = pd.concat([v2_pos[['yt_id', 'segment_type']], 
                             v2_neg[['yt_id', 'segment_type']]])
    yt_to_segment = v2_combined.set_index('yt_id')['segment_type'].to_dict()
    
    # Map v1 yt_ids to segments (copy to avoid warning)
    df_v1 = df_v1.copy()
    df_v1['segment_type'] = df_v1['yt_id'].map(yt_to_segment)
    
    # Fill NaN with 'unknown'
    df_v1['segment_type'] = df_v1['segment_type'].fillna('unknown')
    
    return df_v1


def print_statistics(v1_pos_stats, v1_neg_stats, v2_pos_stats, v2_neg_stats):
    """Print detailed statistics."""
    print("\n" + "=" * 80)
    print("DATASET COMPARISON STATISTICS")
    print("=" * 80)
    
    print("\n--- AUDIOSET_EV_V1_2025 ---")
    print(f"\nPositives:")
    print(f"  Total: {v1_pos_stats['total']}")
    print(f"  Downloaded: {v1_pos_stats['downloaded']}")
    print(f"  Not downloaded: {v1_pos_stats['not_downloaded']}")
    
    if 'by_segment' in v1_pos_stats:
        print("\n  By segment (inferred from v2):")
        for segment, data in sorted(v1_pos_stats['by_segment'].items()):
            if segment != 'unknown':
                print(f"    {segment}: {data['downloaded']}/{data['total']} downloaded")
    
    print(f"\nNegatives:")
    print(f"  Total: {v1_neg_stats['total']}")
    print(f"  Downloaded: {v1_neg_stats['downloaded']}")
    print(f"  Not downloaded: {v1_neg_stats['not_downloaded']}")
    
    if 'by_segment' in v1_neg_stats:
        print("\n  By segment (inferred from v2):")
        for segment, data in sorted(v1_neg_stats['by_segment'].items()):
            if segment != 'unknown':
                print(f"    {segment}: {data['downloaded']}/{data['total']} downloaded")
    
    print("\n--- AUDIOSET_EV_V2_PANNs_2020 ---")
    print(f"\nPositives:")
    print(f"  Total: {v2_pos_stats['total']}")
    print(f"  Downloaded: {v2_pos_stats['downloaded']}")
    print(f"  Not downloaded: {v2_pos_stats['not_downloaded']}")
    
    if 'by_segment' in v2_pos_stats:
        print("\n  By segment:")
        for segment, data in sorted(v2_pos_stats['by_segment'].items()):
            print(f"    {segment}: {data['downloaded']}/{data['total']} downloaded")
    
    print(f"\nNegatives:")
    print(f"  Total: {v2_neg_stats['total']}")
    print(f"  Downloaded: {v2_neg_stats['downloaded']}")
    print(f"  Not downloaded: {v2_neg_stats['not_downloaded']}")
    
    if 'by_segment' in v2_neg_stats:
        print("\n  By segment:")
        for segment, data in sorted(v2_neg_stats['by_segment'].items()):
            print(f"    {segment}: {data['downloaded']}/{data['total']} downloaded")
    
    # Comparison
    print("\n--- COMPARISON ---")
    print(f"\nPositives:")
    print(f"  v1 downloaded: {v1_pos_stats['downloaded']}")
    print(f"  v2 downloaded: {v2_pos_stats['downloaded']}")
    diff = v2_pos_stats['downloaded'] - v1_pos_stats['downloaded']
    print(f"  Difference: {diff:+d} ({diff/v1_pos_stats['downloaded']*100:+.1f}%)")
    
    print(f"\nNegatives:")
    print(f"  v1 downloaded: {v1_neg_stats['downloaded']}")
    print(f"  v2 downloaded (excl. unbalanced): {v2_neg_stats['downloaded']}")
    diff = v2_neg_stats['downloaded'] - v1_neg_stats['downloaded']
    print(f"  Difference: {diff:+d} ({diff/v1_neg_stats['downloaded']*100:+.1f}%)")


def plot_overview_comparison(v1_pos, v1_neg, v2_pos, v2_neg):
    """
    Plot 1: Overview bar plot comparing pos/neg for v1 and v2.
    Shows potential (faded) and available (saturated) bars.
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    datasets = ['AudioSet EV\nv1 (2025)', 'AudioSet EV\nv2 (2020)']
    x = np.arange(len(datasets)) * 1.2  # Reduced spacing
    width = 0.4
    
    # V1 data
    v1_pos_downloaded = v1_pos['downloaded'].sum()
    v1_pos_total = len(v1_pos)
    v1_neg_downloaded = v1_neg['downloaded'].sum()
    v1_neg_total = len(v1_neg)
    
    # V2 data (exclude unbalanced negatives)
    v2_pos_downloaded = v2_pos['downloaded'].sum()
    v2_pos_total = len(v2_pos)
    v2_neg_downloaded = v2_neg[v2_neg['segment_type'] != 'unbalanced_train']['downloaded'].sum()
    v2_neg_total = len(v2_neg[v2_neg['segment_type'] != 'unbalanced_train'])
    
    # Data arrays
    pos_total = [v1_pos_total, v2_pos_total]
    pos_downloaded = [v1_pos_downloaded, v2_pos_downloaded]
    neg_total = [v1_neg_total, v2_neg_total]
    neg_downloaded = [v1_neg_downloaded, v2_neg_downloaded]
    
    # Plot positives (faded then saturated)
    bars_pos_potential = ax.bar(x - width/2, pos_total, width, 
                                 label='Positives (potential)', 
                                 color='#4CAF50', alpha=0.3)
    bars_pos_downloaded = ax.bar(x - width/2, pos_downloaded, width,
                                   label='Positives (available)',
                                   color='#4CAF50', alpha=1.0)
    
    # Plot negatives (faded then saturated)
    bars_neg_potential = ax.bar(x + width/2, neg_total, width,
                                  label='Negatives (potential)',
                                  color='#F44336', alpha=0.3)
    bars_neg_downloaded = ax.bar(x + width/2, neg_downloaded, width,
                                   label='Negatives (available)',
                                   color='#F44336', alpha=1.0)
    
    # Labels and formatting
    ax.set_ylabel('Number of Samples', fontsize=13, fontweight='bold')
    ax.set_title('AudioSet EV Dataset Comparison: v1 vs v2\n(Potential vs Available)', 
                 fontsize=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for bars in [bars_pos_downloaded, bars_neg_downloaded]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height):,}',
                   ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'dataset_comparison_overview.svg'), 
                format='svg', dpi=300, bbox_inches='tight')
    print("✓ Saved: dataset_comparison_overview.svg")
    plt.close()


def plot_label_distribution_positives_comparison(v1_pos, v2_pos):
    """
    Plot: Side-by-side comparison of positives label distribution for v1 and v2.
    Shows only Emergency Vehicle-related labels.
    """
    # Specific MIDs for Emergency Vehicle labels (in desired order)
    desired_mids = [
        '/m/03j1ly',  # Emergency vehicle
        '/m/03kmc9',  # Siren
        '/m/04qvtq',  # Police car (siren)
        '/m/012ndj',  # Fire engine, fire truck (siren)
        '/m/012n7d',  # Ambulance (siren)
    ]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7), sharey=True)
    
    segments = ['balanced_train', 'eval', 'unbalanced_train']
    colors = ['#2196F3', '#FF9800', '#9C27B0']  # Blue, Orange, Purple
    bar_height = 0.25
    
    # Process both datasets
    for ax, df, dataset_name in [(ax1, v1_pos, 'v1 (2025)'), (ax2, v2_pos, 'v2 (2020)')]:
        # Extract label counts by segment
        label_counts_by_segment = {seg: Counter() for seg in segments}
        total_counts = Counter()
        
        for _, row in df[df['downloaded'] == True].iterrows():
            segment = row.get('segment_type', 'unknown')
            if segment in segments:
                try:
                    labels = ast.literal_eval(row['positive_labels'])
                    for mid in labels:
                        label_counts_by_segment[segment][mid] += 1
                        total_counts[mid] += 1
                except:
                    continue
        
        # Filter to only include desired MIDs that exist in the data
        filtered_mids = [mid for mid in desired_mids if mid in total_counts]
        
        # Convert to display names (human-readable) - REVERSED for top to bottom
        filtered_labels = [mid_to_name.get(mid, mid) for mid in filtered_mids]
        filtered_labels = list(reversed(filtered_labels))  # Most common at top
        filtered_mids = list(reversed(filtered_mids))
        
        # Prepare data for plotting
        data_by_segment = {seg: [] for seg in segments}
        for mid in filtered_mids:
            for seg in segments:
                data_by_segment[seg].append(label_counts_by_segment[seg].get(mid, 0))
        
        # Plot bars
        y_pos = np.arange(len(filtered_labels))
        
        for i, (seg, color) in enumerate(zip(segments, colors)):
            offset = (i - 1) * bar_height
            bars = ax.barh(y_pos + offset, data_by_segment[seg], bar_height,
                          label=seg.replace('_', ' ').title(), color=color, alpha=0.8)
            
            # Add value labels
            for bar in bars:
                width = bar.get_width()
                if width > 0:
                    ax.text(width, bar.get_y() + bar.get_height()/2.,
                           f' {int(width)}',
                           ha='left', va='center', fontsize=11)
        
        # Format subplot
        ax.set_yticks(y_pos)
        if ax == ax1:  # Only show y-labels on left subplot
            ax.set_yticklabels(filtered_labels, fontsize=11)
        ax.tick_params(axis='both', which='major', labelsize=11)
        ax.set_xlabel('Number of Samples', fontsize=13, fontweight='bold')
        ax.set_title(f'AudioSet EV {dataset_name}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11, loc='lower right')
        ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Overall title
    fig.suptitle('Emergency Vehicle Labels by Segment: v1 vs v2 Comparison', 
                 fontsize=15, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    filename = 'label_distribution_positives_comparison.svg'
    plt.savefig(os.path.join(OUTPUT_DIR, filename),
                format='svg', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filename}")
    plt.close()


def plot_label_distribution_by_segment(df, label_type, dataset_name):
    """
    Plot 2-5: Horizontal bar plots showing TOP 10 label distribution by segment.
    One for positives, one for negatives.
    """
    # Extract label counts by segment
    # For negatives, exclude unbalanced_train
    if label_type == 'negatives':
        segments = ['balanced_train', 'eval']
    else:
        segments = ['balanced_train', 'eval', 'unbalanced_train']
    
    label_counts_by_segment = {seg: Counter() for seg in segments}
    total_counts = Counter()
    
    for _, row in df[df['downloaded'] == True].iterrows():
        segment = row.get('segment_type', 'unknown')
        if segment in segments:
            try:
                labels = ast.literal_eval(row['positive_labels'])
                for mid in labels:
                    label_counts_by_segment[segment][mid] += 1
                    total_counts[mid] += 1
            except:
                continue
    
    # For positives, show only specific Emergency Vehicle-related labels
    if label_type == 'positives':
        # Specific MIDs for Emergency Vehicle labels (in desired order)
        desired_mids = [
            '/m/03j1ly',  # Emergency vehicle
            '/m/03kmc9',  # Siren
            '/m/04qvtq',  # Police car (siren)
            '/m/012ndj',  # Fire engine, fire truck (siren)
            '/m/012n7d',  # Ambulance (siren)
        ]
        # Filter to only include these MIDs that exist in the data
        filtered_mids = [mid for mid in desired_mids if mid in total_counts]
    else:
        # For negatives, get TOP 10 MIDs by total count
        filtered_mids = [mid for mid, count in total_counts.most_common(10)]
    
    # Convert to display names (human-readable) - REVERSED for top to bottom
    filtered_labels = [mid_to_name.get(mid, mid) for mid in filtered_mids]
    filtered_labels = list(reversed(filtered_labels))  # Most common at top
    filtered_mids = list(reversed(filtered_mids))
    
    # Prepare data for plotting
    data_by_segment = {seg: [] for seg in segments}
    for mid in filtered_mids:
        for seg in segments:
            data_by_segment[seg].append(label_counts_by_segment[seg].get(mid, 0))
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, max(8, len(filtered_labels) * 0.5)))
    
    y_pos = np.arange(len(filtered_labels))
    bar_height = 0.25
    colors = ['#2196F3', '#FF9800', '#9C27B0']  # Blue, Orange, Purple
    
    for i, (seg, color) in enumerate(zip(segments, colors)):
        offset = (i - len(segments)//2) * bar_height if len(segments) == 2 else (i - 1) * bar_height
        bars = ax.barh(y_pos + offset, data_by_segment[seg], bar_height,
                      label=seg.replace('_', ' ').title(), color=color, alpha=0.8)
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            if width > 0:
                ax.text(width, bar.get_y() + bar.get_height()/2.,
                       f' {int(width)}',
                       ha='left', va='center', fontsize=11)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(filtered_labels, fontsize=11)
    ax.tick_params(axis='both', which='major', labelsize=11)
    ax.set_xlabel('Number of Samples', fontsize=13, fontweight='bold')
    
    # Update title for positives
    if label_type == 'positives':
        title = f'Emergency Vehicle Labels by Segment\n({dataset_name})'
    else:
        title = f'Top 10 {label_type.title()} Labels by Segment\n({dataset_name})'
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    filename = f'label_distribution_{label_type}_{dataset_name.lower().replace(" ", "_")}.svg'
    plt.savefig(os.path.join(OUTPUT_DIR, filename),
                format='svg', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filename}")
    plt.close()


def plot_segment_comparison(v1_pos, v1_neg, v2_pos, v2_neg):
    """
    Plot 4: Compare v1 vs v2 by segment.
    Shows downloaded samples for each segment.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    segments = ['balanced_train', 'eval', 'unbalanced_train']
    segment_labels = ['Balanced\nTrain', 'Eval', 'Unbalanced\nTrain']
    x = np.arange(len(segments))
    width = 0.35
    
    # Positives comparison
    v1_pos_by_seg = v1_pos[v1_pos['downloaded'] == True].groupby('segment_type').size()
    v2_pos_by_seg = v2_pos[v2_pos['downloaded'] == True].groupby('segment_type').size()
    
    v1_pos_counts = [v1_pos_by_seg.get(seg, 0) for seg in segments]
    v2_pos_counts = [v2_pos_by_seg.get(seg, 0) for seg in segments]
    
    bars1 = ax1.bar(x - width/2, v1_pos_counts, width, label='v1 (2025)', 
                    color='#2196F3', alpha=0.8)
    bars2 = ax1.bar(x + width/2, v2_pos_counts, width, label='v2 (2020)', 
                    color='#4CAF50', alpha=0.8)
    
    ax1.set_ylabel('Number of Samples', fontsize=13, fontweight='bold')
    ax1.set_title('Positives: Segment Comparison (v1 vs v2)', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(segment_labels, fontsize=12)
    ax1.tick_params(axis='both', which='major', labelsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height):,}',
                        ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Negatives comparison (exclude unbalanced for v2)
    v1_neg_by_seg = v1_neg[v1_neg['downloaded'] == True].groupby('segment_type').size()
    v2_neg_by_seg = v2_neg[(v2_neg['downloaded'] == True) & 
                           (v2_neg['segment_type'] != 'unbalanced_train')].groupby('segment_type').size()
    
    v1_neg_counts = [v1_neg_by_seg.get(seg, 0) for seg in segments[:2]]  # Only balanced and eval
    v2_neg_counts = [v2_neg_by_seg.get(seg, 0) for seg in segments[:2]]
    
    x_neg = np.arange(len(segments[:2]))
    bars3 = ax2.bar(x_neg - width/2, v1_neg_counts, width, label='v1 (2025)', 
                    color='#2196F3', alpha=0.8)
    bars4 = ax2.bar(x_neg + width/2, v2_neg_counts, width, label='v2 (2020)', 
                    color='#F44336', alpha=0.8)
    
    ax2.set_ylabel('Number of Samples', fontsize=13, fontweight='bold')
    ax2.set_title('Negatives: Segment Comparison (v1 vs v2)', fontsize=14, fontweight='bold')
    ax2.set_xticks(x_neg)
    ax2.set_xticklabels(segment_labels[:2], fontsize=12)
    ax2.tick_params(axis='both', which='major', labelsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels
    for bars in [bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height):,}',
                        ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'segment_comparison_v1_vs_v2.svg'), 
                format='svg', dpi=300, bbox_inches='tight')
    print("✓ Saved: segment_comparison_v1_vs_v2.svg")
    plt.close()


def main():
    """Main execution function."""
    
    print("=" * 80)
    print("AUDIOSET EV DATASETS COMPARATIVE ANALYSIS")
    print("=" * 80)
    
    # Load datasets
    print("\n[1/6] Loading datasets...")
    v1_pos, v1_pos_stats = load_and_analyze_csv(V1_POS_CSV, "v1_positives")
    v1_neg, v1_neg_stats = load_and_analyze_csv(V1_NEG_CSV, "v1_negatives")
    v2_pos, v2_pos_stats = load_and_analyze_csv(V2_POS_CSV, "v2_positives")
    v2_neg, v2_neg_stats = load_and_analyze_csv(V2_NEG_CSV, "v2_negatives")
    
    # Infer segment_type for v1 from v2
    print("\n[2/6] Inferring segment types for v1 from v2...")
    v1_pos = infer_segment_type_v1(v1_pos, v2_pos, v2_neg)
    v1_neg = infer_segment_type_v1(v1_neg, v2_pos, v2_neg)
    
    # Recalculate v1 stats with segment info
    print("\n[3/6] Recalculating v1 statistics with segment breakdown...")
    v1_pos_stats = recalculate_stats_with_segments(v1_pos, "v1_positives")
    v1_neg_stats = recalculate_stats_with_segments(v1_neg, "v1_negatives")
    
    # For v2, exclude unbalanced negatives from stats
    v2_neg_filtered = v2_neg[v2_neg['segment_type'] != 'unbalanced_train'].copy()
    v2_neg_stats_filtered = recalculate_stats_with_segments(v2_neg_filtered, "v2_negatives_filtered")
    
    # Print statistics
    print("\n[4/6] Computing and printing statistics...")
    print_statistics(v1_pos_stats, v1_neg_stats, v2_pos_stats, v2_neg_stats_filtered)
    
    # Generate plots
    print("\n[5/6] Generating plots...")
    
    # Plot 1: Overview comparison
    plot_overview_comparison(v1_pos, v1_neg, v2_pos, v2_neg)
    
    # Plot 2: Segment comparison v1 vs v2
    plot_segment_comparison(v1_pos, v1_neg, v2_pos, v2_neg)
    
    # Plot 3: Positives distribution comparison (v1 vs v2 side-by-side)
    plot_label_distribution_positives_comparison(v1_pos, v2_pos)
    
    # Plot 4-5: Negatives distribution by segment (v1 and v2, excl. unbalanced)
    plot_label_distribution_by_segment(v1_neg[v1_neg['segment_type'] != 'unbalanced_train'], 
                                      'negatives', 'AudioSet EV v1')
    plot_label_distribution_by_segment(v2_neg[v2_neg['segment_type'] != 'unbalanced_train'], 
                                      'negatives', 'AudioSet EV v2')
    
    print("\n[6/6] Analysis complete!")
    print("\n" + "=" * 80)
    print("All plots saved to:", OUTPUT_DIR)
    print("=" * 80)


if __name__ == "__main__":
    main()
