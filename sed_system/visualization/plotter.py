"""Visualization Tools for SED Predictions
============================================
Create plots showing:
- Mel-spectrogram (background, semi-transparent)
- Raw prediction probabilities (continuous line)
- Detected events (colored boxes)

Author: Stefano Giacomelli - Ph.D. candidate in ICT (DISIM dpt. - University of L'Aquila)
"""

import logging
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec


def plot_predictions_with_spectrogram(
    spectrogram: np.ndarray,
    audio_duration: float,
    predictions: List[Dict],
    events: List[Tuple[float, float, float]],  # (onset, offset, confidence)
    threshold: float,
    save_path: str,
    sr: int = 32000,
    figsize: Tuple[int, int] = (14, 8)
):
    """
    Create comprehensive visualization of SED predictions.
    
    Args:
        spectrogram: Mel-spectrogram array (n_mels, time_frames)
        audio_duration: Total audio duration in seconds
        predictions: List of prediction dicts with 'timestamp' and 'probability'
        events: List of detected events as (onset, offset) tuples
        threshold: Detection threshold used
        save_path: Path to save the plot
        sr: Sample rate
        figsize: Figure size (width, height)
    """

    # Create figure with custom layout
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(3, 1, height_ratios=[2, 1, 0.3], hspace=0.3)
    
    # Axes (share X axis for perfect alignment)
    ax_spec = fig.add_subplot(gs[0])              # Spectrogram
    ax_pred = fig.add_subplot(gs[1], sharex=ax_spec)    # Predictions
    ax_events = fig.add_subplot(gs[2], sharex=ax_spec)  # Events timeline
    
    # ===========================================================================
    # 1. SPECTROGRAM (background, semi-transparent)
    # ===========================================================================
    time_frames = spectrogram.shape[1]
    time_axis = np.linspace(0, audio_duration, time_frames)
    
    im = ax_spec.imshow(
        spectrogram,
        aspect='auto',
        origin='lower',
        extent=[0, audio_duration, 0, spectrogram.shape[0]],
        cmap='plasma',  # Plasma colormap, nitido
        interpolation='none'  # Sharp, no blur
    )
    
    ax_spec.set_ylabel('Mel Frequency Bins', fontsize=10)
    ax_spec.set_title('Input Mel-Spectrogram', fontsize=11, pad=10)
    ax_spec.grid(True, alpha=0.3, linestyle='--')
    
    # ===========================================================================
    # 2. PREDICTIONS (raw probabilities + threshold line)
    # ===========================================================================
    if predictions:
        pred_times = [p['timestamp'] for p in predictions]
        pred_probs = [p['probability'] for p in predictions]
        
        # Plot raw predictions
        ax_pred.plot(
            pred_times,
            pred_probs,
            color='#2E86AB',
            linewidth=2,
            label='Raw Predictions',
            marker='o',
            markersize=3,
            alpha=0.8
        )
        
        # Threshold line
        ax_pred.axhline(
            y=threshold,
            color='#E63946',
            linestyle='--',
            linewidth=2,
            label=f'Threshold ({threshold})',
            alpha=0.7
        )
        
        # Fill area above threshold
        ax_pred.fill_between(
            pred_times,
            pred_probs,
            threshold,
            where=np.array(pred_probs) >= threshold,
            alpha=0.2,
            color='#06D6A0',
            label='Above Threshold'
        )
    
    ax_pred.set_xlim(0, audio_duration)
    ax_pred.set_ylim(0, 1.0)
    ax_pred.set_ylabel('Probability', fontsize=10)
    ax_pred.set_title('Model Predictions (Emergency Vehicle)', fontsize=11, pad=10)
    ax_pred.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax_pred.grid(True, alpha=0.3, linestyle='--')
    
    # ===========================================================================
    # 3. EVENTS TIMELINE (detected events as colored boxes)
    # ===========================================================================
    ax_events.set_xlim(0, audio_duration)
    ax_events.set_ylim(0, 1)
    ax_events.set_yticks([])
    ax_events.set_xlabel('Time (s)', fontsize=10)
    ax_events.set_title(f'Detected Events ({len(events)} events)', fontsize=11, pad=10)
    ax_events.grid(True, alpha=0.3, axis='x', linestyle='--')
    
    if events:
        for i, event_data in enumerate(events):
            # Events are tuples: (onset, offset, confidence)
            onset, offset, confidence = event_data
            duration = offset - onset
            
            # Event box
            rect = patches.Rectangle(
                (onset, 0.2),
                duration,
                0.6,
                linewidth=2,
                edgecolor='#E63946',
                facecolor='#06D6A0',
                alpha=0.6
            )
            ax_events.add_patch(rect)
            
            # Event number (circle at center)
            center_x = onset + duration / 2
            ax_events.text(
                center_x,
                0.5,
                f'{i+1}',
                ha='center',
                va='center',
                fontsize=10,
                fontweight='bold',
                color='white',
                bbox=dict(boxstyle='circle', facecolor='#E63946', alpha=0.9, pad=0.3)
            )
            
            # Probability (black bold text, positioned to the right of the circle)
            prob_x = center_x + 0.15  # Slight offset to the right
            ax_events.text(
                prob_x,
                0.5,
                f'{confidence:.3f}',
                ha='left',
                va='center',
                fontsize=9,
                fontweight='bold',
                color='black'
            )
    else:
        ax_events.text(
            audio_duration / 2,
            0.5,
            'No events detected',
            ha='center',
            va='center',
            fontsize=11,
            color='gray',
            style='italic'
        )
    
    # ===========================================================================
    # Final layout adjustment
    # ===========================================================================
    plt.tight_layout()
    
    # Save plot
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_dataset_results_summary(
    results_csv_path: str,
    output_path: str,
    figsize: Tuple[int, int] = (12, 10)
):
    """
    Create summary visualization for dataset processing results.
    
    Args:
        results_csv_path: Path to results CSV file
        output_path: Path to save summary plot
        figsize: Figure size
    """
    import pandas as pd
    
    logging.info("Creating dataset results summary plot...")
    
    # Load results
    df = pd.read_csv(results_csv_path)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # 1. SED Metrics Distribution (Segment-based F1)
    ax = axes[0, 0]
    if 'seg_f1' in df.columns:
        df['seg_f1'].hist(bins=20, ax=ax, color='#2E86AB', alpha=0.7, edgecolor='black')
        ax.axvline(df['seg_f1'].mean(), color='#E63946', linestyle='--', linewidth=2, label=f'Mean: {df["seg_f1"].mean():.3f}')
        ax.set_xlabel('Segment F1 Score')
        ax.set_ylabel('Frequency')
        ax.set_title('Segment-based F1 Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 2. Event-based F1 Distribution
    ax = axes[0, 1]
    if 'evt_f1' in df.columns:
        df['evt_f1'].hist(bins=20, ax=ax, color='#06D6A0', alpha=0.7, edgecolor='black')
        ax.axvline(df['evt_f1'].mean(), color='#E63946', linestyle='--', linewidth=2, label=f'Mean: {df["evt_f1"].mean():.3f}')
        ax.set_xlabel('Event F1 Score')
        ax.set_ylabel('Frequency')
        ax.set_title('Event-based F1 Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 3. Throughput Distribution
    ax = axes[1, 0]
    if 'throughput' in df.columns:
        df['throughput'].hist(bins=20, ax=ax, color='#F77F00', alpha=0.7, edgecolor='black')
        ax.axvline(df['throughput'].mean(), color='#E63946', linestyle='--', linewidth=2, label=f'Mean: {df["throughput"].mean():.2f}x')
        ax.set_xlabel('Throughput (× real-time)')
        ax.set_ylabel('Frequency')
        ax.set_title('Processing Throughput Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 4. CPU Usage Distribution
    ax = axes[1, 1]
    if 'cpu_mean' in df.columns:
        df['cpu_mean'].hist(bins=20, ax=ax, color='#A23E48', alpha=0.7, edgecolor='black')
        ax.axvline(df['cpu_mean'].mean(), color='#E63946', linestyle='--', linewidth=2, label=f'Mean: {df["cpu_mean"].mean():.1f}%')
        ax.set_xlabel('CPU Usage (%)')
        ax.set_ylabel('Frequency')
        ax.set_title('CPU Usage Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Summary plot saved: {output_path}")
