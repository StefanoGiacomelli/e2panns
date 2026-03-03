"""
Plotting Utilities for XAI
===========================
Individual plot functions for saliency maps, spectrograms, etc.
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from typing import Optional, Tuple, Dict
import os

# Set matplotlib backend for SVG output
mpl.use('Agg')


class SaliencyPlotter:
    """Plot saliency maps with various styles."""
    
    def __init__(self, dpi: int = 600, output_format: str = "svg"):
        """
        Args:
            dpi: Resolution
            output_format: 'svg', 'png', etc.
        """
        self.dpi = dpi
        self.output_format = output_format
    
    def plot_overlay(
        self,
        spectrogram: np.ndarray,
        saliency_map: np.ndarray,
        title: str = "",
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (14, 5),
        spec_cmap: str = "viridis",
        saliency_cmap: str = "hot",
        alpha: float = 0.6
    ):
        """
        Plot saliency overlay on spectrogram.
        
        Args:
            spectrogram: Spectrogram (T, F)
            saliency_map: Saliency map (T, F)
            title: Plot title
            save_path: Path to save figure
            figsize: Figure size
            spec_cmap: Colormap for spectrogram
            saliency_cmap: Colormap for saliency
            alpha: Transparency of saliency overlay
        """
        fig, ax = plt.subplots(figsize=figsize, dpi=self.dpi)
        
        # Normalize spectrogram (assume dB range)
        spec_norm = np.clip((spectrogram + 80) / 80, 0, 1)
        
        # Plot spectrogram
        ax.imshow(
            spec_norm.T,
            origin='lower',
            aspect='auto',
            cmap=spec_cmap,
            interpolation='bilinear'
        )
        
        # Overlay saliency
        ax.imshow(
            saliency_map.T,
            origin='lower',
            aspect='auto',
            cmap=saliency_cmap,
            alpha=alpha,
            interpolation='bilinear'
        )
        
        ax.set_xlabel('Time Frames', fontsize=12)
        ax.set_ylabel('Frequency Bins', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        plt.colorbar(ax.images[1], ax=ax, label='Saliency', fraction=0.046, pad=0.04)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, format=self.output_format, dpi=self.dpi, bbox_inches='tight')
            plt.close(fig)
        else:
            return fig
    
    def plot_time_series(
        self,
        saliency_map: np.ndarray,
        title: str = "",
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (14, 3)
    ):
        """
        Plot saliency as time series (averaged over frequency).
        
        Args:
            saliency_map: Saliency map (T, F)
            title: Title
            save_path: Save path
            figsize: Figure size
        """
        fig, ax = plt.subplots(figsize=figsize, dpi=self.dpi)
        
        # Average over frequency dimension
        if saliency_map.ndim > 1 and saliency_map.shape[-1] > 1:
            time_series = saliency_map.mean(axis=-1)
        else:
            time_series = saliency_map.flatten()
        
        ax.plot(time_series, linewidth=2, color='#e74c3c')
        ax.fill_between(range(len(time_series)), time_series, alpha=0.3, color='#e74c3c')
        
        ax.set_xlabel('Time Frames', fontsize=12)
        ax.set_ylabel('Average Saliency', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, format=self.output_format, dpi=self.dpi, bbox_inches='tight')
            plt.close(fig)
        else:
            return fig
    
    def plot_multi_layer_saliency(
        self,
        saliency_maps: Dict[str, np.ndarray],
        spectrogram: Optional[np.ndarray] = None,
        title: str = "",
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (16, 12),
        cmap: str = "gray_r"
    ):
        """
        Plot saliency maps from multiple layers stacked vertically.
        
        Args:
            saliency_maps: Dict of {layer_name: saliency_array}
            spectrogram: Optional spectrogram to show at top
            title: Plot title
            save_path: Save path
            figsize: Figure size
            cmap: Colormap for saliency
        """
        n_layers = len(saliency_maps)
        n_rows = n_layers + (1 if spectrogram is not None else 0)
        
        fig, axes = plt.subplots(n_rows, 1, figsize=figsize, dpi=self.dpi, sharex=True)
        
        if n_rows == 1:
            axes = [axes]
        
        row_idx = 0
        
        # Plot spectrogram at top if provided
        if spectrogram is not None:
            ax = axes[row_idx]
            spec_norm = np.clip((spectrogram + 80) / 80, 0, 1)
            # spectrogram is (T, F) - transpose to (F, T) for imshow
            # so X-axis shows time and Y-axis shows frequency
            ax.imshow(spec_norm.T, origin='lower', aspect='auto', cmap='viridis', interpolation='bilinear')
            ax.set_ylabel('Frequency', fontsize=10, fontweight='bold')
            ax.set_yticks([])
            if row_idx == 0:
                ax.set_title('Input Spectrogram', fontsize=11)
            row_idx += 1
        
        # Plot each layer's saliency
        for layer_name, saliency in saliency_maps.items():
            ax = axes[row_idx]
            # saliency is (T, F) - transpose to (F, T) for imshow
            # X-axis = time (horizontal), Y-axis = frequency (vertical)
            im = ax.imshow(saliency.T, origin='lower', aspect='auto', cmap=cmap, vmin=0, vmax=1, interpolation='bilinear')
            
            # Clean up layer name for display
            display_name = layer_name.replace('model.', '').replace('audio_branch.', '').replace('blocks.', 'block_')
            ax.set_ylabel(f'{display_name}\n(Freq)', fontsize=8, fontweight='bold')
            
            # Add colorbar on the side
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="2%", pad=0.05)
            plt.colorbar(im, cax=cax, label='Saliency')
            
            row_idx += 1
        
        # Add time label only on bottom
        axes[-1].set_xlabel('Time (frames) →', fontsize=12, fontweight='bold')
        fig.suptitle(title, fontsize=15, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, format=self.output_format, dpi=self.dpi, bbox_inches='tight')
            plt.close(fig)
        else:
            return fig
    
    def plot_attention_maps(
        self,
        attention_maps: list,
        title: str = "",
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (18, 12),
        cmap: str = "viridis",
        num_to_show: int = 12
    ):
        """
        Plot attention maps from transformer layers.
        
        Args:
            attention_maps: List of attention matrices (seq, seq)
            title: Title
            save_path: Save path
            figsize: Figure size
            cmap: Colormap
            num_to_show: Number of layers to visualize
        """
        n_layers = min(len(attention_maps), num_to_show)
        
        # Create grid
        ncols = 4
        nrows = (n_layers + ncols - 1) // ncols
        
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=self.dpi)
        axes = axes.flatten() if nrows > 1 or ncols > 1 else [axes]
        
        for idx in range(n_layers):
            ax = axes[idx]
            attn = attention_maps[idx]
            
            # If attention is 2D (seq, seq), show it
            if isinstance(attn, np.ndarray) and attn.ndim == 2:
                im = ax.imshow(attn, cmap=cmap, interpolation='nearest')
                ax.set_title(f'Layer {idx}', fontsize=10, fontweight='bold')
                ax.set_xlabel('Key', fontsize=8)
                ax.set_ylabel('Query', fontsize=8)
                plt.colorbar(im, ax=ax, fraction=0.046)
            else:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'Layer {idx}', fontsize=10)
            
            ax.tick_params(labelsize=7)
        
        # Hide unused subplots
        for idx in range(n_layers, len(axes)):
            axes[idx].axis('off')
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, format=self.output_format, dpi=self.dpi, bbox_inches='tight')
            plt.close(fig)
        else:
            return fig


class SpectrogramPlotter:
    """Plot spectrograms and mel-spectrograms."""
    
    def __init__(self, dpi: int = 600, output_format: str = "svg"):
        self.dpi = dpi
        self.output_format = output_format
    
    def plot_spectrogram(
        self,
        spectrogram: np.ndarray,
        title: str = "",
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (14, 5),
        cmap: str = "viridis",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None
    ):
        """
        Plot spectrogram.
        
        Args:
            spectrogram: Spectrogram array (T, F)
            title: Title
            save_path: Save path
            figsize: Figure size
            cmap: Colormap
            vmin, vmax: Value range for colormap
        """
        fig, ax = plt.subplots(figsize=figsize, dpi=self.dpi)
        
        im = ax.imshow(
            spectrogram.T,
            origin='lower',
            aspect='auto',
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation='bilinear'
        )
        
        ax.set_xlabel('Time Frames', fontsize=12)
        ax.set_ylabel('Frequency Bins', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        plt.colorbar(im, ax=ax, label='Magnitude (dB)', fraction=0.046, pad=0.04)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, format=self.output_format, dpi=self.dpi, bbox_inches='tight')
            plt.close(fig)
        else:
            return fig
    
    def plot_comparison(
        self,
        spec1: np.ndarray,
        spec2: np.ndarray,
        title1: str = "Spectrogram 1",
        title2: str = "Spectrogram 2",
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (18, 5),
        cmap: str = "viridis"
    ):
        """
        Plot two spectrograms side by side.
        
        Args:
            spec1, spec2: Spectrograms
            title1, title2: Titles
            save_path: Save path
            figsize: Figure size
            cmap: Colormap
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=self.dpi)
        
        for ax, spec, title in zip(axes, [spec1, spec2], [title1, title2]):
            im = ax.imshow(
                spec.T,
                origin='lower',
                aspect='auto',
                cmap=cmap,
                interpolation='bilinear'
            )
            ax.set_xlabel('Time Frames', fontsize=11)
            ax.set_ylabel('Frequency Bins', fontsize=11)
            ax.set_title(title, fontsize=13, fontweight='bold')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, format=self.output_format, dpi=self.dpi, bbox_inches='tight')
            plt.close(fig)
        else:
            return fig


class ComparisonPlotter:
    """Plot comparisons between methods or models."""
    
    def __init__(self, dpi: int = 600, output_format: str = "svg"):
        self.dpi = dpi
        self.output_format = output_format
    
    def plot_difference_map(
        self,
        saliency_tp: np.ndarray,
        saliency_tn: np.ndarray,
        title: str = "Difference Map (TP - TN)",
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (14, 5),
        cmap: str = "RdBu_r"
    ):
        """
        Plot difference between TP and TN saliency maps.
        
        Args:
            saliency_tp: Saliency for TP sample
            saliency_tn: Saliency for TN sample
            title: Title
            save_path: Save path
            figsize: Figure size
            cmap: Diverging colormap
        """
        fig, ax = plt.subplots(figsize=figsize, dpi=self.dpi)
        
        # Align shapes if needed
        if saliency_tp.shape != saliency_tn.shape:
            min_shape = (
                min(saliency_tp.shape[0], saliency_tn.shape[0]),
                min(saliency_tp.shape[1], saliency_tn.shape[1]) if saliency_tp.ndim > 1 else 1
            )
            saliency_tp = saliency_tp[:min_shape[0], :min_shape[1]] if saliency_tp.ndim > 1 else saliency_tp[:min_shape[0]]
            saliency_tn = saliency_tn[:min_shape[0], :min_shape[1]] if saliency_tn.ndim > 1 else saliency_tn[:min_shape[0]]
        
        # Compute difference
        diff = saliency_tp - saliency_tn
        
        # Symmetric range around zero
        vmax = np.abs(diff).max()
        
        im = ax.imshow(
            diff.T if diff.ndim > 1 else diff.reshape(-1, 1).T,
            origin='lower',
            aspect='auto',
            cmap=cmap,
            vmin=-vmax,
            vmax=vmax,
            interpolation='bilinear'
        )
        
        ax.set_xlabel('Time Frames', fontsize=12)
        ax.set_ylabel('Frequency Bins', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        cbar = plt.colorbar(im, ax=ax, label='Difference', fraction=0.046, pad=0.04)
        cbar.ax.axhline(0, color='black', linewidth=1, linestyle='--')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, format=self.output_format, dpi=self.dpi, bbox_inches='tight')
            plt.close(fig)
        else:
            return fig
    
    def plot_filterbank_comparison(
        self,
        learned_fb: np.ndarray,
        standard_fb: np.ndarray,
        learned_centroids: np.ndarray,
        standard_centroids: np.ndarray,
        title: str = "Filterbank Comparison",
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (18, 10)
    ):
        """
        Plot comparison of learned vs standard filterbanks.
        
        Args:
            learned_fb: Learned filterbank (n_mels, n_freqs)
            standard_fb: Standard filterbank
            learned_centroids: Centroid frequencies (learned)
            standard_centroids: Centroid frequencies (standard)
            title: Title
            save_path: Save path
            figsize: Figure size
        """
        fig = plt.figure(figsize=figsize, dpi=self.dpi)
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # Plot filterbanks
        ax1 = fig.add_subplot(gs[0, 0])
        im1 = ax1.imshow(learned_fb, aspect='auto', cmap='viridis', interpolation='bilinear')
        ax1.set_title('Learned Filterbank', fontweight='bold')
        ax1.set_xlabel('Frequency Bins')
        ax1.set_ylabel('Mel Bins')
        plt.colorbar(im1, ax=ax1, fraction=0.046)
        
        ax2 = fig.add_subplot(gs[0, 1])
        im2 = ax2.imshow(standard_fb, aspect='auto', cmap='viridis', interpolation='bilinear')
        ax2.set_title('Standard Filterbank', fontweight='bold')
        ax2.set_xlabel('Frequency Bins')
        ax2.set_ylabel('Mel Bins')
        plt.colorbar(im2, ax=ax2, fraction=0.046)
        
        # Plot difference
        ax3 = fig.add_subplot(gs[1, :])
        diff = learned_fb - standard_fb
        vmax = np.abs(diff).max()
        im3 = ax3.imshow(diff, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax, interpolation='bilinear')
        ax3.set_title('Difference (Learned - Standard)', fontweight='bold')
        ax3.set_xlabel('Frequency Bins')
        ax3.set_ylabel('Mel Bins')
        cbar3 = plt.colorbar(im3, ax=ax3, fraction=0.046)
        cbar3.ax.axhline(0, color='black', linewidth=1, linestyle='--')
        
        # Plot centroids
        ax4 = fig.add_subplot(gs[2, :])
        mel_indices = np.arange(len(learned_centroids))
        ax4.plot(mel_indices, learned_centroids, 'o-', label='Learned', linewidth=2, markersize=6, color='#e74c3c')
        ax4.plot(mel_indices, standard_centroids, 's--', label='Standard', linewidth=2, markersize=6, color='#3498db')
        ax4.set_xlabel('Mel Bin Index', fontsize=12)
        ax4.set_ylabel('Centroid Frequency (Hz)', fontsize=12)
        ax4.set_title('Filter Centroids Comparison', fontweight='bold')
        ax4.legend(fontsize=11)
        ax4.grid(True, alpha=0.3, linestyle='--')
        
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
        
        if save_path:
            plt.savefig(save_path, format=self.output_format, dpi=self.dpi, bbox_inches='tight')
            plt.close(fig)
        else:
            return fig
    
    def plot_metrics_summary(
        self,
        metrics_dict: Dict[str, Dict[str, float]],
        title: str = "XAI Metrics Summary",
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (14, 8)
    ):
        """
        Plot summary of all computed metrics.
        
        Args:
            metrics_dict: Dict of {model_name: {metric_name: value}}
            title: Title
            save_path: Save path
            figsize: Figure size
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize, dpi=self.dpi)
        axes = axes.flatten()
        
        models = list(metrics_dict.keys())
        
        # Group metrics by type
        metric_groups = {
            'Consistency': ['sparsity', 'peak_to_mean'],
            'Faithfulness': ['deletion_auc', 'insertion_auc', 'average_drop'],
            'Agreement': ['cross_model_correlation'],
            'Localization': ['temporal_iou']
        }
        
        for idx, (group_name, metric_names) in enumerate(metric_groups.items()):
            if idx >= len(axes):
                break
            
            ax = axes[idx]
            
            # Collect data
            data_to_plot = []
            labels = []
            
            for metric in metric_names:
                values = [metrics_dict[m].get(metric, 0) for m in models]
                if any(v != 0 for v in values):
                    data_to_plot.append(values)
                    labels.append(metric.replace('_', ' ').title())
            
            if data_to_plot:
                x = np.arange(len(models))
                width = 0.8 / len(data_to_plot)
                
                for i, (data, label) in enumerate(zip(data_to_plot, labels)):
                    offset = (i - len(data_to_plot) / 2) * width + width / 2
                    ax.bar(x + offset, data, width, label=label, alpha=0.8)
                
                ax.set_xlabel('Model', fontsize=11)
                ax.set_ylabel('Value', fontsize=11)
                ax.set_title(group_name, fontweight='bold')
                ax.set_xticks(x)
                ax.set_xticklabels(models, rotation=15, ha='right')
                ax.legend(fontsize=9)
                ax.grid(True, alpha=0.3, axis='y', linestyle='--')
            else:
                ax.text(0.5, 0.5, f'No {group_name} metrics', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(group_name, fontweight='bold')
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, format=self.output_format, dpi=self.dpi, bbox_inches='tight')
            plt.close(fig)
        else:
            return fig
