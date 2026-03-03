"""
Model Comparison Visualizer
============================
Side-by-side comparison of multiple models' explanations.
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from typing import Dict, List, Optional, Tuple
import os


class ModelComparisonVisualizer:
    """Create comprehensive comparison visualizations across models."""
    
    def __init__(self, dpi: int = 600, output_format: str = "svg"):
        """
        Args:
            dpi: Resolution
            output_format: Output format
        """
        self.dpi = dpi
        self.output_format = output_format
    
    def plot_side_by_side(
        self,
        spectrograms: Dict[str, np.ndarray],
        saliency_maps: Dict[str, np.ndarray],
        sample_name: str = "",
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (20, 12),
        spec_cmap: str = "viridis",
        saliency_cmap: str = "hot"
    ):
        """
        Plot side-by-side comparison of all models.
        
        Args:
            spectrograms: Dict of {model_name: spectrogram}
            saliency_maps: Dict of {model_name: saliency_map}
            sample_name: Sample identifier
            save_path: Save path
            figsize: Figure size
            spec_cmap: Spectrogram colormap
            saliency_cmap: Saliency colormap
        """
        models = list(saliency_maps.keys())
        n_models = len(models)
        
        fig = plt.figure(figsize=figsize, dpi=self.dpi)
        gs = gridspec.GridSpec(n_models, 3, figure=fig, hspace=0.3, wspace=0.25)
        
        for idx, model_name in enumerate(models):
            spec = spectrograms.get(model_name)
            saliency = saliency_maps[model_name]
            
            # Spectrogram
            ax1 = fig.add_subplot(gs[idx, 0])
            if spec is not None:
                spec_norm = np.clip((spec + 80) / 80, 0, 1)
                ax1.imshow(spec_norm.T if spec_norm.ndim > 1 else spec_norm.reshape(-1, 1).T,
                          origin='lower', aspect='auto', cmap=spec_cmap, interpolation='bilinear')
            ax1.set_ylabel(model_name, fontsize=13, fontweight='bold')
            if idx == 0:
                ax1.set_title('Input Spectrogram', fontsize=14, fontweight='bold')
            if idx == n_models - 1:
                ax1.set_xlabel('Time Frames', fontsize=11)
            else:
                ax1.set_xticklabels([])
            
            # Saliency map
            ax2 = fig.add_subplot(gs[idx, 1])
            saliency_2d = saliency.T if saliency.ndim > 1 else saliency.reshape(-1, 1).T
            im2 = ax2.imshow(saliency_2d, origin='lower', aspect='auto', cmap=saliency_cmap, interpolation='bilinear')
            if idx == 0:
                ax2.set_title('Saliency Map', fontsize=14, fontweight='bold')
            if idx == n_models - 1:
                ax2.set_xlabel('Time Frames', fontsize=11)
            else:
                ax2.set_xticklabels([])
            ax2.set_yticklabels([])
            
            # Overlay
            ax3 = fig.add_subplot(gs[idx, 2])
            if spec is not None:
                ax3.imshow(spec_norm.T if spec_norm.ndim > 1 else spec_norm.reshape(-1, 1).T,
                          origin='lower', aspect='auto', cmap=spec_cmap, interpolation='bilinear')
                ax3.imshow(saliency_2d, origin='lower', aspect='auto', cmap=saliency_cmap,
                          alpha=0.6, interpolation='bilinear')
            else:
                ax3.imshow(saliency_2d, origin='lower', aspect='auto', cmap=saliency_cmap, interpolation='bilinear')
            
            if idx == 0:
                ax3.set_title('Overlay', fontsize=14, fontweight='bold')
            if idx == n_models - 1:
                ax3.set_xlabel('Time Frames', fontsize=11)
            else:
                ax3.set_xticklabels([])
            ax3.set_yticklabels([])
            
            # Colorbar for saliency
            if idx == n_models - 1:
                cbar = plt.colorbar(im2, ax=[ax2, ax3], orientation='horizontal', pad=0.15, shrink=0.8)
                cbar.set_label('Saliency', fontsize=10)
        
        fig.suptitle(f'Multi-Model XAI Comparison: {sample_name}', fontsize=16, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, format=self.output_format, dpi=self.dpi, bbox_inches='tight')
            plt.close(fig)
        else:
            return fig
    
    def plot_time_series_comparison(
        self,
        saliency_maps: Dict[str, np.ndarray],
        sample_name: str = "",
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (16, 6)
    ):
        """
        Plot time series comparison of saliency across models.
        
        Args:
            saliency_maps: Dict of {model_name: saliency_map}
            sample_name: Sample name
            save_path: Save path
            figsize: Figure size
        """
        fig, ax = plt.subplots(figsize=figsize, dpi=self.dpi)
        
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
        
        for idx, (model_name, saliency) in enumerate(saliency_maps.items()):
            # Average over frequency if 2D
            if saliency.ndim > 1 and saliency.shape[-1] > 1:
                time_series = saliency.mean(axis=-1)
            else:
                time_series = saliency.flatten()
            
            color = colors[idx % len(colors)]
            ax.plot(time_series, label=model_name, linewidth=2.5, color=color, alpha=0.8)
            ax.fill_between(range(len(time_series)), time_series, alpha=0.2, color=color)
        
        ax.set_xlabel('Time Frames', fontsize=13)
        ax.set_ylabel('Average Saliency', fontsize=13)
        ax.set_title(f'Temporal Saliency Comparison: {sample_name}', fontsize=15, fontweight='bold')
        ax.legend(loc='best', fontsize=12, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, format=self.output_format, dpi=self.dpi, bbox_inches='tight')
            plt.close(fig)
        else:
            return fig
    
    def plot_heatmap_grid(
        self,
        saliency_maps: Dict[str, np.ndarray],
        sample_name: str = "",
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (18, 5),
        cmap: str = "hot"
    ):
        """
        Plot grid of saliency heatmaps.
        
        Args:
            saliency_maps: Dict of saliency maps
            sample_name: Sample name
            save_path: Save path
            figsize: Figure size
            cmap: Colormap
        """
        models = list(saliency_maps.keys())
        n_models = len(models)
        
        fig, axes = plt.subplots(1, n_models, figsize=figsize, dpi=self.dpi, sharey=True)
        
        if n_models == 1:
            axes = [axes]
        
        for ax, model_name in zip(axes, models):
            saliency = saliency_maps[model_name]
            saliency_2d = saliency.T if saliency.ndim > 1 else saliency.reshape(-1, 1).T
            
            im = ax.imshow(saliency_2d, origin='lower', aspect='auto', cmap=cmap, interpolation='bilinear')
            ax.set_title(model_name, fontsize=13, fontweight='bold')
            ax.set_xlabel('Time Frames', fontsize=11)
            
            if ax == axes[0]:
                ax.set_ylabel('Frequency Bins', fontsize=11)
            
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        fig.suptitle(f'Saliency Heatmaps: {sample_name}', fontsize=15, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, format=self.output_format, dpi=self.dpi, bbox_inches='tight')
            plt.close(fig)
        else:
            return fig
    
    def plot_comprehensive_analysis(
        self,
        data_dict: Dict[str, Dict[str, np.ndarray]],
        metrics_dict: Dict[str, Dict[str, float]],
        sample_name: str = "",
        sample_type: str = "",
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (24, 16)
    ):
        """
        Create comprehensive multi-panel analysis figure.
        
        Args:
            data_dict: {model_name: {'spectrogram': arr, 'saliency': arr, ...}}
            metrics_dict: {model_name: {'metric': value, ...}}
            sample_name: Sample name
            sample_type: TP or TN
            save_path: Save path
            figsize: Figure size
        """
        models = list(data_dict.keys())
        n_models = len(models)
        
        fig = plt.figure(figsize=figsize, dpi=self.dpi)
        gs = gridspec.GridSpec(n_models + 2, 3, figure=fig, hspace=0.4, wspace=0.3,
                               height_ratios=[1] * n_models + [0.8, 1])
        
        # Row for each model
        for idx, model_name in enumerate(models):
            model_data = data_dict[model_name]
            spec = model_data.get('spectrogram')
            saliency = model_data['saliency']
            
            # Spectrogram
            ax_spec = fig.add_subplot(gs[idx, 0])
            if spec is not None:
                spec_norm = np.clip((spec + 80) / 80, 0, 1)
                ax_spec.imshow(spec_norm.T if spec_norm.ndim > 1 else spec_norm.reshape(-1, 1).T, origin='lower', aspect='auto', cmap='viridis')
            ax_spec.set_ylabel(model_name, fontsize=12, fontweight='bold')
            if idx == 0:
                ax_spec.set_title('Input', fontsize=13, fontweight='bold')
            if idx < n_models - 1:
                ax_spec.set_xticklabels([])
            else:
                ax_spec.set_xlabel('Time', fontsize=10)
            
            # Saliency
            ax_sal = fig.add_subplot(gs[idx, 1])
            sal_2d = saliency.T if saliency.ndim > 1 else saliency.reshape(-1, 1).T
            im_sal = ax_sal.imshow(sal_2d, origin='lower', aspect='auto', cmap='hot')
            if idx == 0:
                ax_sal.set_title('Saliency', fontsize=13, fontweight='bold')
            if idx < n_models - 1:
                ax_sal.set_xticklabels([])
            else:
                ax_sal.set_xlabel('Time', fontsize=10)
            ax_sal.set_yticklabels([])
            
            # Time series
            ax_ts = fig.add_subplot(gs[idx, 2])
            ts = saliency.mean(axis=-1) if saliency.ndim > 1 and saliency.shape[-1] > 1 else saliency.flatten()
            ax_ts.plot(ts, linewidth=2, color='#e74c3c')
            ax_ts.fill_between(range(len(ts)), ts, alpha=0.3, color='#e74c3c')
            if idx == 0:
                ax_ts.set_title('Temporal Profile', fontsize=13, fontweight='bold')
            if idx < n_models - 1:
                ax_ts.set_xticklabels([])
            else:
                ax_ts.set_xlabel('Time', fontsize=10)
            ax_ts.set_ylabel('Avg.', fontsize=9)
            ax_ts.grid(True, alpha=0.3)
        
        # Comparison time series
        ax_comp = fig.add_subplot(gs[n_models, :])
        colors = ['#e74c3c', '#3498db', '#2ecc71']
        for idx, model_name in enumerate(models):
            saliency = data_dict[model_name]['saliency']
            ts = saliency.mean(axis=-1) if saliency.ndim > 1 and saliency.shape[-1] > 1 else saliency.flatten()
            ax_comp.plot(ts, label=model_name, linewidth=2, color=colors[idx % len(colors)])
        ax_comp.set_xlabel('Time Frames', fontsize=11)
        ax_comp.set_ylabel('Saliency', fontsize=11)
        ax_comp.set_title('Temporal Comparison', fontsize=13, fontweight='bold')
        ax_comp.legend(fontsize=10)
        ax_comp.grid(True, alpha=0.3)
        
        # Metrics
        ax_metrics = fig.add_subplot(gs[n_models + 1, :])
        metric_names = ['sparsity', 'peak_to_mean', 'deletion_auc', 'average_drop']
        x = np.arange(len(models))
        width = 0.2
        
        for i, metric in enumerate(metric_names):
            values = [metrics_dict[m].get(metric, 0) for m in models]
            offset = (i - len(metric_names) / 2) * width + width / 2
            ax_metrics.bar(x + offset, values, width, label=metric.replace('_', ' ').title(), alpha=0.8)
        
        ax_metrics.set_xlabel('Model', fontsize=11)
        ax_metrics.set_ylabel('Value', fontsize=11)
        ax_metrics.set_title('Metrics Summary', fontsize=13, fontweight='bold')
        ax_metrics.set_xticks(x)
        ax_metrics.set_xticklabels(models)
        ax_metrics.legend(fontsize=9, loc='upper right')
        ax_metrics.grid(True, alpha=0.3, axis='y')
        
        fig.suptitle(f'Comprehensive XAI Analysis: {sample_name} ({sample_type})', 
                    fontsize=17, fontweight='bold', y=0.995)
        
        if save_path:
            plt.savefig(save_path, format=self.output_format, dpi=self.dpi, bbox_inches='tight')
            plt.close(fig)
        else:
            return fig
    
    def plot_unified_filterbank_comparison(
        self,
        filterbank_data: Dict[str, Dict[str, np.ndarray]],
        title: str = "Mel Filterbank Comparison Across Models",
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (20, 12)
    ):
        """
        Create unified filterbank comparison plot.
        3 columns (models) x 3 rows (learned heatmap, standard heatmap, spectral profiles).
        
        Args:
            filterbank_data: {model_name: {learned_fb, standard_fb, learned_centroids, standard_centroids, ...}}
            title: Plot title
            save_path: Save path
            figsize: Figure size
        """
        models = list(filterbank_data.keys())
        n_models = len(models)
        
        fig = plt.figure(figsize=figsize, dpi=self.dpi)
        gs = gridspec.GridSpec(3, n_models, figure=fig, hspace=0.35, wspace=0.25)
        
        for col_idx, model_name in enumerate(models):
            data = filterbank_data[model_name]
            learned_fb = data['learned_filterbank']
            standard_fb = data['standard_filterbank']
            learned_centroids = data['learned_centroids']
            standard_centroids = data['standard_centroids']
            
            # Row 0: Learned filterbank heatmap
            ax0 = fig.add_subplot(gs[0, col_idx])
            im0 = ax0.imshow(learned_fb, aspect='auto', cmap='viridis', interpolation='bilinear')
            ax0.set_title(f'{model_name.upper()} Filterbank', fontsize=12, fontweight='bold')
            ax0.set_ylabel('Mel Bins' if col_idx == 0 else '', fontsize=10)
            ax0.set_xlabel('Frequency Bins', fontsize=9)
            plt.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)
            
            # Row 1: Standard (reference) filterbank heatmap
            ax1 = fig.add_subplot(gs[1, col_idx])
            im1 = ax1.imshow(standard_fb, aspect='auto', cmap='viridis', interpolation='bilinear')
            ax1.set_title(f'Reference ({"torchaudio" if model_name == "ced" else "torchlibrosa"})', 
                         fontsize=11, fontweight='bold')
            ax1.set_ylabel('Mel Bins' if col_idx == 0 else '', fontsize=10)
            ax1.set_xlabel('Frequency Bins', fontsize=9)
            plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
            
            # Row 2: Spectral profiles (centroid frequencies)
            ax2 = fig.add_subplot(gs[2, col_idx])
            mel_indices = np.arange(len(learned_centroids))
            ax2.plot(learned_centroids, mel_indices, 'o-', label='Learned', 
                    linewidth=2.5, markersize=4, color='#e74c3c', alpha=0.8)
            ax2.plot(standard_centroids, mel_indices, 's--', label='Standard', 
                    linewidth=2.5, markersize=4, color='#3498db', alpha=0.8)
            ax2.set_title('Filter Centroids', fontsize=11, fontweight='bold')
            ax2.set_xlabel('Centroid Frequency (Hz)', fontsize=10)
            ax2.set_ylabel('Mel Bin Index' if col_idx == 0 else '', fontsize=10)
            ax2.legend(fontsize=9, loc='best')
            ax2.grid(True, alpha=0.3, linestyle='--')
            ax2.set_xlim(left=0)
        
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, format=self.output_format, dpi=self.dpi, bbox_inches='tight')
            plt.close(fig)
        else:
            return fig
