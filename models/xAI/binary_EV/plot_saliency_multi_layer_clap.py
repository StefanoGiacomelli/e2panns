"""
Multi-Layer Saliency Visualization for CLAP
============================================
Self-contained script that extracts and visualizes saliency maps
from all CLAP layers using Guided Backpropagation.

Generates: clap_TP_multi_layer.svg
"""

import sys
import yaml
import torch
import torch.nn as nn
import numpy as np
import torchaudio
import matplotlib.pyplot as plt
from pathlib import Path
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.lightning_models import load_model


# ==============================================================================
# Guided Backpropagation Implementation
# ==============================================================================

class GuidedBackprop:
    """Guided Backpropagation for CNN models."""
    
    def __init__(self, model: nn.Module, target_layer: str):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.hooks = []
        
        # Register backward hook on target layer
        layer = self._get_layer(target_layer)
        hook = layer.register_full_backward_hook(self._save_gradient)
        self.hooks.append(hook)
    
    def _save_gradient(self, module, grad_input, grad_output):
        """Hook function to save gradients (backward hook)."""
        self.gradients = grad_output[0].detach()
    
    def _get_layer(self, layer_path: str):
        """Navigate to target layer by path."""
        layer = self.model
        for attr in layer_path.split('.'):
            if hasattr(layer, attr):
                layer = getattr(layer, attr)
            else:
                try:
                    layer = layer[int(attr)]
                except (ValueError, TypeError, IndexError):
                    raise ValueError(f"Cannot find layer: {layer_path}")
        return layer
    
    def generate(self, waveform: torch.Tensor, target_class: int, device: str = "cuda"):
        """
        Generate saliency map.
        
        Args:
            waveform: Input (1D or batched)
            target_class: Target class index
            device: Device
            
        Returns:
            saliency: Normalized saliency map (T, F)
        """
        # Reset gradients
        self.gradients = None
        
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        waveform = waveform.to(device)
        waveform.requires_grad = True
        
        # Forward
        output = self.model(waveform)
        if isinstance(output, dict):
            output = output.get('clipwise_output', output)
        
        # Backward
        class_score = output[:, target_class]
        self.model.zero_grad()
        class_score.backward()
        
        # Get gradients
        if self.gradients is None:
            raise RuntimeError(f"No gradients captured from layer: {self.target_layer}")
        
        grads = self.gradients.cpu().numpy()
        
        # Process: (1, C, T, F) or (1, 1, T, F) -> (T, F)
        # IMPORTANT: abs() BEFORE squeeze/mean to avoid cancellation
        grads = np.abs(grads)
        grads = grads.squeeze()
        
        # Handle 3D case (channels, time, freq)
        if grads.ndim == 3:
            grads = grads.mean(axis=0)  # Average over channels -> (time, freq)
        
        # Handle (F, T) vs (T, F)
        if grads.ndim == 2 and grads.shape[0] < grads.shape[1]:
            # Likely (F, T) -> transpose
            if grads.shape[0] < 200 and grads.shape[1] > 500:
                grads = grads.T
        
        # Normalize with very small threshold for tiny gradients
        g_min, g_max = grads.min(), grads.max()
        if g_max - g_min > 1e-15:
            saliency = (grads - g_min) / (g_max - g_min)
        else:
            saliency = grads / (g_max + 1e-20)
        
        return saliency
    
    def __del__(self):
        """Remove hooks."""
        for hook in self.hooks:
            hook.remove()


# ==============================================================================
# Spectrogram Extraction
# ==============================================================================

def extract_clap_spectrogram(model, waveform, device="cuda"):
    """Extract log-mel spectrogram from CLAP."""
    model.eval()
    
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    
    waveform = waveform.to(device)
    
    with torch.no_grad():
        # CLAP: model.model.audio_branch.logmel_extractor
        spec = model.model.audio_branch.spectrogram_extractor(waveform)
        logmel = model.model.audio_branch.logmel_extractor(spec)
    
    # (1, 1, T, F) -> (T, F)
    logmel_np = logmel.squeeze().cpu().numpy()
    
    return logmel_np


# ==============================================================================
# Visualization
# ==============================================================================

def plot_multi_layer_saliency(
    saliency_maps: dict,
    spectrogram: np.ndarray,
    title: str,
    save_path: str,
    dpi: int = 600
):
    """
    Plot multi-layer saliency maps stacked vertically.
    
    Args:
        saliency_maps: {layer_name: saliency_array (T, F)}
        spectrogram: Input spectrogram (T, F)
        title: Figure title
        save_path: Output path
        dpi: Figure DPI
    """
    n_layers = len(saliency_maps)
    n_rows = n_layers + 1  # +1 for spectrogram
    
    fig, axes = plt.subplots(n_rows, 1, figsize=(16, 2 * n_rows), dpi=dpi, sharex=True)
    
    if n_rows == 1:
        axes = [axes]
    
    # Plot spectrogram at top
    ax = axes[0]
    spec_norm = np.clip((spectrogram + 80) / 80, 0, 1)
    ax.imshow(spec_norm.T, origin='lower', aspect='auto', cmap='viridis', interpolation='bilinear')
    ax.set_ylabel('Input\nSpectrogram\n(Mel bins)', fontsize=10, fontweight='bold')
    ax.set_yticks([])
    ax.set_title('Input Log-Mel Spectrogram', fontsize=11, pad=10)
    
    # Plot each layer's saliency
    for idx, (layer_name, saliency) in enumerate(saliency_maps.items(), start=1):
        ax = axes[idx]
        
        # Saliency is (T, F) -> transpose to (F, T) for imshow
        im = ax.imshow(saliency.T, origin='lower', aspect='auto', 
                      cmap='gray_r', vmin=0, vmax=1, interpolation='bilinear')
        
        # Clean layer name
        display_name = layer_name.replace('model.audio_branch.', '').replace('_', ' ').title()
        ax.set_ylabel(f'{display_name}\n(Saliency)', fontsize=9, fontweight='bold')
        
        # Colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label('Importance', fontsize=8)
    
    # X-axis only on bottom
    axes[-1].set_xlabel('Time (frames)', fontsize=12, fontweight='bold')
    
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # Save
    plt.savefig(save_path, format='svg', dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    
    print(f"✓ Saved: {save_path}")


# ==============================================================================
# Main
# ==============================================================================

def main():
    """Main execution."""
    # Load config
    config_path = Path(__file__).parent / 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_cfg = config['models']['clap']
    sample_cfg = config['samples']['TP']
    target_class = config['target_class']
    
    print("=" * 70)
    print("CLAP Multi-Layer Saliency Analysis")
    print("=" * 70)
    
    # Load model
    print("\n[1/4] Loading CLAP model...")
    model = load_model('clap', pretrained=False)
    model.load_pretrained(model_cfg['checkpoint_pretrained'])
    # Load finetuned weights
    print(f"  Loading finetuned: {Path(model_cfg['checkpoint_finetuned']).name}")
    finetuned_checkpoint = torch.load(model_cfg['checkpoint_finetuned'], map_location='cpu')
    finetuned_state = finetuned_checkpoint.get('pytorch_model_state_dict', finetuned_checkpoint)
    model.load_state_dict(finetuned_state, strict=True)
    model.to(device)
    model.train()  # IMPORTANT: train mode for gradients
    print("✓ Model loaded (finetuned on AS-EV_v2)")
    
    # Load audio
    print(f"\n[2/4] Loading sample: {sample_cfg['name']}")
    waveform, sr = torchaudio.load(sample_cfg['path'])
    waveform = waveform.mean(dim=0)  # Convert to mono
    
    # Resample if needed
    if sr != model_cfg['sample_rate']:
        resampler = torchaudio.transforms.Resample(sr, model_cfg['sample_rate'])
        waveform = resampler(waveform)
    
    print(f"✓ Audio loaded: {waveform.shape[0] / model_cfg['sample_rate']:.2f}s")
    
    # Extract spectrogram
    print("\n[3/4] Extracting saliency maps from all layers...")
    spectrogram = extract_clap_spectrogram(model, waveform, device)
    print(f"  Spectrogram shape: {spectrogram.shape}")
    
    # Get saliency for each layer
    saliency_maps = {}
    for layer_name in model_cfg['layers']:
        print(f"  Processing: {layer_name}")
        explainer = GuidedBackprop(model, layer_name)
        saliency = explainer.generate(waveform, target_class, device)
        
        # Resize to match spectrogram if needed
        if saliency.shape != spectrogram.shape:
            from scipy.ndimage import zoom
            zoom_factors = (spectrogram.shape[0] / saliency.shape[0],
                           spectrogram.shape[1] / saliency.shape[1])
            saliency = zoom(saliency, zoom_factors, order=1)
        
        saliency_maps[layer_name] = saliency
        del explainer
    
    print(f"✓ Generated {len(saliency_maps)} saliency maps")
    
    # Visualize
    print("\n[4/4] Generating visualization...")
    save_path = output_dir / f"clap_{sample_cfg['name']}_multi_layer.svg"
    
    plot_multi_layer_saliency(
        saliency_maps=saliency_maps,
        spectrogram=spectrogram,
        title=f"CLAP Multi-Layer Saliency Analysis: {sample_cfg['description']}",
        save_path=str(save_path),
        dpi=config['dpi']
    )
    
    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
