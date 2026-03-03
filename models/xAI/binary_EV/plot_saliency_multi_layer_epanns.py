"""
Multi-Layer Saliency Visualization for E-PANNs
==============================================
Self-contained script that extracts and visualizes saliency maps
from all E-PANNs layers using Vanilla Backpropagation.

Generates: epanns_TP_multi_layer.svg
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
# Guided Backpropagation Implementation (EXACT COPY from working code)
# ==============================================================================

class GuidedBackprop:
    """
    Guided Backpropagation - copied exactly from models/xAI/methods/gradients.py
    """
    
    def __init__(self, model: nn.Module, target_layer: str):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.forward_relu_outputs = []
        self.hooks = []
        
        self._register_hooks()
    
    def _register_hooks(self):
        """Register hooks on ReLUs and target layer."""
        # TEMPORARY: Disable ReLU hooks to test if they block gradients
        # # Hook all ReLU layers
        # for module in self.model.modules():
        #     if isinstance(module, nn.ReLU):
        #         self.hooks.append(
        #             module.register_forward_hook(self._relu_forward_hook)
        #         )
        #         self.hooks.append(
        #             module.register_full_backward_hook(self._relu_backward_hook)
        #         )
        
        # Hook target layer only
        layer = self._get_layer(self.target_layer)
        if layer is not None:
            self.hooks.append(
                layer.register_full_backward_hook(self._save_gradient)
            )
    
    def _get_layer(self, name: str):
        """Get layer by name."""
        print(f"    [DEBUG] Looking for layer: {name}")
        parts = name.split('.')
        layer = self.model
        for i, part in enumerate(parts):
            if hasattr(layer, part):
                layer = getattr(layer, part)
                print(f"    [DEBUG]   Part {i}: '{part}' -> {type(layer).__name__}")
            else:
                try:
                    layer = layer[int(part)]
                    print(f"    [DEBUG]   Part {i}: '{part}' (index) -> {type(layer).__name__}")
                except (ValueError, TypeError, IndexError):
                    print(f"    [DEBUG]   Part {i}: '{part}' NOT FOUND!")
                    return None
        print(f"    [DEBUG] Found layer: {type(layer).__name__}")
        return layer
    
    def _relu_forward_hook(self, module, input_tensor, output_tensor):
        """Store ReLU forward outputs."""
        self.forward_relu_outputs.append(output_tensor)
    
    def _relu_backward_hook(self, module, grad_input, grad_output):
        """Modify ReLU backward pass (Guided Backprop)."""
        if len(self.forward_relu_outputs) > 0:
            forward_output = self.forward_relu_outputs.pop()
            
            # Guided backprop: only positive gradients through positive activations
            modified_grad_input = grad_input[0].clone()
            modified_grad_input[forward_output <= 0] = 0
            modified_grad_input = torch.clamp(modified_grad_input, min=0)
            
            return (modified_grad_input,)
        return grad_input
    
    def _save_gradient(self, module, grad_input, grad_output):
        """Hook to save gradients at target layer."""
        print(f"    [HOOK BACKWARD] Called for layer: {self.target_layer}")
        print(f"    [HOOK BACKWARD] grad_output type: {type(grad_output)}, len: {len(grad_output) if isinstance(grad_output, tuple) else 'N/A'}")
        if grad_output[0] is not None:
            self.gradients = grad_output[0].detach()
            print(f"    [HOOK BACKWARD] Gradient shape: {self.gradients.shape}")
            print(f"    [HOOK BACKWARD] Gradient range: [{self.gradients.min():.6f}, {self.gradients.max():.6f}]")
            non_zero = (self.gradients.abs() > 1e-8).sum().item()
            total = self.gradients.numel()
            print(f"    [HOOK BACKWARD] Non-zero: {non_zero}/{total} ({100*non_zero/total:.1f}%)")
        else:
            print(f"    [HOOK BACKWARD] WARNING: grad_output[0] is None!")
    
    def generate(self, waveform: torch.Tensor, target_class: int, device: str = "cuda"):
        """
        Generate guided backprop saliency map.
        
        Args:
            waveform: Input (1D or batched)
            target_class: Target class index
            device: Device
            
        Returns:
            saliency: Normalized saliency map (T, F)
        """
        # Reset
        self.forward_relu_outputs = []
        self.gradients = None
        
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        waveform = waveform.to(device)
        waveform.requires_grad = True
        
        print(f"    [GENERATE] Input shape: {waveform.shape}, requires_grad: {waveform.requires_grad}")
        
        # Forward
        print(f"    [GENERATE] Starting forward pass...")
        output = self.model(waveform)
        if isinstance(output, dict):
            output = output.get('clipwise_output', output)
        
        print(f"    [GENERATE] Output shape: {output.shape}")
        
        # Backward
        class_score = output[:, target_class]
        print(f"    [GENERATE] Class score: {class_score.item():.4f}, target_class: {target_class}")
        print(f"    [GENERATE] Starting backward pass...")
        self.model.zero_grad()
        class_score.backward()  # Don't use .sum() for single batch
        print(f"    [GENERATE] Backward pass completed")
        
        # Get gradients
        if self.gradients is None:
            raise RuntimeError(f"No gradients captured from layer: {self.target_layer}")
        
        print(f"    [GENERATE] Processing gradients...")
        grads = self.gradients.cpu().numpy()
        
        # DEBUG: Print raw gradient stats BEFORE processing
        print(f"    [DEBUG RAW] Shape before processing: {grads.shape}")
        print(f"    [DEBUG RAW] Min: {grads.min():.10f}, Max: {grads.max():.10f}")
        print(f"    [DEBUG RAW] Mean: {grads.mean():.10f}, Std: {grads.std():.10f}")
        
        # Process: (1, C, T, F) or (1, 1, T, F) -> (T, F)
        grads = np.abs(grads.squeeze())
        
        # Handle 3D case
        if grads.ndim == 3:
            grads = grads.mean(axis=0)  # Average over channels
        
        # Handle (F, T) vs (T, F)
        if grads.ndim == 2 and grads.shape[0] < grads.shape[1]:
            # Likely (F, T) -> transpose
            if grads.shape[0] < 200 and grads.shape[1] > 500:
                grads = grads.T
        
        # Normalize
        g_min, g_max = grads.min(), grads.max()
        print(f"    [DEBUG NORM] Before normalize - Min: {g_min:.15f}, Max: {g_max:.15f}")
        if g_max - g_min > 1e-15:  # Use very small threshold for tiny gradients
            saliency = (grads - g_min) / (g_max - g_min)
        else:
            saliency = grads / (g_max + 1e-20)  # Scale by max instead of normalizing
        
        print(f"    [DEBUG NORM] After normalize - Min: {saliency.min():.15f}, Max: {saliency.max():.15f}")
        
        return saliency
    
    def cleanup(self):
        """Explicitly remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def __del__(self):
        """Remove hooks on deletion."""
        self.cleanup()


# ==============================================================================
# Spectrogram Extraction
# ==============================================================================

def extract_epanns_spectrogram(model, waveform, device="cuda"):
    """Extract log-mel spectrogram from E-PANNs."""
    model.eval()
    
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    
    waveform = waveform.to(device)
    
    with torch.no_grad():
        # EPANNs: model.model.spectrogram_extractor -> model.model.logmel_extractor
        spec = model.model.spectrogram_extractor(waveform)
        logmel = model.model.logmel_extractor(spec)
    
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
        display_name = layer_name.replace('model.', '').replace('_', ' ').title()
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
    
    model_cfg = config['models']['epanns']
    sample_cfg = config['samples']['TP']
    target_class = config['target_class']
    
    print("=" * 70)
    print("E-PANNs Multi-Layer Saliency Analysis")
    print("=" * 70)
    
    # Load model
    print("\n[1/4] Loading E-PANNs model...")
    model = load_model('epanns', pretrained=False)
    
    # Load pretrained weights
    model.load_pretrained(model_cfg['checkpoint_pretrained'])
    
    # Load finetuned weights (PyTorch state_dict)
    print(f"  Loading finetuned: {Path(model_cfg['checkpoint_finetuned']).name}")
    finetuned_checkpoint = torch.load(model_cfg['checkpoint_finetuned'], map_location='cpu')
    # Extract state_dict from checkpoint
    finetuned_state = finetuned_checkpoint.get('pytorch_model_state_dict', finetuned_checkpoint)
    model.load_state_dict(finetuned_state, strict=True)
    
    model.to(device)
    # DON'T set to eval() - keep in train mode for gradients
    # model.eval()
    model.train() #  IMPORTANT for saliency extraction
    
    # DEBUG: Check which parameters have requires_grad
    n_total = sum(1 for _ in model.parameters())
    n_trainable = sum(1 for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_trainable}/{n_total} trainable")
    
    print("✓ Model loaded (in train mode for saliency)")
    
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
    spectrogram = extract_epanns_spectrogram(model, waveform, device)
    print(f"  Spectrogram shape: {spectrogram.shape}")
    
    # IMPORTANT: Enable gradients on ALL parameters for saliency computation
    for param in model.parameters():
        param.requires_grad = True
    
    # Get saliency for each layer
    saliency_maps = {}
    for layer_name in model_cfg['layers']:
        print(f"  Processing: {layer_name}")
        explainer = GuidedBackprop(model, layer_name)
        saliency = explainer.generate(waveform, target_class, device)
        
        print(f"    Shape: {saliency.shape}, Min: {saliency.min():.4f}, Max: {saliency.max():.4f}, Non-zero: {(saliency > 0.01).sum()}/{saliency.size}")
        
        # Resize to match spectrogram if needed
        if saliency.shape != spectrogram.shape:
            from scipy.ndimage import zoom
            zoom_factors = (spectrogram.shape[0] / saliency.shape[0],
                           spectrogram.shape[1] / saliency.shape[1])
            saliency = zoom(saliency, zoom_factors, order=1)
            print(f"    Resized to: {saliency.shape}")
        
        saliency_maps[layer_name] = saliency
        
        # Clean up hooks explicitly before next iteration
        explainer.cleanup()
        del explainer
        torch.cuda.empty_cache()
    
    print(f"✓ Generated {len(saliency_maps)} saliency maps")
    
    # Visualize
    print("\n[4/4] Generating visualization...")
    save_path = output_dir / f"epanns_{sample_cfg['name']}_multi_layer.svg"
    
    plot_multi_layer_saliency(
        saliency_maps=saliency_maps,
        spectrogram=spectrogram,
        title=f"E-PANNs Multi-Layer Saliency Analysis: {sample_cfg['description']}",
        save_path=str(save_path),
        dpi=config['dpi']
    )
    
    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
