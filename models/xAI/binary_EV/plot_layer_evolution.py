"""
Layer Evolution Analysis
=========================
Self-contained script that visualizes how feature representations
evolve through the layers of each model.

Generates: 
- epanns_layer_evolution_TP.svg
- ced_layer_evolution_TP.svg
- clap_layer_evolution_TP.svg

Shows: Progressive transformation of spectro-temporal features
"""

import sys
import yaml
import torch
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
# Feature Extraction
# ==============================================================================

def extract_layer_features(model, model_name: str, waveform: torch.Tensor, 
                           layer_names: list, device: str = "cuda"):
    """
    Extract intermediate feature representations from specified layers.
    
    Args:
        model: Model instance
        model_name: Name of the model
        waveform: Input waveform
        layer_names: List of layer names to extract from
        device: Device
        
    Returns:
        features: Dict mapping layer names to feature tensors
    """
    model.eval()
    
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    
    waveform = waveform.to(device)
    
    features = {}
    
    def make_hook(name):
        def hook(module, input, output):
            # Store output features
            if isinstance(output, torch.Tensor):
                features[name] = output.detach().cpu()
            elif isinstance(output, tuple):
                features[name] = output[0].detach().cpu()
        return hook
    
    # Register hooks
    hooks = []
    for layer_name in layer_names:
        # Navigate to the layer
        parts = layer_name.split('.')
        layer = model
        for part in parts:
            if part.isdigit():
                layer = layer[int(part)]
            else:
                layer = getattr(layer, part)
        
        hook = layer.register_forward_hook(make_hook(layer_name))
        hooks.append(hook)
    
    # Forward pass
    with torch.no_grad():
        _ = model(waveform)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    return features


def process_features_for_visualization(features: torch.Tensor, model_name: str):
    """
    Process feature tensor for visualization.
    
    Args:
        features: Feature tensor (various shapes depending on layer)
        model_name: Model name
        
    Returns:
        vis_features: (T, F) array for visualization
    """
    # Remove batch dimension
    if features.dim() == 4:
        # (B, C, H, W) or (B, C, T, F)
        features = features.squeeze(0)  # (C, H, W)
        
        # Average over channels or take first few
        if features.shape[0] > 1:
            # Use mean or max pooling over channels
            vis_features = features.mean(dim=0).numpy()  # (H, W)
        else:
            vis_features = features.squeeze(0).numpy()
            
    elif features.dim() == 3:
        # (B, T, F) or (C, T, F)
        if features.shape[0] == 1:
            vis_features = features.squeeze(0).numpy()  # (T, F)
        else:
            # Could be (C, T, F) - average over channels
            vis_features = features.mean(dim=0).numpy()
            
    elif features.dim() == 2:
        # Already (T, F)
        vis_features = features.numpy()
        
    else:
        # Fallback
        vis_features = features.squeeze().numpy()
        if vis_features.ndim == 1:
            vis_features = vis_features.reshape(-1, 1)
    
    # Ensure we have (T, F) shape
    if vis_features.ndim != 2:
        raise ValueError(f"Cannot process features with shape {vis_features.shape}")
    
    return vis_features


# ==============================================================================
# Visualization
# ==============================================================================

def plot_layer_evolution(
    features_dict: dict,
    layer_labels: list,
    model_label: str,
    save_path: str,
    dpi: int = 600
):
    """
    Plot layer evolution showing progressive feature transformation.
    
    Args:
        features_dict: Dict mapping layer names to processed features (T, F)
        layer_labels: Human-readable labels for layers
        model_label: Model name for title
        save_path: Output path
        dpi: Figure DPI
    """
    n_layers = len(features_dict)
    
    # Create figure with n_layers rows, 1 column (no sharex to handle different aspect ratios)
    fig, axes = plt.subplots(n_layers, 1, figsize=(16, 3 * n_layers), dpi=dpi)
    
    if n_layers == 1:
        axes = [axes]
    
    layer_names = list(features_dict.keys())
    
    for i, (layer_name, label) in enumerate(zip(layer_names, layer_labels)):
        ax = axes[i]
        features = features_dict[layer_name]  # (T, F)
        
        # Normalize for better visualization
        p_low, p_high = np.percentile(features, [2, 98])
        features_norm = np.clip((features - p_low) / (p_high - p_low + 1e-8), 0, 1)
        
        # Plot
        im = ax.imshow(features_norm.T, origin='lower', aspect='auto', 
                      cmap='viridis', interpolation='bilinear')
        
        ax.set_ylabel(f'{label}\n(Feature dim)', fontsize=11, fontweight='bold')
        
        if i == n_layers - 1:
            ax.set_xlabel('Time (frames)', fontsize=12, fontweight='bold')
        
        # Colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label('Activation', fontsize=9)
        
        # Add stats as text
        stats_text = f'Shape: {features.shape}, Mean: {features.mean():.3f}, Std: {features.std():.3f}'
        ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, 
               fontsize=8, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    fig.suptitle(f'{model_label}: Layer-by-Layer Feature Evolution (TP Sample)', 
                 fontsize=16, fontweight='bold', y=0.995)
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
    
    tp_cfg = config['samples']['TP']
    
    print("=" * 70)
    print("Layer Evolution Analysis")
    print("=" * 70)
    
    # Process each model
    for model_name in ['epanns', 'ced', 'clap']:
        model_cfg = config['models'][model_name]
        
        print(f"\n[{model_name.upper()}]")
        
        # Load model
        print(f"  Loading model...")
        model = load_model(model_name, pretrained=False)
        model.load_pretrained(model_cfg['checkpoint_pretrained'])
        # Load finetuned weights
        finetuned_checkpoint = torch.load(model_cfg['checkpoint_finetuned'], map_location='cpu')
        finetuned_state = finetuned_checkpoint.get('pytorch_model_state_dict', finetuned_checkpoint)
        model.load_state_dict(finetuned_state, strict=True)
        model.to(device)
        model.eval()
        print(f"  ✓ Model loaded (finetuned)")
        
        # Load TP audio sample
        print(f"  Loading TP audio sample...")
        tp_waveform, sr = torchaudio.load(tp_cfg['path'])
        tp_waveform = tp_waveform.mean(dim=0)
        if sr != model_cfg['sample_rate']:
            resampler = torchaudio.transforms.Resample(sr, model_cfg['sample_rate'])
            tp_waveform = resampler(tp_waveform)
        print(f"  ✓ Audio loaded: {tp_waveform.shape[0] / model_cfg['sample_rate']:.2f}s")
        
        # Get layer names from config
        layer_names = model_cfg.get('layers', [])
        if not layer_names:
            print(f"  ⚠ No layers specified in config for {model_name}, skipping")
            continue
        
        print(f"  Extracting features from {len(layer_names)} layers...")
        
        # Extract features
        features_dict = extract_layer_features(model, model_name, tp_waveform, 
                                               layer_names, device)
        
        # Process features for visualization
        processed_features = {}
        layer_labels = []
        
        for i, layer_name in enumerate(layer_names):
            if layer_name not in features_dict:
                print(f"    ⚠ Layer {layer_name} not found in features")
                continue
            
            features = features_dict[layer_name]
            print(f"    Layer {i+1}/{len(layer_names)}: {layer_name}, shape={features.shape}")
            
            try:
                vis_features = process_features_for_visualization(features, model_name)
                processed_features[layer_name] = vis_features
                
                # Create human-readable label
                if 'blocks.' in layer_name:
                    # CED transformer blocks
                    block_num = layer_name.split('.')[-1]
                    layer_label = f"Transformer Block {block_num}"
                elif 'layers.' in layer_name:
                    # CLAP HTS layers
                    layer_num = layer_name.split('.')[-1]
                    layer_label = f"HTS Layer {layer_num}"
                elif 'conv_block' in layer_name:
                    # EPANNs conv blocks
                    block_num = layer_name.split('conv_block')[-1]
                    layer_label = f"Conv Block {block_num}"
                elif 'front_end' in layer_name:
                    layer_label = "Front-End (Mel Spectrogram)"
                elif 'logmel_extractor' in layer_name:
                    layer_label = "Log-Mel Extractor"
                elif 'patch_embed' in layer_name:
                    layer_label = "Patch Embedding"
                elif 'init_bn' in layer_name:
                    layer_label = "Initial Batch Norm"
                elif 'bn0' in layer_name:
                    layer_label = "Batch Norm 0"
                else:
                    # Fallback: use last part of layer name
                    layer_label = layer_name.split('.')[-1]
                
                layer_labels.append(layer_label)
                
                print(f"      ✓ Processed to shape {vis_features.shape}")
                
            except Exception as e:
                print(f"      ✗ Failed to process: {e}")
                continue
        
        if not processed_features:
            print(f"  ✗ No features could be processed for {model_name}")
            continue
        
        # Visualize
        print(f"  Generating visualization...")
        save_path = output_dir / f"{model_name}_layer_evolution_TP.svg"
        
        model_label = {
            'epanns': 'E-PANNs',
            'ced': 'CED-Base',
            'clap': 'CLAP'
        }[model_name]
        
        plot_layer_evolution(
            features_dict=processed_features,
            layer_labels=layer_labels,
            model_label=model_label,
            save_path=str(save_path),
            dpi=config['dpi']
        )
    
    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
