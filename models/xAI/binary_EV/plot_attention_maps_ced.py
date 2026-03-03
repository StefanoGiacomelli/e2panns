"""
Attention Maps Visualization for CED (Transformer)
==================================================
Self-contained script that extracts and visualizes attention maps
from all CED transformer layers.

Generates: ced_attention_maps_comparison.svg
Shows: TN | TP | Difference (TP - TN) for each layer
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
# Attention Extraction
# ==============================================================================

def extract_attention_maps(model, waveform, device="cuda"):
    """
    Extract attention maps from all transformer layers.
    
    Returns:
        attention_maps: List of attention maps (one per layer)
                       Each shape: (num_heads, seq_len, seq_len)
        spectrogram: Log-mel spectrogram (T, F)
    """
    model.eval()
    
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    
    waveform = waveform.to(device)
    
    # Storage for attention weights
    attention_weights = []
    
    def attention_hook(module, input, output):
        """Hook to capture attention weights from transformer blocks."""
        # CED uses standard PyTorch MultiheadAttention
        # We need to register hook on the attention mechanism
        # For now, we'll use a simpler approach with model internals
        pass
    
    with torch.no_grad():
        # Get spectrogram first
        x = model.model.front_end(waveform)  # (batch, n_mels, time)
        spectrogram = x.squeeze(0).cpu().numpy().T  # (time, n_mels)
        
        # Forward through transformer blocks and collect attention
        # CED architecture: front_end -> blocks (list of transformer blocks)
        x = x.transpose(1, 2)  # (batch, time, n_mels)
        
        # Add positional encoding if present
        if hasattr(model.model, 'pos_encoder'):
            x = model.model.pos_encoder(x)
        
        # Go through transformer blocks
        for block in model.model.blocks:
            # Save attention before processing
            # MultiheadAttention in PyTorch doesn't expose weights by default
            # We need to use a different approach
            
            # For CED, we can directly access self-attention
            if hasattr(block, 'self_attn'):
                attn_module = block.self_attn
                
                # Manually compute attention weights
                # This is a simplified version - might need adjustment
                batch_size, seq_len, embed_dim = x.shape
                
                # Project to Q, K, V
                q = attn_module.in_proj_weight[:embed_dim] @ x.transpose(0, 1).flatten(1) + attn_module.in_proj_bias[:embed_dim]
                k = attn_module.in_proj_weight[embed_dim:2*embed_dim] @ x.transpose(0, 1).flatten(1) + attn_module.in_proj_bias[embed_dim:2*embed_dim]
                
                # Actually, let's use a simpler approach with modified forward
                # Store input for later
                x_in = x.clone()
                
                # Forward through block
                x = block(x)
            else:
                x = block(x)
    
    # For now, use a placeholder - we'll refine this
    # Let's use the working implementation from the old code
    return extract_attention_rollout(model, waveform, device)


def extract_attention_rollout(model, waveform, device="cuda"):
    """
    Extract attention using rollout method (from working code).
    
    Returns:
        attention_maps: List of (T, T) attention matrices per layer
        spectrogram: (T, F) spectrogram
    """
    model.eval()
    
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    
    waveform = waveform.to(device)
    
    attention_weights_list = []
    
    def make_hook(layer_idx):
        def hook(module, input, output):
            """Hook to capture attention weights from attn_drop."""
            # The input to attn_drop is the attention weights
            # Shape: (batch, heads, seq, seq)
            if isinstance(input, tuple):
                attn_weights = input[0]
            else:
                attn_weights = input
            attention_weights_list.append(attn_weights.detach().cpu())
        return hook
    
    # Register hooks on all transformer blocks' attention dropout
    hooks = []
    for i, block in enumerate(model.blocks):
        if hasattr(block, 'attn') and hasattr(block.attn, 'attn_drop'):
            hook = block.attn.attn_drop.register_forward_hook(make_hook(i))
            hooks.append(hook)
    
    with torch.no_grad():
        # Forward pass
        output = model(waveform)
        
        # Get spectrogram from front_end
        mel_spec = model.front_end(waveform)  # (1, 1, n_mels, T) or (1, n_mels, T)
        spectrogram = mel_spec.squeeze().cpu().numpy()  # (n_mels, T)
        spectrogram = spectrogram.T  # (T, n_mels)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Process attention weights
    attention_maps = []
    for attn in attention_weights_list:
        # Average over heads: (batch, num_heads, seq, seq) -> (seq, seq)
        attn_avg = attn.squeeze(0).mean(dim=0).numpy()
        attention_maps.append(attn_avg)
    
    return attention_maps, spectrogram


# ==============================================================================
# Visualization
# ==============================================================================

def compute_attention_rollout(attention_maps: list) -> list:
    """
    Compute attention rollout (cumulative attention composition).
    
    Args:
        attention_maps: List of (seq, seq) attention matrices
        
    Returns:
        rollout_maps: List of cumulative attention maps
    """
    rollout_maps = []
    
    # Start with identity
    result = np.eye(attention_maps[0].shape[0])
    
    for attn in attention_maps:
        # Add residual connection (self-attention has residual)
        # Attention rollout: A_rollout = A_current @ A_previous
        # With residual: (I + A) where A is attention - I (residual identity)
        attn_with_residual = 0.5 * attn + 0.5 * np.eye(attn.shape[0])
        
        # Compose with previous
        result = attn_with_residual @ result
        
        # Normalize rows to sum to 1
        result = result / (result.sum(axis=-1, keepdims=True) + 1e-15)
        
        rollout_maps.append(result.copy())
    
    return rollout_maps


def apply_attention_rollout_to_spectrogram(spectrogram: np.ndarray, rollout_attn: np.ndarray) -> np.ndarray:
    """
    Apply attention rollout to spectrogram showing what the model focuses on.
    
    Args:
        spectrogram: (T, F) spectrogram  
        rollout_attn: (seq, seq) cumulative attention
        
    Returns:
        attended_spec: (T, F) spectrogram weighted by attention
    """
    # Compute average attention received by each position
    # Sum over queries (rows) to see which positions are attended to
    attention_weights = rollout_attn.mean(axis=0)  # (seq,)
    
    # Resize to match spectrogram time dimension
    if len(attention_weights) != spectrogram.shape[0]:
        from scipy.ndimage import zoom
        zoom_factor = spectrogram.shape[0] / len(attention_weights)
        attention_weights = zoom(attention_weights, zoom_factor, order=1)
    
    # Normalize to [0, 1]
    attention_weights = (attention_weights - attention_weights.min()) / (attention_weights.max() - attention_weights.min() + 1e-15)
    
    # Expand to frequency dimension: (T,) -> (T, F)
    attention_2d = attention_weights[:, np.newaxis] * np.ones((1, spectrogram.shape[1]))
    
    # Apply as mask to show attended regions
    attended_spec = spectrogram * attention_2d
    
    return attended_spec


def plot_attention_on_spectrogram(
    tp_attentions: list,
    tp_spec: np.ndarray,
    save_path: str,
    dpi: int = 600
):
    """
    Plot attention rollout applied to TP spectrogram layer-by-layer.
    
    Shows how cumulative attention focuses on different spectro-temporal regions.
    
    Args:
        tp_attentions: List of attention maps for TP sample
        tp_spec: TP spectrogram (T, F)
        save_path: Output path
        dpi: Figure DPI
    """
    # Compute attention rollout
    rollout_attentions = compute_attention_rollout(tp_attentions)
    
    n_layers = len(rollout_attentions)
    n_rows = n_layers + 1  # +1 for input spectrogram
    
    # Create figure with 1 column, n_rows rows
    fig, axes = plt.subplots(n_rows, 1, figsize=(16, 2 * n_rows), dpi=dpi, sharex=True)
    
    if n_rows == 1:
        axes = [axes]
    
    # Row 0: Input spectrogram
    ax = axes[0]
    spec_norm = np.clip((tp_spec + 80) / 80, 0, 1)
    im = ax.imshow(spec_norm.T, origin='lower', aspect='auto', cmap='viridis', 
                  interpolation='bilinear')
    ax.set_ylabel('Input\nSpectrogram', fontsize=11, fontweight='bold')
    ax.set_title('Input Log-Mel Spectrogram (TP)', fontsize=12, fontweight='bold', pad=10)
    
    # Colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label('Amplitude', fontsize=9)
    
    # Rows 1-N: Attention rollout applied layer-by-layer
    for i, rollout_attn in enumerate(rollout_attentions):
        ax = axes[i + 1]
        
        # Apply cumulative attention to spectrogram
        attended_spec = apply_attention_rollout_to_spectrogram(tp_spec, rollout_attn)
        
        # Normalize for visualization
        attended_norm = np.clip((attended_spec + 80) / 80, 0, 1)
        
        im = ax.imshow(attended_norm.T, origin='lower', aspect='auto', cmap='viridis',
                      interpolation='bilinear')
        
        ax.set_ylabel(f'Layer {i+1}\nAttention\nRollout', fontsize=10, fontweight='bold')
        
        if i == n_layers - 1:
            ax.set_xlabel('Time (frames)', fontsize=11, fontweight='bold')
        
        # Colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label('Attended\nAmplitude', fontsize=8)
    
    fig.suptitle('CED Attention Rollout Applied to Spectrogram (Cumulative Layer-wise)', 
                 fontsize=16, fontweight='bold', y=0.998)
    plt.tight_layout()
    
    # Save
    plt.savefig(save_path, format='svg', dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    
    print(f"✓ Saved: {save_path}")


def plot_attention_comparison(
    tn_attentions: list,
    tp_attentions: list,
    tn_spec: np.ndarray,
    tp_spec: np.ndarray,
    save_path: str,
    dpi: int = 600
):
    """
    Plot attention maps in 3-column layout: TN | TP | Difference.
    
    Args:
        tn_attentions: List of attention maps for TN sample
        tp_attentions: List of attention maps for TP sample
        tn_spec: TN spectrogram (not used in plot, just for context)
        tp_spec: TP spectrogram (not used in plot, just for context)
        save_path: Output path
        dpi: Figure DPI
    """
    n_layers = len(tp_attentions)
    
    # Create figure with 3 columns, n_layers rows
    fig, axes = plt.subplots(n_layers, 3, figsize=(18, 3 * n_layers), dpi=dpi)
    
    if n_layers == 1:
        axes = axes.reshape(1, -1)
    
    for i, (tn_attn, tp_attn) in enumerate(zip(tn_attentions, tp_attentions)):
        # Compute difference
        diff_attn = tp_attn - tn_attn
        
        # Column 1: TN
        ax = axes[i, 0]
        im = ax.imshow(tn_attn, origin='lower', aspect='auto', cmap='viridis', 
                      interpolation='nearest')  # Remove vmin/vmax
        if i == 0:
            ax.set_title('TN (No Siren)', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'Layer {i+1}', fontsize=10, fontweight='bold')
        ax.set_xlabel('Time' if i == n_layers - 1 else '')
        
        # Colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.05)
        plt.colorbar(im, cax=cax)
        
        # Column 2: TP
        ax = axes[i, 1]
        im = ax.imshow(tp_attn, origin='lower', aspect='auto', cmap='viridis',
                      interpolation='nearest')  # Remove vmin/vmax
        if i == 0:
            ax.set_title('TP (With Siren)', fontsize=12, fontweight='bold')
        ax.set_ylabel('')
        ax.set_yticklabels([])
        ax.set_xlabel('Time' if i == n_layers - 1 else '')
        
        # Colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.05)
        plt.colorbar(im, cax=cax)
        
        # Column 3: Difference (TP - TN)
        ax = axes[i, 2]
        vmax_diff = max(abs(diff_attn.min()), abs(diff_attn.max()))
        im = ax.imshow(diff_attn, origin='lower', aspect='auto', cmap='RdBu_r',
                      vmin=-vmax_diff, vmax=vmax_diff, interpolation='nearest')
        if i == 0:
            ax.set_title('Difference (TP - TN)', fontsize=12, fontweight='bold')
        ax.set_ylabel('')
        ax.set_yticklabels([])
        ax.set_xlabel('Time' if i == n_layers - 1 else '')
        
        # Colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.05)
        plt.colorbar(im, cax=cax, label='Attention Change')
    
    fig.suptitle('CED Attention Maps: TN vs TP Comparison', 
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
    
    model_cfg = config['models']['ced']
    tn_cfg = config['samples']['TN']
    tp_cfg = config['samples']['TP']
    
    print("=" * 70)
    print("CED Attention Maps Analysis")
    print("=" * 70)
    
    # Load model
    print("\n[1/5] Loading CED model...")
    model = load_model('ced', pretrained=False)
    model.load_pretrained(model_cfg['checkpoint_pretrained'])
    # Load finetuned weights
    print(f"  Loading finetuned: {Path(model_cfg['checkpoint_finetuned']).name}")
    finetuned_checkpoint = torch.load(model_cfg['checkpoint_finetuned'], map_location='cpu')
    finetuned_state = finetuned_checkpoint.get('pytorch_model_state_dict', finetuned_checkpoint)
    model.load_state_dict(finetuned_state, strict=True)
    model.to(device)
    model.eval()  # Eval mode for attention extraction
    print("✓ Model loaded (finetuned on AS-EV_v2)")
    
    # Load TN audio
    print(f"\n[2/5] Loading TN sample: {tn_cfg['name']}")
    tn_waveform, sr = torchaudio.load(tn_cfg['path'])
    tn_waveform = tn_waveform.mean(dim=0)  # Convert to mono
    if sr != model_cfg['sample_rate']:
        resampler = torchaudio.transforms.Resample(sr, model_cfg['sample_rate'])
        tn_waveform = resampler(tn_waveform)
    print(f"✓ TN audio loaded: {tn_waveform.shape[0] / model_cfg['sample_rate']:.2f}s")
    
    # Load TP audio
    print(f"\n[3/5] Loading TP sample: {tp_cfg['name']}")
    tp_waveform, sr = torchaudio.load(tp_cfg['path'])
    tp_waveform = tp_waveform.mean(dim=0)  # Convert to mono
    if sr != model_cfg['sample_rate']:
        resampler = torchaudio.transforms.Resample(sr, model_cfg['sample_rate'])
        tp_waveform = resampler(tp_waveform)
    print(f"✓ TP audio loaded: {tp_waveform.shape[0] / model_cfg['sample_rate']:.2f}s")
    
    # Extract attention maps
    print("\n[4/5] Extracting attention maps...")
    print("  Processing TN sample...")
    tn_attentions, tn_spec = extract_attention_rollout(model, tn_waveform, device)
    print(f"    ✓ Extracted {len(tn_attentions)} attention layers")
    
    print("  Processing TP sample...")
    tp_attentions, tp_spec = extract_attention_rollout(model, tp_waveform, device)
    print(f"    ✓ Extracted {len(tp_attentions)} attention layers")
    
    # Visualize
    print("\n[5/5] Generating visualizations...")
    save_path = output_dir / "ced_attention_maps_comparison.svg"
    
    plot_attention_comparison(
        tn_attentions=tn_attentions,
        tp_attentions=tp_attentions,
        tn_spec=tn_spec,
        tp_spec=tp_spec,
        save_path=str(save_path),
        dpi=config['dpi']
    )
    
    # Generate second visualization: attention masks on spectrogram
    save_path_spec = output_dir / "ced_attention_on_spectrogram_TP.svg"
    plot_attention_on_spectrogram(
        tp_attentions=tp_attentions,
        tp_spec=tp_spec,
        save_path=str(save_path_spec),
        dpi=config['dpi']
    )
    
    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
