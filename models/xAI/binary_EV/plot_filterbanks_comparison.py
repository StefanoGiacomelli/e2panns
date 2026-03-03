"""
Mel-Filterbanks Comparison Across Models
=========================================
Self-contained script that compares learned mel-filterbanks (from finetuned models)
vs reference implementations (torchaudio/torchlibrosa).

Generates: filterbanks_comparison.svg
Layout: 4 rows x 3 columns
- Row 0: Mel filterbank frequency response (both learned and reference)
- Row 1: Learned filterbank heatmap (from finetuned model)
- Row 2: Reference filterbank heatmap (torchaudio/torchlibrosa)
- Row 3: Centroid comparison (learned vs reference)
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
# Filterbank Parameters (extracted from models)
# ==============================================================================

FILTERBANK_PARAMS = {
    'epanns': {
        'sample_rate': 32000,
        'n_fft': 1024,  # Verified from melW.shape[0] = 513
        'hop_length': 320,
        'n_mels': 64,
        'fmin': 50,
        'fmax': 14000,
        'window_fn': torch.hann_window,
        'power': 2.0
    },
    'ced': {
        'sample_rate': 16000,
        'n_fft': 512,
        'hop_length': 160,
        'n_mels': 64,
        'fmin': 0,
        'fmax': 8000,
        'window_fn': torch.hann_window,
        'power': 2.0
    },
    'clap': {
        'sample_rate': 48000,
        'n_fft': 1024,
        'hop_length': 480,
        'n_mels': 64,
        'fmin': 0,
        'fmax': 14000,
        'window_fn': torch.hann_window,
        'power': 2.0
    }
}


# ==============================================================================
# Filterbank Extraction
# ==============================================================================

def get_learned_filterbank(model, model_name: str):
    """
    Extract learned mel filterbank from model.
    
    Returns:
        mel_fb: (n_mels, n_freqs) learned mel filterbank
        freqs: Frequency bins in Hz
    """
    params = FILTERBANK_PARAMS[model_name]
    
    if model_name == 'epanns':
        # EPANNs uses torchlibrosa LogmelFilterBank
        # Extract melW: (n_freqs, n_mels) -> transpose to (n_mels, n_freqs)
        mel_fb = model.model.logmel_extractor.melW.T.detach().cpu().numpy()
        
    elif model_name == 'ced':
        # CED uses torchaudio MelSpectrogram (first module in Sequential)
        # Extract mel_scale.fb: (n_freqs, n_mels) -> transpose
        mel_spec_module = model.front_end[0]  # MelSpectrogram is first in Sequential
        mel_fb = mel_spec_module.mel_scale.fb.T.detach().cpu().numpy()
        
    elif model_name == 'clap':
        # CLAP uses torchlibrosa LogmelFilterBank
        # Extract melW: (n_freqs, n_mels) -> transpose
        mel_fb = model.model.audio_branch.logmel_extractor.melW.T.detach().cpu().numpy()
    
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Frequency bins
    freqs = np.linspace(0, params['sample_rate'] / 2, params['n_fft'] // 2 + 1)
    
    return mel_fb, freqs


def get_reference_filterbank(model_name: str):
    """
    Get reference mel filterbank using torchaudio (not trained).
    
    Returns:
        mel_fb: (n_mels, n_freqs) reference mel filterbank
        freqs: Frequency bins in Hz
    """
    params = FILTERBANK_PARAMS[model_name]
    
    # Create mel filterbank using torchaudio
    mel_fb = torchaudio.functional.melscale_fbanks(
        n_freqs=params['n_fft'] // 2 + 1,
        f_min=params['fmin'],
        f_max=params['fmax'],
        n_mels=params['n_mels'],
        sample_rate=params['sample_rate'],
        norm='slaney'
    ).T.numpy()  # (n_mels, n_freqs)
    
    # Frequency bins
    freqs = np.linspace(0, params['sample_rate'] / 2, params['n_fft'] // 2 + 1)
    
    return mel_fb, freqs


def compute_filter_centroids(mel_fb: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    """
    Compute centroid (center of mass) for each mel filter.
    
    Args:
        mel_fb: (n_mels, n_freqs) filterbank
        freqs: Frequency bins in Hz
        
    Returns:
        centroids: (n_mels,) centroid frequencies in Hz
    """
    centroids = []
    for i in range(mel_fb.shape[0]):
        # Weighted average: sum(freq * weight) / sum(weight)
        filter_weights = mel_fb[i, :]
        if filter_weights.sum() > 0:
            centroid = np.sum(freqs * filter_weights) / filter_weights.sum()
        else:
            centroid = 0.0
        centroids.append(centroid)
    
    return np.array(centroids)


# ==============================================================================
# Visualization
# ==============================================================================

def plot_filterbanks_comparison(
    models_data: dict,
    save_path: str,
    dpi: int = 600
):
    """
    Plot learned vs reference mel filterbanks comparison.
    
    Args:
        models_data: Dict with keys ['epanns', 'ced', 'clap'], each containing:
                    - 'learned_fb': Learned filterbank
                    - 'reference_fb': Reference filterbank
                    - 'freqs': Frequency bins
                    - 'learned_centroids': Centroid frequencies (learned)
                    - 'reference_centroids': Centroid frequencies (reference)
        save_path: Output path
        dpi: Figure DPI
    """
    model_names = ['epanns', 'ced', 'clap']
    model_labels = ['E-PANNs (32kHz, Finetuned)', 'CED-Base (16kHz, Finetuned)', 'CLAP (48kHz, Finetuned)']
    reference_impl = ['torchlibrosa', 'torchaudio', 'torchlibrosa']  # Reference implementation used
    
    # Create 4 rows × 3 columns
    fig, axes = plt.subplots(4, 3, figsize=(18, 16), dpi=dpi)
    
    for col, (model_name, label, ref_impl) in enumerate(zip(model_names, model_labels, reference_impl)):
        data = models_data[model_name]
        
        # Row 0: Mel filterbank frequency response (learned + reference overlay)
        ax = axes[0, col]
        learned_fb = data['learned_fb']  # (n_mels, n_freqs)
        reference_fb = data['reference_fb']
        freqs = data['freqs']
        
        # Plot learned filters
        for i in range(learned_fb.shape[0]):
            ax.plot(freqs, learned_fb[i, :], alpha=0.4, linewidth=0.8, color='blue', label='Learned' if i == 0 else '')
        
        # Plot reference filters (dashed)
        for i in range(reference_fb.shape[0]):
            ax.plot(freqs, reference_fb[i, :], alpha=0.3, linewidth=0.6, color='red', linestyle='--', label='Reference' if i == 0 else '')
        
        ax.set_xlabel('Frequency (Hz)', fontsize=10)
        ax.set_ylabel('Magnitude', fontsize=10)
        ax.set_title(f'{label}\nFilterbank Response', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([freqs[1], freqs[-1]])  # Start from freqs[1] to avoid log(0)
        ax.set_xscale('log')  # Logarithmic scale for better visualization
        if col == 0:
            ax.legend(fontsize=8, loc='upper left')
        
        # Row 1: Learned filterbank heatmap
        ax = axes[1, col]
        im = ax.imshow(learned_fb, origin='lower', aspect='auto', cmap='viridis', interpolation='bilinear')
        if col == 0:
            ax.set_ylabel('Learned\nFilterbank\n(Mel bins)', fontsize=11, fontweight='bold')
        else:
            ax.set_yticklabels([])
        ax.set_xlabel('Frequency bins', fontsize=9)
        
        # Colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax)
        if col == 2:
            cbar.set_label('Weight', fontsize=9)
        
        # Row 2: Reference filterbank heatmap
        ax = axes[2, col]
        im = ax.imshow(reference_fb, origin='lower', aspect='auto', cmap='viridis', interpolation='bilinear')
        if col == 0:
            ax.set_ylabel('Reference\nFilterbank\n(Mel bins)', fontsize=11, fontweight='bold')
        else:
            ax.set_yticklabels([])
        ax.set_xlabel('Frequency bins', fontsize=9)
        ax.set_title(f'Reference ({ref_impl})', fontsize=10, style='italic')
        
        # Colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax)
        if col == 2:
            cbar.set_label('Weight', fontsize=9)
        
        # Row 3: Centroid comparison
        ax = axes[3, col]
        learned_centroids = data['learned_centroids']
        reference_centroids = data['reference_centroids']
        mel_bins = np.arange(len(learned_centroids))
        
        ax.plot(mel_bins, learned_centroids, 'o-', label='Learned (Finetuned)', color='blue', markersize=4, linewidth=1.5)
        ax.plot(mel_bins, reference_centroids, 's--', label='Reference', color='red', markersize=3, linewidth=1)
        
        ax.set_xlabel('Mel bin index', fontsize=10)
        ax.set_ylabel('Centroid Frequency (Hz)', fontsize=10)
        if col == 0:
            ax.legend(fontsize=9, loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_title('Filter Centroid Comparison', fontsize=10, fontweight='bold')
    
    fig.suptitle('Mel-Filterbank Comparison: Learned (Finetuned) vs Reference', 
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
    
    print("=" * 70)
    print("Mel-Filterbank Comparison Analysis")
    print("=" * 70)
    
    models_data = {}
    
    # Process each model
    for model_name in ['epanns', 'ced', 'clap']:
        model_cfg = config['models'][model_name]
        
        print(f"\n[{model_name.upper()}]")
        
        # Load model
        print(f"  Loading model...")
        model = load_model(model_name, pretrained=False)
        model.load_pretrained(model_cfg['checkpoint_pretrained'])
        
        # Load finetuned weights
        print(f"  Loading finetuned: {Path(model_cfg['checkpoint_finetuned']).name}")
        finetuned_checkpoint = torch.load(model_cfg['checkpoint_finetuned'], map_location='cpu')
        finetuned_state = finetuned_checkpoint.get('pytorch_model_state_dict', finetuned_checkpoint)
        model.load_state_dict(finetuned_state, strict=True)
        
        model.to(device)
        model.eval()
        print(f"  ✓ Model loaded (finetuned)")
        
        # Extract learned filterbank
        print(f"  Extracting learned filterbank...")
        learned_fb, freqs = get_learned_filterbank(model, model_name)
        print(f"  ✓ Learned filterbank: {learned_fb.shape}")
        
        # Get reference filterbank
        print(f"  Extracting reference filterbank...")
        reference_fb, _ = get_reference_filterbank(model_name)
        print(f"  ✓ Reference filterbank: {reference_fb.shape}")
        
        # Compute centroids
        print(f"  Computing centroids...")
        learned_centroids = compute_filter_centroids(learned_fb, freqs)
        reference_centroids = compute_filter_centroids(reference_fb, freqs)
        
        # Calculate difference statistics
        centroid_diff = np.abs(learned_centroids - reference_centroids)
        mean_diff = centroid_diff.mean()
        max_diff = centroid_diff.max()
        print(f"  ✓ Centroid diff: mean={mean_diff:.2f} Hz, max={max_diff:.2f} Hz")
        
        # Store data
        models_data[model_name] = {
            'learned_fb': learned_fb,
            'reference_fb': reference_fb,
            'freqs': freqs,
            'learned_centroids': learned_centroids,
            'reference_centroids': reference_centroids
        }
    
    # Visualize
    print("\n[VISUALIZATION]")
    save_path = output_dir / "filterbanks_comparison.svg"
    
    plot_filterbanks_comparison(
        models_data=models_data,
        save_path=str(save_path),
        dpi=config['dpi']
    )
    
    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("=" * 70)
    print("\nNote: Using FINETUNED models (trained on AS-EV_v2)")
    print("Comparison shows learned filterbanks vs reference implementations")


if __name__ == "__main__":
    main()
