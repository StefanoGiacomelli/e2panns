"""
Temporal Dynamics of Model Predictions
=======================================
Self-contained script that analyzes how model predictions evolve over time
using sliding windows.

Generates: temporal_dynamics_comparison.svg
Shows: Frame-by-frame confidence for siren class across all 3 models
"""

import sys
import yaml
import torch
import numpy as np
import torchaudio
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import uniform_filter1d

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.lightning_models import load_model


# ==============================================================================
# Temporal Analysis
# ==============================================================================

def sliding_window_predictions(model, model_name: str, waveform: torch.Tensor, 
                               target_class: int, window_size: float, 
                               hop_size: float, sample_rate: int, device: str = "cuda"):
    """
    Compute predictions using sliding windows over the audio.
    
    Args:
        model: Model instance
        model_name: Name of the model
        waveform: Input waveform (1D tensor)
        target_class: Target class index
        window_size: Window size in seconds
        hop_size: Hop size in seconds
        sample_rate: Sample rate
        device: Device
        
    Returns:
        times: Time points (seconds)
        confidences: Confidence scores for target class
    """
    model.eval()
    
    # Convert to samples
    window_samples = int(window_size * sample_rate)
    hop_samples = int(hop_size * sample_rate)
    
    # Ensure waveform is long enough
    if waveform.shape[0] < window_samples:
        # Pad if needed
        padding = window_samples - waveform.shape[0]
        waveform = torch.nn.functional.pad(waveform, (0, padding))
    
    # Compute number of windows
    n_windows = (waveform.shape[0] - window_samples) // hop_samples + 1
    
    times = []
    confidences = []
    
    with torch.no_grad():
        for i in range(n_windows):
            start = i * hop_samples
            end = start + window_samples
            
            # Extract window
            window = waveform[start:end].unsqueeze(0).to(device)
            
            # Forward pass
            output = model(window)
            if isinstance(output, dict):
                output = output.get('clipwise_output', output)
            
            # Get confidence for target class
            confidence = torch.sigmoid(output[0, target_class]).item()
            
            # Time point (center of window)
            time = (start + window_samples / 2) / sample_rate
            
            times.append(time)
            confidences.append(confidence)
    
    return np.array(times), np.array(confidences)


# ==============================================================================
# Spectrogram Extraction (for background overlay)
# ==============================================================================

def extract_spectrogram_for_overlay(waveform: torch.Tensor, sample_rate: int):
    """
    Extract a simple mel spectrogram for background visualization.
    
    Returns:
        spectrogram: (T, F) mel spectrogram
        duration: Duration in seconds
    """
    # Use torchaudio for quick extraction
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=2048,
        hop_length=512,
        n_mels=128,
        f_min=0,
        f_max=sample_rate // 2
    )
    
    mel_spec = mel_transform(waveform)  # (n_mels, T)
    
    # Convert to dB
    mel_spec_db = torchaudio.transforms.AmplitudeToDB(top_db=80)(mel_spec)
    
    # Transpose to (T, n_mels)
    spec_np = mel_spec_db.numpy().T
    
    duration = waveform.shape[0] / sample_rate
    
    return spec_np, duration


# ==============================================================================
# Visualization
# ==============================================================================

def plot_temporal_dynamics(
    models_data_adaptive: dict,
    models_data_fixed: dict,
    tp_spectrogram: np.ndarray,
    tp_duration: float,
    save_path: str,
    dpi: int = 600
):
    """
    Plot temporal dynamics comparison with spectrogram background.
    
    Args:
        models_data_adaptive: Data with model-specific window sizes
        models_data_fixed: Data with fixed 500ms windows
        tp_spectrogram: Spectrogram of TP sample for background (T, F)
        tp_duration: Duration of TP sample in seconds
        save_path: Output path
        dpi: Figure DPI
    """
    model_names = ['epanns', 'ced', 'clap']
    model_labels = ['E-PANNs', 'CED-Base', 'CLAP']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    # Create figure with 2 rows
    fig, axes = plt.subplots(2, 1, figsize=(18, 10), dpi=dpi, sharex=True)
    
    # Prepare spectrogram for background (normalize with better contrast)
    # Use percentile-based normalization for better visibility
    p_low, p_high = np.percentile(tp_spectrogram, [5, 95])
    spec_norm = np.clip((tp_spectrogram - p_low) / (p_high - p_low + 1e-8), 0, 1)
    extent = [0, tp_duration, 0, 1]
    
    # Row 0: Adaptive window size (model-specific minimum input size)
    ax = axes[0]
    
    # Plot spectrogram as background
    ax.imshow(spec_norm.T, origin='lower', aspect='auto', cmap='gray', 
              extent=extent, alpha=0.4, interpolation='bilinear', zorder=0)
    
    for model_name, label, color in zip(model_names, model_labels, colors):
        data = models_data_adaptive[model_name]
        times = data['times']
        conf = data['conf']
        window_size = data['window_size']
        
        # Smooth with moving average for better visualization
        conf_smooth = uniform_filter1d(conf, size=min(5, len(conf)), mode='nearest')
        
        ax.plot(times, conf_smooth, label=f'{label} (window={window_size:.2f}s)', 
                linewidth=2.5, color=color, alpha=0.9, zorder=2)
        ax.fill_between(times, 0, conf_smooth, alpha=0.15, color=color, zorder=1)
    
    ax.axhline(0.5, color='red', linestyle='--', linewidth=1.5, alpha=0.7, 
               label='Decision Threshold', zorder=10)
    ax.set_ylabel('Siren Confidence', fontsize=13, fontweight='bold')
    ax.set_title('TP Sample: Adaptive Window Size (Model-Specific Minimum Input Length)', 
                 fontsize=14, fontweight='bold', pad=12)
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, zorder=0)
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlim([0, tp_duration])
    
    # Row 1: Fixed 500ms window
    ax = axes[1]
    
    # Plot spectrogram as background
    ax.imshow(spec_norm.T, origin='lower', aspect='auto', cmap='gray',
              extent=extent, alpha=0.4, interpolation='bilinear', zorder=0)
    
    for model_name, label, color in zip(model_names, model_labels, colors):
        data = models_data_fixed[model_name]
        times = data['times']
        conf = data['conf']
        
        # Smooth with moving average
        conf_smooth = uniform_filter1d(conf, size=min(5, len(conf)), mode='nearest')
        
        ax.plot(times, conf_smooth, label=label, linewidth=2.5, color=color, alpha=0.9, zorder=2)
        ax.fill_between(times, 0, conf_smooth, alpha=0.15, color=color, zorder=1)
    
    ax.axhline(0.5, color='red', linestyle='--', linewidth=1.5, alpha=0.7, 
               label='Decision Threshold', zorder=10)
    ax.set_xlabel('Time (seconds)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Siren Confidence', fontsize=13, fontweight='bold')
    ax.set_title('TP Sample: Fixed Window Size (500ms for all models)', 
                 fontsize=14, fontweight='bold', pad=12)
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, zorder=0)
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlim([0, tp_duration])
    
    fig.suptitle('Temporal Dynamics: Frame-by-Frame Siren Confidence with Spectrogram Background', 
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
    target_class = config['target_class']
    
    # Fixed window parameters for second analysis
    fixed_window_size = 0.5  # 500ms
    hop_size = 0.25  # 250ms (50% overlap)
    
    print("=" * 70)
    print("Temporal Dynamics Analysis")
    print("=" * 70)
    
    models_data_adaptive = {}
    models_data_fixed = {}
    
    # Extract TP spectrogram for background (use first model's sample rate)
    tp_spectrogram = None
    tp_duration = None
    
    # Process each model
    for model_name in ['epanns', 'ced', 'clap']:
        model_cfg = config['models'][model_name]
        
        print(f"\n[{model_name.upper()}]")
        
        # Get model-specific minimum window size
        adaptive_window_size = model_cfg.get('min_length_seconds', 1.0)
        print(f"  Minimum input length: {adaptive_window_size}s")
        print(f"  Fixed window length: {fixed_window_size}s")
        
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
        
        # Extract spectrogram for background (only once, using first model)
        if tp_spectrogram is None:
            print(f"  Extracting spectrogram for background...")
            tp_spectrogram, tp_duration = extract_spectrogram_for_overlay(
                tp_waveform, model_cfg['sample_rate']
            )
            print(f"  ✓ Spectrogram extracted: {tp_spectrogram.shape}")
        
        # Compute temporal predictions with adaptive window
        print(f"  Computing predictions (adaptive window={adaptive_window_size}s)...")
        tp_times_adaptive, tp_conf_adaptive = sliding_window_predictions(
            model, model_name, tp_waveform, target_class, 
            adaptive_window_size, hop_size, model_cfg['sample_rate'], device
        )
        print(f"    ✓ {len(tp_times_adaptive)} windows, mean={tp_conf_adaptive.mean():.3f}, max={tp_conf_adaptive.max():.3f}")
        
        # Compute temporal predictions with fixed window
        print(f"  Computing predictions (fixed window={fixed_window_size}s)...")
        tp_times_fixed, tp_conf_fixed = sliding_window_predictions(
            model, model_name, tp_waveform, target_class,
            fixed_window_size, hop_size, model_cfg['sample_rate'], device
        )
        print(f"    ✓ {len(tp_times_fixed)} windows, mean={tp_conf_fixed.mean():.3f}, max={tp_conf_fixed.max():.3f}")
        
        # Store data
        models_data_adaptive[model_name] = {
            'times': tp_times_adaptive,
            'conf': tp_conf_adaptive,
            'window_size': adaptive_window_size
        }
        
        models_data_fixed[model_name] = {
            'times': tp_times_fixed,
            'conf': tp_conf_fixed
        }
    
    # Visualize
    print("\n[VISUALIZATION]")
    save_path = output_dir / "temporal_dynamics_comparison.svg"
    
    plot_temporal_dynamics(
        models_data_adaptive=models_data_adaptive,
        models_data_fixed=models_data_fixed,
        tp_spectrogram=tp_spectrogram,
        tp_duration=tp_duration,
        save_path=str(save_path),
        dpi=config['dpi']
    )
    
    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
