"""SED Processing Pipeline
===========================
Unified processing pipeline for single audio file SED analysis.

Integrates:
- Audio loading and preprocessing
- Model inference with adaptive window
- SED metrics computation (if GT available)
- Performance monitoring
- Event post-processing

Author: Stefano Giacomelli - Ph.D. candidate in ICT (DISIM dpt. - University of L'Aquila)
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from sed_system.core import load_audio, run_inference_simulation
from sed_system.monitoring import SEDMetrics, PerformanceMonitor


def process_audio_file(
    model: torch.nn.Module,
    audio_path: str,
    config: dict,
    target_sr: int,
    class_index: int,
    ground_truth_events: Optional[List[Tuple[float, float]]] = None,
    return_spectrogram: bool = False
) -> Dict:
    """
    Process a single audio file for Sound Event Detection.
    
    Args:
        model: Loaded PyTorch model (EPANNs/CED/CLAP)
        audio_path: Path to audio file
        config: Configuration dictionary with inference settings
        target_sr: Target sample rate for model
        class_index: Target class index for detection
        ground_truth_events: Optional list of (onset, offset) GT events
        return_spectrogram: If True, compute and return spectrogram for visualization
    
    Returns:
        Dictionary with:
            - 'predictions': List of prediction dicts with timestamp and probability
            - 'events': List of detected events (onset, offset)
            - 'sed_metrics': SED metrics dict (if GT provided)
            - 'performance': Performance metrics dict
            - 'audio_info': Audio metadata
            - 'spectrogram': Mel-spectrogram (if return_spectrogram=True)
    """
    # Extract config parameters
    threshold = config['inference']['threshold']
    chunk_duration = config['inference']['chunk_duration']
    buffer_duration = config['inference']['buffer_duration']
    device = config['model']['device']
    
    # Adaptive window settings
    adaptive_config = config['inference']['adaptive_window']
    adaptive_enabled = adaptive_config['enabled']
    
    if adaptive_enabled:
        frame_duration_min = chunk_duration
        frame_duration_max = adaptive_config['frame_duration_max']
        adapt_coeff = adaptive_config['adapt_coeff']
    else:
        frame_duration_min = chunk_duration
        frame_duration_max = chunk_duration
        adapt_coeff = 0.0
    
    # Load and resample audio
    audio_data = load_audio(audio_path, target_sr=target_sr)
    audio_duration = len(audio_data) / target_sr
    
    #Pad to multiple of chunk size
    chunk_samples = int(chunk_duration * target_sr)
    if len(audio_data) % chunk_samples != 0:
        num_chunks = (len(audio_data) // chunk_samples) + 1
        target_length = num_chunks * chunk_samples
        padding = target_length - len(audio_data)
        audio_data = np.pad(audio_data, (0, padding), mode='constant')
    
    # Initialize monitoring
    sed_metrics = SEDMetrics(class_name="Emergency_vehicle", threshold=threshold)
    perf_monitor = PerformanceMonitor()
    perf_monitor.set_audio_duration(audio_duration)
    
    # Start monitoring
    perf_monitor.start()
    
    # Run inference (frame-by-frame logging inside inference engine)
    results = run_inference_simulation(
        audio_data=audio_data,
        model=model,
        class_index=class_index,
        sampling_rate=target_sr,
        buffer_duration=buffer_duration,
        chunk_duration=chunk_duration,
        frame_duration_min=frame_duration_min,
        frame_duration_max=frame_duration_max,
        threshold=threshold,
        adapt_width_coeff=adapt_coeff,
        device=device
    )
    
    # Stop monitoring
    perf_monitor.stop()
    perf_stats = perf_monitor.get_stats()
    
    # Process results for SED metrics
    for result in results:
        sed_metrics.add_prediction(
            timestamp=result['timestamp'],
            probability=result['probability']
        )
    
    # Get detected events
    detected_events = sed_metrics.get_detected_events()
    logging.info(f"Detected {len(detected_events)} events")
    
    # Compute SED metrics if ground truth provided
    sed_metrics_dict = None
    if ground_truth_events is not None:
        logging.info("Computing SED metrics against ground truth...")
        
        # Get segment and event resolution from config
        segment_resolution = config['sed_metrics']['segment_time_resolution']
        event_tolerance = config['sed_metrics']['event_tolerance']
        
        sed_metrics_dict = sed_metrics.compute_metrics(
            ground_truth_events=ground_truth_events,
            time_resolution=segment_resolution,
            tolerance=event_tolerance
        )
        
        seg = sed_metrics_dict['segment_based']
        evt = sed_metrics_dict['event_based']
        logging.info(f"SED Metrics - Segment F1: {seg['f1']:.3f}, Event F1: {evt['f1']:.3f}")
    
    # Compute spectrogram if requested
    spectrogram = None
    if return_spectrogram:
        logging.info("Computing spectrogram for visualization...")
        spectrogram = compute_mel_spectrogram(audio_data, target_sr)
    
    # Build result dictionary
    result_dict = {
        'predictions': results,
        'events': detected_events,
        'sed_metrics': sed_metrics_dict,
        'performance': perf_stats,
        'audio_info': {
            'path': audio_path,
            'duration': audio_duration,
            'sample_rate': target_sr,
            'num_inferences': len(results)
        },
        'spectrogram': spectrogram
    }
    
    return result_dict


def compute_mel_spectrogram(audio_data: np.ndarray, sr: int, n_mels: int = 128) -> np.ndarray:
    """
    Compute Mel-spectrogram for visualization.
    
    Args:
        audio_data: Audio waveform
        sr: Sample rate
        n_mels: Number of mel bands
    
    Returns:
        Mel-spectrogram array (n_mels, time_frames)
    """
    import librosa
    
    mel_spec = librosa.feature.melspectrogram(
        y=audio_data,
        sr=sr,
        n_mels=n_mels,
        n_fft=2048,
        hop_length=512,
        fmax=sr // 2
    )
    
    # Convert to dB scale
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    return mel_spec_db
