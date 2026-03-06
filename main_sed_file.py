#!/usr/bin/env python3
"""Main Script for Single File SED Processing
==============================================
Process a single audio file for Emergency Vehicle Sound Event Detection.

Features:
- Load audio file
- Run SED inference with EPANNs/CED/CLAP
- Monitor performance (CPU, RAM, throughput)
- Visualize predictions with spectrogram overlay
- Print formatted results to terminal

Usage:
    python main_file.py configs/example_file.yaml

Author: Stefano Giacomelli - Ph.D. candidate in ICT (DISIM dpt. - University of L'Aquila)
"""

import sys
import logging
import argparse
from pathlib import Path

import yaml

from sed_system.core import load_inference_model
from sed_system.pipeline import process_audio_file
from sed_system.visualization import plot_predictions_with_spectrogram


def setup_logging(level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Silence noisy libraries in DEBUG mode
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('numba').setLevel(logging.WARNING)
    logging.getLogger('numba').setLevel(logging.WARNING)


def validate_config(config: dict):
    """Validate configuration file."""
    required_keys = ['audio_file', 'model', 'inference']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")
    
    # Check audio file exists
    audio_path = Path(config['audio_file'])
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    # Validate adaptive window params
    if config['inference']['adaptive_window']['enabled']:
        chunk_dur = config['inference']['chunk_duration']
        max_dur = config['inference']['adaptive_window']['frame_duration_max']
        
        if max_dur < chunk_dur:
            raise ValueError(f"frame_duration_max ({max_dur}) must be >= chunk_duration ({chunk_dur})")
        
        adapt_coeff = config['inference']['adaptive_window']['adapt_coeff']
        if not (0.0 <= adapt_coeff <= 1.0):
            raise ValueError(f"adapt_coeff ({adapt_coeff}) must be in [0.0, 1.0]")


def print_results(config: dict, results: dict):
    """Print formatted results to terminal."""
    audio_info = results['audio_info']
    performance = results['performance']
    events = results['events']
    
    # Header
    print("\n" + "="*80)
    print(f"SED PROCESSING: {Path(audio_info['path']).name}")
    print("="*80)
    
    # Model info
    print(f"Model: {config['model']['name'].upper()}")
    if 'checkpoint' in config['model']:
        checkpoint_name = Path(config['model']['checkpoint']).name
        print(f"Checkpoint: {checkpoint_name}")
    print(f"Audio duration: {audio_info['duration']:.1f}s")
    print(f"Sample rate: {audio_info['sample_rate']}Hz")
    
    # Inference settings
    print(f"\nINFERENCE SETTINGS:")
    print(f"  Threshold: {config['inference']['threshold']:.2f}")
    
    adaptive_config = config['inference']['adaptive_window']
    if adaptive_config['enabled']:
        print(f"  Adaptive window: ENABLED")
        print(f"    - Frame duration: {config['inference']['chunk_duration']:.3f}s - {adaptive_config['frame_duration_max']:.3f}s")
        print(f"    - Adapt coefficient: {adaptive_config['adapt_coeff']:.2f}")
    else:
        print(f"  Adaptive window: DISABLED (fixed {config['inference']['chunk_duration']:.3f}s)")
    
    print(f"  Buffer duration: {config['inference']['buffer_duration']:.1f}s")
    
    # Inference stats
    print(f"\nINFERENCE:")
    print(f"  Chunks processed: {audio_info['num_inferences']}")
    total_time = performance['total_time']
    print(f"  Inference time: {total_time:.2f}s")
    print(f"  Throughput: {performance['throughput']:.2f}x real-time")
    
    # Performance
    print(f"\nPERFORMANCE:")
    cpu_stats = performance.get('cpu', {})
    ram_stats = performance.get('ram_mb', {})
    
    if cpu_stats:
        print(f"  CPU usage: {cpu_stats.get('mean', 0):.1f}% (mean)")
        print(f"             {cpu_stats.get('min', 0):.1f}% - {cpu_stats.get('max', 0):.1f}% (min-max)")
    
    if ram_stats:
        print(f"  RAM usage: {ram_stats.get('mean', 0):.0f} MB (mean)")
        print(f"             {ram_stats.get('min', 0):.0f} - {ram_stats.get('max', 0):.0f} MB (min-max)")
    
    # Detected events
    print(f"\nDETECTED EVENTS ({len(events)} events):")
    if events:
        for i, event_data in enumerate(events, 1):
            # Events are tuples: (onset, offset, confidence)
            onset, offset, confidence = event_data
            duration = offset - onset
            
            print(f"  {i}. [{onset:.2f}s - {offset:.2f}s]  duration={duration:.2f}s  confidence={confidence:.3f}")
    else:
        print("  No events detected")
    
    # Visualization
    if config.get('visualization', {}).get('plot_predictions', False):
        plot_path = config['visualization'].get('save_plot')
        if plot_path:
            print(f"\nVISUALIZATION:")
            print(f"  Plot saved: {plot_path}")
    
    print("="*80 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Single File SED Processing")
    parser.add_argument('config', type=str, help='Path to YAML configuration file')
    args = parser.parse_args()
    
    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup logging
    log_level = config.get('logging', {}).get('level', 'INFO')
    setup_logging(log_level)
    
    # Print configuration header
    print("\n" + "="*80)
    print("SED SINGLE FILE PROCESSING")
    print("="*80)
    print(f"Config: {config_path.name}")
    print(f"Audio: {Path(config['audio_file']).name}")
    print(f"Model: {config['model']['name'].upper()} ({config['model']['device'].upper()})")
    if 'checkpoint' in config['model']:
        print(f"Checkpoint: {Path(config['model']['checkpoint']).name}")
    
    # Inference settings
    print(f"\nInference Settings:")
    print(f"  Threshold: {config['inference']['threshold']}")
    print(f"  Chunk duration: {config['inference']['chunk_duration']}s")
    
    adaptive_cfg = config['inference']['adaptive_window']
    if adaptive_cfg['enabled']:
        print(f"  Adaptive window: ENABLED (min={config['inference']['chunk_duration']}s, max={adaptive_cfg['frame_duration_max']}s, coeff={adaptive_cfg['adapt_coeff']})")
    else:
        print(f"  Adaptive window: DISABLED")
    
    print(f"\nSED Metrics:")
    print(f"  Segment resolution: {config['sed_metrics']['segment_time_resolution']}s")
    print(f"  Event tolerance: {config['sed_metrics']['event_tolerance']}s")
    
    if config.get('visualization', {}).get('plot_predictions'):
        print(f"\nVisualization: ENABLED")
        print(f"  Output: {config['visualization']['save_plot']}")
    
    print(f"\nLogging level: {log_level}")
    print("="*80 + "\n")
    
    # Validate config
    try:
        validate_config(config)
    except (ValueError, FileNotFoundError) as e:
        print(f"\nERROR: {e}")
        sys.exit(1)
    
    # Load model
    model, target_sr, class_index = load_inference_model(
        model_name=config['model']['name'],
        checkpoint_path=config['model'].get('checkpoint'),
        device=config['model']['device']
    )
    
    # Process audio file
    audio_path = config['audio_file']
    
    # Check if visualization is requested
    return_spec = config.get('visualization', {}).get('plot_predictions', False)
    
    results = process_audio_file(
        model=model,
        audio_path=audio_path,
        config=config,
        target_sr=target_sr,
        class_index=class_index,
        ground_truth_events=None,  # No GT for single file
        return_spectrogram=return_spec
    )
    
    # Create visualization if requested
    if config.get('visualization', {}).get('plot_predictions', False):
        plot_config = config['visualization']
        save_path = plot_config.get('save_plot')
        
        if save_path and results['spectrogram'] is not None:
            plot_predictions_with_spectrogram(
                spectrogram=results['spectrogram'],
                audio_duration=results['audio_info']['duration'],
                predictions=results['predictions'],
                events=results['events'],
                threshold=config['inference']['threshold'],
                save_path=save_path,
                sr=target_sr
            )
    
    # Print results
    print_results(config, results)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
