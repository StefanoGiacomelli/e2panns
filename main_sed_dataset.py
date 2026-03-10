#!/usr/bin/env python3
"""Main Script for Dataset SED Processing
==========================================
Process entire dataset for Emergency Vehicle Sound Event Detection.

Features:
- Load AudioSet_EV_Strong dataset (v1 or v2, positives only)
- Process all samples with SED inference
- Compute SED metrics (segment + event based)
- Monitor performance across dataset
- Support multi-processing for faster processing
- Save comprehensive results (CSV + JSON)

Usage:
    python main_dataset.py configs/example_dataset.yaml

Author: Stefano Giacomelli - Ph.D. candidate in ICT (DISIM dpt. - University of L'Aquila)
"""

import sys
import logging
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict

import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm

from sed_system.core import load_inference_model
from sed_system.pipeline import process_audio_file
from datasets.AudioSet_EV_Strong.dataloader import AudioSetEV_Strong_Dataset
from datasets.KineScaper_EV.dataloader import KineScaper_EV_DetectionDataset


def setup_logging(level: str = "INFO", log_file: str = None):
    """Setup logging configuration - logs only to file, keeping console clean for tqdm."""
    handlers = []
    
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    else:
        # If no log file specified, use NullHandler to suppress logs
        handlers.append(logging.NullHandler())
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=handlers,
        force=True  # Override any existing config
    )
    
    # Silence noisy libraries
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('numba').setLevel(logging.WARNING)
    
    # Silence noisy libraries in DEBUG mode
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('numba').setLevel(logging.WARNING)


def validate_config(config: dict):
    """Validate configuration file."""
    required_keys = ['dataset', 'model', 'inference', 'output']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")
    
    # Validate dataset name
    valid_datasets = ['AudioSet_EV_v1', 'AudioSet_EV_v2', 'KineScaper_EV']
    if config['dataset']['name'] not in valid_datasets:
        raise ValueError(f"Invalid dataset name. Must be one of: {valid_datasets}")
    
    # Validate device
    valid_devices = ['cpu', 'cuda', 'mps']
    device = config['model'].get('device', 'cpu')
    if device not in valid_devices:
        raise ValueError(f"Invalid device. Must be one of: {valid_devices}")
    
    # Validate adaptive window params
    if config['inference']['adaptive_window']['enabled']:
        chunk_dur = config['inference']['chunk_duration']
        max_dur = config['inference']['adaptive_window']['frame_duration_max']
        
        if max_dur < chunk_dur:
            raise ValueError(f"frame_duration_max ({max_dur}) must be >= chunk_duration ({chunk_dur})")
        
        adapt_coeff = config['inference']['adaptive_window']['adapt_coeff']
        if not (0.0 <= adapt_coeff <= 1.0):
            raise ValueError(f"adapt_coeff ({adapt_coeff}) must be in [0.0, 1.0]")


def extract_ground_truth_events(sample_events: List[Dict], ev_mids: List[str]) -> List:
    """Extract EV ground truth events from sample metadata.
    
    Supports both AudioSet Strong (filter by MID) and KineScaper (all events are EV).
    """
    gt_events = []
    
    for event in sample_events:
        mid = event.get('mid', '')
        # KineScaper events use 'kinescaper_ev' as MID - always include
        # AudioSet events: filter by EV MIDs
        if mid == 'kinescaper_ev' or mid in ev_mids:
            gt_events.append((event['start'], event['end']))
    
    return gt_events


def process_single_sample(args):
    """
    Process a single sample (for multiprocessing).
    
    Args:
        args: Tuple of (sample_idx, sample, dataset, model, config, target_sr, class_index, ev_mids)
    
    Returns:
        Dictionary with sample results
    """
    sample_idx, sample, model, config, target_sr, class_index, ev_mids = args
    
    segment_id = sample['segment_id']
    file_path = sample['file_path']
    sample_events = sample['events']
    
    # Extract ground truth
    gt_events = extract_ground_truth_events(sample_events, ev_mids)
    
    # Process audio
    results = process_audio_file(
        model=model,
        audio_path=file_path,
        config=config,
        target_sr=target_sr,
        class_index=class_index,
        ground_truth_events=gt_events,
        return_spectrogram=False  # No visualization for dataset processing
    )
    
    # Build result dict
    sample_result = {
        'sample_idx': sample_idx,
        'segment_id': segment_id,
        'file_path': file_path,
        'audio_duration': results['audio_info']['duration'],
        'num_inferences': results['audio_info']['num_inferences'],
        'num_gt_events': len(gt_events),
        'num_detected_events': len(results['events']),
        'sed_metrics': results['sed_metrics'],
        'performance': results['performance']
    }
    
    return sample_result


def aggregate_metrics(all_results: List[Dict]) -> Dict:
    """Aggregate metrics across all samples."""
    # Collect lists
    seg_precision, seg_recall, seg_f1 = [], [], []
    seg_accuracy, seg_balanced_acc = [], []
    seg_error_rate, seg_del_rate, seg_ins_rate = [], [], []
    
    evt_precision, evt_recall, evt_f1 = [], [], []
    evt_error_rate, evt_del_rate, evt_ins_rate = [], [], []
    
    throughputs, cpu_means, ram_means = [], [], []
    
    for result in all_results:
        if result['sed_metrics']:
            seg = result['sed_metrics']['segment_based']
            seg_precision.append(seg.get('precision', 0))
            seg_recall.append(seg.get('recall', 0))
            seg_f1.append(seg.get('f1', 0))
            seg_accuracy.append(seg.get('accuracy', 0))
            seg_balanced_acc.append(seg.get('balanced_accuracy', 0))
            seg_error_rate.append(seg.get('error_rate', 0))
            seg_del_rate.append(seg.get('deletion_rate', 0))
            seg_ins_rate.append(seg.get('insertion_rate', 0))
            
            evt = result['sed_metrics']['event_based']
            evt_precision.append(evt.get('precision', 0))
            evt_recall.append(evt.get('recall', 0))
            evt_f1.append(evt.get('f1', 0))
            if evt.get('error_rate') is not None:
                evt_error_rate.append(evt.get('error_rate', 0))
            if evt.get('deletion_rate') is not None:
                evt_del_rate.append(evt.get('deletion_rate', 0))
            if evt.get('insertion_rate') is not None:
                evt_ins_rate.append(evt.get('insertion_rate', 0))
        
        if result['performance']:
            perf = result['performance']
            throughputs.append(perf.get('throughput', 0))
            
            cpu_stats = perf.get('cpu', {})
            if cpu_stats:
                cpu_means.append(cpu_stats.get('mean', 0))
            
            ram_stats = perf.get('ram_mb', {})
            if ram_stats:
                ram_means.append(ram_stats.get('mean', 0))
    
    # Compute aggregates (use nanmean/nanstd to ignore NaN values)
    aggregated = {
        'segment_based': {
            'precision': float(np.nanmean(seg_precision)) if seg_precision else 0,
            'recall': float(np.nanmean(seg_recall)) if seg_recall else 0,
            'f1': float(np.nanmean(seg_f1)) if seg_f1 else 0,
            'accuracy': float(np.nanmean(seg_accuracy)) if seg_accuracy else 0,
            'balanced_accuracy': float(np.nanmean(seg_balanced_acc)) if seg_balanced_acc else 0,
            'error_rate': float(np.nanmean(seg_error_rate)) if seg_error_rate else 0,
            'deletion_rate': float(np.nanmean(seg_del_rate)) if seg_del_rate else 0,
            'insertion_rate': float(np.nanmean(seg_ins_rate)) if seg_ins_rate else 0,
            'std_precision': float(np.nanstd(seg_precision)) if seg_precision else 0,
            'std_recall': float(np.nanstd(seg_recall)) if seg_recall else 0,
            'std_f1': float(np.nanstd(seg_f1)) if seg_f1 else 0,
            'std_accuracy': float(np.nanstd(seg_accuracy)) if seg_accuracy else 0,
            'std_balanced_accuracy': float(np.nanstd(seg_balanced_acc)) if seg_balanced_acc else 0
        },
        'event_based': {
            'precision': float(np.nanmean(evt_precision)) if evt_precision else 0,
            'recall': float(np.nanmean(evt_recall)) if evt_recall else 0,
            'f1': float(np.nanmean(evt_f1)) if evt_f1 else 0,
            'error_rate': float(np.nanmean(evt_error_rate)) if evt_error_rate else 0,
            'deletion_rate': float(np.nanmean(evt_del_rate)) if evt_del_rate else 0,
            'insertion_rate': float(np.nanmean(evt_ins_rate)) if evt_ins_rate else 0,
            'std_precision': float(np.nanstd(evt_precision)) if evt_precision else 0,
            'std_recall': float(np.nanstd(evt_recall)) if evt_recall else 0,
            'std_f1': float(np.nanstd(evt_f1)) if evt_f1 else 0
        },
        'performance': {
            'mean_throughput': float(np.nanmean(throughputs)) if throughputs else 0,
            'std_throughput': float(np.nanstd(throughputs)) if throughputs else 0,
            'mean_cpu': float(np.nanmean(cpu_means)) if cpu_means else 0,
            'std_cpu': float(np.nanstd(cpu_means)) if cpu_means else 0,
            'mean_ram_mb': float(np.nanmean(ram_means)) if ram_means else 0,
            'std_ram_mb': float(np.nanstd(ram_means)) if ram_means else 0
        }
    }
    
    return aggregated


def print_summary(config: dict, num_samples: int, aggregated_metrics: Dict, total_audio: float, total_time: float):
    """Print summary results to terminal."""
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Dataset: {config['dataset']['name']}")
    print(f"Model: {config['model']['name'].upper()}")
    print(f"Samples processed: {num_samples}")
    print(f"Total audio: {total_audio:.1f}s")
    print(f"Total inference time: {total_time:.1f}s")
    
    print(f"\nSED Metrics (Segment-based, averaged over {num_samples} samples):")
    seg = aggregated_metrics['segment_based']
    print(f"  Precision: {seg['precision']:.3f} ± {seg['std_precision']:.3f}")
    print(f"  Recall: {seg['recall']:.3f} ± {seg['std_recall']:.3f}")
    print(f"  F1: {seg['f1']:.3f} ± {seg['std_f1']:.3f}")
    print(f"  Accuracy: {seg['accuracy']:.3f} ± {seg['std_accuracy']:.3f}")
    print(f"  Balanced Accuracy: {seg['balanced_accuracy']:.3f} ± {seg['std_balanced_accuracy']:.3f}")
    print(f"  Error Rate: {seg['error_rate']:.3f} (del={seg['deletion_rate']:.3f}, ins={seg['insertion_rate']:.3f})")
    
    print(f"\nSED Metrics (Event-based, averaged over {num_samples} samples):")
    evt = aggregated_metrics['event_based']
    print(f"  Precision: {evt['precision']:.3f} ± {evt['std_precision']:.3f}")
    print(f"  Recall: {evt['recall']:.3f} ± {evt['std_recall']:.3f}")
    print(f"  F1: {evt['f1']:.3f} ± {evt['std_f1']:.3f}")
    if evt['error_rate'] > 0:
        print(f"  Error Rate: {evt['error_rate']:.3f} (del={evt['deletion_rate']:.3f}, ins={evt['insertion_rate']:.3f})")
    
    print(f"\nPerformance:")
    perf = aggregated_metrics['performance']
    print(f"  Mean throughput: {perf['mean_throughput']:.2f}x ± {perf['std_throughput']:.2f}x")
    if perf['mean_cpu'] > 0:
        print(f"  CPU: {perf['mean_cpu']:.1f}% ± {perf['std_cpu']:.1f}%")
    if perf['mean_ram_mb'] > 0:
        print(f"  RAM: {perf['mean_ram_mb']:.0f} MB ± {perf['std_ram_mb']:.0f} MB")
    
    print(f"\nResults saved to: {config['output']['dir']}")
    print("="*80 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Dataset SED Processing")
    parser.add_argument('config', type=str, help='Path to YAML configuration file')
    args = parser.parse_args()
    
    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup output directory
    output_dir = Path(config['output']['dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Setup logging
    log_level = config.get('logging', {}).get('level', 'INFO')
    log_file = config.get('logging', {}).get('log_file')
    if log_file:
        log_file = str(output_dir / Path(log_file).name)
    
    setup_logging(log_level, log_file)
    
    # Validate model device availability
    import torch
    device = config['model'].get('device', 'cpu')
    if device == 'cuda' and not torch.cuda.is_available():
        print(f"WARNING: CUDA requested but not available, falling back to CPU")
        config['model']['device'] = 'cpu'
    elif device == 'mps' and not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
        print(f"WARNING: MPS requested but not available, falling back to CPU")
        config['model']['device'] = 'cpu'
    
    # Print configuration header
    print("\n" + "="*80)
    print("SED DATASET PROCESSING")
    print("="*80)
    print(f"Config: {config_path.name}")
    print(f"Dataset: {config['dataset']['name']}")
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
    
    print(f"\nOutput directory: {config['output']['dir']}")
    print(f"Logging level: {log_level}")
    if log_file:
        print(f"Log file: {Path(log_file).name}")
    print("="*80 + "\n")
    
    # Validate config
    try:
        validate_config(config)
    except ValueError as e:
        print(f"\nERROR: {e}")
        sys.exit(1)
    
    # Print header
    print("\n" + "="*80)
    print(f"LOADING DATASET")
    print("="*80)
    
    # Load dataset
    project_root = Path(__file__).parent
    dataset_name = config['dataset']['name']
    
    if dataset_name == 'KineScaper_EV':
        # Load KineScaper-EV detection dataset
        dataset_root = config['dataset'].get('dataset_root', '/mnt/ssd/Kinescaper_EV/dataset/')
        
        dataset = KineScaper_EV_DetectionDataset(
            dataset_root=dataset_root,
            metadata_format=config['dataset'].get('metadata_format', 'json'),
            window_size=config['inference']['chunk_duration'],
            target_sr=32000,
            target_duration=config['dataset'].get('target_duration', 40.0),
            is_positive=True,
            seed=42
        )
        
        total_samples = len(dataset.samples)
        max_samples = config['dataset'].get('max_samples')
        num_samples = min(total_samples, max_samples) if max_samples else total_samples
        
        # Warn if processing a very large number of samples
        estimated_hours = num_samples * config['dataset'].get('target_duration', 40.0) / 3600
        print(f"Total samples: {total_samples:,}")
        if estimated_hours > 1.0:
            print(f"WARNING: {num_samples:,} samples = ~{estimated_hours:.1f}h of audio. Consider setting max_samples in config.")
        print(f"Processing: {num_samples} samples")
        print("="*80 + "\n")
        
        # KineScaper: all events are EV, no MID filtering needed
        ev_mids = []
        
    else:
        # AudioSet EV (v1 or v2)
        dataset_version = 'v2' if dataset_name == 'AudioSet_EV_v2' else 'v1'
        
        # Strong metadata
        strong_metadata = [
            project_root / "datasets/AudioSet_EV_Strong/audioset_strong_metadata/audioset_train_strong.tsv",
            project_root / "datasets/AudioSet_EV_Strong/audioset_strong_metadata/audioset_eval_strong.tsv"
        ]
        
        # Audio folders
        if dataset_version == 'v2':
            audio_folders = [
                project_root / "datasets/AudioSet_EV_v2PANNs_2020/Positive_files/balanced_train",
                project_root / "datasets/AudioSet_EV_v2PANNs_2020/Positive_files/eval",
                project_root / "datasets/AudioSet_EV_v2PANNs_2020/Positive_files/unbalanced",
                project_root / "datasets/AudioSet_EV_v2PANNs_2020/Negative_files/balanced_train",
                project_root / "datasets/AudioSet_EV_v2PANNs_2020/Negative_files/eval"
            ]
        else:  # v1
            audio_folders = [
                project_root / "datasets/AudioSet_EV_v1_2025/Positive_files",
                project_root / "datasets/AudioSet_EV_v1_2025/Negative_files"
            ]
        
        # EV MIDs
        ev_mids = [
            '/m/03j1ly',  # Emergency vehicle
            '/m/04qvtq',  # Police car (siren)
            '/m/012n7d',  # Ambulance (siren)
            '/m/012ndj'   # Fire engine
        ]
        
        # Initialize dataset (positives only)
        dataset = AudioSetEV_Strong_Dataset(
            strong_metadata_paths=[str(p) for p in strong_metadata],
            audio_folders=[str(p) for p in audio_folders],
            ev_mids=ev_mids,
            window_size=config['inference']['chunk_duration'],
            target_sr=32000,
            target_duration=10.0,
            is_positive=True,  # Only positives
            seed=42
        )
        
        total_samples = len(dataset.samples)
        max_samples = config['dataset'].get('max_samples')
        num_samples = min(total_samples, max_samples) if max_samples else total_samples
        
        print(f"Total positive samples: {total_samples}")
        print(f"Processing: {num_samples} samples")
        print("="*80 + "\n")
    
    # Load model
    print("Loading model...")
    model, target_sr, class_index = load_inference_model(
        model_name=config['model']['name'],
        checkpoint_path=config['model'].get('checkpoint'),
        device=config['model']['device']
    )
    print(f"Model loaded (sr={target_sr}Hz, class_idx={class_index})\n")
    
    # Process samples
    all_results = []
    total_audio_duration = 0
    total_inference_time = 0
    
    print("="*80)
    print(f"PROCESSING {num_samples} SAMPLES")
    print("="*80 + "\n")
    
    # Sequential processing (multiprocessing would require model serialization)
    for i in tqdm(range(num_samples), desc="Processing"):
        sample = dataset.samples[i]
        
        args = (i, sample, model, config, target_sr, class_index, ev_mids)
        result = process_single_sample(args)
        
        all_results.append(result)
        total_audio_duration += result['audio_duration']
        total_inference_time += result['performance']['total_time']
    
    print("\n")
    
    # Aggregate metrics
    aggregated_metrics = aggregate_metrics(all_results)
    
    # Save results
    print("="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    # Summary JSON
    summary = {
        'dataset': config['dataset']['name'],
        'model': config['model']['name'],
        'timestamp': timestamp,
        'num_samples': num_samples,
        'total_audio_duration': total_audio_duration,
        'total_inference_time': total_inference_time,
        'aggregated_metrics': aggregated_metrics
    }
    
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  ✅ summary.json")
    
    # Results CSV
    csv_rows = []
    for result in all_results:
        row = {
            'segment_id': result['segment_id'],
            'audio_duration': result['audio_duration'],
            'num_inferences': result['num_inferences'],
            'num_gt_events': result['num_gt_events'],
            'num_detected': result['num_detected_events']
        }
        
        if result['sed_metrics']:
            seg = result['sed_metrics']['segment_based']
            row.update({
                'seg_precision': seg.get('precision', 0),
                'seg_recall': seg.get('recall', 0),
                'seg_f1': seg.get('f1', 0),
                'seg_accuracy': seg.get('accuracy', 0),
                'seg_balanced_accuracy': seg.get('balanced_accuracy', 0),
                'seg_error_rate': seg.get('error_rate', 0),
                'seg_deletion_rate': seg.get('deletion_rate', 0),
                'seg_insertion_rate': seg.get('insertion_rate', 0)
            })
            
            evt = result['sed_metrics']['event_based']
            row.update({
                'evt_precision': evt.get('precision', 0),
                'evt_recall': evt.get('recall', 0),
                'evt_f1': evt.get('f1', 0),
                'evt_error_rate': evt.get('error_rate', 0),
                'evt_deletion_rate': evt.get('deletion_rate', 0),
                'evt_insertion_rate': evt.get('insertion_rate', 0)
            })
        
        if result['performance']:
            perf = result['performance']
            cpu_stats = perf.get('cpu', {})
            ram_stats = perf.get('ram_mb', {})
            row.update({
                'throughput': perf.get('throughput', 0),
                'cpu_mean': cpu_stats.get('mean', 0),
                'ram_mean_mb': ram_stats.get('mean', 0)
            })
        
        csv_rows.append(row)
    
    df = pd.DataFrame(csv_rows)
    df.to_csv(output_dir / 'results.csv', index=False)
    print(f"  ✅ results.csv ({len(df)} rows)")
    
    # Sample metrics JSON (detailed)
    with open(output_dir / 'sample_metrics.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"  ✅ sample_metrics.json")
    
    # Print summary
    print_summary(config, num_samples, aggregated_metrics, total_audio_duration, total_inference_time)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
