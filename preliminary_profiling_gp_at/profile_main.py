"""
Model Profiling Main Script (READ ONLY)
=======================================
Main script for profiling Neural Network models across different devices.

This script orchestrates the complete profiling pipeline:
1. Model discovery and checkpoint loading
2. Parameter counting
3. Minimum input length discovery (binary search)
4. FLOPs/MACs computation
5. Performance profiling on CPU (always)
6. Performance profiling on MPS/CUDA (if available)
7. Results persistence with incremental saving

Usage:
    python profile_main.py [options]
    
    Options:
        --models-dir: Path to models directory (default: ./models)
        --output-dir: Path to output directory (default: ./models/profiling_results)
        --models: Specific models to profile (space-separated)
        --skip-cpu: Skip CPU profiling
        --force-rerun: Force re-run even if results exist
        --no-binary-search: Skip binary search for min length
        --no-flops: Skip FLOPs/MACs computation
"""

import sys
import argparse
import traceback
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from profile_utils import (ModelConfig,
                           DeviceManager,
                           StatisticsCalculator,
                           ParameterProfiler,
                           MemoryProfiler,
                           PerformanceProfiler,
                           FLOPsProfiler,
                           InputLengthFinder,
                           ModelRegistry,
                           ResultsManager)


# =============================================================================
# Argument Parsing
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Profile neural network models for audio processing',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--models-dir',
        type=str,
        default='models',
        help='Path to models directory (default: ./models)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='preliminary_profiling_gp_at/results',
        help='Path to output directory (default: ./preliminary_profiling_gp_at/results)'
    )
    
    parser.add_argument(
        '--models',
        nargs='+',
        help='Specific models to profile (space-separated model names)'
    )
    
    parser.add_argument(
        '--skip-cpu',
        action='store_true',
        help='Skip CPU profiling'
    )
    
    parser.add_argument(
        '--force-rerun',
        action='store_true',
        help='Force re-run profiling even if results exist'
    )
    
    parser.add_argument(
        '--no-binary-search',
        action='store_true',
        help='Skip binary search for minimum input length'
    )
    
    parser.add_argument(
        '--no-flops',
        action='store_true',
        help='Skip FLOPs/MACs computation'
    )
    
    parser.add_argument(
        '--num-runs',
        type=int,
        default=10,
        help='Number of profiling runs per device (default: 10)'
    )
    
    parser.add_argument(
        '--warmup-runs',
        type=int,
        default=3,
        help='Number of warm-up runs (default: 3)'
    )
    
    return parser.parse_args()


# =============================================================================
# Device-Specific Profiling
# =============================================================================

def profile_model_on_device(model: nn.Module,
                            config: ModelConfig,
                            device: str,
                            num_runs: int = 10,
                            warmup_runs: int = 3) -> Dict[str, Any]:
    """
    Profile a model on a specific device.
    
    Args:
        model: PyTorch model to profile
        config: ModelConfig with model information
        device: Device string ('cpu', 'mps', 'cuda')
        num_runs: Number of profiling runs
        warmup_runs: Number of warm-up runs
        
    Returns:
        Dictionary with profiling results for this device
    """
    print(f"  Profiling on {device.upper()}...")
    
    # Move model to device
    device_manager = DeviceManager()
    model, torch_device = device_manager.move_model_to_device(model, device)
    
    # Create 10-second input
    num_samples = int(10.0 * config.sample_rate)
    input_tensor = torch.randn(1, num_samples, device=torch_device)
    
    # Initialize profiler
    profiler = PerformanceProfiler(model, torch_device)
    
    # Reset memory stats if GPU
    if device in ['cuda', 'mps']:
        MemoryProfiler.reset_peak_memory(device)
    
    # Warm-up
    print(f"    Warming up ({warmup_runs} runs)...")
    profiler.warm_up(input_tensor, num_runs=warmup_runs)
    
    # Reset memory stats after warm-up
    if device in ['cuda', 'mps']:
        MemoryProfiler.reset_peak_memory(device)
    
    # Profile forward time
    print(f"    Measuring forward time ({num_runs} runs)...")
    times = profiler.measure_forward_time(input_tensor, num_runs=num_runs)
    
    if not times:
        raise RuntimeError(f"No successful timing measurements on {device}")
    
    # Calculate statistics
    stats = StatisticsCalculator.compute_stats(times)
    
    # Calculate throughput
    throughput = profiler.calculate_throughput(times, num_samples)
    
    # Get peak memory (if available)
    peak_memory = None
    if device in ['cuda', 'mps']:
        peak_memory = MemoryProfiler.get_peak_memory(device)
    
    # Build results
    results = {
        'fwd_times': [round(t, 6) for t in times],
        'fwd_times_stats': {k: round(v, 6) for k, v in stats.items()},
        'throughput_samples_per_sec': round(throughput, 2)
    }
    
    if peak_memory is not None:
        results['peak_memory_mb'] = round(peak_memory, 2)
    
    return results


# =============================================================================
# Main Profiling Function
# =============================================================================

def profile_single_model(config: ModelConfig,
                         args: argparse.Namespace,
                         results_manager: ResultsManager,
                         available_devices: list) -> bool:
    """
    Profile a single model across all available devices.
    
    Args:
        config: ModelConfig for the model
        args: Command line arguments
        results_manager: ResultsManager instance
        available_devices: List of available devices
        
    Returns:
        True if successful, False otherwise
    """
    print(f"\nProfiling: {config.name}")
    print(f"  Sample rate: {config.sample_rate} Hz")
    print(f"  Checkpoint: {config.checkpoint_path}")
    
    try:
        # Load existing results
        existing_results = results_manager.load_existing_results(config.name)
        
        # Load model
        print(f"  Loading model...")
        registry = ModelRegistry(args.models_dir)
        model = registry.load_model(config)
        print(f"  Model loaded successfully")
        
        # Store checkpoint info
        existing_results['checkpoint'] = config.checkpoint_path
        existing_results['sample_rate'] = config.sample_rate
        
        # ==================================================================
        # STEP 1: Parameter Counting (device-agnostic, done once)
        # ==================================================================
        if 'parameters' not in existing_results or args.force_rerun:
            print(f"  Counting parameters...")
            param_info = ParameterProfiler.count_parameters(model)
            existing_results['parameters'] = param_info
            print(f"    Total: {param_info['total']}M parameters ({param_info['total_mb']} MB)")
            print(f"    Trainable: {param_info['trainable']}M ({param_info['trainable_percent']}%)")
        else:
            print(f"  Parameters already counted (use --force-rerun to re-compute)")
        
        # ==================================================================
        # STEP 2: Binary Search for Minimum Input Length (done once on CPU)
        # ==================================================================
        if ('input' not in existing_results or args.force_rerun) and not args.no_binary_search:
            print(f"  Finding minimum input length...")
            try:
                finder = InputLengthFinder(model, config.sample_rate, 'cpu')
                min_length = finder.find_min_length(min_seconds=0.020, max_seconds=15.0)
                existing_results['input'] = {
                    'sample_rate': config.sample_rate,
                    'channels': 1,
                    'min_length_samples': min_length,
                    'min_length_seconds': round(min_length / config.sample_rate, 3)
                }
                print(f"    Min length: {min_length} samples ({existing_results['input']['min_length_seconds']}s)")
            except Exception as e:
                print(f"    Warning: Could not find minimum length: {e}")
                existing_results['input'] = {
                    'sample_rate': config.sample_rate,
                    'channels': 1,
                    'min_length_samples': None,
                    'min_length_seconds': None,
                    'error': str(e)
                }
        elif 'input' in existing_results:
            print(f"  Minimum input length already found (use --force-rerun to re-compute)")
        
        # ==================================================================
        # STEP 3: FLOPs and MACs Computation (done once on CPU)
        # ==================================================================
        if (('flops' not in existing_results or existing_results.get('flops') is None) 
            or args.force_rerun) and not args.no_flops:
            print(f"  Computing FLOPs and MACs...")
            try:
                # Use 10-second input for FLOPs computation
                input_shape = (1, int(10.0 * config.sample_rate))
                flops_info = FLOPsProfiler.compute_flops_macs(model, input_shape, config.sample_rate)
                existing_results.update(flops_info)
                if flops_info['gflops'] is not None:
                    print(f"    FLOPs: {flops_info['gflops']} GFLOPs")
                    print(f"    MACs: {flops_info['gmacs']} GMACs")
                else:
                    print(f"    FLOPs/MACs computation not available (install fvcore or thop)")
            except Exception as e:
                print(f"    Warning: Could not compute FLOPs/MACs: {e}")
        elif 'flops' in existing_results and existing_results['flops'] is not None:
            print(f"  FLOPs/MACs already computed (use --force-rerun to re-compute)")
        
        # ==================================================================
        # STEP 4: Profile on Each Device
        # ==================================================================
        for device in available_devices:
            # Skip if already profiled (unless force-rerun)
            if device in existing_results and not args.force_rerun:
                print(f"  {device.upper()} profiling already done (use --force-rerun to re-run)")
                continue
            
            # Skip CPU if requested
            if device == 'cpu' and args.skip_cpu:
                print(f"  Skipping CPU profiling (--skip-cpu flag)")
                continue
            
            try:
                # Profile on this device
                device_results = profile_model_on_device(model=model,
                                                         config=config,
                                                         device=device,
                                                         num_runs=args.num_runs,
                                                         warmup_runs=args.warmup_runs)
                
                # Store results
                existing_results[device] = device_results
                
                # Print summary
                mean_time = device_results['fwd_times_stats']['mean']
                throughput = device_results['throughput_samples_per_sec']
                print(f"    Mean time: {mean_time*1000:.2f} ms")
                print(f"    Throughput: {throughput:.0f} samples/sec")
                
                if 'peak_memory_mb' in device_results:
                    print(f"    Peak memory: {device_results['peak_memory_mb']:.2f} MB")
                
            except Exception as e:
                print(f"    Error profiling on {device}: {e}")
                traceback.print_exc()
                continue
        
        # ==================================================================
        # STEP 5: Save Results
        # ==================================================================
        results_manager.save_results(config.name, existing_results)
        
        # Cleanup
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return True
        
    except Exception as e:
        print(f"  Error profiling {config.name}: {e}")
        traceback.print_exc()
        return False


# =============================================================================
# Main Function
# =============================================================================

def main():
    """Main profiling orchestration."""
    args = parse_args()
    
    print("=" * 80)
    print("Model Profiling Tool")
    print("=" * 80)
    
    # Setup
    print("\n[1/3] Initializing...")
    device_manager = DeviceManager()
    available_devices = device_manager.get_available_devices()
    print(f"  Available devices: {', '.join(available_devices)}")
    
    model_registry = ModelRegistry(args.models_dir)
    results_manager = ResultsManager(args.output_dir)
    print(f"  Output directory: {args.output_dir}")
    
    # Discover models
    print("\n[2/3] Discovering models...")
    # When filtering by specific models, suppress warnings for other models
    silent_discovery = args.models is not None
    all_models = model_registry.discover_models(silent=silent_discovery)
    
    # Filter models if specified
    if args.models:
        all_models = [m for m in all_models if m.name in args.models]
        if not all_models:
            print(f"Error: No models found matching: {args.models}")
            return
    
    print(f"  Found {len(all_models)} model(s) to profile:")
    for config in all_models:
        print(f"    - {config.name}")
    
    # Profile each model
    print("\n[3/3] Profiling models...")
    print(f"  Configuration:")
    print(f"    Profiling runs: {args.num_runs}")
    print(f"    Warm-up runs: {args.warmup_runs}")
    print(f"    Force re-run: {args.force_rerun}")
    print(f"    Skip CPU: {args.skip_cpu}")
    print(f"    Skip binary search: {args.no_binary_search}")
    print(f"    Skip FLOPs: {args.no_flops}")
    
    success_count = 0
    failure_count = 0
    
    with tqdm(total=len(all_models), desc="Overall Progress", position=0) as pbar:
        for config in all_models:
            success = profile_single_model(config=config,
                                           args=args,
                                           results_manager=results_manager,
                                           available_devices=available_devices)
            
            if success:
                success_count += 1
            else:
                failure_count += 1
            
            pbar.update(1)
    
    # Summary
    print("\n" + "=" * 80)
    print("Profiling Complete!")
    print("=" * 80)
    print(f"  Successfully profiled: {success_count} model(s)")
    print(f"  Failed: {failure_count} model(s)")
    print(f"  Results saved to: {args.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
