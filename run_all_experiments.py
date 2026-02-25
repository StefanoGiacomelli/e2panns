"""
Batch Experiment Runner for E2PANNs
====================================
Execute multiple training experiments sequentially from config files.

Usage:
    python run_all_experiments.py --config_dir configs/binary_EV
    python run_all_experiments.py --config_dir configs/binary_EV --mode test
    python run_all_experiments.py --config_dir configs/multiclass_EV --continue_on_error

Author: Stefano Giacomelli - Ph.D. candidate in ICT (DISIM dpt. - University of L'Aquila)
"""

import sys
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Tuple


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run multiple E2PANNs experiments sequentially',
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    
    parser.add_argument('--config_dir', type=str, required=True,
                        help='Path to directory containing config YAML files (e.g., configs/binary_EV)')
    
    parser.add_argument('--mode', type=str, default='fit+test',
                        choices=['fit', 'test', 'fit+test', 'benchmark'],
                        help='Execution mode for main.py (default: fit+test)')
    
    parser.add_argument('--python', type=str, default='.venv/bin/python',
                        help='Path to Python interpreter (default: .venv/bin/python)')
    
    parser.add_argument('--continue_on_error', action='store_true',
                        help='Continue running experiments even if one fails')
    
    return parser.parse_args()


def find_config_files(config_dir: str) -> List[Path]:
    """
    Find all YAML config files in directory.
    
    Args:
        config_dir: Path to config directory
        
    Returns:
        Sorted list of config file paths
    """
    config_path = Path(config_dir)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config directory not found: {config_dir}")
    
    if not config_path.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {config_dir}")
    
    # Find all .yaml and .yml files
    yaml_files = list(config_path.glob('*.yaml')) + list(config_path.glob('*.yml'))
    
    if not yaml_files:
        raise FileNotFoundError(f"No YAML config files found in: {config_dir}")
    
    return sorted(yaml_files)


def run_experiment(exp_num: int, total: int, config_file: Path, mode: str, python_path: str) -> Tuple[int, float]:
    """
    Run single experiment.
    
    Args:
        exp_num: Experiment number
        total: Total number of experiments
        config_file: Path to config file
        mode: Execution mode
        python_path: Path to Python interpreter
        
    Returns:
        Tuple of (return_code, duration_seconds)
    """
    print("\n" + "#"*80)
    print(f"# EXPERIMENT {exp_num}/{total}: {config_file.name}")
    print("#"*80)
    print()
    
    # Build command
    cmd = [python_path, 'main.py', '--config', str(config_file), '--mode', mode]
    
    print(f"Running: {' '.join(cmd)}")
    print()
    
    # Start timing
    start_time = datetime.now()
    
    # Run process with direct output to terminal
    try:
        process = subprocess.run(cmd, check=False)
        return_code = process.returncode
        
    except Exception as e:
        print(f"\n✗ ERROR: Failed to run experiment: {e}")
        return_code = 1
    
    # Calculate duration
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    # Print status
    print()
    print("="*80)
    if return_code == 0:
        print(f"Experiment {exp_num}/{total} ✓ SUCCESS")
    else:
        print(f"Experiment {exp_num}/{total} ✗ FAILED (exit code: {return_code})")
    print(f"Duration: {format_duration(duration)}")
    print("="*80)
    
    return return_code, duration


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes}:{secs:02d}"


def print_summary(results: List[Tuple[Path, int, float]], total_duration: float):
    """
    Print final summary of all experiments.
    
    Args:
        results: List of (config_file, return_code, duration) tuples
        total_duration: Total duration in seconds
    """
    print("\n" + "="*80)
    print("BATCH EXPERIMENT SUMMARY")
    print("="*80)
    
    total = len(results)
    success_count = sum(1 for _, rc, _ in results if rc == 0)
    failed_count = total - success_count
    
    print(f"Total experiments: {total}")
    print(f"Completed successfully: {success_count}")
    if failed_count > 0:
        print(f"Failed: {failed_count}")
    print(f"Total duration: {format_duration(total_duration)}")
    print()
    print("Results:")
    
    for config_file, return_code, duration in results:
        status = "✓ SUCCESS" if return_code == 0 else "✗ FAILED "
        duration_str = format_duration(duration)
        print(f"  {status}  {config_file.name:<50} ({duration_str})")
    
    print("="*80)


def main():
    """Main execution function."""
    args = parse_args()
    
    # Print header
    print("="*80)
    print("BATCH EXPERIMENT RUNNER")
    print("="*80)
    print(f"Config directory: {args.config_dir}")
    print(f"Mode: {args.mode}")
    print(f"Python: {args.python}")
    print(f"Continue on error: {args.continue_on_error}")
    print()
    
    # Find config files
    try:
        config_files = find_config_files(args.config_dir)
    except Exception as e:
        print(f"✗ ERROR: {e}")
        sys.exit(1)
    
    print(f"Found {len(config_files)} config file(s):")
    for cf in config_files:
        print(f"  - {cf.name}")
    print("="*80)
    
    # Run experiments
    results = []
    start_time = datetime.now()
    
    for i, config_file in enumerate(config_files, 1):
        # Run experiment
        return_code, duration = run_experiment(i, len(config_files), config_file, args.mode, args.python)
        
        results.append((config_file, return_code, duration))
        
        # Check if should stop on error
        if return_code != 0 and not args.continue_on_error:
            print()
            print("="*80)
            print("✗ STOPPING: Experiment failed and --continue_on_error not set")
            print("="*80)
            break
    
    # Calculate total duration
    end_time = datetime.now()
    total_duration = (end_time - start_time).total_seconds()
    
    # Print summary
    print_summary(results, total_duration)
    
    # Exit with appropriate code
    failed = any(rc != 0 for _, rc, _ in results)
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
