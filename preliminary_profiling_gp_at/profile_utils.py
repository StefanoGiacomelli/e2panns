"""
Model Profiling Utilities (READ ONLY)
=====================================
Utilities and classes for profiling Neural Network models on different devices.

This module provides comprehensive profiling capabilities including:
- Parameter counting (total, trainable, memory)
- Forward pass timing with statistical analysis
- Memory profiling (device-specific)
- FLOPs and MACs computation
- Binary search for minimum input length estimation
- Device management with fallback
"""

import json
import time
import importlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any

import numpy as np
import torch
import torch.nn as nn
from scipy import stats as scipy_stats


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ModelConfig:
    """Configuration for a model to be profiled."""
    name: str
    module_path: str
    class_name: str
    sample_rate: int
    checkpoint_path: str
    init_kwargs: dict = None
    
    def __post_init__(self):
        if self.init_kwargs is None:
            self.init_kwargs = {}


# =============================================================================
# Device Management
# =============================================================================

class DeviceManager:
    """Manages device availability and model transfer."""
    
    @staticmethod
    def get_available_devices() -> List[str]:
        """
        Get list of available devices in order: CPU, MPS, CUDA.
        
        Returns:
            List of device strings (e.g., ['cpu', 'mps'] or ['cpu', 'cuda'])
        """
        devices = ['cpu']  # CPU always available
        
        # Check for MPS (Apple Silicon)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            devices.append('mps')
        
        # Check for CUDA
        if torch.cuda.is_available():
            devices.append('cuda')
        
        return devices
    
    @staticmethod
    def move_model_to_device(model: nn.Module, device: str) -> Tuple[nn.Module, torch.device]:
        """
        Move model to specified device.
        
        Args:
            model: PyTorch model
            device: Device string ('cpu', 'mps', 'cuda')
            
        Returns:
            Tuple of (model, torch.device)
        """
        torch_device = torch.device(device)
        model = model.to(torch_device)
        model.eval()
        return model, torch_device


# =============================================================================
# Statistics Calculation
# =============================================================================

class StatisticsCalculator:
    """Calculates comprehensive statistics for timing measurements."""
    
    @staticmethod
    def compute_stats(times: List[float]) -> Dict[str, float]:
        """
        Compute statistical measures for a list of timing values.
        
        Args:
            times: List of timing measurements in seconds
            
        Returns:
            Dictionary with statistical measures
        """
        if not times:
            return {}
        
        times_array = np.array(times)
        
        # Basic statistics
        mean_val = float(np.mean(times_array))
        max_val = float(np.max(times_array))
        min_val = float(np.min(times_array))
        std_val = float(np.std(times_array, ddof=1)) if len(times_array) > 1 else 0.0
        stderr_val = std_val / np.sqrt(len(times_array)) if len(times_array) > 1 else 0.0
        median_val = float(np.median(times_array))
        
        # Percentiles
        q75, q25 = np.percentile(times_array, [75, 25])
        iqr_val = float(q75 - q25)
        p75_val = float(q75)
        
        # Shape statistics
        skewness_val = float(scipy_stats.skew(times_array)) if len(times_array) > 2 else 0.0
        kurtosis_val = float(scipy_stats.kurtosis(times_array)) if len(times_array) > 3 else 0.0
        
        return {
            'mean': mean_val,
            'max': max_val,
            'min': min_val,
            'std': std_val,
            'stderr': stderr_val,
            'median': median_val,
            'iqr': iqr_val,
            'p75': p75_val,
            'skewness': skewness_val,
            'kurtosis': kurtosis_val
        }


# =============================================================================
# Parameter Profiling
# =============================================================================

class ParameterProfiler:
    """Profiles model parameters."""
    
    @staticmethod
    def count_parameters(model: nn.Module) -> Dict[str, float]:
        """
        Count total and trainable parameters.
        
        Args:
            model: PyTorch model
            
        Returns:
            Dictionary with parameter counts and memory estimates
        """
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Estimate memory (assuming float32 = 4 bytes)
        total_mb = (total_params * 4) / (1024 ** 2)
        
        trainable_percent = (trainable_params / total_params * 100) if total_params > 0 else 0.0
        
        return {
            'total': round(total_params / 1e6, 2),  # in millions
            'total_mb': round(total_mb, 2),
            'trainable': round(trainable_params / 1e6, 2),
            'trainable_percent': round(trainable_percent, 2)
        }


# =============================================================================
# Memory Profiling
# =============================================================================

class MemoryProfiler:
    """Profiles peak memory usage on GPU/MPS devices."""
    
    @staticmethod
    def reset_peak_memory(device: str):
        """Reset peak memory stats for the device."""
        if device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        elif device == 'mps':
            # MPS doesn't have a reset function, we'll track manually
            pass
    
    @staticmethod
    def get_peak_memory(device: str) -> Optional[float]:
        """
        Get peak memory usage in MB.
        
        Args:
            device: Device string ('cuda' or 'mps')
            
        Returns:
            Peak memory in MB, or None if not available
        """
        try:
            if device == 'cuda':
                peak_bytes = torch.cuda.max_memory_allocated()
                return peak_bytes / (1024 ** 2)
            elif device == 'mps':
                # MPS memory tracking
                if hasattr(torch.mps, 'current_allocated_memory'):
                    current_bytes = torch.mps.current_allocated_memory()
                    return current_bytes / (1024 ** 2)
        except Exception as e:
            print(f"Warning: Could not get peak memory for {device}: {e}")
        
        return None


# =============================================================================
# Performance Profiling
# =============================================================================

class PerformanceProfiler:
    """Profiles forward pass performance."""
    
    def __init__(self, model: nn.Module, device: torch.device):
        """
        Initialize performance profiler.
        
        Args:
            model: PyTorch model to profile
            device: Device to run on
        """
        self.model = model
        self.device = device
        self.device_str = str(device).split(':')[0]  # 'cuda:0' -> 'cuda'
    
    def warm_up(self, input_tensor: torch.Tensor, num_runs: int = 3):
        """
        Perform warm-up runs to stabilize performance.
        
        Args:
            input_tensor: Input tensor for the model
            num_runs: Number of warm-up iterations
        """
        self.model.eval()
        with torch.no_grad():
            for _ in range(num_runs):
                try:
                    _ = self.model(input_tensor)
                    if self.device_str in ['cuda', 'mps']:
                        if self.device_str == 'cuda':
                            torch.cuda.synchronize()
                except Exception:
                    pass
    
    def measure_forward_time(self, input_tensor: torch.Tensor, num_runs: int = 10) -> List[float]:
        """
        Measure forward pass time over multiple runs.
        
        Args:
            input_tensor: Input tensor for the model
            num_runs: Number of profiling runs
            
        Returns:
            List of timing measurements in seconds
        """
        times = []
        self.model.eval()
        
        with torch.no_grad():
            for _ in range(num_runs):
                try:
                    # Clear cache before each run
                    if self.device_str == 'cuda':
                        torch.cuda.empty_cache()
                    
                    # Time the forward pass
                    start_time = time.perf_counter()
                    _ = self.model(input_tensor)
                    
                    # Synchronize if GPU
                    if self.device_str == 'cuda':
                        torch.cuda.synchronize()
                    
                    end_time = time.perf_counter()
                    times.append(end_time - start_time)
                    
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    continue
                except Exception as e:
                    print(f"Warning: Error during timing run: {e}")
                    continue
        
        return times
    
    def calculate_throughput(self, times: List[float], num_samples: int) -> float:
        """
        Calculate throughput in samples per second.
        
        Args:
            times: List of timing measurements
            num_samples: Number of samples in input
            
        Returns:
            Throughput in samples/second
        """
        if not times:
            return 0.0
        
        mean_time = np.mean(times)
        if mean_time > 0:
            return num_samples / mean_time
        return 0.0


# =============================================================================
# FLOPs and MACs Profiling
# =============================================================================

class FLOPsProfiler:
    """Computes FLOPs and MACs for models."""
    
    @staticmethod
    def compute_flops_macs(model: nn.Module, input_shape: Tuple[int, ...], 
                          sample_rate: int) -> Dict[str, Any]:
        """
        Compute FLOPs and MACs for the model.
        
        Args:
            model: PyTorch model
            input_shape: Input tensor shape (batch, samples)
            sample_rate: Audio sample rate
            
        Returns:
            Dictionary with FLOPs and MACs information
        """
        try:
            from fvcore.nn import FlopCountAnalysis
            
            dummy_input = torch.randn(input_shape)
            flops_analyzer = FlopCountAnalysis(model, dummy_input)
            total_flops = flops_analyzer.total()
            
            # MACs are approximately FLOPs / 2 for most operations
            total_macs = total_flops // 2
            
            return {
                'flops': int(total_flops),
                'macs': int(total_macs),
                'gflops': round(total_flops / 1e9, 3),
                'gmacs': round(total_macs / 1e9, 3)
            }
        
        except Exception as e:
            print(f"Warning: Could not compute FLOPs/MACs: {e}")
            return {
                'flops': None,
                'macs': None,
                'gflops': None,
                'gmacs': None
            }


# =============================================================================
# Input Length Finder (Binary Search)
# =============================================================================

class InputLengthFinder:
    """Finds minimum supported input length using binary search."""
    
    def __init__(self, model: nn.Module, sample_rate: int, device: str):
        """
        Initialize input length finder.
        
        Args:
            model: PyTorch model
            sample_rate: Audio sample rate
            device: Device to test on
        """
        self.model = model
        self.sample_rate = sample_rate
        self.device = torch.device(device)
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def find_min_length(self, min_seconds: float = 0.020, 
                       max_seconds: float = 15.0) -> int:
        """
        Find minimum input length using binary search.
        
        Args:
            min_seconds: Minimum duration to test (default: 20ms)
            max_seconds: Maximum duration to test (default: 15s)
            
        Returns:
            Minimum input length in samples
        """
        min_samples = int(min_seconds * self.sample_rate)
        max_samples = int(max_seconds * self.sample_rate)
        
        # First check if max works
        print(f"  Testing max length ({max_seconds}s = {max_samples} samples)...")
        if not self._test_length(max_samples):
            raise RuntimeError(f"Model doesn't work even with {max_seconds}s input")
        
        print(f"  Binary searching between {min_seconds}s and {max_seconds}s...")
        
        # Binary search
        while min_samples < max_samples - 1:
            mid_samples = (min_samples + max_samples) // 2
            
            if self._test_length(mid_samples):
                max_samples = mid_samples  # Works, try shorter
            else:
                min_samples = mid_samples + 1  # Doesn't work, need longer
        
        # Verify final result
        if self._test_length(max_samples):
            print(f"  Found minimum length: {max_samples} samples ({max_samples/self.sample_rate:.3f}s)")
            return max_samples
        else:
            # Try the next value
            if self._test_length(max_samples + 1):
                print(f"  Found minimum length: {max_samples + 1} samples ({(max_samples+1)/self.sample_rate:.3f}s)")
                return max_samples + 1
            else:
                raise RuntimeError("Binary search failed to find valid minimum length")
    
    def _test_length(self, length: int, max_retries: int = 2) -> bool:
        """
        Test if a given input length works for the model.
        
        Args:
            length: Input length in samples
            max_retries: Maximum number of retry attempts
            
        Returns:
            True if length is valid, False otherwise
        """
        for attempt in range(max_retries):
            try:
                dummy_input = torch.randn(1, length, device=self.device)
                
                with torch.no_grad():
                    output = self.model(dummy_input)
                
                # Check if output is valid
                if output is not None and output.numel() > 0:
                    # Clear memory
                    del dummy_input, output
                    if str(self.device).startswith('cuda'):
                        torch.cuda.empty_cache()
                    return True
                else:
                    return False
                    
            except torch.cuda.OutOfMemoryError:
                # OOM means input is too long (or model too large)
                if str(self.device).startswith('cuda'):
                    torch.cuda.empty_cache()
                if attempt < max_retries - 1:
                    continue
                return False
                
            except RuntimeError as e:
                # Runtime errors usually mean input is too short or incompatible
                error_msg = str(e).lower()
                if 'out of memory' in error_msg:
                    if str(self.device).startswith('cuda'):
                        torch.cuda.empty_cache()
                    return False
                # Other runtime errors likely mean input is invalid
                return False
                
            except Exception as e:
                # Any other exception means this length doesn't work
                return False
        
        return False


# =============================================================================
# Model Registry and Discovery
# =============================================================================

class ModelRegistry:
    """Registry for discovering and loading models."""
    
    # Mapping of model folder names to their configurations
    MODEL_CONFIGS = {
        'ast': {
            'class_name': 'ASTModel',
            'sample_rate': 16000,
            'init_kwargs': {}
        },
        'beats': {
            'class_name': 'BEATs',
            'sample_rate': 16000,
            'init_kwargs': {'sample_rate': 16000}
        },
        'yamnet': {
            'class_name': 'YAMNet',
            'sample_rate': 16000,
            'init_kwargs': {}
        },
        'clap': {
            'class_name': 'CLAP',
            'sample_rate': 48000,
            'init_kwargs': {}
        },
        'ced': {
            'class_name': 'CEDBase',
            'sample_rate': 16000,
            'init_kwargs': {}
        },
        'psla': {
            'class_name': 'EffNetAttention',
            'sample_rate': 16000,
            'init_kwargs': {'label_dim': 527, 'b': 2, 'pretrain': False, 'head_num': 4}
        },
        'efficientat': {
            'variants': {
                'model_mn': {'checkpoint': 'mn40_as_ext_mAP_487.pt'},
                'model_dymn': {'checkpoint': 'dymn20_as_mAP_493.pt'}
            }
        },
        'audioclip': {
            'class_name': 'AudioCLIP',
            'sample_rate': 44100,
            'init_kwargs': {}
        },
        'htsat': {
            'class_name': 'HTSAT',
            'sample_rate': 32000,
            'init_kwargs': {}
        },
        'vggish': {
            'class_name': 'VGGish',
            'sample_rate': 16000,
            'init_kwargs': {}
        },
        'm2d': {
            'class_name': 'M2D',
            'sample_rate': 32000,
            'init_kwargs': {}
        },
        'panns': {
            'variants': {
                'model_wavegram_logmel_cnn14': {'checkpoint': 'Wavegram_Logmel_Cnn14_mAP=0.439.pth'},
                'model_resnet38': {'checkpoint': 'ResNet38_mAP=0.434.pth'}
            }
        },
        'audiomae': {
            'class_name': 'AudioMAE',
            'sample_rate': 16000,
            'init_kwargs': {}
        },
        'epanns': {
            'class_name': 'Cnn14Pruned',
            'sample_rate': 32000,
            'init_kwargs': {}
        },
        'convnext': {
            'class_name': 'ConvNeXt',
            'sample_rate': 32000,
            'init_kwargs': {}
        },
        'passt': {
            'class_name': 'PaSST',
            'sample_rate': 32000,
            'init_kwargs': {}
        },
    }
    
    def __init__(self, models_dir: str):
        """
        Initialize model registry.
        
        Args:
            models_dir: Path to models directory
        """
        self.models_dir = Path(models_dir)
    
    def discover_models(self, silent: bool = False) -> List[ModelConfig]:
        """
        Discover all available models in the models directory.
        
        Args:
            silent: If True, suppress warning messages
        
        Returns:
            List of ModelConfig objects
        """
        configs = []
        
        for model_name, config_info in self.MODEL_CONFIGS.items():
            model_dir = self.models_dir / model_name
            
            if not model_dir.exists():
                if not silent:
                    print(f"Warning: Model directory not found: {model_dir}")
                continue
            
            # Handle models with variants
            if 'variants' in config_info:
                variants_info = config_info['variants']
                if isinstance(variants_info, dict):
                    # Format with checkpoint specification
                    for variant, variant_data in variants_info.items():
                        variant_config = self._create_variant_config(model_name, variant, 
                                                                     model_dir, 
                                                                     checkpoint_name=variant_data.get('checkpoint'),
                                                                     silent=silent)
                        if variant_config:
                            configs.append(variant_config)
                else:
                    # List of variant names
                    for variant in variants_info:
                        variant_config = self._create_variant_config(model_name, variant, model_dir, silent=silent)
                        if variant_config:
                            configs.append(variant_config)
            else:
                # Single model
                checkpoint = self._find_checkpoint(model_dir)
                if checkpoint:
                    config = ModelConfig(name=model_name,
                                         module_path=f"models.{model_name}.model",
                                         class_name=config_info['class_name'],
                                         sample_rate=config_info['sample_rate'],
                                         checkpoint_path=str(checkpoint),
                                         init_kwargs=config_info.get('init_kwargs', {}))
                    configs.append(config)
                else:
                    if not silent:
                        print(f"Warning: No checkpoint found for {model_name}")
        
        return configs
    
    def _create_variant_config(self, model_name: str, variant: str, 
                               model_dir: Path, checkpoint_name: Optional[str] = None,
                               silent: bool = False) -> Optional[ModelConfig]:
        """Create config for model variant."""
        checkpoint = self._find_checkpoint(model_dir, specific_name=checkpoint_name)
        if not checkpoint:
            if not silent:
                print(f"Warning: No checkpoint found for {model_name}/{variant}")
            return None
        
        # Special handling for different variants
        if model_name == 'efficientat':
            if variant == 'model_mn':
                class_name = 'EfficientAT_MN'
                name = 'efficientat_mn'
                sample_rate = 32000
            elif variant == 'model_dymn':
                class_name = 'EfficientAT_DyMN'
                name = 'efficientat_dymn'
                sample_rate = 32000
            else:
                return None
        elif model_name == 'panns':
            if variant == 'model_wavegram_logmel_cnn14':
                class_name = 'Wavegram_Logmel_Cnn14'
                name = 'panns_wavegram_logmel_cnn14'
                sample_rate = 32000
            elif variant == 'model_resnet38':
                class_name = 'ResNet38'
                name = 'panns_resnet38'
                sample_rate = 32000
            else:
                return None
        else:
            return None
        
        return ModelConfig(name=name,
                           module_path=f"models.{model_name}.{variant}",
                           class_name=class_name,
                           sample_rate=sample_rate,
                           checkpoint_path=str(checkpoint),
                           init_kwargs={})
    
    def _find_checkpoint(self, model_dir: Path, specific_name: Optional[str] = None) -> Optional[Path]:
        """
        Find checkpoint file in model directory.
        
        Args:
            model_dir: Path to model directory
            specific_name: Specific checkpoint filename to look for
            
        Returns:
            Path to checkpoint file or None
        """
        # If specific name provided, look for it directly
        if specific_name:
            specific_path = model_dir / specific_name
            if specific_path.exists():
                return specific_path
        
        # Priority keywords for checkpoint selection
        priority_keywords = ['audioset', 'pretrained', 'best', 'final', 'finetuned']
        
        # Search patterns (added .ckpt for htsat)
        patterns = ['*.pth', '*.pt', '*.ckpt']
        
        for pattern in patterns:
            matches = list(model_dir.glob(pattern))
            
            if matches:
                # Try to find checkpoint with priority keywords
                for keyword in priority_keywords:
                    for checkpoint in matches:
                        if keyword in checkpoint.name.lower():
                            return checkpoint
                
                # If no priority match, return first one
                return matches[0]
        
        return None
    
    def load_model(self, config: ModelConfig) -> nn.Module:
        """
        Load and initialize a model from its configuration.
        
        Args:
            config: ModelConfig object
            
        Returns:
            Initialized PyTorch model
        """
        # Import the model module
        try:
            module = importlib.import_module(config.module_path)
            model_class = getattr(module, config.class_name)
        except (ImportError, AttributeError) as e:
            raise RuntimeError(f"Failed to import {config.class_name} from {config.module_path}: {e}")
        
        # Initialize model
        model = model_class(**config.init_kwargs)
        
        # Load checkpoint if it has load_pretrained method
        if hasattr(model, 'load_pretrained'):
            model.load_pretrained(config.checkpoint_path)
        else:
            # Load state dict directly
            state_dict = torch.load(config.checkpoint_path, map_location='cpu')
            # Handle different checkpoint formats
            if 'model' in state_dict:
                state_dict = state_dict['model']
            elif 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            model.load_state_dict(state_dict, strict=False)
        
        model.eval()
        return model


# =============================================================================
# Results Management
# =============================================================================

class ResultsManager:
    """Manages loading, saving, and merging of profiling results."""
    
    def __init__(self, output_dir: str):
        """
        Initialize results manager.
        
        Args:
            output_dir: Directory to store results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def load_existing_results(self, model_name: str) -> Dict[str, Any]:
        """
        Load existing results for a model if they exist.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary with existing results or empty dict
        """
        result_file = self.output_dir / f"{model_name}_stats.json"
        
        if result_file.exists():
            try:
                with open(result_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Could not load existing results for {model_name}: {e}")
        
        return {}
    
    def save_results(self, model_name: str, results: Dict[str, Any]):
        """
        Save results for a model.
        
        Args:
            model_name: Name of the model
            results: Dictionary with profiling results
        """
        result_file = self.output_dir / f"{model_name}_stats.json"
        
        # Add metadata
        results['profiling_date'] = datetime.now().isoformat()
        results['model_name'] = model_name
        
        try:
            with open(result_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"  Saved results to {result_file}")
        except Exception as e:
            print(f"Error: Could not save results for {model_name}: {e}")
    
    def merge_results(self, existing: Dict[str, Any], 
                     new_data: Dict[str, Any], 
                     device: str) -> Dict[str, Any]:
        """
        Merge new profiling data with existing results.
        
        Args:
            existing: Existing results dictionary
            new_data: New data to merge
            device: Device name for the new data
            
        Returns:
            Merged results dictionary
        """
        # Update device-specific results
        existing[device] = new_data
        return existing
