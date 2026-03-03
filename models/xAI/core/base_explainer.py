"""
Base Explainer Abstract Class
==============================
Base class for all model explainers with common interfaces.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any
import torch
import torch.nn as nn
import torchaudio
import numpy as np
from pathlib import Path


class BaseExplainer(ABC):
    """
    Abstract base class for model explainability.
    
    Provides common interface for all explainers (CNN, Transformer, CLAP).
    """
    
    def __init__(
        self,
        model: nn.Module,
        model_name: str,
        sample_rate: int,
        device: str = "cuda",
        target_class: int = 322,  # Emergency Vehicle class in AudioSet
    ):
        """
        Initialize base explainer.
        
        Args:
            model: PyTorch model
            model_name: Name of the model (epanns, ced, clap)
            sample_rate: Audio sample rate
            device: Device to run computations
            target_class: Target class index for explanation
        """
        self.model = model
        self.model_name = model_name
        self.sample_rate = sample_rate
        self.device = device
        self.target_class = target_class
        
        self.model.to(device)
        self.model.eval()
        
        # Storage for activations and gradients
        self.activations = {}
        self.gradients = {}
        self.hooks = []
        
    def load_audio(
        self,
        audio_path: str,
        target_duration: float = 10.0
    ) -> torch.Tensor:
        """
        Load and preprocess audio file.
        
        Args:
            audio_path: Path to audio file
            target_duration: Target duration in seconds
            
        Returns:
            Preprocessed waveform tensor (1D)
        """
        waveform, sr = torchaudio.load(audio_path)
        
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        # Pad or truncate
        target_length = int(self.sample_rate * target_duration)
        if waveform.shape[1] < target_length:
            waveform = torch.nn.functional.pad(
                waveform, (0, target_length - waveform.shape[1])
            )
        else:
            waveform = waveform[:, :target_length]
        
        return waveform.squeeze(0)  # Return 1D tensor
    
    def get_prediction(self, waveform: torch.Tensor) -> Dict[str, float]:
        """
        Get model prediction for waveform.
        
        Args:
            waveform: Input waveform (1D tensor)
            
        Returns:
            Dictionary with prediction info
        """
        waveform = waveform.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(waveform)
            
            # Handle different output types
            if isinstance(output, dict):
                probs = output.get('clipwise_output', output)
            else:
                probs = output
            
            target_prob = probs[0, self.target_class].item()
            predicted_class = torch.argmax(probs[0]).item()
            predicted_prob = probs[0, predicted_class].item()
        
        return {
            'target_probability': target_prob,
            'predicted_class': predicted_class,
            'predicted_probability': predicted_prob,
            'is_target_predicted': predicted_class == self.target_class
        }
    
    def register_hooks(self, layer_names: List[str]):
        """
        Register forward and backward hooks on specified layers.
        
        Args:
            layer_names: List of layer names to hook
        """
        self.remove_hooks()  # Clean up existing hooks
        
        for name in layer_names:
            layer = self._get_layer_by_name(name)
            if layer is not None:
                # Forward hook
                h_fwd = layer.register_forward_hook(
                    self._make_forward_hook(name)
                )
                # Backward hook
                h_bwd = layer.register_full_backward_hook(
                    self._make_backward_hook(name)
                )
                self.hooks.extend([h_fwd, h_bwd])
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.activations = {}
        self.gradients = {}
    
    def _make_forward_hook(self, name: str):
        """Create forward hook function."""
        def hook(module, input, output):
            self.activations[name] = output.detach()
        return hook
    
    def _make_backward_hook(self, name: str):
        """Create backward hook function."""
        def hook(module, grad_input, grad_output):
            self.gradients[name] = grad_output[0].detach()
        return hook
    
    def _get_layer_by_name(self, name: str) -> Optional[nn.Module]:
        """
        Get layer by name using dot notation.
        
        Args:
            name: Layer name (e.g., 'conv_block1', 'blocks.0')
            
        Returns:
            Layer module or None if not found
        """
        try:
            # Handle nested attributes
            parts = name.split('.')
            layer = self.model
            for part in parts:
                if hasattr(layer, part):
                    layer = getattr(layer, part)
                else:
                    # Try numeric indexing
                    try:
                        layer = layer[int(part)]
                    except (ValueError, TypeError, IndexError):
                        return None
            return layer
        except Exception as e:
            print(f"Warning: Could not find layer '{name}': {e}")
            return None
    
    @abstractmethod
    def extract_spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Extract spectrogram/mel-spectrogram from waveform.
        Model-specific implementation.
        
        Args:
            waveform: Input waveform
            
        Returns:
            Spectrogram tensor
        """
        pass
    
    @abstractmethod
    def get_attention_weights(self, waveform: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Extract attention weights if applicable.
        
        Args:
            waveform: Input waveform
            
        Returns:
            Attention weights or None if not applicable
        """
        pass
    
    def normalize_saliency(
        self,
        saliency_map: np.ndarray,
        method: str = "minmax"
    ) -> np.ndarray:
        """
        Normalize saliency map to [0, 1].
        
        Args:
            saliency_map: Raw saliency values
            method: Normalization method ('minmax', 'percentile')
            
        Returns:
            Normalized saliency map
        """
        if method == "minmax":
            smap_min = saliency_map.min()
            smap_max = saliency_map.max()
            if smap_max - smap_min < 1e-8:
                return np.zeros_like(saliency_map)
            normalized = (saliency_map - smap_min) / (smap_max - smap_min)
            
        elif method == "percentile":
            p1, p99 = np.percentile(saliency_map, [1, 99])
            normalized = np.clip((saliency_map - p1) / (p99 - p1 + 1e-8), 0, 1)
            
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        return normalized
    
    def __del__(self):
        """Cleanup hooks on deletion."""
        self.remove_hooks()
