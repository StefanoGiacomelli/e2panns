"""
CLAP Explainer for CLAP/HTSAT
===============================
Explainability methods for CLAP (Swin Transformer-based).
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List
import numpy as np

from .base_explainer import BaseExplainer


class CLAPExplainer(BaseExplainer):
    """Explainer for CLAP (HTSAT Swin Transformer)."""
    
    def __init__(
        self,
        model: nn.Module,
        model_name: str = "clap",
        sample_rate: int = 48000,
        device: str = "cuda",
        target_class: int = 322
    ):
        super().__init__(model, model_name, sample_rate, device, target_class)
        
        # Access the audio branch (HTSAT)
        if hasattr(model, 'model') and hasattr(model.model, 'audio_branch'):
            self.audio_model = model.model.audio_branch
        elif hasattr(model, 'audio_branch'):
            self.audio_model = model.audio_branch
        else:
            self.audio_model = model
        
        # Storage for attention (window attention in Swin)
        self.window_attention_maps = []
    
    def extract_spectrogram(self, waveform: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract spectrogram and log-mel from CLAP/HTSAT.
        
        Args:
            waveform: Input waveform (1D)
            
        Returns:
            Dictionary with 'spectrogram' and 'logmel'
        """
        waveform = waveform.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Extract spectrogram (similar to EPANNs)
            spec = self.audio_model.spectrogram_extractor(waveform)  # (1, 1, T, F)
            
            # Extract log-mel
            logmel = self.audio_model.logmel_extractor(spec)  # (1, 1, T, mel_bins)
        
        return {
            'spectrogram': spec.squeeze().cpu(),  # (T, F)
            'logmel': logmel.squeeze().cpu()  # (T, mel_bins)
        }
    
    def get_attention_weights(
        self,
        waveform: torch.Tensor,
        layer_idx: Optional[int] = None
    ) -> List[torch.Tensor]:
        """
        Extract window attention weights from Swin Transformer layers.
        
        Args:
            waveform: Input waveform
            layer_idx: Specific layer index or None for all layers
            
        Returns:
            List of attention weight tensors
        """
        self.window_attention_maps = []
        
        # Register hooks on Swin attention modules
        # This is architecture-specific
        hooks = []
        
        if hasattr(self.audio_model, 'layers'):
            layers_to_hook = [self.audio_model.layers[layer_idx]] if layer_idx is not None else self.audio_model.layers
            
            for layer in layers_to_hook:
                # Swin has WindowAttention in each block
                # Need to traverse the structure
                hook = layer.register_forward_hook(self._swin_attention_hook)
                hooks.append(hook)
        
        # Forward pass
        waveform = waveform.unsqueeze(0).to(self.device)
        with torch.no_grad():
            _ = self.model(waveform)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        return self.window_attention_maps
    
    def _swin_attention_hook(self, module, input, output):
        """Hook for capturing Swin window attention."""
        # Store output (may need refinement based on actual structure)
        self.window_attention_maps.append(output.detach().cpu())
    
    def get_filterbank_weights(self) -> np.ndarray:
        """
        Get learned mel filterbank weights from CLAP.
        
        Returns:
            Filterbank matrix (n_mels, n_freqs)
        """
        with torch.no_grad():
            for name, param in self.audio_model.logmel_extractor.named_parameters():
                if 'melW' in name:
                    return param.cpu().numpy().T  # Transpose to (n_mels, n_freqs)
        
        raise ValueError("Could not find melW weights in logmel_extractor")
    
    def get_tscam_features(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Extract Time-Semantic CAM features from HTSAT.
        
        Args:
            waveform: Input waveform
            
        Returns:
            TSCAM features
        """
        waveform = waveform.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Forward through model up to TSCAM layer
            # This requires accessing intermediate features
            # Simplified version - may need adjustment
            
            # Get mel spectrogram
            spec = self.audio_model.spectrogram_extractor(waveform)
            logmel = self.audio_model.logmel_extractor(spec)
            
            # Process through Swin layers
            # (Implementation depends on exact HTSAT structure)
            
            # Apply TSCAM convolution if exists
            if hasattr(self.audio_model, 'tscam_conv'):
                # Get features before TSCAM
                # This is a simplified placeholder
                features = logmel  # Replace with actual intermediate features
                tscam = self.audio_model.tscam_conv(features)
                return tscam.squeeze().cpu()
        
        return None
    
    def get_layer_names(self) -> List[str]:
        """Get names of Swin transformer layers."""
        if hasattr(self.audio_model, 'layers'):
            return [f"layers.{i}" for i in range(len(self.audio_model.layers))]
        return []
