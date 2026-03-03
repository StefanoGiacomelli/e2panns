"""
CNN Explainer for EPANNs
=========================
Explainability methods for CNN-based models (EPANNs).
"""

import torch
import torch.nn as nn
from typing import Optional, Dict
import numpy as np

from .base_explainer import BaseExplainer


class CNNExplainer(BaseExplainer):
    """Explainer for CNN-based models (EPANNs)."""
    
    def __init__(
        self,
        model: nn.Module,
        model_name: str = "epanns",
        sample_rate: int = 32000,
        device: str = "cuda",
        target_class: int = 322
    ):
        super().__init__(model, model_name, sample_rate, device, target_class)
        
        # Access the underlying CNN14 model
        if hasattr(model, 'model'):
            self.cnn_model = model.model
        else:
            self.cnn_model = model
    
    def extract_spectrogram(self, waveform: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract spectrogram and log-mel features from EPANNs.
        
        Args:
            waveform: Input waveform (1D)
            
        Returns:
            Dictionary with 'spectrogram' and 'logmel' tensors
        """
        waveform = waveform.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Extract spectrogram
            spec = self.cnn_model.spectrogram_extractor(waveform)  # (1, 1, T, F)
            
            # Extract log-mel
            logmel = self.cnn_model.logmel_extractor(spec)  # (1, 1, T, mel_bins)
        
        return {
            'spectrogram': spec.squeeze().cpu(),  # (T, F)
            'logmel': logmel.squeeze().cpu()  # (T, mel_bins)
        }
    
    def get_attention_weights(self, waveform: torch.Tensor) -> Optional[torch.Tensor]:
        """CNNs don't have attention. Returns None."""
        return None
    
    def get_filterbank_weights(self) -> np.ndarray:
        """
        Get learned mel filterbank weights from the model.
        
        Returns:
            Filterbank matrix (n_mels, n_freqs)
        """
        with torch.no_grad():
            for name, param in self.cnn_model.logmel_extractor.named_parameters():
                if 'melW' in name:
                    return param.cpu().numpy().T  # Transpose to (n_mels, n_freqs)
        
        raise ValueError("Could not find melW weights in logmel_extractor")
    
    def get_conv_block_names(self) -> list:
        """Get names of all conv blocks."""
        return [f"conv_block{i}" for i in range(1, 7)]
