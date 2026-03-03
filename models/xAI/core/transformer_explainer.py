"""
Transformer Explainer for CED
==============================
Explainability methods for Transformer-based models (CED-Base).
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List
import numpy as np

from .base_explainer import BaseExplainer


class TransformerExplainer(BaseExplainer):
    """Explainer for Transformer-based models (CED)."""
    
    def __init__(
        self,
        model: nn.Module,
        model_name: str = "ced",
        sample_rate: int = 16000,
        device: str = "cuda",
        target_class: int = 322
    ):
        super().__init__(model, model_name, sample_rate, device, target_class)
        
        # Storage for attention weights
        self.attention_maps = []
    
    def extract_spectrogram(self, waveform: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract mel-spectrogram from CED front-end.
        
        Args:
            waveform: Input waveform (1D)
            
        Returns:
            Dictionary with 'mel_spectrogram'
        """
        waveform = waveform.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # CED uses torchaudio MelSpectrogram + AmplitudeToDB
            mel_spec = self.model.front_end(waveform)  # (1, 1, n_mels, T)
        
        
        # Transpose to match EPANNs/CLAP format: (T, n_mels)
        mel_spec_np = mel_spec.squeeze().cpu().numpy()  # (n_mels, T)
        
        mel_spec_np = mel_spec_np.T  # (T, n_mels)
        
        return {
            'mel_spectrogram': mel_spec_np
        }
    
    def get_attention_weights(
        self,
        waveform: torch.Tensor,
        average_heads: bool = True
    ) -> List[torch.Tensor]:
        """
        Extract attention weights from all transformer blocks.
        
        Args:
            waveform: Input waveform
            average_heads: Whether to average over attention heads
            
        Returns:
            List of attention weight tensors (one per layer)
        """
        self.attention_maps = []
        hooks = []
        
        # Register hooks on each block's attention module
        for i, block in enumerate(self.model.blocks):
            if hasattr(block, 'attn'):
                # Hook the attention module's forward
                # We need to capture the attention weights before matmul with V
                # Since attention weights aren't returned, we hook attn_drop
                def make_hook(layer_idx):
                    def hook(module, input, output):
                        # The input to attn_drop is the attention weights
                        # Shape: (batch, heads, seq, seq)
                        if isinstance(input, tuple):
                            attn_weights = input[0]
                        else:
                            attn_weights = input
                        self.attention_maps.append(attn_weights.detach().cpu())
                    return hook
                
                # Hook the dropout that comes after softmax
                hook_handle = block.attn.attn_drop.register_forward_hook(make_hook(i))
                hooks.append(hook_handle)
        
        # Forward pass
        waveform = waveform.unsqueeze(0).to(self.device)
        with torch.no_grad():
            _ = self.model(waveform)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Process attention maps
        if average_heads and len(self.attention_maps) > 0:
            attention_maps = []
            for attn in self.attention_maps:
                if attn.dim() == 4:  # (batch, heads, seq, seq)
                    attn_avg = attn.mean(dim=1)  # Average over heads
                    attention_maps.append(attn_avg.squeeze(0))  # Remove batch dim
                else:
                    attention_maps.append(attn.squeeze(0))
        else:
            attention_maps = [attn.squeeze(0) for attn in self.attention_maps]
        
        return attention_maps
    
    def get_patch_embeddings(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Get patch embeddings from the model.
        
        Args:
            waveform: Input waveform
            
        Returns:
            Patch embeddings tensor
        """
        waveform = waveform.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Extract mel spectrogram
            mel_spec = self.model.front_end(waveform)
            
            # Apply batch norm
            mel_spec = self.model.init_bn(mel_spec)
            
            # Get patch embeddings
            patches = self.model.patch_embed(mel_spec)
        
        return patches.squeeze(0).cpu()
    
    def get_block_names(self) -> List[str]:
        """Get names of all transformer blocks."""
        num_blocks = len(self.model.blocks)
        return [f"blocks.{i}" for i in range(num_blocks)]
