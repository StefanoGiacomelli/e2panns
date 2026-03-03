"""
Gradient-based Explainability Methods
======================================
Vanilla and Guided Backpropagation implementations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np


class VanillaBackprop:
    """
    Vanilla Backpropagation for gradient computation.
    Computes raw gradients w.r.t. input or intermediate layers.
    """
    
    def __init__(self, model: nn.Module, target_layer: Optional[str] = None):
        """
        Args:
            model: PyTorch model
            target_layer: Layer name to compute gradients for (None = input)
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.hook = None
        
        if target_layer:
            self._register_hook()
    
    def _register_hook(self):
        """Register backward hook on target layer."""
        layer = self._get_layer(self.target_layer)
        if layer is not None:
            self.hook = layer.register_full_backward_hook(self._save_gradient)
    
    def _get_layer(self, name: str) -> Optional[nn.Module]:
        """Get layer by name."""
        parts = name.split('.')
        layer = self.model
        for part in parts:
            if hasattr(layer, part):
                layer = getattr(layer, part)
            else:
                try:
                    layer = layer[int(part)]
                except (ValueError, TypeError, IndexError):
                    return None
        return layer
    
    def _save_gradient(self, module, grad_input, grad_output):
        """Hook function to save gradients."""
        self.gradients = grad_output[0].detach()
    
    def generate(
        self,
        waveform: torch.Tensor,
        target_class: int,
        device: str = "cuda"
    ) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Generate vanilla backprop gradients.
        
        Args:
            waveform: Input waveform (1D or batched)
            target_class: Target class for gradient computation
            device: Device to use
            
        Returns:
            raw_gradients: Raw gradient tensor
            normalized_gradients: Normalized gradients as numpy array
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        waveform = waveform.to(device)
        waveform.requires_grad = True
        
        # For intermediate layers, capture activations and compute grads directly
        if self.target_layer:
            activation_to_save = None
            
            def forward_hook(module, input, output):
                nonlocal activation_to_save
                activation_to_save = output
                if isinstance(activation_to_save, tuple):
                    activation_to_save = activation_to_save[0]
                # Retain gradient on this tensor
                if hasattr(activation_to_save, 'retain_grad'):
                    activation_to_save.retain_grad()
            
            layer = self._get_layer(self.target_layer)
            if layer is None:
                raise ValueError(f"Layer {self.target_layer} not found")
            
            hook_handle = layer.register_forward_hook(forward_hook)
            
            # Forward pass
            output = self.model(waveform)
            if isinstance(output, dict):
                output = output.get('clipwise_output', output)
            
            # Get target class score
            class_score = output[:, target_class]
            
            # Backward to compute gradients
            self.model.zero_grad()
            class_score.sum().backward()
            
            # Get the gradient from the retained tensor
            if activation_to_save is not None and hasattr(activation_to_save, 'grad') and activation_to_save.grad is not None:
                raw_grads = activation_to_save.grad
            else:
                # Fallback: create zero gradients with same shape as activation
                if activation_to_save is not None:
                    raw_grads = torch.zeros_like(activation_to_save)
                else:
                    raw_grads = torch.zeros(1, 1)
            
            hook_handle.remove()
        else:
            # For input gradients
            # Forward pass
            output = self.model(waveform)
            if isinstance(output, dict):
                output = output.get('clipwise_output', output)
            
            # Get target class score
            class_score = output[:, target_class]
            
            # Backward pass
            self.model.zero_grad()
            class_score.sum().backward()
            raw_grads = waveform.grad
        
        if raw_grads is None:
            raise RuntimeError("No gradients computed")
        
        # Process gradients
        raw_grads = raw_grads.detach().cpu()
        
        # Handle different gradient shapes
        # - Waveform: (batch, time) -> (time,)
        # - After spectrogram: (batch, channels, time, freq) -> need to process
        # - CED front_end: (batch, n_mels, time) -> (n_mels, time) -> need transpose to (time, n_mels)
        
        grads_np = raw_grads.squeeze(0).numpy()  # Remove batch dimension
        
        if grads_np.ndim == 3:
            # Shape: (channels, time, freq)
            grads_np = np.abs(grads_np)
            grads_np = grads_np.mean(axis=0)  # -> (time, freq)
        elif grads_np.ndim == 2:
            # Could be (time, freq) or (freq, time)
            grads_np = np.abs(grads_np)
            
            # Detect if this is (freq, time) format (like CED)
            # Heuristic: if first dimension is small (like n_mels=64) and second is large (time frames)
            if grads_np.shape[0] < 200 and grads_np.shape[1] > 500:
                # Likely (freq, time) -> transpose to (time, freq)
                grads_np = grads_np.T
        elif grads_np.ndim == 1:
            # Linear (time,) - reshape to column
            grads_np = grads_np.reshape(-1, 1)
        
        # Normalize
        grads_min = grads_np.min()
        grads_max = grads_np.max()
        
        if grads_max - grads_min > 1e-8:
            normalized = (grads_np - grads_min) / (grads_max - grads_min)
        else:
            normalized = np.zeros_like(grads_np)
        
        return raw_grads, normalized
    
    def __del__(self):
        """Remove hook on deletion."""
        if self.hook is not None:
            self.hook.remove()


class GuidedBackprop:
    """
    Guided Backpropagation.
    Modifies ReLU backward pass to only propagate positive gradients.
    """
    
    def __init__(self, model: nn.Module, target_layer: Optional[str] = None):
        """
        Args:
            model: PyTorch model
            target_layer: Layer name to compute gradients for (None = input)
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.forward_relu_outputs = []
        self.hooks = []
        
        self._register_hooks()
    
    def _register_hooks(self):
        """Register hooks on ReLUs and target layer."""
        # Hook all ReLU layers
        for module in self.model.modules():
            if isinstance(module, nn.ReLU):
                self.hooks.append(
                    module.register_forward_hook(self._relu_forward_hook)
                )
                self.hooks.append(
                    module.register_full_backward_hook(self._relu_backward_hook)
                )
        
        # Hook target layer if specified
        if self.target_layer:
            layer = self._get_layer(self.target_layer)
            if layer is not None:
                self.hooks.append(
                    layer.register_full_backward_hook(self._save_gradient)
                )
    
    def _get_layer(self, name: str) -> Optional[nn.Module]:
        """Get layer by name."""
        parts = name.split('.')
        layer = self.model
        for part in parts:
            if hasattr(layer, part):
                layer = getattr(layer, part)
            else:
                try:
                    layer = layer[int(part)]
                except (ValueError, TypeError, IndexError):
                    return None
        return layer
    
    def _relu_forward_hook(self, module, input_tensor, output_tensor):
        """Store ReLU forward outputs."""
        self.forward_relu_outputs.append(output_tensor)
    
    def _relu_backward_hook(self, module, grad_input, grad_output):
        """Modify ReLU backward pass (Guided Backprop)."""
        if len(self.forward_relu_outputs) > 0:
            forward_output = self.forward_relu_outputs.pop()
            
            # Guided backprop: set negative gradients to zero
            # and only propagate through positive activations
            modified_grad_input = grad_input[0].clone()
            modified_grad_input[forward_output <= 0] = 0  # Positive activations
            modified_grad_input = torch.clamp(modified_grad_input, min=0)  # Positive gradients
            
            return (modified_grad_input,)
        return grad_input
    
    def _save_gradient(self, module, grad_input, grad_output):
        """Hook to save gradients at target layer."""
        self.gradients = grad_output[0].detach()
    
    def generate(
        self,
        waveform: torch.Tensor,
        target_class: int,
        device: str = "cuda",
        return_spectrogram: bool = False
    ) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Generate guided backprop gradients.
        
        Args:
            waveform: Input waveform
            target_class: Target class
            device: Device
            return_spectrogram: If True, also return log-mel spectrogram
            
        Returns:
            raw_gradients: Raw gradient tensor
            normalized_gradients: Normalized numpy array
        """
        self.forward_relu_outputs = []
        
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        waveform = waveform.to(device)
        waveform.requires_grad = True
        
        # Forward pass
        output = self.model(waveform)
        if isinstance(output, dict):
            output = output.get('clipwise_output', output)
        
        # Get target class score
        class_score = output[:, target_class]
        
        # Backward pass
        self.model.zero_grad()
        class_score.sum().backward()
        
        # Get gradients
        if self.target_layer:
            raw_grads = self.gradients
        else:
            raw_grads = waveform.grad
        
        if raw_grads is None:
            raise RuntimeError("No gradients computed")
        
        # Process gradients
        raw_grads = raw_grads.detach().cpu()
        
        
        # Handle different gradient shapes based on target layer
        # Expected shapes:
        # - Waveform: (batch, time) -> (time,)
        # - After spectrogram: (batch, channels, time, freq) -> (time, freq)
        # - After conv: (batch, channels, time, freq) -> (time, freq)
        # - CED front_end: (batch, n_mels, time) -> (time, n_mels)
        
        grads_np = raw_grads.squeeze(0).numpy()  # Remove batch dimension
        
        if grads_np.ndim == 3:
            # Shape: (channels, time, freq)
            # Take absolute value and mean over channels
            grads_np = np.abs(grads_np)
            grads_np = grads_np.mean(axis=0)  # -> (time, freq)
        elif grads_np.ndim == 2:
            # Could be (time, freq), (freq, time), or (channels, time)
            grads_np = np.abs(grads_np)
            
            # Detect if this is (freq, time) format (like CED)
            # Heuristic: if first dimension is small (like n_mels=64) and second is large (time frames)
            if grads_np.shape[0] < 200 and grads_np.shape[1] > 500:
                # Likely (freq, time) -> transpose to (time, freq)
                grads_np = grads_np.T
        elif grads_np.ndim == 1:
            # Linear (time,) - reshape to column
            grads_np = grads_np.reshape(-1, 1)
        
        # Normalize to [0, 1]
        grads_min = grads_np.min()
        grads_max = grads_np.max()
        
        if grads_max - grads_min > 1e-8:
            normalized = (grads_np - grads_min) / (grads_max - grads_min)
        else:
            normalized = np.zeros_like(grads_np)
        
        return raw_grads, normalized
    
    def __del__(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
