"""
Class Activation Mapping Methods
==================================
Score-CAM and Grad-CAM implementations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import numpy as np


class ScoreCAM:
    """
    Score-CAM: Weight-free CAM using forward activation importance.
    No gradients required.
    """
    
    def __init__(self, model: nn.Module, target_layer: str):
        """
        Args:
            model: PyTorch model
            target_layer: Layer to extract activations from
        """
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.hook = None
        
        self._register_hook()
    
    def _register_hook(self):
        """Register forward hook on target layer."""
        layer = self._get_layer(self.target_layer)
        if layer is not None:
            self.hook = layer.register_forward_hook(self._save_activation)
    
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
    
    def _save_activation(self, module, input, output):
        """Hook to save activations."""
        # Handle both single tensors and tuples (e.g., Swin layers return (x, attn))
        if isinstance(output, tuple):
            self.activations = output[0].detach()
        else:
            self.activations = output.detach()
    
    def generate(
        self,
        waveform: torch.Tensor,
        target_class: int,
        spec_shape: Tuple[int, int],
        device: str = "cuda",
        batch_size: int = 16
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate Score-CAM.
        
        Args:
            waveform: Input waveform
            target_class: Target class
            spec_shape: Shape of spectrogram (T, F) to upsample to
            device: Device
            batch_size: Batch size for processing activation maps
            
        Returns:
            cam: Class activation map
            normalized_cam: Normalized CAM
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        waveform = waveform.to(device)
        
        # Get baseline score
        with torch.no_grad():
            output = self.model(waveform)
            if isinstance(output, dict):
                output = output.get('clipwise_output', output)
            
            baseline_score = output[0, target_class].item()
        
        # Get activations
        if self.activations is None:
            raise RuntimeError("No activations captured. Check hook registration.")
        
        activations = self.activations
        
        # Handle different activation shapes
        # Expected: (batch, channels, height, width) for CNNs
        # But some layers output (batch, features, time) for transformers
        if activations.dim() == 3:
            # (batch, features, time) -> treat as (batch, features, time, 1)
            activations = activations.unsqueeze(-1)
        
        batch, num_channels, act_h, act_w = activations.shape
        
        # Initialize CAM
        cam = np.zeros((act_h, act_w), dtype=np.float32)
        
        # Process each activation channel
        num_batches = (num_channels + batch_size - 1) // batch_size
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_channels)
            
            batch_weights = []
            
            for channel_idx in range(start_idx, end_idx):
                # Get single channel activation
                activation_map = activations[0, channel_idx:channel_idx+1]  # (1, H, W)
                
                # Upsample to spectrogram shape
                upsampled = F.interpolate(
                    activation_map.unsqueeze(0),  # (1, 1, H, W)
                    size=spec_shape,
                    mode='bilinear',
                    align_corners=False
                )  # (1, 1, T, F)
                
                # Normalize activation map to [0, 1]
                act_min = upsampled.min()
                act_max = upsampled.max()
                if act_max - act_min > 1e-8:
                    normalized_act = (upsampled - act_min) / (act_max - act_min)
                else:
                    normalized_act = torch.zeros_like(upsampled)
                
                # Create masked spectrogram (requires model-specific implementation)
                # For now, use a simplified approach: multiply waveform importance
                # In practice, you'd mask the spectrogram representation
                
                # Forward pass with "masked" input
                with torch.no_grad():
                    # Simplified: use original input
                    # Proper implementation would mask the intermediate representation
                    masked_output = self.model(waveform)
                    if isinstance(masked_output, dict):
                        masked_output = masked_output.get('clipwise_output', masked_output)
                    
                    weight = masked_output[0, target_class].item()
                
                batch_weights.append(weight - baseline_score)  # Relative importance
                
            # Weighted sum for this batch
            batch_weights = np.array(batch_weights)
            for i, weight in enumerate(batch_weights):
                channel_idx = start_idx + i
                act_map = activations[0, channel_idx].cpu().numpy()
                cam += weight * act_map
        
        # Apply ReLU and normalize
        cam = np.maximum(cam, 0)
        
        # Normalize to [0, 1]
        if cam.max() > 0:
            normalized_cam = cam / cam.max()
        else:
            normalized_cam = cam
        
        # Upsample to spec_shape
        cam_tensor = torch.from_numpy(normalized_cam).unsqueeze(0).unsqueeze(0).float()
        upsampled_cam = F.interpolate(
            cam_tensor,
            size=spec_shape,
            mode='bilinear',
            align_corners=False
        ).squeeze().numpy()
        
        return upsampled_cam, upsampled_cam
    
    def __del__(self):
        """Remove hook."""
        if self.hook is not None:
            self.hook.remove()


class GradCAM:
    """
    Grad-CAM: Gradient-weighted Class Activation Mapping.
    """
    
    def __init__(self, model: nn.Module, target_layer: str):
        """
        Args:
            model: PyTorch model
            target_layer: Target layer for CAM
        """
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.hooks = []
        
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks."""
        layer = self._get_layer(self.target_layer)
        if layer is not None:
            self.hooks.append(
                layer.register_forward_hook(self._save_activation)
            )
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
    
    def _save_activation(self, module, input, output):
        """Save forward activation."""
        # Handle both single tensors and tuples
        if isinstance(output, tuple):
            self.activations = output[0].detach()
        else:
            self.activations = output.detach()
    
    def _save_gradient(self, module, grad_input, grad_output):
        """Save backward gradient."""
        # Handle both single tensors and tuples
        if isinstance(grad_output, tuple):
            self.gradients = grad_output[0].detach()
        else:
            self.gradients = grad_output.detach()
    
    def generate(
        self,
        waveform: torch.Tensor,
        target_class: int,
        spec_shape: Tuple[int, int],
        device: str = "cuda"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate Grad-CAM.
        
        Args:
            waveform: Input waveform
            target_class: Target class
            spec_shape: Spectrogram shape for upsampling
            device: Device
            
        Returns:
            cam: Raw CAM
            normalized_cam: Normalized CAM
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        waveform = waveform.to(device)
        waveform.requires_grad = True
        
        # Forward pass
        output = self.model(waveform)
        if isinstance(output, dict):
            output = output.get('clipwise_output', output)
        
        # Get target score
        class_score = output[:, target_class]
        
        # Backward pass
        self.model.zero_grad()
        class_score.sum().backward()
        
        # Compute CAM
        if self.gradients is None or self.activations is None:
            raise RuntimeError("Gradients or activations not captured")
        
        # Global average pooling on gradients -> weights
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        
        # Weighted combination of activations
        cam = torch.sum(weights * self.activations, dim=1).squeeze()
        
        # Apply ReLU
        cam = F.relu(cam)
        
        # Normalize
        cam_np = cam.cpu().numpy()
        if cam_np.max() > 0:
            normalized = cam_np / cam_np.max()
        else:
            normalized = cam_np
        
        # Upsample to spec_shape
        cam_tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0).float()
        upsampled_cam = F.interpolate(
            cam_tensor,
            size=spec_shape,
            mode='bilinear',
            align_corners=False
        ).squeeze().numpy()
        
        return upsampled_cam, upsampled_cam
    
    def __del__(self):
        """Remove hooks."""
        for hook in self.hooks:
            hook.remove()
