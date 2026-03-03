"""
Sensitivity Metrics
===================
Faithfulness metrics: Deletion, Insertion, Average Drop.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List, Callable
from tqdm import tqdm


class DeletionMetric:
    """
    Deletion metric: progressively remove most salient regions and measure drop in confidence.
    Lower area under curve (AUC) = better explanation.
    """
    
    def __init__(self, model: nn.Module, target_class: int, num_steps: int = 20):
        """
        Args:
            model: PyTorch model
            target_class: Target class index
            num_steps: Number of deletion steps
        """
        self.model = model
        self.target_class = target_class
        self.num_steps = num_steps
    
    def compute(
        self,
        waveform: torch.Tensor,
        saliency_map: np.ndarray,
        forward_func: Callable,
        device: str = "cuda"
    ) -> Tuple[List[float], float]:
        """
        Compute deletion curve.
        
        Args:
            waveform: Input waveform
            saliency_map: Saliency map (same temporal resolution)
            forward_func: Function to get model output
            device: Device
            
        Returns:
            scores: Confidence scores at each deletion step
            auc: Area under deletion curve
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        waveform = waveform.to(device)
        
        # Get baseline score
        with torch.no_grad():
            baseline_output = forward_func(waveform)
            baseline_score = baseline_output[0, self.target_class].item()
        
        # Flatten and sort saliency
        if saliency_map.ndim > 1:
            # For 2D saliency (time-frequency), average over frequency
            saliency_1d = saliency_map.mean(axis=-1) if saliency_map.shape[-1] > 1 else saliency_map.flatten()
        else:
            saliency_1d = saliency_map
        
        # Resize to match waveform length if needed
        if len(saliency_1d) != waveform.shape[1]:
            # Interpolate
            saliency_1d = np.interp(
                np.linspace (0, len(saliency_1d) - 1, waveform.shape[1]),
                np.arange(len(saliency_1d)),
                saliency_1d
            )
        
        # Get indices sorted by importance (descending)
        sorted_indices = np.argsort(saliency_1d)[::-1].copy()  # .copy() to avoid negative strides
        
        # Progressively delete
        scores = [baseline_score]
        masked_waveform = waveform.clone()
        
        indices_per_step = len(sorted_indices) // self.num_steps
        
        for step in range(1, self.num_steps + 1):
            # Delete top salient indices
            end_idx = step * indices_per_step
            delete_indices = sorted_indices[:end_idx]
            
            # Zero out these regions
            masked_waveform[0, delete_indices] = 0
            
            # Get new score
            with torch.no_grad():
                output = forward_func(masked_waveform)
                score = output[0, self.target_class].item()
            
            scores.append(score)
        
        # Compute AUC (normalized)
        # Handle edge case where baseline_score is very small
        if baseline_score < 1e-6:
            # If baseline is already near zero, AUC is meaningless
            # Return 1.0 (max) as deletion has no effect
            auc = 1.0
        else:
            auc = np.trapz(scores, dx=1.0 / self.num_steps) / baseline_score
        
        return scores, auc


class AverageDropMetric:
    """
    Average Drop: measures drop in confidence when removing top-k% salient regions.
    """
    
    def __init__(self, model: nn.Module, target_class: int, top_k_percent: float = 0.1):
        """
        Args:
            model: Model
            target_class: Target class
            top_k_percent: Percentage of most salient regions to remove (0-1)
        """
        self.model = model
        self.target_class = target_class
        self.top_k_percent = top_k_percent
    
    def compute(
        self,
        waveform: torch.Tensor,
        saliency_map: np.ndarray,
        forward_func: Callable,
        device: str = "cuda"
    ) -> float:
        """
        Compute average drop.
        
        Args:
            waveform: Input waveform
            saliency_map: Saliency map
            forward_func: Forward function
            device: Device
            
        Returns:
            average_drop: AD metric value
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        waveform = waveform.to(device)
        
        # Original score
        with torch.no_grad():
            orig_output = forward_func(waveform)
            orig_score = orig_output[0, self.target_class].item()
        
        # Process saliency
        if saliency_map.ndim > 1:
            saliency_1d = saliency_map.mean(axis=-1) if saliency_map.shape[-1] > 1 else saliency_map.flatten()
        else:
            saliency_1d = saliency_map
        
        # Resize
        if len(saliency_1d) != waveform.shape[1]:
            saliency_1d = np.interp(
                np.linspace(0, len(saliency_1d) - 1, waveform.shape[1]),
                np.arange(len(saliency_1d)),
                saliency_1d
            )
        
        # Get top-k indices
        k = int(len(saliency_1d) * self.top_k_percent)
        top_k_indices = np.argsort(saliency_1d)[::-1][:k].copy()  # .copy() to avoid negative strides
        
        # Mask top-k
        masked_waveform = waveform.clone()
        masked_waveform[0, top_k_indices] = 0
        
        # New score
        with torch.no_grad():
            masked_output = forward_func(masked_waveform)
            masked_score = masked_output[0, self.target_class].item()
        
        # Compute AD
        if orig_score < 1e-6:
            # If original score is near zero, average drop is not meaningful
            average_drop = 0.0
        elif orig_score > 0:
            average_drop = max(0, orig_score - masked_score) / orig_score
        else:
            average_drop = 0.0
        
        return average_drop
        return average_drop
