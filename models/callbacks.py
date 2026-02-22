"""
Custom PyTorch Lightning Callbacks
===================================
Custom callbacks extending Lightning functionality.

Author: Stefano Giacomelli - Ph.D. candidate in ICT (DISIM dpt. - University of L'Aquila)
"""

import os
import torch
from pytorch_lightning.callbacks import ModelCheckpoint as _ModelCheckpoint


class ModelCheckpoint(_ModelCheckpoint):
    """
    Extended ModelCheckpoint that saves both .ckpt and .pt files.
    
    This callback extends Lightning's ModelCheckpoint to automatically save
    a standalone PyTorch .pt file alongside each .ckpt file. The .pt file
    contains only the model state_dict for easy standalone usage.
    
    The .pt file mirrors the .ckpt file:
    - Same filename (with .pt extension)
    - Same save_top_k logic
    - Same monitor/mode behavior
    - Saved at the same time
    
    Usage:
        callbacks = [
            ModelCheckpoint(
                dirpath='./checkpoints',
                monitor='val_f1',
                save_top_k=1,
                mode='max',
                filename='best'
            )
        ]
    
    Output structure:
        checkpoints/
        ├── best.ckpt          # Lightning checkpoint (full training state)
        ├── best.pt            # PyTorch model only (for inference)
        ├── last.ckpt
        └── last.pt
    """
    
    def _save_checkpoint(self, trainer, filepath: str):
        """
        Override to save both .ckpt and .pt files.
        
        Args:
            trainer: PyTorch Lightning Trainer
            filepath: Path where checkpoint will be saved (.ckpt)
        """
        # 1. Save .ckpt normally using Lightning's logic
        super()._save_checkpoint(trainer, filepath)
        
        # 2. Save standalone .pt with only model state_dict
        pt_filepath = filepath.replace('.ckpt', '.pt')
        
        # Extract model and metadata
        lightning_module = trainer.lightning_module
        
        checkpoint_data = {'epoch': trainer.current_epoch,
                           'global_step': trainer.global_step,
                           'pytorch_model_state_dict': lightning_module.model.state_dict(),
                           'model_name': getattr(lightning_module, 'model_name', 'unknown')}
        
        # Add task-specific metadata
        if hasattr(lightning_module, 'threshold'):
            # Binary classifier
            checkpoint_data['threshold'] = lightning_module.threshold
        
        if hasattr(lightning_module, 'num_classes'):
            # Multi-class classifier
            checkpoint_data['num_classes'] = lightning_module.num_classes
        
        # Add monitored metric value if available
        if self.monitor and self.best_model_score is not None:
            checkpoint_data['monitor_metric'] = self.monitor
            checkpoint_data['monitor_value'] = self.best_model_score.item()
        
        # Save .pt file
        torch.save(checkpoint_data, pt_filepath)
        
        # Optional: print confirmation
        if trainer.is_global_zero:
            print(f"Saved checkpoint pair: {os.path.basename(filepath)} + {os.path.basename(pt_filepath)}")
    
    def _remove_checkpoint(self, trainer, filepath: str):
        """
        Override to remove both .ckpt and .pt files when purging old checkpoints.
        
        This ensures that when Lightning removes an old .ckpt (due to save_top_k),
        the corresponding .pt file is also removed, maintaining consistency.
        
        Args:
            trainer: PyTorch Lightning Trainer
            filepath: Path to checkpoint being removed (.ckpt)
        """
        # 1. Remove .ckpt using Lightning's logic
        super()._remove_checkpoint(trainer, filepath)
        
        # 2. Remove corresponding .pt file
        pt_filepath = filepath.replace('.ckpt', '.pt')
        
        if os.path.exists(pt_filepath):
            try:
                os.remove(pt_filepath)
                if trainer.is_global_zero:
                    print(f"Removed old checkpoint pair: {os.path.basename(filepath)} + {os.path.basename(pt_filepath)}")
            except Exception as e:
                if trainer.is_global_zero:
                    print(f"Warning: Failed to remove {pt_filepath}: {e}")
