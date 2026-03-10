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

# =============================================================================
# EPOCH RESAMPLING CALLBACK (for KineScaper unified training)
# =============================================================================

class EpochResamplingCallback(_ModelCheckpoint.__bases__[0]):
    """
    Callback to re-sample KineScaper positive chunks at the start of each epoch.
    
    This enables epoch-to-epoch diversity in unified training by randomly
    sampling different positive chunks from KineScaper-EV while maintaining
    stratification across siren classes.
    
    The callback looks for datasets with a set_epoch() method and calls it
    with the current epoch number. This triggers re-sampling in:
    - KineScaper_PositiveChunkDataset
    
    Usage:
        callbacks = [
            EpochResamplingCallback(),
            ModelCheckpoint(...)
        ]
    
    Benefits:
    - Increases diversity of seen samples over training
    - Prevents overfitting to specific subset
    - Maintains class balance through stratified sampling
    
    Example:
        With 234K available positives, using 82K per epoch:
        - Epoch 0: samples indices [1,5,9,...]
        - Epoch 1: samples indices [3,8,12,...]  (different!)
        - Epoch 2: samples indices [2,7,10,...]  (different again!)
        
        Over ~3 epochs, all 234K samples seen at least once.
        Over 50 epochs, each sample seen ~17x on average.
    """
    
    def on_train_epoch_start(self, trainer, pl_module):
        """
        Called at the start of each training epoch.
        Triggers re-sampling in datasets that support it.
        
        Args:
            trainer: PyTorch Lightning Trainer
            pl_module: LightningModule being trained
        """
        current_epoch = trainer.current_epoch
        
        # Get train dataloader
        if hasattr(trainer, 'train_dataloader') and trainer.train_dataloader is not None:
            train_dataloader = trainer.train_dataloader
            
            # Handle DataLoader or DataLoaderIterator
            if hasattr(train_dataloader, 'dataset'):
                dataset = train_dataloader.dataset
            elif hasattr(train_dataloader, 'loaders'):
                # CombinedLoader case
                dataset = train_dataloader.loaders.dataset if hasattr(train_dataloader.loaders, 'dataset') else None
            else:
                dataset = None
            
            # Recursively search for datasets with set_epoch method
            self._trigger_resampling(dataset, current_epoch, trainer)
    
    def _trigger_resampling(self, dataset, epoch: int, trainer):
        """
        Recursively trigger re-sampling in nested datasets.
        
        Handles:
        - ConcatDataset (searches sub-datasets)
        - Subset (searches underlying dataset)
        - Custom datasets with set_epoch() method
        
        Args:
            dataset: Dataset to check/trigger
            epoch: Current epoch number
            trainer: Trainer instance for logging
        """
        if dataset is None:
            return
        
        # Check if dataset has set_epoch method
        if hasattr(dataset, 'set_epoch') and callable(dataset.set_epoch):
            dataset.set_epoch(epoch)
            if trainer.is_global_zero:
                dataset_name = dataset.__class__.__name__
                print(f"  ↻ Re-sampled {dataset_name} for epoch {epoch}")
        
        # Handle ConcatDataset (search all sub-datasets)
        if hasattr(dataset, 'datasets'):
            for sub_dataset in dataset.datasets:
                self._trigger_resampling(sub_dataset, epoch, trainer)
        
        # Handle Subset (check underlying dataset)
        if hasattr(dataset, 'dataset'):
            self._trigger_resampling(dataset.dataset, epoch, trainer)