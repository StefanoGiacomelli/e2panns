"""
Lightning Modules for Emergency Vehicle Recognition
===================================================
PyTorch Lightning wrappers for GP-AT models (E-PANNs, CED, CLAP).

Author: Stefano Giacomelli - Ph.D. candidate in ICT (DISIM dpt. - University of L'Aquila)
"""

import os
import json
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
import pytorch_lightning as pl
from torchmetrics import Accuracy, Precision, Recall, F1Score, Specificity, AUROC, ConfusionMatrix, FBetaScore

# Import custom scheduler
from models.scheduler import CyclicCosineDecayLR

# Import models
from models.epanns.model import EPANNs
from models.ced.model import CEDBase
from models.clap.model import CLAP


# =============================================================================
# MODEL FACTORY
# =============================================================================

def load_model(model_name: str, pretrained: bool = True) -> nn.Module:
    """
    Load a GP-AT model with optional pretrained weights.
    
    Args:
        model_name: One of 'epanns', 'ced', 'clap'
        pretrained: Whether to load pretrained AudioSet weights
    
    Returns:
        Initialized model (torch.nn.Module)
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    if model_name == 'epanns':
        model = EPANNs(sample_rate=32000)
        if pretrained:
            checkpoint_path = os.path.join(script_dir, 'epanns', 'checkpoint_closeto_.44.pt')
            model.load_pretrained(checkpoint_path)
            print(f"✓ Loaded E-PANNs pretrained weights from {checkpoint_path}")
    
    elif model_name == 'ced':
        model = CEDBase()
        if pretrained:
            checkpoint_path = os.path.join(script_dir, 'ced', 'audiotransformer_base_mAP_4999.pt')
            model.load_pretrained(checkpoint_path)
            print(f"✓ Loaded CED pretrained weights from {checkpoint_path}")
    
    elif model_name == 'clap':
        model = CLAP(sample_rate=48000)
        if pretrained:
            checkpoint_path = os.path.join(script_dir, 'clap', '630k-audioset-fusion-best.pt')
            model.load_pretrained(checkpoint_path)
            print(f"✓ Loaded CLAP pretrained weights from {checkpoint_path}")
    
    else:
        raise ValueError(f"Unknown model_name: {model_name}. Must be 'epanns', 'ced', or 'clap'.")
    
    return model


# =============================================================================
# BINARY EV CLASSIFIER
# =============================================================================

class BinaryEVClassifier(pl.LightningModule):
    """
    Binary Emergency Vehicle Classifier.
    
    Task: Classify audio as Emergency Vehicle (EV) or Non-EV.
    Uses AudioSet class 322 (/m/03j1ly - "Emergency vehicle").
    """
    
    def __init__(self,
                 model_name: str,
                 pretrained: bool = True,
                 threshold: float = 0.5,
                 optimizer_kwargs: Optional[Dict] = None,
                 scheduler_type: str = 'cyclic_cosine',
                 scheduler_kwargs: Optional[Dict] = None,
                 results_path: str = './results',
                 f_beta: float = 0.5):
        """
        Args:
            model_name: 'epanns', 'ced', or 'clap'
            pretrained: Load pretrained AudioSet weights
            threshold: Classification threshold for binary decision
            optimizer_kwargs: Adam optimizer parameters
            scheduler_type: 'cyclic_cosine' or 'exponential'
            scheduler_kwargs: Scheduler parameters
            results_path: Path to save results
            f_beta: Beta value for F-beta score
        """
        super().__init__()
        
        self.model_name = model_name
        self.threshold = threshold
        self.f_beta = f_beta
        self.results_path = results_path
        
        # Ensure results directory exists
        os.makedirs(self.results_path, exist_ok=True)
        os.makedirs(os.path.join(self.results_path, 'validation'), exist_ok=True)
        os.makedirs(os.path.join(self.results_path, 'test'), exist_ok=True)
        
        # AudioSet class index for Emergency Vehicle
        self.ev_class_idx = 322
        
        # Load model
        self.model = load_model(model_name, pretrained)
        
        # Optimizer and scheduler configuration
        self.optimizer_kwargs = optimizer_kwargs or {'lr': 1e-3,
                                                     'weight_decay': 1e-6,
                                                     'betas': (0.9, 0.999),
                                                     'eps': 1e-8,
                                                     'amsgrad': True}
        self.scheduler_type = scheduler_type
        self.scheduler_kwargs = scheduler_kwargs or {}
        
        # Loss
        self.criterion = nn.BCELoss()
        
        # Initialize metrics
        self._init_metrics()
        
        # Storage for validation/test predictions
        self.val_epoch_outputs = []
        self.test_predictions = []
        self.test_targets = []
        self.test_file_paths = []
        
        # Save hyperparameters
        self.save_hyperparameters(ignore=['model'])
    
    def _init_metrics(self):
        """Initialize torchmetrics for train/val/test stages."""
        # Training metrics
        self.train_accuracy = Accuracy(task='binary', threshold=self.threshold)
        
        # Validation metrics
        self.val_accuracy = Accuracy(task='binary', threshold=self.threshold)
        self.val_precision = Precision(task='binary', threshold=self.threshold)
        self.val_recall = Recall(task='binary', threshold=self.threshold)
        self.val_f1 = F1Score(task='binary', threshold=self.threshold)
        
        # Test metrics
        self.test_accuracy = Accuracy(task='binary', threshold=self.threshold)
        self.test_precision = Precision(task='binary', threshold=self.threshold)
        self.test_recall = Recall(task='binary', threshold=self.threshold)
        self.test_specificity = Specificity(task='binary', threshold=self.threshold)
        self.test_f1 = F1Score(task='binary', threshold=self.threshold)
        self.test_auroc = AUROC(task='binary')
        self.test_confusion_matrix = ConfusionMatrix(task='binary', num_classes=2, threshold=self.threshold)
        self.test_fbeta = FBetaScore(task='binary', beta=self.f_beta, threshold=self.threshold)
    
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: extract Emergency Vehicle probability.
        
        Args:
            waveform: (batch, samples) or (batch, 1, samples) audio tensor
        
        Returns:
            ev_prob: (batch,) Emergency Vehicle probabilities
        """
        # Handle shape: squeeze if (batch, 1, samples) -> (batch, samples)
        if waveform.ndim == 3 and waveform.shape[1] == 1:
            waveform = waveform.squeeze(1)
        
        # Get full AudioSet predictions
        full_probs = self.model(waveform)  # (batch, 527)
        
        # Extract EV probability (class 322)
        ev_prob = full_probs[:, self.ev_class_idx]  # (batch,)
        
        return ev_prob
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        waveform, target = batch
        
        # Forward pass
        ev_prob = self(waveform)
        
        # Loss
        loss = self.criterion(ev_prob, target.float())
        
        # Metrics
        preds = (ev_prob >= self.threshold).float()
        self.train_accuracy(preds, target)
        
        # Logging
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_accuracy', self.train_accuracy, on_epoch=True, prog_bar=True)
        
        # Log learning rate
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('learning_rate', current_lr, on_step=True, on_epoch=False)
        
        return loss
    
    def on_train_epoch_end(self):
        """End of training epoch."""
        self.train_accuracy.reset()
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        waveform, target = batch
        
        # Forward pass
        ev_prob = self(waveform)
        
        # Loss
        loss = self.criterion(ev_prob, target.float())
        
        # Metrics
        preds = (ev_prob >= self.threshold).float()
        self.val_accuracy(preds, target)
        self.val_precision(preds, target)
        self.val_recall(preds, target)
        self.val_f1(preds, target)
        
        # Store outputs for epoch-end processing
        self.val_epoch_outputs.append({'loss': loss.item(),
                                       'preds': preds.detach().cpu(),
                                       'targets': target.detach().cpu(),
                                       'probs': ev_prob.detach().cpu()})
        
        # Logging
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def on_validation_epoch_end(self):
        """End of validation epoch - compute and save metrics."""
        # Compute metrics
        val_acc = self.val_accuracy.compute()
        val_prec = self.val_precision.compute()
        val_rec = self.val_recall.compute()
        val_f1 = self.val_f1.compute()
        
        # Log to TensorBoard
        self.log('val_accuracy', val_acc, prog_bar=True)
        self.log('val_precision', val_prec, prog_bar=True)
        self.log('val_recall', val_rec, prog_bar=True)
        self.log('val_f1', val_f1, prog_bar=True)
        
        # Save metrics to JSON
        metrics = {'epoch': self.current_epoch,
                   'accuracy': val_acc.item(),
                   'precision': val_prec.item(),
                   'recall': val_rec.item(),
                   'f1_score': val_f1.item()}
        
        metrics_path = os.path.join(self.results_path, 'validation', 
                                    f'epoch_{self.current_epoch:03d}_metrics.json')
        
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Reset metrics
        self.val_accuracy.reset()
        self.val_precision.reset()
        self.val_recall.reset()
        self.val_f1.reset()
        self.val_epoch_outputs.clear()
    
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        """Test step - collect predictions for final evaluation."""
        waveform, target = batch
        
        # Forward pass
        ev_prob = self(waveform)
        
        # Store predictions and targets
        self.test_predictions.append(ev_prob.detach().cpu())
        self.test_targets.append(target.detach().cpu())
    
    def on_test_epoch_end(self):
        """End of test epoch - compute all metrics and save results."""
        # Aggregate predictions and move to CPU for metric computation
        all_probs = torch.cat(self.test_predictions).cpu()  # (N,)
        all_targets = torch.cat(self.test_targets).cpu()  # (N,)
        all_preds = (all_probs >= self.threshold).float()
        
        # Move all test metrics to CPU to avoid device mismatch
        self.test_accuracy = self.test_accuracy.to('cpu')
        self.test_precision = self.test_precision.to('cpu')
        self.test_recall = self.test_recall.to('cpu')
        self.test_specificity = self.test_specificity.to('cpu')
        self.test_f1 = self.test_f1.to('cpu')
        self.test_auroc = self.test_auroc.to('cpu')
        self.test_fbeta = self.test_fbeta.to('cpu')
        self.test_confusion_matrix = self.test_confusion_matrix.to('cpu')
        
        # Compute all test metrics on CPU
        test_acc = self.test_accuracy(all_preds, all_targets)
        test_prec = self.test_precision(all_preds, all_targets)
        test_rec = self.test_recall(all_preds, all_targets)
        test_spec = self.test_specificity(all_preds, all_targets)
        test_f1 = self.test_f1(all_preds, all_targets)
        test_auroc = self.test_auroc(all_probs, all_targets)
        test_fbeta = self.test_fbeta(all_preds, all_targets)
        conf_matrix = self.test_confusion_matrix(all_preds, all_targets)
        
        # Extract TP, FP, TN, FN from confusion matrix
        tn = conf_matrix[0, 0]
        fp = conf_matrix[0, 1]
        fn = conf_matrix[1, 0]
        tp = conf_matrix[1, 1]
        
        # Build metrics dictionary
        metrics = {'accuracy': test_acc.item(),
                   'precision': test_prec.item(),
                   'recall': test_rec.item(),
                   'specificity': test_spec.item(),
                   'f1_score': test_f1.item(),
                   'auroc': test_auroc.item(),
                   'fbeta_score': test_fbeta.item(),
                   'confusion_matrix': {'tn': int(conf_matrix[0, 0]),
                                        'fp': int(conf_matrix[0, 1]),
                                        'fn': int(conf_matrix[1, 0]),
                                        'tp': int(conf_matrix[1, 1])}
                   }
        
        # Log to TensorBoard
        for key, value in metrics.items():
            if not isinstance(value, dict):
                self.log(f'test_{key}', value)
        
        # Print to terminal
        print("\n" + "=" * 80)
        print("TEST RESULTS - BINARY EV CLASSIFICATION")
        print("=" * 80)
        print(f"Accuracy:      {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"Precision:     {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
        print(f"Recall:        {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
        print(f"Specificity:   {metrics['specificity']:.4f} ({metrics['specificity']*100:.2f}%)")
        print(f"F1 Score:      {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.2f}%)")
        print(f"F-beta Score:  {metrics['fbeta_score']:.4f} ({metrics['fbeta_score']*100:.2f}%)")
        print(f"AUROC:         {metrics['auroc']:.4f} ({metrics['auroc']*100:.2f}%)")
        print("\nConfusion Matrix:")
        print(f"  TN: {metrics['confusion_matrix']['tn']:<6} FP: {metrics['confusion_matrix']['fp']}")
        print(f"  FN: {metrics['confusion_matrix']['fn']:<6} TP: {metrics['confusion_matrix']['tp']}")
        print("=" * 80 + "\n")
        
        # Save metrics to JSON
        metrics_json_path = os.path.join(self.results_path, 'test', 'test_metrics.json')
        with open(metrics_json_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"✓ Test metrics saved to: {metrics_json_path}")
        
        # Save predictions to NPZ
        predictions_npz_path = os.path.join(self.results_path, 'test', 'test_predictions.npz')
        np.savez_compressed(predictions_npz_path,
                            probs=all_probs.numpy(),
                            targets=all_targets.numpy())
        print(f"✓ Test predictions saved to: {predictions_npz_path}")
        
        # Clear storage
        self.test_predictions.clear()
        self.test_targets.clear()
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        # Optimizer
        optimizer = optim.Adam(self.model.parameters(), **self.optimizer_kwargs)
        
        # Scheduler
        if self.scheduler_type == 'cyclic_cosine':
            scheduler = CyclicCosineDecayLR(optimizer, **self.scheduler_kwargs)
        elif self.scheduler_type == 'exponential':
            scheduler = ExponentialLR(optimizer, **self.scheduler_kwargs)
        else:
            raise ValueError(f"Unknown scheduler_type: {self.scheduler_type}")
        
        return [optimizer], [scheduler]
    
    def on_save_checkpoint(self, checkpoint):
        """Override to save only PyTorch model state_dict."""
        # Save full Lightning checkpoint as usual
        # But also save standalone PyTorch model weights
        checkpoint['pytorch_model_state_dict'] = self.model.state_dict()


# =============================================================================
# MULTI-CLASS SIREN CLASSIFIER
# =============================================================================

class MultiClassSirenClassifier(pl.LightningModule):
    """
    Multi-Class Siren Type Classifier.
    
    Task: 4-way classification:
      - Class 0: Negative (Non-EV)
      - Class 1: Police car siren
      - Class 2: Ambulance siren
      - Class 3: Fire engine siren
    
    Uses AudioSet classes:
      - 322: /m/03j1ly - "Emergency vehicle" (for deriving class 0)
      - 323: /m/04qvtq - "Police car (siren)"
      - 324: /m/012n7d - "Ambulance (siren)"
      - 325: /m/012ndj - "Fire engine, fire truck (siren)"
    """
    
    def __init__(self,
                 model_name: str,
                 pretrained: bool = True,
                 num_classes: int = 4,
                 optimizer_kwargs: Optional[Dict] = None,
                 scheduler_type: str = 'cyclic_cosine',
                 scheduler_kwargs: Optional[Dict] = None,
                 results_path: str = './results',
                 f_beta: float = 0.5):
        """
        Args:
            model_name: 'epanns', 'ced', or 'clap'
            pretrained: Load pretrained AudioSet weights
            num_classes: Number of classes (always 4)
            optimizer_kwargs: Adam optimizer parameters
            scheduler_type: 'cyclic_cosine' or 'exponential'
            scheduler_kwargs: Scheduler parameters
            results_path: Path to save results
            f_beta: Beta value for F-beta score
        """
        super().__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        self.f_beta = f_beta
        self.results_path = results_path
        
        # Ensure results directory exists
        os.makedirs(self.results_path, exist_ok=True)
        os.makedirs(os.path.join(self.results_path, 'validation'), exist_ok=True)
        os.makedirs(os.path.join(self.results_path, 'test'), exist_ok=True)
        
        # AudioSet class indices
        self.ev_general_idx = 322  # Emergency vehicle (for deriving class 0)
        self.police_idx = 323      # Police car (siren) - class 1
        self.ambulance_idx = 324   # Ambulance (siren) - class 2
        self.fire_idx = 325        # Fire engine (siren) - class 3
        
        # Load model
        self.model = load_model(model_name, pretrained)
        
        # Optimizer and scheduler configuration
        self.optimizer_kwargs = optimizer_kwargs or {'lr': 1e-3,
                                                     'weight_decay': 1e-6,
                                                     'betas': (0.9, 0.999),
                                                     'eps': 1e-8,
                                                     'amsgrad': True}
        self.scheduler_type = scheduler_type
        self.scheduler_kwargs = scheduler_kwargs or {}
        
        # Loss
        self.criterion = nn.CrossEntropyLoss()
        
        # Initialize metrics
        self._init_metrics()
        
        # Storage for validation/test predictions
        self.val_epoch_outputs = []
        self.test_predictions = []
        self.test_targets = []
        
        # Save hyperparameters
        self.save_hyperparameters(ignore=['model'])
    
    def _init_metrics(self):
        """Initialize torchmetrics for train/val/test stages."""
        # Training metrics
        self.train_accuracy = Accuracy(task='multiclass', num_classes=self.num_classes)
        
        # Validation metrics
        self.val_accuracy = Accuracy(task='multiclass', num_classes=self.num_classes)
        self.val_precision = Precision(task='multiclass', num_classes=self.num_classes, average='macro')
        self.val_recall = Recall(task='multiclass', num_classes=self.num_classes, average='macro')
        self.val_f1 = F1Score(task='multiclass', num_classes=self.num_classes, average='macro')
        
        # Test metrics
        self.test_accuracy = Accuracy(task='multiclass', num_classes=self.num_classes)
        self.test_precision = Precision(task='multiclass', num_classes=self.num_classes, average='macro')
        self.test_recall = Recall(task='multiclass', num_classes=self.num_classes, average='macro')
        self.test_specificity = Specificity(task='multiclass', num_classes=self.num_classes, average='macro')
        self.test_f1 = F1Score(task='multiclass', num_classes=self.num_classes, average='macro')
        self.test_auroc = AUROC(task='multiclass', num_classes=self.num_classes, average='macro')
        self.test_confusion_matrix = ConfusionMatrix(task='multiclass', num_classes=self.num_classes)
        self.test_fbeta = FBetaScore(task='multiclass', num_classes=self.num_classes, beta=self.f_beta, average='macro')
    
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: extract 4-class probabilities and convert to logits.
        
        Args:
            waveform: (batch, samples) or (batch, 1, samples) audio tensor
        
        Returns:
            logits_4way: (batch, 4) logits for CrossEntropyLoss
        """
        # Handle shape: squeeze if (batch, 1, samples) -> (batch, samples)
        if waveform.ndim == 3 and waveform.shape[1] == 1:
            waveform = waveform.squeeze(1)
        
        # Get full AudioSet predictions
        full_probs = self.model(waveform)  # (batch, 527)
        
        # Extract relevant probabilities
        ev_general_prob = full_probs[:, self.ev_general_idx]  # (batch,)
        police_prob = full_probs[:, self.police_idx]          # (batch,)
        ambulance_prob = full_probs[:, self.ambulance_idx]    # (batch,)
        fire_prob = full_probs[:, self.fire_idx]              # (batch,)
        
        # Build 4-way probabilities: [negative, police, ambulance, fire]
        probs_4way = torch.stack([1 - ev_general_prob,  # Class 0: Negative (inverse of EV)
                                  police_prob,          # Class 1: Police
                                  ambulance_prob,       # Class 2: Ambulance
                                  fire_prob             # Class 3: Fire
                                  ], dim=1)  # (batch, 4)
        
        # Convert to logits (clamp for numerical stability)
        probs_clamped = torch.clamp(probs_4way, 1e-7, 1 - 1e-7)
        logits_4way = torch.logit(probs_clamped)  # (batch, 4)
        
        return logits_4way
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        waveform, target = batch
        
        # Forward pass
        logits = self(waveform)
        
        # Loss
        loss = self.criterion(logits, target)
        
        # Metrics
        preds = torch.argmax(logits, dim=1)
        self.train_accuracy(preds, target)
        
        # Logging
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_accuracy', self.train_accuracy, on_epoch=True, prog_bar=True)
        
        # Log learning rate
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('learning_rate', current_lr, on_step=True, on_epoch=False)
        
        return loss
    
    def on_train_epoch_end(self):
        """End of training epoch."""
        self.train_accuracy.reset()
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        waveform, target = batch
        
        # Forward pass
        logits = self(waveform)
        
        # Loss
        loss = self.criterion(logits, target)
        
        # Metrics
        preds = torch.argmax(logits, dim=1)
        self.val_accuracy(preds, target)
        self.val_precision(preds, target)
        self.val_recall(preds, target)
        self.val_f1(preds, target)
        
        # Store outputs
        self.val_epoch_outputs.append({'loss': loss.item(),
                                       'preds': preds.detach().cpu(),
                                       'targets': target.detach().cpu(),
                                       'logits': logits.detach().cpu()})
        
        # Logging
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def on_validation_epoch_end(self):
        """End of validation epoch - compute and save metrics."""
        # Compute metrics
        val_acc = self.val_accuracy.compute()
        val_prec = self.val_precision.compute()
        val_rec = self.val_recall.compute()
        val_f1 = self.val_f1.compute()
        
        # Log to TensorBoard
        self.log('val_accuracy', val_acc, prog_bar=True)
        self.log('val_precision', val_prec, prog_bar=True)
        self.log('val_recall', val_rec, prog_bar=True)
        self.log('val_f1', val_f1, prog_bar=True)
        
        # Save metrics to JSON
        metrics = {'epoch': self.current_epoch,
                   'accuracy': val_acc.item(),
                   'precision': val_prec.item(),
                   'recall': val_rec.item(),
                   'f1_score': val_f1.item()}
        
        metrics_path = os.path.join(self.results_path, 'validation', f'epoch_{self.current_epoch:03d}_metrics.json')
        
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Reset metrics
        self.val_accuracy.reset()
        self.val_precision.reset()
        self.val_recall.reset()
        self.val_f1.reset()
        self.val_epoch_outputs.clear()
    
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        """Test step - collect predictions for final evaluation."""
        waveform, target = batch
        
        # Forward pass
        logits = self(waveform)
        
        # Store predictions and targets
        self.test_predictions.append(logits.detach().cpu())
        self.test_targets.append(target.detach().cpu())
    
    def on_test_epoch_end(self):
        """End of test epoch - compute all metrics and save results."""
        # Aggregate predictions and move to CPU for metric computation
        all_logits = torch.cat(self.test_predictions).cpu()   # (N, 4)
        all_targets = torch.cat(self.test_targets).cpu()      # (N,)
        all_preds = torch.argmax(all_logits, dim=1)
        all_probs = torch.softmax(all_logits, dim=1)    # Convert logits to probs for AUROC
        
        # Move all test metrics to CPU to avoid device mismatch
        self.test_accuracy = self.test_accuracy.to('cpu')
        self.test_precision = self.test_precision.to('cpu')
        self.test_recall = self.test_recall.to('cpu')
        self.test_specificity = self.test_specificity.to('cpu')
        self.test_f1 = self.test_f1.to('cpu')
        self.test_auroc = self.test_auroc.to('cpu')
        self.test_fbeta = self.test_fbeta.to('cpu')
        self.test_confusion_matrix = self.test_confusion_matrix.to('cpu')
        
        # Compute all test metrics on CPU
        test_acc = self.test_accuracy(all_preds, all_targets)
        test_prec = self.test_precision(all_preds, all_targets)
        test_rec = self.test_recall(all_preds, all_targets)
        test_spec = self.test_specificity(all_preds, all_targets)
        test_f1 = self.test_f1(all_preds, all_targets)
        test_auroc = self.test_auroc(all_probs, all_targets)
        test_fbeta = self.test_fbeta(all_preds, all_targets)
        conf_matrix = self.test_confusion_matrix(all_preds, all_targets)
        
        # Build metrics dictionary
        metrics = {'accuracy': test_acc.item(),
                   'precision': test_prec.item(),
                   'recall': test_rec.item(),
                   'specificity': test_spec.item(),
                   'f1_score': test_f1.item(),
                   'auroc': test_auroc.item(),
                   'fbeta_score': test_fbeta.item(),
                   'confusion_matrix': conf_matrix.tolist()
                   }
        
        # Log to TensorBoard
        for key, value in metrics.items():
            if not isinstance(value, (dict, list)):
                self.log(f'test_{key}', value)
        
        # Print to terminal
        print("\n" + "=" * 80)
        print("TEST RESULTS - MULTI-CLASS SIREN CLASSIFICATION")
        print("=" * 80)
        print(f"Accuracy:      {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"Precision:     {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
        print(f"Recall:        {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
        print(f"Specificity:   {metrics['specificity']:.4f} ({metrics['specificity']*100:.2f}%)")
        print(f"F1 Score:      {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.2f}%)")
        print(f"F-beta Score:  {metrics['fbeta_score']:.4f} ({metrics['fbeta_score']*100:.2f}%)")
        print(f"AUROC:         {metrics['auroc']:.4f} ({metrics['auroc']*100:.2f}%)")
        print("\nConfusion Matrix (rows=true, cols=pred):")
        conf_matrix_np = conf_matrix.numpy()
        print("       Neg  Police  Amb  Fire")
        for i, label in enumerate(['Neg', 'Police', 'Amb', 'Fire']):
            print(f"{label:6s} {conf_matrix_np[i, 0]:4.0f}  {conf_matrix_np[i, 1]:6.0f}  {conf_matrix_np[i, 2]:3.0f}  {conf_matrix_np[i, 3]:4.0f}")
        print("=" * 80 + "\n")
        
        # Save metrics to JSON
        metrics_json_path = os.path.join(self.results_path, 'test', 'test_metrics.json')
        with open(metrics_json_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"✓ Test metrics saved to: {metrics_json_path}")
        
        # Save predictions to NPZ
        predictions_npz_path = os.path.join(self.results_path, 'test', 'test_predictions.npz')
        np.savez_compressed(predictions_npz_path, probs=all_probs.numpy(), targets=all_targets.numpy())
        print(f"✓ Test predictions saved to: {predictions_npz_path}")
        
        # Clear storage
        self.test_predictions.clear()
        self.test_targets.clear()
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        # Optimizer
        optimizer = optim.Adam(self.model.parameters(), **self.optimizer_kwargs)
        
        # Scheduler
        if self.scheduler_type == 'cyclic_cosine':
            scheduler = CyclicCosineDecayLR(optimizer, **self.scheduler_kwargs)
        elif self.scheduler_type == 'exponential':
            scheduler = ExponentialLR(optimizer, **self.scheduler_kwargs)
        else:
            raise ValueError(f"Unknown scheduler_type: {self.scheduler_type}")
        
        return [optimizer], [scheduler]
    
    def on_save_checkpoint(self, checkpoint):
        """Override to save PyTorch model state_dict."""
        # Save full Lightning checkpoint as usual
        # But also save standalone PyTorch model weights
        checkpoint['pytorch_model_state_dict'] = self.model.state_dict()
