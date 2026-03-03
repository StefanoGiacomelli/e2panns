import os
import csv
import time
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics import (Accuracy, ConfusionMatrix, Precision, Recall, F1Score, 
                          Specificity, AUROC, AveragePrecision, MatthewsCorrCoef, 
                          FBetaScore, StatScores)
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pytorch_lightning as pl
from scheduler import CyclicCosineDecayLR


def load_lightning2pt(checkpoint_path, model, verbose=False, validate_updates=True):
    """
    Loads a PyTorch Lightning checkpoint's state_dict into a plain PyTorch model 
    on the CPU and optionally verifies parameter updates.

    :param checkpoint_path: Absolute Path to the Lightning checkpoint file (.ckpt).
    :param model: The plain PyTorch model instance to load the checkpoint into (already on CPU).
    :param verbose: Whether to print detailed information about the loading process.
    :param validate_updates: Whether to validate which layers were updated (default: True).
    :return tuple:
        (model, updated_layers) if validate_updates=True
        (model, None) otherwise
    """
    # Always load on CPU
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    except FileNotFoundError:
        raise ValueError(f"Checkpoint file not found at: {checkpoint_path}")
    except Exception as e:
        raise ValueError(f"Failed to load checkpoint: {e}")

    if "state_dict" not in checkpoint:
        raise ValueError(f"Checkpoint does not contain a 'state_dict'. "
                         f"Keys found: {list(checkpoint.keys())}")

    lightning_state_dict = checkpoint["state_dict"]

    # Attempt to detect & strip prefix
    prefix = None
    for key in lightning_state_dict.keys():
        if "." in key:
            prefix = key.split(".")[0] + "."
            break

    if prefix:
        stripped_state_dict = {key.replace(prefix, ""): value for key, value in lightning_state_dict.items()}
        if verbose:
            print(f"Detected prefix '{prefix}'. Stripped from state_dict keys.")
    else:
        stripped_state_dict = lightning_state_dict
        if verbose:
            print("No prefix detected in state_dict keys.")

    updated_layers = []
    if validate_updates:
        # Compare old vs new params on CPU
        for name, param in model.state_dict().items():
            if name in stripped_state_dict:
                old_param = param.clone()
                new_param = stripped_state_dict[name]
                if verbose:
                    print(f"Validating layer: {name}")
                    print(f"  Old Param: Type: {type(old_param)}, DType: {old_param.dtype}")
                    print(f"  New Param: Type: {type(new_param)}, DType: {new_param.dtype}")
                if not torch.equal(old_param, new_param):
                    updated_layers.append(name)
                    if verbose:
                        diff = (old_param - new_param).float()
                        print(f"  Layer: {name} has changes!")
                        print(f"    Min Diff: {diff.abs().min().item():.6f}")
                        print(f"    Max Diff: {diff.abs().max().item():.6f}")
                        print(f"    Mean Diff: {diff.abs().mean().item():.6f}")
                        print(f"    Std-Dev of Diff: {diff.abs().std().item():.6f}")
                        print(f"    Sample Differences: {diff.flatten()[:5].tolist()}...")
                print('---------------------------------------------------------------------------------')

    try:
        model.load_state_dict(stripped_state_dict)
        if verbose:
            print("State dict successfully loaded into the model!")
    except Exception as e:
        raise ValueError(f"Failed to load state_dict into the model: {e}")

    # Clean up memory
    del checkpoint
    del lightning_state_dict

    if verbose and validate_updates:
        if updated_layers:
            print("The following layers were updated during fine-tuning:")
            for layer in updated_layers:
                print(f" - {layer}")
        else:
            print("No layers were updated. (Possible that no fine-tuning changes occurred.)")

    print('Model correctly loaded from checkpoint.')
    
    return model, updated_layers if validate_updates else None


class E2PANNs_Model(pl.LightningModule):
    def __init__(self,
                 model,
                 threshold=0.5,
                 output_mode='bin_raw',
                 overall_training=False,
                 eta_max=1e-3,
                 eta_min=1e-6,
                 decay_epochs=50,
                 restart_eta=1e-5,
                 restart_interval=10,
                 warmup_epochs=10,
                 warmup_eta=1e-4,
                 weight_decay=1e-6,
                 f_beta=0.8):
        """
        :param model: base-model to fine-tune.
        :param threshold: Probability threshold for predicting the positive class.
        :param output_mode: 'bin_only' => return only the EV prob; 'bin_raw' => store full clipwise output.
        :param overall_training: If False, only last layers are trainable; if True, entire model is trainable.
        :param eta_max: Max LR for Adam.
        :param eta_min: Min LR for Adam (Cosine Annealing).
        :param decay_epochs: Number of epochs before the first LR restart or decay.
        :param restart_eta: LR to use after a restart.
        :param restart_interval: Interval (in epochs) for LR restarts.
        :param warmup_epochs: Number of warmup epochs.
        :param warmup_eta: LR for warmup.
        :param weight_decay: Weight decay for Adam.
        :param f_beta: Beta for F-beta metric in test metrics.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        # --- Ensure the base model is on CPU initially ---
        self.model = model.cpu()

        self.class_idx = 322
        self.threshold = threshold
        self.output_mode = output_mode
        self.overall = overall_training

        # F-beta
        self.beta = f_beta

        # Scheduler / Optim Params
        self.decay_epochs = decay_epochs
        self.eta_min = eta_min
        self.eta_max = eta_max
        self.restart_interval = restart_interval
        self.restart_eta = restart_eta
        self.warmup_epochs = warmup_epochs
        self.warmup_eta = warmup_eta
        self.weight_decay = weight_decay
        self.betas = (0.9, 0.999)
        self.eps = 1e-08

        # Allow partial or full finetuning
        self.setup_model()

        # Loss function
        self.criterion = nn.BCELoss()

        # Init metrics for train/val/test
        self.init_metrics()

        # Buffers for train & val steps
        self.train_step_outputs = []
        self.val_step_outputs = []

        # Buffers for test predictions
        self.preds = []
        self.targets = []
        self.all_preds_storage = []
        self.tot_inference_time = 0.0

        # Prepare results path
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
        self.model_results_path = f"./experiments/{current_time}/model_results/"
        os.makedirs(self.model_results_path, exist_ok=True)

    def setup_model(self):
        if not self.overall:
            print('Finetuning only the last layers.')
            for name, param in self.model.named_parameters():
                if name.startswith("fc1") or name.startswith("fc_audioset"):
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        else:
            print('Full model training: all parameters are trainable.')
            for _, param in self.model.named_parameters():
                param.requires_grad = True

    def init_metrics(self):
        # ------------------ TRAIN METRICS ------------------
        self.train_accuracy = Accuracy(task="binary", num_classes=2, threshold=self.threshold)

        # ------------------ VAL METRICS --------------------
        self.val_accuracy = Accuracy(task="binary", num_classes=2, threshold=self.threshold)
        self.val_precision = Precision(task="binary", num_classes=2, threshold=self.threshold)
        self.val_recall = Recall(task="binary", num_classes=2, threshold=self.threshold)
        self.val_f1 = F1Score(task="binary", num_classes=2, threshold=self.threshold)

        # ------------------ TEST METRICS -------------------
        self.test_accuracy = Accuracy(task="binary", num_classes=2, threshold=self.threshold)
        self.test_confusion_matrix = ConfusionMatrix(task="binary", num_classes=2, threshold=self.threshold)
        self.test_precision = Precision(task="binary", num_classes=2, threshold=self.threshold)
        self.test_recall = Recall(task="binary", num_classes=2, threshold=self.threshold)
        self.test_f1 = F1Score(task="binary", num_classes=2, threshold=self.threshold)
        self.test_specificity = Specificity(task="binary", num_classes=2, threshold=self.threshold)
        self.test_auroc = AUROC(task="binary", num_classes=2)
        self.test_auprc = AveragePrecision(task="binary", num_classes=2)
        self.test_mcc = MatthewsCorrCoef(task="binary", num_classes=2, threshold=self.threshold)
        self.test_fbeta = FBetaScore(task="binary", num_classes=2, threshold=self.threshold, beta=self.beta)
        self.test_stat_scores = StatScores(task="binary", num_classes=2, threshold=self.threshold)

    # ----------------------------------------------------------------
    #                  LOAD TRAINED WEIGHTS (from .ckpt)
    # ----------------------------------------------------------------
    def load_trained_weights(self, checkpoint_path: str, verbose: bool = False, validate_updates: bool = True):
        """
        Load a Lightning .ckpt into self.model parameters on CPU using 'load_lightning2pt'.
        
        :param checkpoint_path: Path to the Lightning checkpoint.
        :param verbose: Print detailed info if True.
        :param validate_updates: Compare old vs new params to see which changed.
        :return: List of updated layers (if validate_updates=True), else None.
        """
        self.model, updated_layers = load_lightning2pt(checkpoint_path=checkpoint_path,
                                                       model=self.model,
                                                       verbose=verbose,
                                                       validate_updates=validate_updates)

        print(f'Model correctly loaded from checkpoint: {checkpoint_path}')
        return updated_layers

    # ----------------------------------------------------------------
    #                       FORWARD PASS
    # ----------------------------------------------------------------
    def forward(self, x):
        logits_dict = self.model(x.float().squeeze())  # e.g., {'clipwise_output': [B x #classes]}
        logits = logits_dict['clipwise_output']
        emergency_prob = logits[:, self.class_idx]
        return emergency_prob

    # ----------------------------------------------------------------
    #                       TRAINING LOOP
    # ----------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        x, y = batch
        emergency_prob = self(x)
        loss = self.criterion(emergency_prob, y.float())
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        # Track training accuracy
        preds = (emergency_prob >= self.threshold).float()
        self.train_accuracy(preds, y)
        self.log("train_accuracy", self.train_accuracy, on_epoch=True, prog_bar=True)
        self.train_step_outputs.append({"preds": preds, "targets": y})

        # Log LR
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log("learning_rate", current_lr, on_step=True, on_epoch=False)

        # (Optional) Log gradient norms
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad_norm = torch.norm(param.grad).item()
                self.log(f"grad_norm_{name}", grad_norm, on_step=False, on_epoch=True)

        return loss

    def on_train_epoch_end(self):
        self.log("epoch_train_accuracy", self.train_accuracy.compute(), prog_bar=True)
        self.train_accuracy.reset()
        self.train_step_outputs.clear()

    # ----------------------------------------------------------------
    #                      VALIDATION LOOP
    # ----------------------------------------------------------------
    def validation_step(self, batch, batch_idx):
        x, y = batch
        emergency_prob = self(x)
        loss = self.criterion(emergency_prob, y.float())

        # Log val loss
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        # Compute and log validation metrics
        preds = (emergency_prob >= self.threshold).float()
        self.val_step_outputs.append({"preds": preds, "targets": y})

        self.val_accuracy(preds, y)
        self.val_precision(preds, y)
        self.val_recall(preds, y)
        self.val_f1(preds, y)

        self.log("epoch_val_accuracy", self.val_accuracy.compute(), prog_bar=True)
        self.log("epoch_val_precision", self.val_precision.compute(), prog_bar=True)
        self.log("epoch_val_recall", self.val_recall.compute(), prog_bar=True)
        self.log("epoch_val_f1", self.val_f1.compute(), prog_bar=True)

        return loss

    def on_validation_epoch_end(self):
        self.val_accuracy.reset()
        self.val_precision.reset()
        self.val_recall.reset()
        self.val_f1.reset()
        self.val_step_outputs.clear()

    # ----------------------------------------------------------------
    #                         TEST LOOP
    # ----------------------------------------------------------------
    def test_step(self, batch, batch_idx):
        x, y = batch
        if self.output_mode == "bin_only":
            start_time = time.perf_counter()
            emergency_prob = self(x)
            end_time = time.perf_counter()
            self.tot_inference_time += (end_time - start_time)
            preds = (emergency_prob >= self.threshold).float()

            # Test Loss
            loss = self.criterion(emergency_prob, y.float())
            self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        elif self.output_mode == "bin_raw":
            start_time = time.perf_counter()
            full_out = self.model(x.float().squeeze())['clipwise_output']
            end_time = time.perf_counter()
            self.tot_inference_time += (end_time - start_time)

            emergency_prob = full_out[:, self.class_idx]
            preds = (emergency_prob >= self.threshold).float()
            self.all_preds_storage.append(full_out.detach().cpu())

            # Test Loss
            loss = self.criterion(emergency_prob, y.float())
            self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        else:
            raise ValueError(f"Invalid output_mode={self.output_mode}. Use 'bin_only' or 'bin_raw'.")

        # Store test predictions and ground truths
        self.preds.append(preds)
        self.targets.append(y.float())

    def on_test_epoch_end(self):
        # Aggregate predictions
        preds = torch.cat(self.preds, dim=0)
        targets = torch.cat(self.targets, dim=0).squeeze()

        # If we used 'bin_raw', save entire clipwise outputs
        if self.all_preds_storage:
            test_set_logits = torch.cat(self.all_preds_storage, dim=0).numpy()
            np.savez_compressed(os.path.join(self.model_results_path, "test_full_predictions.npz"),
                                logits=test_set_logits,
                                tot_inference_time_sec=self.tot_inference_time)

        # Compute final test metrics
        self.compute_metrics(preds, targets)
        self.log_metrics()
        self.save_test_metrics_to_csv(os.path.join(self.model_results_path, "test_metrics.csv"))

        # Generate plots
        self.plot_confusion_matrix(targets, preds, save_dir=self.model_results_path)
        self.plot_precision_recall_f1_fbeta(self.test_precision_val,
                                            self.test_recall_val,
                                            self.test_f1_val,
                                            self.test_fbeta_val,
                                            save_dir=self.model_results_path)
        self.plot_roc_pr_det_curves(targets, preds, save_dir=self.model_results_path)

        # Clear buffers
        self.preds.clear()
        self.targets.clear()
        self.all_preds_storage.clear()
        self.tot_inference_time = 0.0

    # ----------------------------------------------------------------
    #                 METRICS COMPUTATION & LOGGING
    # ----------------------------------------------------------------
    def compute_metrics(self, preds, targets):
        self.test_acc_val = self.test_accuracy(preds, targets)
        self.test_conf_matrix = self.test_confusion_matrix(preds, targets)
        self.test_precision_val = self.test_precision(preds, targets)
        self.test_recall_val = self.test_recall(preds, targets)
        self.test_f1_val = self.test_f1(preds, targets)
        self.test_specificity_val = self.test_specificity(preds, targets)
        self.test_auroc_val = self.test_auroc(preds, targets)
        self.test_auprc_val = self.test_auprc(preds, targets.int())
        self.test_mcc_val = self.test_mcc(preds, targets)
        self.test_fbeta_val = self.test_fbeta(preds, targets)
        tp, fp, tn, fn, support = self.test_stat_scores(preds, targets)
        self.test_tp_acc = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    def log_metrics(self):
        self.log("Test_Accuracy", self.test_acc_val)
        self.log("Test_Precision", self.test_precision_val)
        self.log("Test_Recall", self.test_recall_val)
        self.log("Test_F1_score", self.test_f1_val)
        self.log("Test_F-beta_score", self.test_fbeta_val)
        self.log("Test_Sensitivity (TP-Accuracy)", self.test_tp_acc)
        self.log("Test_Specificity (TN-Accuracy)", self.test_specificity_val)
        self.log("Test_AuROC", self.test_auroc_val)
        self.log("Test_AuPRC", self.test_auprc_val)
        self.log("Test_MCC", self.test_mcc_val)

    def save_test_metrics_to_csv(self, csv_path):
        metrics = {"Accuracy": self.test_acc_val.item(),
                   "Precision": self.test_precision_val.item(),
                   "Recall": self.test_recall_val.item(),
                   "F1_Score": self.test_f1_val.item(),
                   "F_Beta_Score": self.test_fbeta_val.item(),
                   "Sensitivity (TP-Accuracy)": self.test_tp_acc.item(),
                   "Specificity (TN-Accuracy)": self.test_specificity_val.item(),
                   "AUROC": self.test_auroc_val.item(),
                   "AUPRC": self.test_auprc_val.item(),
                   "MCC": self.test_mcc_val.item()}
        
        with open(csv_path, mode="w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=metrics.keys())
            writer.writeheader()
            writer.writerow(metrics)
        
        print(f"Test metrics saved to: {csv_path}")

    # ----------------------------------------------------------------
    #                            PLOTTING
    # ----------------------------------------------------------------
    def plot_confusion_matrix(self, targets, predictions, save_dir):
        tp = ((predictions.int() == 1) & (targets.int() == 1)).sum().item()
        fp = ((predictions.int() == 1) & (targets.int() == 0)).sum().item()
        tn = ((predictions.int() == 0) & (targets.int() == 0)).sum().item()
        fn = ((predictions.int() == 0) & (targets.int() == 1)).sum().item()
        conf_matrix = np.array([[tp, fn], [fp, tn]])

        cm_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1, keepdims=True)

        plt.figure(figsize=(6, 5))
        ax = sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="flare",
                         cbar=True, annot_kws={"size": 12}, square=True,
                         linewidths=1.5, linecolor="black")

        cell_labels = np.array([["TP", "FN"], ["FP", "TN"]])
        for i in range(2):
            for j in range(2):
                ax.text(j + 0.5, i + 0.3, cell_labels[i, j],
                        ha="center", va="center", color="black",
                        fontsize=10, fontweight="bold")

        plt.xlabel("Ground Truth", labelpad=10)
        plt.ylabel("Predictions", labelpad=10)
        plt.title("Confusion Matrix", pad=20, fontsize=14)

        plt.xticks([0.5, 1.5], ["Positive", "Negative"])
        plt.gca().xaxis.tick_top()
        plt.gca().xaxis.set_label_position('top')
        plt.yticks([0.5, 1.5], ["Positive", "Negative"], rotation=0)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "Confusion_matrix.svg"), format="svg")
        plt.savefig(os.path.join(save_dir, "Confusion_matrix.png"), format="png")
        plt.close()

    def plot_precision_recall_f1_fbeta(self, precision, recall, f1_score, f_beta, save_dir):
        metrics = ["Precision", "Recall", "$F_1$", f"$F_{{{self.beta}}}$"]
        values = [precision.cpu(), recall.cpu(), f1_score.cpu(), f_beta.cpu()]

        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.bar(metrics, values, width=0.5, color="white", edgecolor="black", zorder=3)

        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.yaxis.set_ticks(np.arange(0, 1.05, 0.05), minor=True)
        ax.grid(True, which='major', axis='y', linestyle='--', linewidth=0.8, color='gray', zorder=0)
        ax.grid(True, which='minor', axis='y', linestyle='--', linewidth=0.5, color='grey', zorder=0)

        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2,
                    f"{value * 100:.2f}%", ha="center", va="center",
                    fontsize=10, fontweight="bold", color="black")

        ax.set_ylim(0, 1)
        ax.set_ylabel("Score (Normalized)")
        ax.set_title("Classification Performance Metrics", fontsize=13)
        ax.tick_params(axis='y', which='both', direction='in')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "Classification_performance_quality.svg"), format="svg")
        plt.savefig(os.path.join(save_dir, "Classification_performance_quality.png"), format="png")
        plt.close()

    def plot_roc_pr_det_curves(self, targets, predictions, save_dir):
        fpr, tpr, _ = roc_curve(targets.cpu().numpy(), predictions.cpu().numpy())
        roc_auc = auc(fpr, tpr)

        precision, recall, _ = precision_recall_curve(targets.cpu().numpy(), predictions.cpu().numpy())
        pr_auc = auc(recall, precision)

        fnr = 1 - tpr

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

        # ROC
        ax1.plot(fpr, tpr, color="black", lw=1.5, label=f"ROC-AuC = {roc_auc:.3f}")
        ax1.fill_between(fpr, tpr, alpha=0.3, color="grey")
        ax1.plot([0, 1], [0, 1], color="grey", linestyle="--", linewidth=1, label="Random classifier")
        ax1.set_title("ROC Curve", fontsize=12)
        ax1.set_xlabel("False Positive Rate")
        ax1.set_ylabel("True Positive Rate")
        ax1.legend(loc="lower right", fontsize=10)
        ax1.grid(True, linestyle='--', color='grey', alpha=0.6)

        # Precision-Recall
        ax2.plot(recall, precision, color="black", lw=1.5, label=f"PR-AuC = {pr_auc:.3f}")
        ax2.fill_between(recall, precision, alpha=0.3, color="grey")
        ax2.set_title("Precision-Recall Curve", fontsize=12)
        ax2.set_xlabel("Recall")
        ax2.set_ylabel("Precision")
        ax2.legend(loc="lower left", fontsize=10)
        ax2.grid(True, linestyle='--', color='grey', alpha=0.6)

        # DET
        ax3.plot(fpr, fnr, color="black", lw=1.5)
        ax3.set_xscale("log")
        ax3.set_yscale("log")
        ticks = [0.001, 0.01, 0.1, 0.2, 0.5, 1.0]
        tick_labels = [f"{int(t * 100)}%" for t in ticks]
        ax3.set_xticks(ticks)
        ax3.set_xticklabels(tick_labels)
        ax3.set_yticks(ticks)
        ax3.set_yticklabels(tick_labels)
        ax3.set_title("DET Curve", fontsize=12)
        ax3.set_xlabel("False Positive Rate (%)")
        ax3.set_ylabel("False Negative Rate (%)")
        ax3.grid(True, linestyle='--', color='grey', alpha=0.6)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "ROC_PR_DET_curves.svg"), format="svg")
        plt.savefig(os.path.join(save_dir, "ROC_PR_DET_curves.png"), format="png")
        plt.close()

    # ----------------------------------------------------------------
    #                 OPTIMIZER & LR SCHEDULER
    # ----------------------------------------------------------------
    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.eta_max,
                               weight_decay=self.weight_decay,
                               betas=self.betas,
                               eps=self.eps,
                               amsgrad=True)
        
        scheduler = CyclicCosineDecayLR(optimizer,
                                        init_decay_epochs=self.decay_epochs,
                                        min_decay_lr=self.eta_min,
                                        restart_interval=self.restart_interval,
                                        restart_lr=self.restart_eta,
                                        warmup_epochs=self.warmup_epochs,
                                        warmup_start_lr=self.warmup_eta)
        
        return [optimizer], [scheduler]


class E2PANNs_Model_DatasetAware(pl.LightningModule):
    """
    E2PANNs model with:
      - F1-based dataset weighting logic (update_dataset_weight_by_f1).
      - A history of dataset weights, appended after each update (dataset_weights_over_time).
      - plot_dataset_weights_history(...) to create a single black-and-white line plot
        with distinct linestyles for each dataset.
    """
    def __init__(self,
                 model,
                 threshold=0.5,
                 output_mode='bin_raw',
                 overall_training=False,
                 eta_max=1e-3,
                 eta_min=1e-6,
                 decay_epochs=50,
                 restart_eta=1e-5,
                 restart_interval=10,
                 warmup_epochs=10,
                 warmup_eta=1e-4,
                 weight_decay=1e-6,
                 f_beta=0.8,
                 num_datasets=6):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        # Base model
        self.model = model.cpu()
        self.class_idx = 322
        self.threshold = threshold
        self.output_mode = output_mode
        self.overall = overall_training

        # F-beta
        self.beta = f_beta

        # Scheduler / Optim
        self.decay_epochs = decay_epochs
        self.eta_min = eta_min
        self.eta_max = eta_max
        self.restart_interval = restart_interval
        self.restart_eta = restart_eta
        self.warmup_epochs = warmup_epochs
        self.warmup_eta = warmup_eta
        self.weight_decay = weight_decay
        self.betas = (0.9, 0.999)
        self.eps = 1e-08

        # Freeze/unfreeze base model layers
        self.setup_model()

        # Loss function
        self.criterion = nn.BCELoss()

        # Metrics
        self.init_metrics()

        # Buffers for training & validation
        self.train_step_outputs = []
        self.val_step_outputs   = []

        # Test Buffers
        self.preds = []
        self.targets = []
        self.all_preds_storage = []
        self.tot_inference_time = 0.0

        # Output path
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
        self.model_results_path = f"./experiments/{current_time}/model_results/"
        os.makedirs(self.model_results_path, exist_ok=True)

        # === Dataset Weights & Tracking ===
        self.dataset_weights = nn.Parameter(torch.ones(num_datasets), requires_grad=False)
        self.current_dataset_idx = 0  # which dataset is currently training

        # F1-based weighting approach:
        self.prev_val_f1 = [0.0] * num_datasets

        # Keep track of the entire weight vector at each update
        self.dataset_weights_over_time = []
        self.dataset_weights_over_time.append(self.dataset_weights.clone().detach().cpu().tolist())

    def setup_model(self):
        if not self.overall:
            print('Finetuning only the last layers of base model.')
            for name, param in self.model.named_parameters():
                if name.startswith("fc1") or name.startswith("fc_audioset"):
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        else:
            print('Full model training (all parameters un-frozen).')
            for _, param in self.model.named_parameters():
                param.requires_grad = True

    def init_metrics(self):
        # TRAIN METRICS
        self.train_accuracy = Accuracy(task="binary", num_classes=2, threshold=self.threshold)

        # VAL METRICS
        self.val_accuracy  = Accuracy(task="binary", num_classes=2, threshold=self.threshold)
        self.val_precision = Precision(task="binary", num_classes=2, threshold=self.threshold)
        self.val_recall    = Recall(task="binary", num_classes=2, threshold=self.threshold)
        self.val_f1        = F1Score(task="binary", num_classes=2, threshold=self.threshold)

    # ----------------------------------------------------------------
    #                  LOAD TRAINED WEIGHTS (from .ckpt)
    # ----------------------------------------------------------------
    def load_trained_weights(self, checkpoint_path: str, verbose: bool = False, validate_updates: bool = True):
        """
        Load a Lightning .ckpt into self.model parameters on CPU using 'load_lightning2pt'.
        :param checkpoint_path: Path to the Lightning checkpoint.
        :param verbose: Print detailed info if True.
        :param validate_updates: Compare old vs new params to see which changed.
        :return: List of updated layers (if validate_updates=True), else None.
        """
        self.model, updated_layers = load_lightning2pt(checkpoint_path=checkpoint_path,
                                                       model=self.model,
                                                       verbose=verbose,
                                                       validate_updates=validate_updates)

        print(f'Model correctly loaded from checkpoint: {checkpoint_path}')
        return updated_layers
    
    # ----------------------------------------------------------------
    #   DATASET-WEIGHTING LOGIC (F1-based)
    # ----------------------------------------------------------------
    def set_current_dataset_idx(self, idx: int):
        """
        Indicate which dataset is active in training_step.
        """
        self.current_dataset_idx = idx

    def update_dataset_weight_by_f1(self, dataset_idx: int, new_val_f1: float, small_threshold=0.0005, large_threshold=0.005):
        improvement = new_val_f1 - self.prev_val_f1[dataset_idx]

        if improvement < small_threshold:
            self.dataset_weights[dataset_idx] *= 1.05
        elif improvement > large_threshold:
            self.dataset_weights[dataset_idx] *= 0.95

        # Clamp the new weight
        self.dataset_weights.data[dataset_idx] = torch.clamp(self.dataset_weights[dataset_idx], 0.1, 10.0)

        # Print a console log of this update
        print(f"[UPDATE DATASET {dataset_idx}]"
              f" prev_F1={self.prev_val_f1[dataset_idx]:.4f}, new_F1={new_val_f1:.4f},"
              f" improvement={improvement:.4f},"
              f" updated_weight={self.dataset_weights[dataset_idx].item():.4f}")

        # Store the new F1 and snapshot of the entire weight vector
        self.prev_val_f1[dataset_idx] = new_val_f1
        self.dataset_weights_over_time.append(self.dataset_weights.clone().detach().cpu().tolist())

    # ----------------------------------------------------------------
    #                       FORWARD
    # ----------------------------------------------------------------
    def forward(self, x):
        logits_dict = self.model(x.float().squeeze())
        logits = logits_dict['clipwise_output']
        return logits[:, self.class_idx]

    # ----------------------------------------------------------------
    #                      TRAINING STEP
    # ----------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        x, y = batch
        emergency_prob = self(x)

        bce_loss = self.criterion(emergency_prob, y.float())
        ds_weight = self.dataset_weights[self.current_dataset_idx]
        loss = ds_weight * bce_loss

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        preds = (emergency_prob >= self.threshold).float()
        self.train_accuracy(preds, y)
        self.log("train_accuracy", self.train_accuracy, on_epoch=True, prog_bar=True)

        self.train_step_outputs.append({"preds": preds, "targets": y})

        # Log LR
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log("learning_rate", current_lr, on_step=True, on_epoch=False)

        return loss

    def on_train_epoch_end(self):
        self.log("epoch_train_accuracy", self.train_accuracy.compute(), prog_bar=True)
        self.train_accuracy.reset()
        self.train_step_outputs.clear()

    # ----------------------------------------------------------------
    #                     VALIDATION STEP
    # ----------------------------------------------------------------
    def validation_step(self, batch, batch_idx):
        x, y = batch
        emergency_prob = self(x)
        loss = self.criterion(emergency_prob, y.float())

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        preds = (emergency_prob >= self.threshold).float()
        self.val_step_outputs.append({"preds": preds, "targets": y})

        self.val_accuracy(preds, y)
        self.val_precision(preds, y)
        self.val_recall(preds, y)
        self.val_f1(preds, y)

        self.log("epoch_val_accuracy", self.val_accuracy.compute(), prog_bar=True)
        self.log("epoch_val_precision", self.val_precision.compute(), prog_bar=True)
        self.log("epoch_val_recall", self.val_recall.compute(), prog_bar=True)
        self.log("epoch_val_f1", self.val_f1.compute(), prog_bar=True)

        return loss

    def on_validation_epoch_end(self):
        self.val_accuracy.reset()
        self.val_precision.reset()
        self.val_recall.reset()
        self.val_f1.reset()
        self.val_step_outputs.clear()

    # ----------------------------------------------------------------
    # PLOTs
    # ----------------------------------------------------------------
    def plot_dataset_weights_history(self, save_dir, dataset_names=None):
        """
        Single black-and-white line plot of dataset weights over time, 
        with distinct linestyles for each dataset.
        """
        weight_array = np.array(self.dataset_weights_over_time)  # shape: (updates, num_datasets)
        num_updates, num_ds = weight_array.shape
        x_axis = np.arange(num_updates)

        # Linestyles
        line_styles = ["-", "--", "-.", ":", (0, (5, 2, 1, 2)), (0, (1, 1))]

        fig, ax = plt.subplots(figsize=(7, 4))
        for d in range(num_ds):
            if dataset_names and d < len(dataset_names):
                label = dataset_names[d]
            else:
                label = f"Dataset {d+1}"

            style = line_styles[d % len(line_styles)]
            ax.plot(x_axis, weight_array[:, d], color="black", linestyle=style, linewidth=1.5, label=label)

        ax.set_title("Dataset Weights Over Time", fontsize=13)
        ax.set_xlabel("Update Step")
        ax.set_ylabel("Weight Value")
        ax.grid(True, linestyle='--', color='grey', alpha=0.6)
        ax.legend(loc="best")

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "dataset_weights_history.svg"), format="svg")
        plt.savefig(os.path.join(save_dir, "dataset_weights_history.png"), format="png")
        plt.close()

    # ----------------------------------------------------------------
    #                OPTIMIZER & LR SCHEDULER
    # ----------------------------------------------------------------
    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.eta_max,
                               weight_decay=self.weight_decay,
                               betas=self.betas,
                               eps=self.eps,
                               amsgrad=True)
        
        scheduler = CyclicCosineDecayLR(optimizer,
                                        init_decay_epochs=self.decay_epochs,
                                        min_decay_lr=self.eta_min,
                                        restart_interval=self.restart_interval,
                                        restart_lr=self.restart_eta,
                                        warmup_epochs=self.warmup_epochs,
                                        warmup_start_lr=self.warmup_eta)
        
        return [optimizer], [scheduler]
