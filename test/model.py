import os
import csv
import time
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchmetrics import (Accuracy, ConfusionMatrix, Precision, Recall, F1Score,
                          Specificity, AUROC, AveragePrecision, MatthewsCorrCoef,
                          FBetaScore, StatScores)
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import pytorch_lightning as pl


def load_lightning2pt(checkpoint_path, model, device="cpu", verbose=False, validate_updates=True):
    """
    Loads a PyTorch Lightning checkpoint's state_dict into a plain PyTorch model 
    and optionally verifies parameter updates.

    :param checkpoint_path: Absolute Path to the Lightning checkpoint file (.ckpt).
    :param model: The plain PyTorch model instance to load the checkpoint into.
    :param device: Device to load the model onto ('cpu' or 'cuda').
    :param verbose: Whether to print detailed information about the loading process.
    :param validate_updates: Whether to validate which layers were updated (default: True).
    :return tuple:
        (model, updated_layers) if validate_updates=True
        (model, None) otherwise
    """
    # Step 1: Load the Lightning checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except FileNotFoundError:
        raise ValueError(f"Checkpoint file not found at: {checkpoint_path}")
    except Exception as e:
        raise ValueError(f"Failed to load checkpoint: {e}")

    # Step 2: Extract the Lightning state_dict
    if "state_dict" not in checkpoint:
        raise ValueError(f"Checkpoint does not contain a 'state_dict'. Keys found: {list(checkpoint.keys())}")

    lightning_state_dict = checkpoint["state_dict"]

    # Step 3: Attempt prefix removal if needed
    stripped_state_dict = {}
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

    # Move model to specified device
    model.to(device)
    if verbose:
        print(f"Model moved to device: {device}")

    # Step 5: Optionally validate parameter updates
    updated_layers = []
    if validate_updates:
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

    # Step 6: Load the stripped state_dict into the plain model
    try:
        model.load_state_dict(stripped_state_dict)
        if verbose:
            print("State dict successfully loaded into the model!")
    except Exception as e:
        raise ValueError(f"Failed to load state_dict into the model: {e}")

    # Print updated layers if validated
    if verbose and validate_updates:
        if updated_layers:
            print("The following layers were updated during fine-tuning:")
            for layer in updated_layers:
                print(f" - {layer}")
        else:
            print("No layers were updated. (Possible that no fine-tuning changes occurred.)")

    return model, updated_layers if validate_updates else None


class E2PANNs_Model(pl.LightningModule):
    """
    A test-only variant of E2PANNs Emergency Vehicle Sirens Classifier
    """
    def __init__(self,
                 model,
                 threshold=0.5,
                 output_mode='bin_raw',
                 class_idx=322,
                 f_beta=0.8,
                 results_path=None):
        """
        :param model: Base-model (e.g., EPANNs) that returns a dict with 'clipwise_output'.
        :param threshold: Probability threshold for positive class (EV).
        :param output_mode: 'bin_only' uses just EV probability for decision;
                            'bin_raw' also collects entire clipwise output array.
        :param class_idx: Index of "Emergency Vehicle" class in model's output dimension.
        :param f_beta: Beta for F-beta metric.
        :param results_path: Directory to store test metrics, plots, and CSV logs.
        """
        super().__init__()
        # Don't save entire submodel in hparams
        self.save_hyperparameters(ignore=["model"])

        self.model = model
        self.threshold = threshold
        self.output_mode = output_mode
        self.class_idx = class_idx
        self.beta = f_beta

        # Results path for test artifacts
        self.results_path = results_path if results_path else "./test_results/"
        os.makedirs(self.results_path, exist_ok=True)

        # Optional test loss function
        self.criterion = nn.BCELoss()

        # Initialize test metrics
        self.init_test_metrics()

        # Buffers for predictions, raw clipwise outputs, etc.
        self.preds = []
        self.targets = []
        self.all_preds_storage = []
        self.tot_inference_time = 0.0

    def load_trained_weights(self,
                             checkpoint_path: str,
                             device: str = "cpu",
                             verbose: bool = False,
                             validate_updates: bool = True):
        """
        Load a Lightning .ckpt into self.model parameters using 'load_lightning2pt'.
        :param checkpoint_path: Path to the Lightning checkpoint.
        :param device: 'cpu' or 'cuda'.
        :param verbose: Print detailed info if True.
        :param validate_updates: Compare old vs new params to see which changed.
        :return: List of updated layers (if validate_updates=True), else None.
        """
        self.model, updated_layers = load_lightning2pt(checkpoint_path=checkpoint_path,
                                                       model=self.model,
                                                       device=device,
                                                       verbose=verbose,
                                                       validate_updates=validate_updates)
        return updated_layers

    # --------------------- METRICS INIT ---------------------
    def init_test_metrics(self):
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

    # --------------------- FORWARD --------------------------
    def forward(self, x):
        """
        Compute EV probability => clipwise_output[:, class_idx].
        'x' is a batch of waveforms (BxCxT).
        """
        logits_dict = self.model(x.float().squeeze())  # e.g. {'clipwise_output': [B x #classes], ...}
        logits = logits_dict['clipwise_output']
        emergency_prob = logits[:, self.class_idx]
        return emergency_prob

    # --------------------- TEST STEP ------------------------
    def test_step(self, batch, batch_idx):
        """
        Inference on each test batch. Optionally store raw clipwise outputs.
        """
        x, y = batch

        if self.output_mode == "bin_only":
            emergency_prob = self(x)
            preds = (emergency_prob >= self.threshold).float()

        elif self.output_mode == "bin_raw":
            start_time = time.perf_counter()
            full_out = self.model(x.float().squeeze())['clipwise_output']
            end_time = time.perf_counter()

            self.tot_inference_time += (end_time - start_time)
            emergency_prob = full_out[:, self.class_idx]
            preds = (emergency_prob >= self.threshold).float()
            self.all_preds_storage.append(full_out.detach().cpu())
        else:
            raise ValueError(f"Invalid output_mode={self.output_mode}. Use 'bin_only' or 'bin_raw'.")

        # Optional test loss
        loss = self.criterion(emergency_prob, y.float())
        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        # Collect preds/targets
        self.preds.append(preds)
        self.targets.append(y.float())

    def on_test_epoch_end(self):
        """
        After all test batches:
          - Aggregate predictions
          - Compute metrics
          - Save plots & CSV
        """
        # Aggregate
        preds = torch.cat(self.preds, dim=0)
        targets = torch.cat(self.targets, dim=0).squeeze()

        # If we used 'bin_raw', save the entire clipwise output
        if self.all_preds_storage:
            test_set_logits = torch.cat(self.all_preds_storage, dim=0).numpy()
            np.savez_compressed(
                os.path.join(self.results_path, "test_full_predictions.npz"),
                logits=test_set_logits,
                tot_inference_time_sec=self.tot_inference_time
            )

        # Compute final metrics
        self.compute_metrics(preds, targets)
        self.log_metrics()
        self.save_test_metrics_to_csv(os.path.join(self.results_path, "test_metrics.csv"))

        # Generate plots
        self.plot_confusion_matrix(targets, preds, save_dir=self.results_path)
        self.plot_precision_recall_f1_fbeta(self.test_precision_val,
                                            self.test_recall_val,
                                            self.test_f1_val,
                                            self.test_fbeta_val,
                                            save_dir=self.results_path)
        self.plot_roc_pr_det_curves(targets, preds, save_dir=self.results_path)

        # Clear buffers
        self.preds.clear()
        self.targets.clear()
        self.all_preds_storage.clear()
        self.tot_inference_time = 0.0

    # --------------------- METRICS + LOGGING ----------------
    def compute_metrics(self, preds, targets):
        """
        Compute all test metrics from aggregated preds & targets.
        """
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
        """
        Log final test metrics to Lightning's default logger (if any).
        """
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
        """
        Save final test metrics to a CSV file.
        """
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

    # --------------------- PLOTTING UTILS -------------------
    def plot_confusion_matrix(self, targets, predictions, save_dir):
        import seaborn as sns

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
        import numpy as np

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
        import numpy as np

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

        # DET (FNR vs FPR)
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
