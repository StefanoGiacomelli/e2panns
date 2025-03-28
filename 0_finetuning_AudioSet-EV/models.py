import os
import csv
import time
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from scheduler import CyclicCosineDecayLR
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torchmetrics import Accuracy, ConfusionMatrix, Precision, Recall, F1Score, Specificity, AUROC, AveragePrecision, MatthewsCorrCoef, FBetaScore, StatScores
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import pytorch_lightning as pl


# Models --------------------------------------------------------------------------------------------
class EPANNs_Binarized_Model(pl.LightningModule):
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
        :param model: Base-model to fine-tune.
        :param threshold: Probabilities Threshold for binary classification.
        :param output_mode: 'bin_only' for binary output only, 'bin_raw' for full output.
        :param overall_training: full model or last stages fine-tuning condition.
        :param eta_max: Maximum learning rate.
        :param eta_min: Minimum learning rate.
        :param decay_epochs: Number of epochs for Cosine Annealing scheduler.
        :param restart_eta: Restart learning rate.
        :param restart_interval: Restart interval (in epochs) for learning rate.
        :param warmup_epochs: Number of epochs for warm-up.
        :param warmup_eta: Warm-up learning rate.
        :param weight_decay: Weight decay (for Adam optimizer).
        :param f_beta: Beta value for F-beta score.
        """
        super(EPANNs_Binarized_Model, self).__init__()
        
        # Model parameters
        self.model = model
        self.class_idx = 322
        self.threshold = threshold
        self.output_mode = output_mode      # 'bin_only' or 'bin_raw'
        self.overall = overall_training
        
        self.beta = f_beta
        
        # Learning Rate Scheduler parameters
        self.decay_epochs = decay_epochs
        self.eta_min = eta_min
        self.eta_max = eta_max
        self.restart_interval = restart_interval 
        self.restart_eta = restart_eta
        self.warmup_epochs = warmup_epochs
        self.warmup_eta = warmup_eta
        
        # Adam parameters
        self.weight_decay = weight_decay
        self.betas=(0.9, 0.999) 
        self.eps=1e-08
        
        # Put (parts of) model on training
        self.setup_model()

        # Loss function
        self.criterion = nn.BCELoss()

        # Initialize metrics
        self.init_metrics()
        self.train_step_outputs = []    # For collecting outputs from training step (Lightning=2.0+ consistency)
        self.val_step_outputs = []      # For collecting validation outputs (Lightning=2.0+ consistency)

        # Placeholders for predictions and ground truths (for test phase)
        self.preds = []
        self.targets = []

        # Ensure results directory exists
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
        self.model_results_path = f"./experiments/{current_time}/model_results/"
        os.makedirs(self.model_results_path, exist_ok=True)


    def setup_model(self):
        if self.overall == False:
            print('finetuning')
            for name, param in self.model.named_parameters():
                if name.startswith("fc1") or name.startswith("fc_audioset"):
                    param.requires_grad = True      # Enable training
                else:
                    param.requires_grad = False     # Freeze weights


    def init_metrics(self):
        # Metrics for Training
        self.train_accuracy = Accuracy(task="binary", num_classes=2, threshold=self.threshold)

        # Metrics for Validation
        self.val_accuracy = Accuracy(task="binary", num_classes=2, threshold=self.threshold)
        self.val_precision = Precision(task="binary", num_classes=2, threshold=self.threshold)
        self.val_recall = Recall(task="binary", num_classes=2, threshold=self.threshold)
        self.val_f1 = F1Score(task="binary", num_classes=2, threshold=self.threshold)

        # Metrics for Testing
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


    def forward(self, x):
        logits = self.model(x.float().squeeze())['clipwise_output']
        emergency_prob = logits[:, self.class_idx]  # Extract Emergency Vehicle probability
        return emergency_prob


    # ---------------- Training Loop ----------------
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

        # Log learning rate
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log("learning_rate", current_lr, on_step=True, on_epoch=False)

        # Log gradient norms
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad_norm = torch.norm(param.grad).item()
                self.log(f"grad_norm_{name}", grad_norm, on_step=False, on_epoch=True)

        return loss


    def on_train_epoch_end(self):
        self.log("epoch_train_accuracy", self.train_accuracy.compute(), prog_bar=True)
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.logger.experiment.add_histogram(f"{name}_weights", param, self.current_epoch)
        
        self.train_accuracy.reset()
        self.train_step_outputs.clear()


    # ---------------- Validation Loop ----------------
    def validation_step(self, batch, batch_idx):
        x, y = batch
        emergency_prob = self(x)
        loss = self.criterion(emergency_prob, y.float())

        # Log validation loss
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


    # ---------------- Test Loop ----------------
    def test_step(self, batch, batch_idx):
        x, y = batch
        # Retrieve `output` parameter from the model
        output_mode = getattr(self, "output_mode", "bin_only")

        # Extract Emergency Vehicle probability (task-specific)
        if output_mode == "bin_only":
            emergency_prob = self(x)
            preds = (emergency_prob >= self.threshold).float()
        # Extract Full classification set & Data-TPs probabilities (for task-specific profiling)
        elif output_mode == "bin_raw":  
            start_time = time.perf_counter()
            full_preds_set = self.model(x.float().squeeze())['clipwise_output']
            end_time = time.perf_counter()
            inference_time = end_time - start_time

            preds = (full_preds_set[:, self.class_idx] >= self.threshold).float()
        else:
            raise ValueError(f"Invalid output_mode: {output_mode}. Use 'bin_only' or 'bin_raw'.")

        # Store predictions and targets (and inference time)
        self.preds.append(preds)
        self.targets.append(y.float())
        
        if output_mode == "bin_raw":
            if not hasattr(self, 'all_preds_storage'):
                self.all_preds_storage = []
            self.all_preds_storage.append(full_preds_set.detach().cpu())
            if not hasattr(self, 'tot_inference_time'):
                self.tot_inference_time = 0.0                                       # in sec.
            self.tot_inference_time += inference_time


    def on_test_epoch_end(self):
        preds, targets = self.aggregate_predictions()

        # Save Raw Predictions & File-Paths to NumPy compressed file
        if hasattr(self, 'all_preds_storage'):
            test_set_logits = torch.cat(self.all_preds_storage, dim=0).numpy()
            np.savez_compressed(self.model_results_path + "test_full_predictions.npz", 
                                logits=test_set_logits, 
                                tot_inference_time_sec=self.tot_inference_time)            
            self.all_preds_storage.clear()
            del self.all_preds_storage
            del self.tot_inference_time

        # Compute and log metrics
        self.compute_metrics(preds, targets)
        self.log_metrics()
        self.save_test_metrics_to_csv(save_path=self.model_results_path + "test_metrics.csv")

        # Generate and save plots
        self.plot_confusion_matrix(targets, preds, save_path=self.model_results_path)
        self.plot_precision_recall_f1_fbeta(self.test_precision_val, 
                                            self.test_recall_val, 
                                            self.test_f1_val, 
                                            self.test_fbeta_val, 
                                            save_path=self.model_results_path)
        self.plot_roc_pr_det_curves(targets, preds, save_path=self.model_results_path)

        # Clear storage for predictions and targets
        self.preds.clear()
        self.targets.clear()


    # ---------------- Optimization and Scheduler ----------------
    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), 
                               lr=self.eta_max, 
                               weight_decay=self.weight_decay,
                               betas=self.betas, 
                               eps=self.eps,
                               amsgrad=True)
        #scheduler = CosineAnnealingLR(optimizer, T_max=self.decay_epochs, eta_min=self.eta_min)
        scheduler = CyclicCosineDecayLR(optimizer, 
                                        init_decay_epochs=self.decay_epochs, 
                                        min_decay_lr=self.eta_min, 
                                        restart_interval=self.restart_interval, 
                                        restart_lr=self.restart_eta, 
                                        warmup_epochs=self.warmup_epochs, 
                                        warmup_start_lr=self.warmup_eta)
        return [optimizer], [scheduler]


    # ---------------- Helpers ----------------
    def aggregate_predictions(self):
        preds = torch.cat(self.preds)
        targets = torch.cat(self.targets).squeeze()
        return preds, targets


    # ---------------- Metrics Logging and Plotting ----------------
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
        self.test_tp_acc = tp / (tp + fn) if tp + fn > 0 else 0.0


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
    

    def save_test_metrics_to_csv(self, save_path="./model_results/test_metrics.csv"):
        """
        Save test metrics into a CSV file.
        
        :param save_path: File path to save the metrics CSV.
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
        
        with open(save_path, mode="w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=metrics.keys())
            writer.writeheader()
            writer.writerow(metrics)
        print(f"Test metrics saved to: {save_path}")
    

    def plot_confusion_matrix(self, targets: torch.Tensor, predictions: torch.Tensor, save_path: str):
        """
        Plot a confusion matrix (with normalized values).

        :param predictions: Predicted binary labels.
        :param targets: True binary labels.
        :param save_path: Path where the SVG and PNG plots will be saved.
        """
        # Calculate confusion matrix values manually
        tp = ((predictions.int() == 1) & (targets.int() == 1)).sum().item()
        fp = ((predictions.int() == 1) & (targets.int() == 0)).sum().item()
        tn = ((predictions.int() == 0) & (targets.int() == 0)).sum().item()
        fn = ((predictions.int() == 0) & (targets.int() == 1)).sum().item()
        conf_matrix = np.array([[tp, fn], [fp, tn]])
        cm_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1, keepdims=True)  # Normalized values

        # Create a figure for the heat-mapped confusion matrix
        plt.figure(figsize=(6, 5))
        ax = sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="flare", cbar=True,
                         annot_kws={"size": 12}, square=True, linewidths=1.5, linecolor="black")

        # Add TP, TN, FP, FN labels to the cells
        cell_labels = np.array([["TP", "FP"], ["FN", "TN"]])
        for i in range(2):
            for j in range(2):
                ax.text(j + 0.5, i + 0.3, cell_labels[i, j], 
                        ha="center", va="center", color="black", fontsize=10, fontweight="bold")

        # Add figure labels and titles
        plt.xlabel("Ground Truth", labelpad=10)
        plt.ylabel("Predictions", labelpad=10)
        plt.title("Confusion Matrix", pad=20, fontsize=14)

        # Move the x-axis labels and ticks to the top and invert Positive/Negative order
        plt.xticks([0.5, 1.5], ["Positive", "Negative"])
        plt.gca().xaxis.tick_top()
        plt.gca().xaxis.set_label_position('top')
        plt.yticks([0.5, 1.5], ["Positive", "Negative"], rotation=0)

        # Adjust layout and save the plot
        plt.tight_layout()
        plt.savefig(save_path + "Confusion_matrix.svg", format="svg")
        plt.savefig(save_path + "Confusion_matrix.png", format="png")
        plt.close()
    

    def plot_precision_recall_f1_fbeta(self,
                                       precision: float, 
                                       recall: float, 
                                       f1_score: float, 
                                       f_beta: float, 
                                       save_path: str):
        """
        Plot a B&W bar-plot for Precision, Recall, F1-Score, and F-beta.

        :param precision: The Precision value.
        :param recall: The Recall value.
        :param f1_score: The F1-score.
        :param f_beta: The F-beta score.
        :param save_path: Path where the SVG bar-plot needs to be saved.
        """
        # Define metrics
        metrics = ["Precision", "Recall", "$F_1$-Score", "$F_{\\beta}$"]
        values = [precision.cpu(), recall.cpu(), f1_score.cpu(), f_beta.cpu()]

        # Create the figure and axis
        fig, ax = plt.subplots(figsize=(10, 5))

        # Create bar-plot
        bars = ax.bar(metrics, values, width=0.5, color="white", edgecolor="black", zorder=3)

        # Grid and ticks handling
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])  # Main y-axis ticks
        ax.yaxis.set_ticks(np.arange(0, 1.05, 0.05), minor=True)  # Grid lines at 0.05 intervals
        ax.grid(True, which='major', axis='y', linestyle='--', linewidth=0.8, color='gray', zorder=0)
        ax.grid(True, which='minor', axis='y', linestyle='--', linewidth=0.5, color='grey', zorder=0)

        # Add text labels inside each bar to show the metric as a percentage (%)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, 
                    bar.get_height() / 2,
                    f"{value * 100:.2f}%", 
                    ha="center", va="center", fontsize=10, fontweight="bold", color="black")

        # Set plot labels and title
        ax.set_ylim(0, 1)
        ax.set_ylabel("Score (Normalized)")
        ax.set_title("Classification Performance Metrics", fontsize=14)
        ax.tick_params(axis='y', which='both', direction='in')

        # Adjust layout and save the plot
        plt.tight_layout()
        plt.savefig(save_path + 'Classification_performance_quality.svg', format="svg")
        plt.savefig(save_path + 'Classification_performance_quality.png', format="png")
        plt.close()
    
    
    def plot_roc_pr_det_curves(self, targets: torch.Tensor, predictions: torch.Tensor, save_path: str):
        """
        Plot B&W ROC, Precision-Recall, and DET curves with highlighted areas under curves.

        :param tragets: True binary labels.
        :param predictions: Target scores, probability estimates of the positive class.
        :param save_path: Path where the SVG and PNG plots will be saved.
        """
        # Calculate ROC curve and AUC
        fpr, tpr, _ = roc_curve(targets.cpu().numpy(), predictions.cpu().numpy())
        roc_auc = auc(fpr, tpr)

        # Calculate Precision-Recall curve and AUC
        precision, recall, _ = precision_recall_curve(targets.cpu().numpy(), predictions.cpu().numpy())
        pr_auc = auc(recall, precision)

        # Calculate DET curve data (using FNR = 1 - TPR)
        fnr = 1 - tpr

        # Create a figure with three subplots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

        # Apply black and white style
        fig.patch.set_facecolor('white')

        # ---- Plot ROC Curve ----
        ax1.plot(fpr, tpr, color="black", lw=1.5, label=f"ROC-AuC = {roc_auc:.3f}")
        ax1.fill_between(fpr, tpr, alpha=0.3, color="grey")
        ax1.plot([0, 1], [0, 1], color="grey", linestyle="--", linewidth=1, label="Random classifier")

        # Set ROC plot labels and style
        ax1.set_title(f"ROC Curve", fontsize=14)
        ax1.set_xlabel(f"False Positive Rate")
        ax1.set_ylabel(f"True Positive Rate")
        ax1.legend(loc="lower right", fontsize=12)
        ax1.tick_params(axis='both', which='both', direction='in')
        ax1.grid(True, linestyle='--', color='grey', alpha=0.6)

        # ---- Plot Precision-Recall Curve ----
        ax2.plot(recall, precision, color="black", lw=1.5, label=f"PR-AuC = {pr_auc:.3f}")
        ax2.fill_between(recall, precision, alpha=0.3, color="grey")

        # Set Precision-Recall plot labels and style
        ax2.set_title(f"Precision-Recall Curve", fontsize=14)
        ax2.set_xlabel(f"Recall")
        ax2.set_ylabel(f"Precision")
        ax2.legend(loc="lower left", fontsize=12)
        ax2.tick_params(axis='both', which='both', direction='in')
        ax2.grid(True, linestyle='--', color='grey', alpha=0.6)

        # ---- Plot DET Curve ----
        ax3.plot(fpr, fnr, color="black", lw=1.5)
        ax3.set_xscale("log")
        ax3.set_yscale("log")

        # Set custom ticks for FPR and FNR in percentage
        ticks = [0.001, 0.01, 0.1, 0.2, 0.5, 1.0]
        tick_labels = [f"{int(t * 100)}%" for t in ticks]
        ax3.set_xticks(ticks)
        ax3.set_xticklabels(tick_labels)
        ax3.set_yticks(ticks)
        ax3.set_yticklabels(tick_labels)

        # Set DET plot labels and style
        ax3.set_title(f"Detection Error Tradeoff (DET) Curve", fontsize=14)
        ax3.set_xlabel(f"False Positive Rate (%)")
        ax3.set_ylabel(f"False Negative Rate (%)")
        ax3.tick_params(axis='both', which='both', direction='in')
        ax3.grid(True, linestyle='--', color='grey', alpha=0.6)

        # Adjust layout and save the plot
        plt.tight_layout()
        plt.savefig(save_path + 'ROC_PR_DET_curves.svg', format="svg")
        plt.savefig(save_path + 'ROC_PR_DET_curves.png', format="png")
        plt.close()
