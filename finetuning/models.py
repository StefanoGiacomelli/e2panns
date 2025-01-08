import os
import csv
import time
from datetime import datetime
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torchmetrics import Accuracy, ConfusionMatrix, Precision, Recall, F1Score, Specificity, AUROC, AveragePrecision, MatthewsCorrCoef, FBetaScore, StatScores
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import pytorch_lightning as pl


# Dataset ------------------------------------------------------------------------------------------------
class AudioDataset(Dataset):
    def __init__(self, file_path, folder_path, target_size=320000, binary_label=1):
        self.cwd = os.getcwd()
        self.file_path = os.path.abspath(os.path.join(self.cwd, file_path))
        self.folder_path = os.path.abspath(os.path.join(self.cwd, folder_path))
        self.filenames = self.get_filenames(self.folder_path)
        self.target_size = target_size
        self.skipped_files = []
        self.label = binary_label

    def __len__(self):
        return len(self.filenames)
    
    def get_filenames(self, path):
        return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.wav')]

    def __getitem__(self, idx):
        file_path = self.filenames[idx]
        try:
            waveform_tensor, _ = torchaudio.load(file_path)
        except Exception as e:
            self.skipped_files.append((idx, file_path))
            print(f"Skipping Error loading {file_path}: {e}")
            return None

        # Pad or truncate waveform_tensor to target_size
        current_size = waveform_tensor.size(1)
        if current_size < self.target_size:
            padding = self.target_size - current_size
            waveform_tensor = F.pad(waveform_tensor, (0, padding), "constant", 0)
        elif current_size > self.target_size:
            waveform_tensor = waveform_tensor[:, :self.target_size]

        return waveform_tensor, self.label

def test_collate_fn(batch):
    """
    Custom collate function to filter out None values from a torch.Dataset batch.
    """
    batch = [item for item in batch if item is not None]
    if not batch:
        return None, None
    
    return torch.utils.data.default_collate(batch)


class AudioDataModule(pl.LightningDataModule):
    def __init__(self, TP_file, TP_folder, TN_file, TN_folder, batch_size=32, split_ratios=(0.8, 0.1, 0.1), shuffle=True):
        super().__init__()
        self.pos_folder = TP_folder
        self.neg_folder = TN_folder
        self.pos_file = TP_file
        self.neg_file = TN_file
        self.batch_size = batch_size
        self.split_ratios = split_ratios
        self.train_shuffle = shuffle

        self.train_dataset = None
        self.dev_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        # Load the full datasets for TP and TN
        pos_dataset = AudioDataset(self.pos_file, self.pos_folder, binary_label=1)
        neg_dataset = AudioDataset(self.neg_file, self.neg_folder, binary_label=0)

        # Combine datasets
        combined_dataset = ConcatDataset([pos_dataset, neg_dataset])

        # Compute split sizes
        total_size = len(combined_dataset)
        train_size = int(self.split_ratios[0] * total_size)
        dev_size = int(self.split_ratios[1] * total_size)
        test_size = total_size - train_size - dev_size

        # Train/Dev/Test split
        self.train_dataset, self.dev_dataset, self.test_dataset = random_split(combined_dataset,
                                                                               [train_size, dev_size, test_size])

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          collate_fn=test_collate_fn,
                          shuffle=self.train_shuffle,
                          num_workers=2)

    def val_dataloader(self):
        return DataLoader(self.dev_dataset,
                          batch_size=self.batch_size,
                          collate_fn=test_collate_fn,
                          shuffle=False,
                          num_workers=2)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          collate_fn=test_collate_fn,
                          shuffle=False,
                          num_workers=2)

    def save_skipped_files_to_csv(self, file_path="./reading_fails.csv"):
        if self.train_dataset is None or self.dev_dataset is None or self.test_dataset is None:
            raise ValueError("The datasets are not initialized. Call setup() first.")

        skipped_details = []
        datasets = [("train", self.train_dataset), ("dev", self.dev_dataset), ("test", self.test_dataset)]
        for split_name, dataset in datasets:
            for idx, subdataset in enumerate(dataset.datasets):
                if hasattr(subdataset, "skipped_files"):
                    for file_idx, filename in subdataset.skipped_files:
                        skipped_details.append({"split": split_name,
                                                "dataset": "TPs" if isinstance(subdataset, AudioDataset) and subdataset.label == 1 else "TNs",
                                                "batch_idx": idx // self.batch_size,
                                                "sample_idx": file_idx % self.batch_size,
                                                "filepath": filename})

        # Write details to CSV
        with open(file_path, mode="w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=["split", "dataset", "batch_idx", "sample_idx", "filename"])
            writer.writeheader()
            writer.writerows(skipped_details)

        print(f"Skipped files saved to {file_path}")


# Models --------------------------------------------------------------------------------------------
class EPANNs_Binarized_Model(pl.LightningModule):
    def __init__(self,
                 model,
                 threshold=0.5,
                 output_mode='bin_raw',
                 overall_training=False,
                 learning_rate=1e-3,
                 weight_decay=1e-6,
                 t_max=50,
                 eta_min=1e-4,
                 f_beta=0.8):
        """
        :param model: Base-model to fine-tune.
        :param threshold: Probabilities Threshold for binary classification.
        :param output_mode: 'bin_only' for binary output only, 'bin_raw' for full output.
        :param overall_training: full model or last stages fine-tuning condition.
        :param learning_rate: Initial learning rate.
        :param weight_decay: Weight decay (for Adam optimizer).
        :param t_max: Maximum number of epochs for Cosine Annealing scheduler.
        :param eta_min: Minimum learning rate for Cosine Annealing scheduler.
        :param f_beta: Beta value for F-beta score.
        """
        super(EPANNs_Binarized_Model, self).__init__()
        self.model = model
        self.threshold = threshold
        self.beta = f_beta
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.t_max = t_max
        self.eta_min = eta_min
        self.output_mode = output_mode  # 'bin_only' or 'bin_raw'
        self.overall = overall_training

        # Prepare embedding and last layers for training
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
                    param.requires_grad = True  # Enable training
                else:
                    param.requires_grad = False  # Freeze weights


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
        emergency_prob = logits[:, 322]  # Extract Emergency Vehicle probability
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

        return loss


    def on_validation_epoch_end(self):
        self.log("epoch_val_accuracy", self.val_accuracy.compute(), prog_bar=True)
        self.log("epoch_val_precision", self.val_precision.compute(), prog_bar=True)
        self.log("epoch_val_recall", self.val_recall.compute(), prog_bar=True)
        self.log("epoch_val_f1", self.val_f1.compute(), prog_bar=True)

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
        # Extract FULL classification set & Data-TPs probabilities (for task-specific profiling)
        elif output_mode == "bin_raw":  
            start_time = time.perf_counter()
            full_preds_set = self.model(x.float().squeeze())['clipwise_output']
            end_time = time.perf_counter()
            inference_time = end_time - start_time

            preds = (full_preds_set[:, 322] >= self.threshold).float()
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
                               lr=self.learning_rate, 
                               weight_decay=self.weight_decay,
                               betas=(0.9, 0.999), 
                               eps=1e-08,
                               amsgrad=True)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.t_max, eta_min=self.eta_min)
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
