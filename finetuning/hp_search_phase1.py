############################################################################################################
#
#  Hyper-Parameters Search Phase-1: finding the best HPs combination which maximizes validation accuracy
#
############################################################################################################
import gc
from itertools import product
import csv
import torch
from epanns_inference import models
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping

from globals import Positives_csv, Positives, Negatives_csv, Negatives
from dataloaders import AudioDataModule
from models import EPANNs_Binarized_Model
from pytorch_lightning import seed_everything
torch.set_float32_matmul_precision('high')


# Hyperparameters definition
EPOCHS = 50
PATIENCE = 15
params_grid = {"overall_training": [True, False],
               "batch_size": [16, 32],
               "learning_rate": [1e-5, 1e-4, 1e-3, 1e-2],
               "threshold": [0.5, 0.6, 0.7],
               "weight_decay": [1e-6],
               "t_max": [EPOCHS],
               "eta_min": [1e-6]}
param_combinations = list(product(*params_grid.values()))   # Generate all combinations


# Early Stopping Callback (w. best validation metrics tracker)
class EarlyStoppingTracker(EarlyStopping):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.best_val_prec = 0.0
        self.best_val_rec = 0.0
        self.best_val_f1 = 0.0
        self.best_val_epoch = None

    def on_validation_epoch_end(self, trainer, pl_module):
        super().on_validation_epoch_end(trainer, pl_module)
        # Get current epoch and metrics
        current_epoch = trainer.current_epoch
        metrics = trainer.callback_metrics
        # Track best metrics and corresponding epoch          
        if metrics["epoch_val_accuracy"].item() > self.best_val_acc:
            self.best_val_loss = metrics["val_loss"].item()
            self.best_val_acc = metrics["epoch_val_accuracy"].item()
            self.best_val_prec = metrics["epoch_val_precision"].item()
            self.best_val_rec = metrics["epoch_val_recall"].item()
            self.best_val_f1 = metrics["epoch_val_f1"].item()
            self.best_val_epoch = current_epoch


# Output file creation
csv_file = "hp_search_phase1_results.csv"
with open(csv_file, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["run", "overall_training", "batch_size", "threshold",
                     "learning_rate", "weight_decay", "t_max", "eta_min",
                     "best_val_epoch", "best_val_loss", "best_val_acc", "best_val_prec", "best_val_rec", "best_val_F1", 
                     "stop_epoch", "val_loss", "val_acc", "val_prec", "val_rec", "val_F1"])


# Simplified HPs-Search
run = 0
for params in param_combinations:
    # Clean states & set Seed
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.reset_accumulated_memory_stats()
    gc.collect()
    seed_everything(42)
    
    run += 1 
    overall_training, batch_size, learning_rate, threshold, weight_decay, t_max, eta_min = params
    print(f"RUN {run}/{len(param_combinations)}: Overall Training: {overall_training}, Binary Threshold: {threshold}, Epochs: {EPOCHS}, Batch Size: {batch_size} samps, Learning Rate: {learning_rate}, Weight Decay: {weight_decay}, T_max: {t_max}, Eta_min: {eta_min}")
    print('\n' * 2)

    # Init model and dataset
    base_model = models.Cnn14_pruned(pre_trained=True)
    print('\n' * 2)
    model = EPANNs_Binarized_Model(model=base_model,
                                   threshold=threshold,
                                   output_mode="bin_raw",
                                   overall_training=overall_training,
                                   learning_rate=learning_rate,
                                   weight_decay=weight_decay,
                                   t_max=t_max,
                                   eta_min=eta_min)
    
    data_module = AudioDataModule(TP_file=Positives_csv, TP_folder=Positives,
                                  TN_file=Negatives_csv, TN_folder=Negatives,
                                  batch_size=batch_size)
    data_module.setup()

    # Train the model
    early_stopping = EarlyStoppingTracker(monitor="epoch_val_accuracy",
                                          mode="max",
                                          patience=PATIENCE,
                                          verbose=True)
    trainer = Trainer(max_epochs=EPOCHS, 
                      accelerator="gpu", 
                      devices=1, 
                      precision=32, 
                      callbacks=[early_stopping])
    print('\n' * 2) 
    trainer.fit(model, datamodule=data_module)
    stop_epoch = trainer.current_epoch

    # Metrics retrieval
    val_loss = trainer.callback_metrics["val_loss"].item()
    epoch_val_accuracy = trainer.callback_metrics["epoch_val_accuracy"].item()
    epoch_val_precision = trainer.callback_metrics["epoch_val_precision"].item()
    epoch_val_recall = trainer.callback_metrics["epoch_val_recall"].item()
    epoch_val_f1 = trainer.callback_metrics["epoch_val_f1"].item()

    # Append results to CSV
    with open(csv_file, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([run, overall_training, batch_size, threshold, 
                         learning_rate, weight_decay, t_max, eta_min,
                         early_stopping.best_val_epoch, early_stopping.best_val_loss, early_stopping.best_val_acc, early_stopping.best_val_prec, early_stopping.best_val_rec, early_stopping.best_val_f1, 
                         stop_epoch, val_loss, epoch_val_accuracy, epoch_val_precision, epoch_val_recall, epoch_val_f1])
    
    del base_model, model, data_module, early_stopping, trainer
    print('-----------------------------------------------------------------------------------------------------------------')
    print('\n' * 4)
