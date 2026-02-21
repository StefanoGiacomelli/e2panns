import warnings
# Ignore "Can't initialize NVML" user warning
warnings.filterwarnings(action="ignore",
                        message=".*Can't initialize NVML.*",
                        category=UserWarning)

# Ignore the PyTorch future warning about torch.load weights_only
warnings.filterwarnings(action="ignore",
                        message=".*You are using `torch.load` with `weights_only=False`.*",
                        category=FutureWarning)

# Ignore the softmax dimension warning
warnings.filterwarnings(action="ignore",
                        message=".*Implicit dimension choice for softmax has been deprecated.*",
                        category=UserWarning)

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
from epanns_inference import models
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import seed_everything

torch.set_float32_matmul_precision('high')
from model import E2PANNs_Model
from torch.utils.data import ConcatDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from dataloaders import (AudioSetEV_DataModule, sireNNet_DataModule, LSSiren_DataModule, ESC50_DataModule, 
                         UrbanSound8K_DataModule, FSD50K_DataModule)
from globals import *

# Reset all CPU and CUDA memories
torch.cuda.empty_cache()
torch.cuda.reset_max_memory_allocated()
torch.cuda.reset_max_memory_cached()
torch.cuda.empty_cache()

# Set the seed for reproducibility
seed_everything(42)


# ---------------- Logger ----------------
def start_tensorboard(logdir):    
    try:
        print(f"Starting TensorBoard at log directory: {logdir}")
        os.system(f"tensorboard --logdir {logdir} --host 0.0.0.0 &")
    except Exception as e:
        print(f"Failed to start TensorBoard: {e}")

logger = TensorBoardLogger(save_dir="./experiments/tb_logs", name="EPANNs_Binarized")
print('Logs will be saved in: ', logger.log_dir)
start_tensorboard(logger.log_dir)
print('--------------------------------------------------------------------------')
print("\n" * 2, end="")


##################################################################################################################
# DATASETS
##################################################################################################################
train_datasets, val_datasets, test_datasets = [], [], []

def audio_collate_fn(batch):
    waveforms, labels = zip(*batch)  # Separate waveforms and labels
    # Transpose waveforms to (num_samples, num_channels) for padding
    waveforms = [w.transpose(0, 1) for w in waveforms]  # shape now (num_samples, num_channels)
    # Pad waveforms batch-wise (pad_sequence pads along first dim)
    padded_waveforms = pad_sequence(waveforms, batch_first=True, padding_value=0.0)
    # After padding, transpose back to original shape: (batch_size, num_channels, num_samples)
    padded_waveforms = padded_waveforms.transpose(1, 2)
    labels = torch.tensor(labels, dtype=torch.long)

    return padded_waveforms, labels


# --- AudioSetEV ---
dm = AudioSetEV_DataModule(TP_file="./datasets/AudioSet_EV/EV_Positives.csv",
                           TP_folder="./datasets/AudioSet_EV/Positive_files/",
                           TN_file="./datasets/AudioSet_EV/EV_Negatives.csv",
                           TN_folder="./datasets/AudioSet_EV/Negative_files/",
                           batch_size=32, 
                           split_ratios=(0.8, 0.1, 0.1), 
                           shuffle=True)
dm.setup()
train_datasets.append(dm.train_dataloader().dataset)
val_datasets.append(dm.val_dataloader().dataset)
test_datasets.append(dm.test_dataloader().dataset)

# --- sireNNet ---
dm = sireNNet_DataModule(folder_path="./datasets/sireNNet/",
                         batch_size=32,
                         split_ratios=(0.8, 0.1, 0.1),
                         shuffle=True,
                         target_size=96000,
                         target_sr=32000)
dm.setup()
train_datasets.append(dm.train_dataloader().dataset)
val_datasets.append(dm.val_dataloader().dataset)
test_datasets.append(dm.test_dataloader().dataset)

# --- LSSiren ---
dm = LSSiren_DataModule(folder_path="./datasets/Large-Scale_Audio_Dataset_for_Emergency_Vehicle_Sirens_and_Road_Noises/",
                        batch_size=32,
                        split_ratios=(0.8, 0.1, 0.1),
                        shuffle=True,
                        target_sr=32000,
                        min_length=32000)
dm.setup()
train_datasets.append(dm.train_dataloader().dataset)
val_datasets.append(dm.val_dataloader().dataset)
test_datasets.append(dm.test_dataloader().dataset)

# --- ESC-50 ---
dm = ESC50_DataModule(file_path="./datasets/ESC-50/esc50.csv",
                      folder_path="./datasets/ESC-50/cross_val_folds/",
                      batch_size=32,
                      split_ratios=(0.8, 0.1, 0.1),
                      shuffle=True,
                      target_size=160000,
                      target_sr=32000)
dm.setup()
train_datasets.append(dm.train_dataloader().dataset)
val_datasets.append(dm.val_dataloader().dataset)
test_datasets.append(dm.test_dataloader().dataset)

# --- UrbanSound8K ---
dm = UrbanSound8K_DataModule(folder_path="./datasets/UrbanSound8K/audio",
                             metadata_path="./datasets/UrbanSound8K/metadata/UrbanSound8K.csv",
                             batch_size=32,
                             split_ratios=(0.8, 0.1, 0.1),
                             shuffle=True,
                             target_sr=32000,
                             min_length=32000)
dm.setup()
train_datasets.append(dm.train_dataloader().dataset)
val_datasets.append(dm.val_dataloader().dataset)
test_datasets.append(dm.test_dataloader().dataset)

# --- FSD50K ---
dm = FSD50K_DataModule(pos_dev_csv="./datasets/FSD50K/FSD-dev_positives.csv",
                       neg_dev_csv="./datasets/FSD50K/FSD-dev_negatives.csv",
                       dev_folder_path="./datasets/FSD50K/FSD50K.dev_audio/",
                       pos_eval_csv="./datasets/FSD50K/FSD-eval_positives.csv",
                       neg_eval_csv="./datasets/FSD50K/FSD-eval_negatives.csv",
                       eval_folder_path="./datasets/FSD50K/FSD50K.eval_audio/",
                       batch_size=32,
                       split_ratios=(0.8, 0.2),
                       target_sr=32000,
                       shuffle=True)
dm.setup()
train_datasets.append(dm.train_dataloader().dataset)
val_datasets.append(dm.val_dataloader().dataset)
test_datasets.append(dm.test_dataloader().dataset)

# ----------------------
# Concatenate datasets
# ----------------------
combined_train_dataset = ConcatDataset(train_datasets)
combined_val_dataset   = ConcatDataset(val_datasets)
combined_test_dataset  = ConcatDataset(test_datasets)

# ----------------------
# Unified DataLoaders
# ----------------------
train_loader = DataLoader(combined_train_dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=4,
                          persistent_workers=False,
                          collate_fn=audio_collate_fn)

val_loader = DataLoader(combined_val_dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=4,
                        persistent_workers=False,
                        collate_fn=audio_collate_fn)

test_loader = DataLoader(combined_test_dataset,
                         batch_size=batch_size,
                         shuffle=False,
                         num_workers=4,
                         persistent_workers=False,
                         collate_fn=audio_collate_fn)

print(f"Unified TRAIN       dataset size: {len(combined_train_dataset)}")
print(f"Unified VALIDATION  dataset size: {len(combined_val_dataset)}")
print(f"Unified TEST        dataset size: {len(combined_test_dataset)}")
print('--------------------------------------------------------------------------')
print("\n" * 2, end="")


##################################################################################################################
# MODEL & TRAINER
##################################################################################################################
base_model = models.Cnn14_pruned(pre_trained=pre_trained)

# --> Instantiate the new E2PANNs_Model (instead of EPANNs_Binarized_Model)
model = E2PANNs_Model(base_model,
                      threshold=threshold,
                      output_mode=output_mode,
                      overall_training=overall_training,
                      eta_max=eta_max,
                      eta_min=eta_min,
                      decay_epochs=decay_epochs,
                      restart_eta=restart_eta,
                      restart_interval=restart_interval,
                      warmup_epochs=warmup_epochs,
                      warmup_eta=warmup_eta,
                      weight_decay=weight_decay,
                      f_beta=f_beta)

# Load the checkpoint's weights into the model
model.load_trained_weights(checkpoint_path=CHECKPOINT_PATH,
                           verbose=False,
                           validate_updates=False)
print('--------------------------------------------------------------------------')
print("\n" * 2, end="")

# ---------------- Trainer & Callbacks ----------------
checkpoint_callback = ModelCheckpoint(dirpath=CHECKPOINT_DIR,
                                      filename="{epoch}_{epoch_val_accuracy:.4f}",
                                      monitor="epoch_val_accuracy",
                                      mode="max",
                                      save_top_k=1,
                                      save_weights_only=False,
                                      verbose=True)

early_stopping = EarlyStopping(monitor="epoch_val_accuracy",
                               mode="max",
                               patience=PATIENCE,
                               verbose=True)

trainer = Trainer(max_epochs=EPOCHS,
                  accelerator="auto",
                  devices=1,
                  precision=32,
                  callbacks=[checkpoint_callback, early_stopping],
                  logger=logger,
                  log_every_n_steps=5,
                  default_root_dir=RESULTS_DIR)

################################################################################
if __name__ == '__main__':
    print("Training Model...")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print('--------------------------------------------------------------------------')
    print("\n" * 2, end="")

    print("Testing Model...")
    trainer.test(model, dataloaders=test_loader)
    print('EOF')
