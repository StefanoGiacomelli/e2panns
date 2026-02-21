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
import torch
import pytorch_lightning as pl
from pytorch_lightning import seed_everything

# Your model + data loading
from epanns_inference import models
from model import E2PANNs_Model
from EV_benchmark_test_factory import (get_audioset_ev_testloaders,
                                       get_audioset_ev_aug_testloaders,
                                       get_sirennet_testloaders,
                                       get_lssiren_testloader,
                                       get_fsd50k_testloader,
                                       get_esc50_testloaders,
                                       get_urbansound8k_testloaders)

# Ensure reproducibility
seed_everything(42)

###############################################################################
# PARAMETERS
###############################################################################
#CHECKPOINT_PATH = "experiments/2025-01-20_20-36_lr_fix_aug/checkpoints/epoch=64_epoch_val_accuracy=0.8480.ckpt"
#RESULTS_DIR_ROOT = "./test_results_lr_fix_aug"
CHECKPOINT_PATH = "2_multi-dataset_results/2025-03-29_13-21_unified/checkpoints/epoch=119_epoch_val_accuracy=0.8831.ckpt"
RESULTS_DIR_ROOT = "./test_results_multi_unified"
CLASS_IDX = 322
THRESHOLD = 0.5
OUTPUT_MODE = 'bin_raw'
F_BETA = 2.0

###############################################################################
# TEST DATALOADERS
###############################################################################
# AudioSet-EV (Standard)
test_dl_audioset_ev = get_audioset_ev_testloaders(TP_file="./datasets/AudioSet_EV/EV_Positives.csv",
                                                  TP_folder="./datasets/AudioSet_EV/Positive_files/",
                                                  TN_file="./datasets/AudioSet_EV/EV_Negatives.csv",
                                                  TN_folder="./datasets/AudioSet_EV/Negative_files/",
                                                  batch_size=32,
                                                  split_ratios=(0.8, 0.1, 0.1))

# AudioSet-EV (w. Augmentations)
test_dl_audioset_ev_aug = get_audioset_ev_aug_testloaders(TP_file="./datasets/AudioSet_EV/EV_Positives.csv",
                                                          TP_folder="./datasets/AudioSet_EV/Positive_files/",
                                                          TN_file="./datasets/AudioSet_EV/EV_Negatives.csv",
                                                          TN_folder="./datasets/AudioSet_EV/Negative_files/",
                                                          batch_size=32,
                                                          split_ratios=(0.8, 0.1, 0.1),
                                                          aug_prob=0.7)

# sireNNet (returns multiple test loaders -> different fraction splits)
test_dls_sirennet = get_sirennet_testloaders(folder_path="./datasets/sireNNet/",
                                             batch_size=32,
                                             target_size=96000,
                                             target_sr=32000)

# LSSiren
test_dl_lssiren = get_lssiren_testloader(folder_path="./datasets/Large-Scale_Audio_Dataset_for_Emergency_Vehicle_Sirens_and_Road_Noises/",
                                         batch_size=32,
                                         target_sr=32000,
                                         min_length=32000)

# FSD50K
test_dl_fsd50k = get_fsd50k_testloader(pos_csv="./datasets/FSD50K/FSD-eval_positives.csv",
                                       neg_csv="./datasets/FSD50K/FSD-eval_negatives.csv",
                                       folder_path="./datasets/FSD50K/FSD50K.eval_audio/",
                                       batch_size=32,
                                       target_sr=32000)

# 6) ESC-50 (5 folds)
test_dls_esc50 = get_esc50_testloaders(csv_path="./datasets/ESC-50/esc50.csv",
                                       wavs_folder="./datasets/ESC-50/cross_val_folds/",
                                       batch_size=32,
                                       target_size=160000,
                                       target_sr=32000)

# 7) UrbanSound8K (10 folds)
test_dls_urbansound8k = get_urbansound8k_testloaders(folder_path="./datasets/UrbanSound8K/audio/",
                                                     metadata_path="./datasets/UrbanSound8K/metadata/UrbanSound8K.csv",
                                                     batch_size=32,
                                                     target_sr=32000,
                                                     min_length=32000)

###############################################################################
# SET UP
###############################################################################
trainer = pl.Trainer(accelerator="auto",
                     devices=1,
                     logger=False,
                     enable_checkpointing=False)

# 1) Instantiate the base model (PANNs Cnn14_pruned --> EPANNs)
base_model = models.Cnn14_pruned(pre_trained=False)

# 2) Instantiate the E2PANNs Model
model = E2PANNs_Model(model=base_model,
                      threshold=THRESHOLD,
                      output_mode=OUTPUT_MODE,
                      class_idx=CLASS_IDX,
                      f_beta=F_BETA,
                      results_path=None)

# 3) Load the checkpoint's weights into the base model
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model.load_trained_weights(checkpoint_path=CHECKPOINT_PATH,
                           device=device,
                           verbose=False,
                           validate_updates=False)

###############################################################################
# MAIN TESTING LOOP
###############################################################################
if not os.path.exists(CHECKPOINT_PATH):
    print(f"Checkpoint not found: {CHECKPOINT_PATH}")
    exit(0)

print(f"\n=== Using checkpoint: {CHECKPOINT_PATH} ===")

# 1) AudioSet-EV Standard
dataset_name = "audioset_ev_standard"
sub_results_dir = os.path.join(RESULTS_DIR_ROOT, dataset_name)
os.makedirs(sub_results_dir, exist_ok=True)
model.results_path = sub_results_dir + '/'
print(f"\n--- Testing on {dataset_name} ---")
trainer.test(model, dataloaders=test_dl_audioset_ev)
print(f"Results saved to: {sub_results_dir}")

# 2) AudioSet-EV Augmented
dataset_name = "audioset_ev_aug"
sub_results_dir = os.path.join(RESULTS_DIR_ROOT, dataset_name)
os.makedirs(sub_results_dir, exist_ok=True)
model.results_path = sub_results_dir + '/'
print(f"\n--- Testing on {dataset_name} ---")
trainer.test(model, dataloaders=test_dl_audioset_ev_aug)
print(f"Results saved to: {sub_results_dir}")

# 3) sireNNet
dataset_name = "sireNNet"
for i, single_test_dl in enumerate(test_dls_sirennet):
    loader_subdir = os.path.join(RESULTS_DIR_ROOT, dataset_name, f"loader_{i}")
    os.makedirs(loader_subdir, exist_ok=True)
    model.results_path = loader_subdir + '/'
    print(f"\n--- Testing on {dataset_name} [loader_{i}] ---")
    trainer.test(model, dataloaders=single_test_dl)
    print(f"Results saved to: {loader_subdir}")

# 4) LSSiren
dataset_name = "lssiren"
sub_results_dir = os.path.join(RESULTS_DIR_ROOT, dataset_name)
os.makedirs(sub_results_dir, exist_ok=True)
model.results_path = sub_results_dir + '/'
print(f"\n--- Testing on {dataset_name} ---")
trainer.test(model, dataloaders=test_dl_lssiren)
print(f"Results saved to: {sub_results_dir}")

# 5) FSD50K
dataset_name = "fsd50k"
sub_results_dir = os.path.join(RESULTS_DIR_ROOT, dataset_name)
os.makedirs(sub_results_dir, exist_ok=True)
model.results_path = sub_results_dir + '/'
print(f"\n--- Testing on {dataset_name} ---")
trainer.test(model, dataloaders=test_dl_fsd50k)
print(f"Results saved to: {sub_results_dir}")

# 6) ESC-50
dataset_name = "esc50"
for fold_idx, esc50_dl in enumerate(test_dls_esc50, start=1):
    fold_subdir = os.path.join(RESULTS_DIR_ROOT, dataset_name, f"fold_{fold_idx}")
    os.makedirs(fold_subdir, exist_ok=True)
    model.results_path = fold_subdir + '/'
    print(f"\n--- Testing on {dataset_name} [fold_{fold_idx}] ---")
    trainer.test(model, dataloaders=esc50_dl)
    print(f"Results saved to: {fold_subdir}")

# 7) UrbanSound8K
dataset_name = "urbansound8k"
for fold_idx, urban_dl in enumerate(test_dls_urbansound8k, start=1):
    fold_subdir = os.path.join(RESULTS_DIR_ROOT, dataset_name, f"fold_{fold_idx}")
    os.makedirs(fold_subdir, exist_ok=True)
    
    model.results_path = fold_subdir + '/'
    
    print(f"\n--- Testing on {dataset_name} [fold_{fold_idx}] ---")
    trainer.test(model, dataloaders=urban_dl)
    print(f"Results saved to: {fold_subdir}")

print("\nAll testing complete!")
print("Check the subfolders under:")
print(RESULTS_DIR_ROOT)
print("for per-dataset (and per-loader) metrics and plots.")