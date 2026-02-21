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

import csv
import json
import time
from pathlib import Path

import torch
from epanns_inference import models
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import seed_everything

torch.set_float32_matmul_precision('high')

from model import E2PANNs_Model
from dataloaders import (AudioSetEV_DataModule, sireNNet_DataModule, LSSiren_DataModule,
                         ESC50_DataModule, UrbanSound8K_DataModule, FSD50K_DataModule)
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
# Auxiliary functions
##################################################################################################################
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p

def save_test_metrics_csv(csv_path: Path, row: dict):
    ensure_dir(csv_path.parent)
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


##################################################################################################################
# DATASET TARGET DI TEST: AudioSet-EV
##################################################################################################################
as_test_dm = AudioSetEV_DataModule(TP_file="./datasets/AudioSet_EV/EV_Positives.csv",
                                   TP_folder="./datasets/AudioSet_EV/Positive_files/",
                                   TN_file="./datasets/AudioSet_EV/EV_Negatives.csv",
                                   TN_folder="./datasets/AudioSet_EV/Negative_files/",
                                   batch_size=32,
                                   split_ratios=(0.8, 0.1, 0.1),
                                   shuffle=True)
as_test_dm.setup()
as_test_loader = as_test_dm.test_dataloader()


##################################################################################################################
# LIST OF TRAINING DATASETS (ALL EV-BENCHMARKS except AudioSet-EV)
##################################################################################################################
def build_training_dm(dataset_name: str):
    if dataset_name == "sireNNet":
        dm = sireNNet_DataModule(folder_path="./datasets/sireNNet/",
                                 batch_size=32,
                                 split_ratios=(0.8, 0.1, 0.1),
                                 shuffle=True,
                                 target_size=96000,
                                 target_sr=32000)
    elif dataset_name == "LSSiren":
        dm = LSSiren_DataModule(folder_path="./datasets/Large-Scale_Audio_Dataset_for_Emergency_Vehicle_Sirens_and_Road_Noises/",
                                batch_size=32,
                                split_ratios=(0.8, 0.1, 0.1),
                                shuffle=True,
                                target_sr=32000,
                                min_length=32000)
    elif dataset_name == "ESC-50":
        dm = ESC50_DataModule(file_path="./datasets/ESC-50/esc50.csv",
                              folder_path="./datasets/ESC-50/cross_val_folds/",
                              batch_size=32,
                              split_ratios=(0.8, 0.1, 0.1),
                              shuffle=True,
                              target_size=160000,
                              target_sr=32000)
    elif dataset_name == "UrbanSound8K":
        dm = UrbanSound8K_DataModule(folder_path="./datasets/UrbanSound8K/audio",
                                     metadata_path="./datasets/UrbanSound8K/metadata/UrbanSound8K.csv",
                                     batch_size=32,
                                     split_ratios=(0.8, 0.1, 0.1),
                                     shuffle=True,
                                     target_sr=32000,
                                     min_length=32000)
    elif dataset_name == "FSD50K":
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
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return dm


TRAIN_DATASETS = ["sireNNet", "LSSiren", "ESC-50", "UrbanSound8K", "FSD50K"]


##################################################################################################################
# LOOP: for each dataset, train -> test on AudioSet-EV (test split)
##################################################################################################################
CSV_ROOT = Path(RESULTS_DIR) / "bi_dir_valid_csv"
ensure_dir(CSV_ROOT)

for ds_name in TRAIN_DATASETS:
    print('==========================================================================')
    print(f">>> RUN: train on '{ds_name}'  ->  test on 'AudioSet-EV' (test)")
    print('==========================================================================')

    # --- DATALOADERS ---
    dm = build_training_dm(ds_name)
    dm.setup()
    train_loader = dm.train_dataloader()

    has_val = True
    try:
        val_loader = dm.val_dataloader()
        if val_loader is None:
            has_val = False
    except Exception:
        val_loader = None
        has_val = False

    # --- MODEL ---
    base_model = models.Cnn14_pruned(pre_trained=pre_trained)
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

    # --- CALLBACKS & TRAINER ---
    callbacks_list = []
    if has_val:
        # Use EarlyStopping only if validation exists (everytime except for FSD50K)
        callbacks_list.append(EarlyStopping(monitor="epoch_val_accuracy",
                                            mode="max",
                                            patience=PATIENCE,
                                            verbose=True))

    trainer = Trainer(max_epochs=EPOCHS,
                      accelerator="auto",
                      devices=1,
                      precision=32,
                      callbacks=callbacks_list,
                      logger=logger,
                      log_every_n_steps=5,
                      default_root_dir=RESULTS_DIR)

    # --- TRAIN ---
    print("Training Model...")
    if has_val:
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    else:
        trainer.fit(model, train_dataloaders=train_loader)
    print('--------------------------------------------------------------------------')
    print("\n" * 2, end="")

    # --- TEST on AudioSet-EV (Test split) ---
    print("Testing Model on AudioSet-EV (test split)...")
    test_results = trainer.test(model, dataloaders=as_test_loader)
    metrics = test_results[0] if isinstance(test_results, list) and len(test_results) > 0 else {}

    # --- Save results CSV (one for each source dataset) ---
    csv_path = CSV_ROOT / f"test_on_AudioSetEV_from_{ds_name}.csv"
    row = {"source_dataset": ds_name,
           "target_dataset": "AudioSet-EV",
           "epochs": EPOCHS,
           "patience": PATIENCE,
           "used_validation": has_val,
           "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
           **{f"metric_{k}": v for k, v in metrics.items()}}
    save_test_metrics_csv(csv_path, row)

    # --- Save also a JSON with the raw results (useful for debugging) ---
    json_path = csv_path.with_suffix(".json")
    with json_path.open("w", encoding="utf-8") as jf:
        json.dump({"run": row, "raw_metrics": metrics}, jf, indent=2)

    # --- Cleanup memory ---
    del model, base_model, dm
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.reset_max_memory_cached()
    torch.cuda.empty_cache()

print('EOF')
