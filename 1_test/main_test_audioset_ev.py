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

from epanns_inference import models
from EV_benchmark_test_factory import get_audioset_ev_testloaders
from model import E2PANNs_Model

# Ensure reproducibility
seed_everything(42)

#######################################################################################################################
paths = [('experiments/2025-01-08_20-06_finetune/checkpoints/epoch=30_epoch_val_accuracy=0.8524.ckpt', 'finetune'),
         ('experiments/2025-01-08_20-39_train/checkpoints/epoch=3_epoch_val_accuracy=0.8602.ckpt', 'train'),
         ('experiments/2025-01-19_22-07_lr_warmup/checkpoints/epoch=6_epoch_val_accuracy=0.8571.ckpt', 'lr_warmup'),
         ('experiments/2025-01-19_23-17_lr_schedule/checkpoints/epoch=7_epoch_val_accuracy=0.8594.ckpt', 'lr_schedule'),
         ('experiments/2025-01-20_16-41_aug/checkpoints/epoch=11_epoch_val_accuracy=0.8353.ckpt', 'aug'),
         ('experiments/2025-01-20_20-36_lr_fix_aug/checkpoints/epoch=64_epoch_val_accuracy=0.8480.ckpt', 'lr_fix_aug')]

# Root directory to store all AudioSet_EV test results
results_dir_root = "./test_results/audioset_ev_standard"

# Create the test dataloader for AudioSet_EV (train/dev/test split)
test_dl_audioset_ev = get_audioset_ev_testloaders(TP_file="./datasets/AudioSet_EV/EV_Positives.csv",
                                                  TP_folder="./datasets/AudioSet_EV/Positive_files/",
                                                  TN_file="./datasets/AudioSet_EV/EV_Negatives.csv",
                                                  TN_folder="./datasets/AudioSet_EV/Negative_files/",
                                                  batch_size=32,
                                                  split_ratios=(0.8, 0.1, 0.1))

# Create a Lightning Trainer for testing only
trainer = pl.Trainer(accelerator="auto",     # "cuda" if available, else "cpu"
                     devices=1,
                     logger=False,
                     enable_checkpointing=False)

#######################################################################################################################
# MAIN LOOP: TEST EACH E2PANNs CHECKPOINT
#######################################################################################################################
for checkpoint_path, subfolder_name in paths:
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        continue
    
    # 1) Init folders and names
    ckpt_name = os.path.basename(checkpoint_path).replace(".ckpt", "")
    sub_results_dir = os.path.join(results_dir_root, subfolder_name)
    os.makedirs(sub_results_dir, exist_ok=True)

    # 2) Instantiate a new base model (PANNs Cnn14_pruned --> EPANNs)
    base_model = models.Cnn14_pruned(pre_trained=False)

    # 3) Instantiate the E2PANNs_Model wrapper pointing to the sub_results_dir
    model = E2PANNs_Model(model=base_model,
                          threshold=0.5,
                          output_mode='bin_raw',
                          class_idx=322,  # e.g., AudioSet "Emergency Vehicle" class index
                          f_beta=0.8,
                          results_path=sub_results_dir + '/')

    # 4) Load the checkpoint's weights into the base model
    print(f"\n=== Loading checkpoint: {checkpoint_path} ===")
    model.load_trained_weights(checkpoint_path=checkpoint_path,
                               device='cuda:0' if torch.cuda.is_available() else 'cpu',
                               verbose=False,
                               validate_updates=False)

    # 5) Test on AudioSet_EV_Standard
    print(f"\n=== Testing {ckpt_name} on AudioSet_EV ===")
    trainer.test(model, dataloaders=test_dl_audioset_ev)
    print(f"Done testing {ckpt_name}. Results saved to '{sub_results_dir}'\n")

print("\nAll checkpoints have been tested. Check each subfolder in:")
print(results_dir_root)
print("for metrics and plots.")
