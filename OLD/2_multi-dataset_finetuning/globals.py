# DATASET
batch_size = 16

# MODEL
pre_trained = False
threshold = 0.5
output_mode = "bin_raw"
overall_training= True
f_beta = 0.8

# LEARNING RATE (ETA)
eta_max = 1e-4
eta_min = 1e-4
decay_epochs = 1
restart_eta = 1e-4
restart_interval = 1
warmup_epochs = 1
warmup_eta = 1e-4
weight_decay = 1e-6

# OPTIMIZER (Adam)
betas = (0.9, 0.999) 
weight_decay = 1e-6
eps = 1e-08

SMALL_THRESHOLD = 0.01
LARGE_THRESHOLD = 0.1
                            
# TRAINING CONSTANTS
EPOCHS = 100                                    # For Sequential Dataset-Aware Training
PATIENCE = 30                                   # For Sequential Dataset-Aware Training                       
ROUNDS = 5                                      # For Sequential Dataset-Aware Training
#EPOCHS = 1000                                   # For Unified Training
#PATIENCE = 50                                   # For Unified Training                                
CHECKPOINT_PATH = "0_finetuning_results/2025-01-20_20-36_lr_fix_aug/checkpoints/epoch=64_epoch_val_accuracy=0.8480.ckpt"                       
CHECKPOINT_DIR = "./experiments/checkpoints"    # Created by the Lightning Trainer (init)
RESULTS_DIR = "./experiments/model_results"     # Created by the Lightning Model (init)
