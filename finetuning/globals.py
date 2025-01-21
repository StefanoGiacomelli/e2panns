# DATASET
Positives_csv = './datasets/AudioSet_EV/EV_Positives.csv'
Positives = "./datasets/AudioSet_EV/Positive_files/"
Negatives_csv = './datasets/AudioSet_EV/EV_Negatives.csv'
Negatives = "./datasets/AudioSet_EV/Negative_files/"
split_ratios = (0.8, 0.1, 0.1)
batch_size = 32
augmentation = True

# MODEL
threshold = 0.5
output_mode = "bin_raw"
overall_training=False
f_beta = 0.8

# LEARNING RATE (ETA)
eta_max = 1e-3
eta_min = 1e-6
decay_epochs = 50
restart_eta = 1e-4
restart_interval = 10
warmup_epochs = 10
warmup_eta = 1e-4
weight_decay = 1e-6

# OPTIMIZER (Adam)
betas = (0.9, 0.999) 
weight_decay = 1e-6
eps = 1e-08
                            
# TRAINING CONSTANTS
EPOCHS = 1000
PATIENCE = 100                                  # Extended (from 100) to enter inside cyclical-LR scheduling                          
CHECKPOINT_DIR = "./experiments/checkpoints"    # Created by the Lightning Trainer (init)
RESULTS_DIR = "./experiments/model_results"     # Created by the Lightning Model (init)
