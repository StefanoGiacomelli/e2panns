# DATASET
batch_size = 32

# MODEL
threshold = 0.5
output_mode = "bin_raw"
overall_training= True
f_beta = 0.8

# LEARNING RATE (ETA)
eta_max = 1e-3
eta_min = 1e-3
decay_epochs = 1
restart_eta = 1e-3
restart_interval = 1
warmup_epochs = 1
warmup_eta = 1e-3
weight_decay = 1e-6

# OPTIMIZER (Adam)
betas = (0.9, 0.999) 
weight_decay = 1e-6
eps = 1e-08
                            
# TRAINING CONSTANTS
EPOCHS = 1000
PATIENCE = 50                       
CHECKPOINT_DIR = "./experiments/checkpoints"    # Created by the Lightning Trainer (init)
RESULTS_DIR = "./experiments/model_results"     # Created by the Lightning Model (init)
