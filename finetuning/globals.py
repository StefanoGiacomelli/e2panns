# DATASET
Positives_csv = './datasets/AudioSet_EV/EV_Positives.csv'
Positives = "./datasets/AudioSet_EV/Positive_files/"
Negatives_csv = './datasets/AudioSet_EV/EV_Negatives.csv'
Negatives = "./datasets/AudioSet_EV/Negative_files/"

# MODEL PARAMETERS
batch_size = 32
threshold = 0.5
output_mode = "bin_raw"
learning_rate = 1e-3
weight_decay = 1e-6
t_max = 10
eta_min = 1e-6

# TRAINING CONSTANTS
overall_training = True
EPOCHS = 100
PATIENCE = 25
CHECKPOINT_DIR = "./experiments/checkpoints"    # Created by the Lightning Trainer (init)
RESULTS_DIR = "./experiments/model_results"     # Created by the Lightning Model (init)
