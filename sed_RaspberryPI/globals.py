# Real-Time Inference Parameters
AUDIO_FILE_PATH = "./files/AudioSet_EV_Positives_Debug/"
AUDIO_FILE = "_6YdBNupJEo_Original.wav"
TORCH_NUM_THREADS = 4

# Frame and Buffer Configuration
frame_duration_min = 0.310
frame_duration_max = 1.0
buffer_duration = 20.0
chunk_duration = 0.310

# Adaptive Inference Logic
output_threshold = 0.5
adapt_width_coeff = 0.4 # 0 disables adaptive_width / 1 takes the whole sample_rate as frame width increment

# Audio & Inference Settings
SAMPLING_RATE = 32000
DEVICE = "cpu"
CLASS_INDEX = 322  # "Emergency vehicle" index in AudioSet
RANDOM_SEED = 42
CHECKPOINT_PATH = "./files/checkpoints/"
CHECKPOINT = "audioset_ev_best.ckpt"

# Logging
HK_LOGGING_ENABLED = True
# Output Folder Root (used by HousekeepingLogger)
HKL_PATH = "./outputs/"  # Base directory for logs: outputs/<checkpoint>/<log_type>/<filename>.csv

# Performance Monitoring
ENABLE_MONITORING = True
MONITORING_INTERVAL = 0.1

# Plotting
SAVE_FIGURES = True
