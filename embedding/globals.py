# RT parameters
FILES_PATH = "./datasets/audio_files/"
audio_filename = "ambulanza2.wav"
audio_file_path = FILES_PATH + audio_filename
SIMULATE_REAL_TIME = True   # with this flag enabled write_to_buffer waits for the chunk duration. HANDLE WITH CARE!!!
TORCH_NUM_THREADS = 1       # number of physical cores implied in the inference computation

# Audio parameters
SAMPLING_RATE = 32000       # Hz
frame_duration_min = 0.310  # Minimum frame duration (seconds)
frame_duration_max = 1.0    # Maximum frame duration (seconds)
buffer_duration = 10.0      # Buffer duration (seconds)
chunk_duration = 0.310      # Duration of audio chunks to be written (seconds)

# Model parameters
RANDOM_SEED = 42            
DEVICE = "cpu"              
output_threshold = 0.8      # Model's binary threshold (for inference probability)
CHECKPOINT_PATH = "./model/checkpoints/epoch=0_epoch_val_accuracy=0.8454.ckpt"
CLASS_INDEX = 322           # Class-IDX of interest in AudioSet (322 = "Emergency vehicle")

# Housekeeping logging parameters
HK_LOGGING_ENABLED = False  # Wheter to gather a set of figures related to inference and simulation metrics 
HKL_PATH = "./embedding/runtime_outputs/"

# Experiment parameters
ENABLE_MONITORING = True    # Performance monitor
MONITORING_INTERVAL = 0.1   # seconds
MONITORING_FILE = HKL_PATH + "perf_on_" + audio_filename.split('.')[0] + ".csv"
SAVE_FIGURES = True
