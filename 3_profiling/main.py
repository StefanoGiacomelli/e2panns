import warnings
# Ignore the PyTorch future warning about torch.load weights_only
warnings.filterwarnings(action="ignore",
                        message=".*You are using `torch.load` with `weights_only=False`.*",
                        category=FutureWarning)

# Ignore the softmax dimension warning
warnings.filterwarnings(action="ignore",
                        message=".*Implicit dimension choice for softmax has been deprecated.*",
                        category=UserWarning)
import os
import datetime
from epanns_inference import models
from utils import (your_gpu, your_hardware, load_lightning2pt, set_seeds, 
                   min_binary_search, overall_time, process_time, memory_and_cache, cpu_usage, energy_co2, 
                   cuda_time, e2e_inference_time, gpu_memory, gpu_usage)


# Ensure reproducibility
set_seeds(42)


# Output Parameters -----------------------------------------------------------------------------
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M")
RESULTS_DIR = "./3_profiling_results"
os.makedirs(RESULTS_DIR, exist_ok=True)


# GPU & Hardware profiling ----------------------------------------------------------------------
gpu_info_path = os.path.join(RESULTS_DIR, f"gpu_info_{TIMESTAMP}.json")
device, gpu_info = your_gpu(verbose=True, save_path=gpu_info_path)
print(f"Torch device: {device}")
print('\n')

hardware_info_path = os.path.join(RESULTS_DIR, f"hardware_info_{TIMESTAMP}.json")
hardware_info = your_hardware(verbose=True, save_path=hardware_info_path)
print('\n')

# CPU inference profiling -----------------------------------------------------------------------
CHECKPOINT_PATH = "./2_multi-dataset_results/2025-03-29_13-21_unified/checkpoints/epoch=119_epoch_val_accuracy=0.8831.ckpt"
#CHECKPOINT_PATH = "./0_finetuning_results/2025-01-19_22-07_lr_warmup/checkpoints/epoch=6_epoch_val_accuracy=0.8571.ckpt"
#CHECKPOINT_PATH = "./0_finetuning_results/2025-01-20_20-36_lr_fix_aug/checkpoints/epoch=64_epoch_val_accuracy=0.8480.ckpt"
SAMPLE_RATE = 32000
INPUT_DURATION_SEC = 10.0
ITERATIONS = 100

# Load the model
model = models.Cnn14_pruned(pre_trained=False)
model, _ = load_lightning2pt(CHECKPOINT_PATH, model, device="cpu", verbose=True, validate_updates=False)
print('\n')
model.cpu()
model_info_path = os.path.join(RESULTS_DIR, f"model_info_{TIMESTAMP}.json")

# Minimum inpout size
min_binary_search(model=model,
                  sample_rate=SAMPLE_RATE,
                  device="cpu",
                  save_path=model_info_path,
                  verbose=True)
print('\n')

# Benchmark CPU Overall Time
npz_path = os.path.join(RESULTS_DIR, f"cpu_overall_times_{TIMESTAMP}.npz")
overall_time(model=model,
             device="cpu",
             sample_rate=SAMPLE_RATE,
             input_duration_sec=INPUT_DURATION_SEC,
             iterations=ITERATIONS,
             verbose=True,
             save_path=model_info_path,
             npz_save_path=npz_path)
print('\n')

# CPU Process Time
npz_path = os.path.join(RESULTS_DIR, f"cpu_process_times_{TIMESTAMP}.npz")
process_time(model=model,
             device="cpu",
             sample_rate=SAMPLE_RATE,
             input_duration_sec=INPUT_DURATION_SEC,
             iterations=ITERATIONS,
             verbose=True,
             save_path=model_info_path,
             npz_save_path=npz_path)
print('\n')

# Memory/Cache Usage
npz_path = os.path.join(RESULTS_DIR, f"memory_and_cache_{TIMESTAMP}.npz")
memory_and_cache(model=model,
                 device="cpu",
                 sample_rate=SAMPLE_RATE,
                 input_duration_sec=INPUT_DURATION_SEC,
                 iterations=ITERATIONS,
                 verbose=True,
                 save_path=model_info_path,
                 npz_save_path=npz_path)
print('\n')

# CPU Usage
npz_path = os.path.join(RESULTS_DIR, f"cpu_usage_{TIMESTAMP}.npz")
cpu_usage(model=model,
          device="cpu",
          sample_rate=SAMPLE_RATE,
          input_duration_sec=INPUT_DURATION_SEC,
          iterations=ITERATIONS,
          verbose=True,
          save_path=model_info_path,
          npz_save_path=npz_path)
print('\n')

# GPU Inference Profiling -------------------------------------------------------------------
model.cuda()

# Cuda Time
npz_path = os.path.join(RESULTS_DIR, f"cuda_times_{TIMESTAMP}.npz")
cuda_time(model=model,
          device="cuda:0",
          sample_rate=SAMPLE_RATE,
          input_duration_sec=INPUT_DURATION_SEC,
          iterations=ITERATIONS,
          verbose=True,
          save_path=model_info_path,
          npz_save_path=npz_path)
print('\n')

# E2E Inference Time
npz_path = os.path.join(RESULTS_DIR, f"e2e_inference_times_{TIMESTAMP}.npz")
e2e_inference_time(model=model,
                   device="cuda:0",
                   sample_rate=SAMPLE_RATE,
                   input_duration_sec=INPUT_DURATION_SEC,
                   iterations=ITERATIONS,
                   verbose=True,
                   save_path=model_info_path,
                   npz_save_path=npz_path)
print('\n')

# GPU Memory Usage
gpu_memory(model=model,
           device="cuda:0",
           sample_rate=SAMPLE_RATE,
           input_duration_sec=INPUT_DURATION_SEC,
           iterations=ITERATIONS,
           verbose=True,
           save_path=model_info_path)
print('\n')

# GPU Usage
gpu_usage(model=model,
          device="cuda:0",
          sample_rate=SAMPLE_RATE,
          input_duration_sec=INPUT_DURATION_SEC,
          iterations=ITERATIONS,
          verbose=True,
          save_path=model_info_path)
print('\n')

# Energy and CO2 Emissions
npz_path = os.path.join(RESULTS_DIR, f"energy_co2_{TIMESTAMP}.npz")
energy_co2(model=model,
           device="cpu",
           sample_rate=SAMPLE_RATE,
           input_duration_sec=INPUT_DURATION_SEC,
           iterations=ITERATIONS,
           verbose=True,
           save_path=model_info_path,
           npz_save_path=npz_path)
print('\n')

print("EOF")
