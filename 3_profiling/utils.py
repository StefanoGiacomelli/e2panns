import os
import sys
import shutil
import subprocess
import glob
import platform
import signal
import psutil
import tracemalloc
from queue import Queue
import threading
from concurrent.futures import ThreadPoolExecutor
import time
from tqdm import tqdm
import random
import logging
import json
import csv
import numpy as np
import torch
from scipy.stats import iqr, skew, kurtosis
from torch.profiler import profile, record_function, ProfilerActivity
from codecarbon import EmissionsTracker


def set_seeds(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# Lightning-2-PyTorch Checkpoints Loader ----------------------------------------------------------------------
def load_lightning2pt(checkpoint_path, model, device="cpu", verbose=True, validate_updates=True):
    """
    Loads a PyTorch Lightning checkpoint's state_dict into a plain PyTorch model and optionally verifies parameter updates.

    :param checkpoint_path: Absolute Path to the Lightning checkpoint file (.ckpt).
    :param model: The plain PyTorch model instance to load the checkpoint into.
    :param device: Device to load the model onto ('cpu' or 'cuda').
    :param verbose: Whether to print detailed information about the loading process (default: True).
    :param validate_updates: Whether to validate which layers were updated during fine-tuning (default: True).
    :return: The plain PyTorch model with weights loaded from the checkpoint, and a list of updated layers (if validated).
    """
    # Step 1: Load the Lightning checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except FileNotFoundError:
        raise ValueError(f"Checkpoint file not found at: {checkpoint_path}")
    except Exception as e:
        raise ValueError(f"Failed to load checkpoint: {e}")

    # Step 2: Extract the Lightning state_dict
    if "state_dict" not in checkpoint:
        raise ValueError(f"Checkpoint does not contain a 'state_dict'. Keys found: {list(checkpoint.keys())}")

    lightning_state_dict = checkpoint["state_dict"]

    # Step 3: Generalize prefix removal
    stripped_state_dict = {}
    prefix = None

    for key in lightning_state_dict.keys():
        if "." in key:
            prefix = key.split(".")[0] + "."
            break

    if prefix:
        stripped_state_dict = {key.replace(prefix, ""): value for key, value in lightning_state_dict.items()}
        if verbose:
            print(f"Detected prefix '{prefix}'. Stripped from state_dict keys.")
    else:
        stripped_state_dict = lightning_state_dict
        if verbose:
            print("No prefix detected in state_dict keys.")

    # Step 4: Move the model to the specified device
    model.to(device)
    if verbose:
        print(f"Model moved to device: {device}")

    # Step 5: Optionally validate parameter updates
    updated_layers = []
    if validate_updates:
        for name, param in model.state_dict().items():
            if name in stripped_state_dict:
                old_param = param.clone()
                new_param = stripped_state_dict[name]

                # Print data type information
                if verbose:
                    print(f"Validating layer: {name}")
                    print(f"  Old Param: Type: {type(old_param)}, DType: {old_param.dtype}")
                    print(f"  New Param: Type: {type(new_param)}, DType: {new_param.dtype}")

                # Compare old and new parameters
                if not torch.equal(old_param, new_param):
                    updated_layers.append(name)
                    
                    # Compute and display parameter differences
                    diff = (old_param - new_param).float()
                    if verbose:
                        print(f"  Layer: {name} has changes!")
                        print(f"    Min Difference: {diff.abs().min().item():.6f}")
                        print(f"    Max Difference: {diff.abs().max().item():.6f}")
                        print(f"    Mean Difference: {diff.abs().mean().item():.6f}")
                        print(f"    Std-Dev of Differences: {diff.abs().std().item():.6f}")

                        # Optionally, display a small set of differences
                        print(f"    Sample Differences: {diff.flatten()[:5].tolist()}...")
                print('---------------------------------------------------------------------------------')

    # Load the stripped state_dict into the plain model
    try:
        model.load_state_dict(stripped_state_dict)
        if verbose:
            print("State dict successfully loaded into the model!")
    except Exception as e:
        raise ValueError(f"Failed to load state_dict into the model: {e}")

    # Step 6: Print updated layers if validated
    if verbose and validate_updates:
        if updated_layers:
            print("The following layers were updated during fine-tuning:")
            for layer in updated_layers:
                print(f" - {layer}")
        else:
            print("No layers were updated. Fine-tuning may not have modified the model.")

    # Return the model and optionally updated layers
    return model, updated_layers if validate_updates else None


# Hardware Profiling ------------------------------------------------------------------------------------------
def your_gpu(verbose=True, save_path=None):
    """
    Verify NVIDIA GPU(s) availability, return PyTorch device string and detailed GPU info.
    Optionally saves the information into a JSON file if save_path is provided.

    :param verbose: Whether to print detailed logging.
    :param save_path: Optional path to save JSON output.
    :return: PyTorch device string ('cuda:ID' or 'cpu') and GPU details dictionary.
    """

    def bytes_to_gb(bytes_val):
        return bytes_val * 1e-9

    def fetch_gpu_info(gpu_id):
        gpu_info = {"id": gpu_id}
        device = f"cuda:{gpu_id}"
        gpu_info["name"] = torch.cuda.get_device_name(gpu_id)
        try:
            free_mem, total_mem = torch.cuda.mem_get_info(device)
            gpu_info["total_memory_gb"] = bytes_to_gb(total_mem)
            gpu_info["free_memory_gb"] = bytes_to_gb(free_mem)
            if verbose:
                print(f"[GPU-{gpu_id}] Name: {gpu_info['name']}, Free: {gpu_info['free_memory_gb']:.2f} GB, Total: {gpu_info['total_memory_gb']:.2f} GB")
        except Exception as e:
            gpu_info["total_memory_gb"] = None
            gpu_info["free_memory_gb"] = None
            if verbose:
                print(f"[GPU-{gpu_id}] Memory info retrieval failed: {e}")
        return gpu_info

    gpu_details = {"gpu_driver_version": None,
                   "cuda_compiler_version": None,
                   "count": 0,
                   "devices": []}

    try:
        if torch.cuda.is_available():
            device_string = f"cuda:{torch.cuda.current_device()}"
            gpu_details["count"] = torch.cuda.device_count()

            with ThreadPoolExecutor() as executor:
                gpu_details["devices"] = list(executor.map(fetch_gpu_info, range(gpu_details["count"])))

            if shutil.which("nvidia-smi"):
                try:
                    smi_output = subprocess.check_output(["nvidia-smi"], encoding="utf-8")
                    for line in smi_output.splitlines():
                        if "Driver Version" in line:
                            parts = line.split()
                            idx = parts.index("Version:") + 1 if "Version:" in parts else None
                            if idx and idx < len(parts):
                                gpu_details["gpu_driver_version"] = parts[idx]
                            break
                    if verbose:
                        print(f"Driver Version: {gpu_details['gpu_driver_version']}")
                except Exception as e:
                    if verbose:
                        print(f"nvidia-smi output parsing failed: {e}")
            else:
                if verbose:
                    print("'nvidia-smi' not found.")

            if shutil.which("nvcc"):
                try:
                    nvcc_output = subprocess.check_output(["nvcc", "--version"], encoding="utf-8")
                    for line in nvcc_output.splitlines():
                        if "release" in line:
                            parts = line.strip().split()
                            for part in parts:
                                if "release" in part:
                                    gpu_details["cuda_compiler_version"] = part.split("release")[-1].strip(",")
                                    break
                            break
                    if verbose:
                        print(f"CUDA Compiler Version: {gpu_details['cuda_compiler_version']}")
                except Exception as e:
                    if verbose:
                        print(f"nvcc output parsing failed: {e}")
            else:
                if verbose:
                    print("'nvcc' not found.")

            if verbose:
                print(f"PyTorch Version: {torch.__version__}")

        else:
            device_string = "cpu"
            if verbose:
                print("No GPU detected. Using CPU.")
                print(f"PyTorch Version: {torch.__version__}")

    except Exception as e:
        device_string = "cpu"
        if verbose:
            print(f"Error during GPU detection: {e}")

    # Optional: Save GPU info to JSON
    if save_path is not None:
        try:
            with open(save_path, "w") as f:
                json.dump({"device": device_string, "gpu_info": gpu_details}, f, indent=4)
            if verbose:
                print(f"Saved GPU info JSON to: {save_path}")
        except Exception as e:
            if verbose:
                print(f"Failed to save GPU info JSON: {e}")

    return device_string, gpu_details


def your_hardware(verbose=True, save_path=None):
    """
    Inspect and log hardware details (CPU, RAM, Disk) with cross-platform support.
    Optionally saves the results into a JSON file if save_path is provided.
    """
    hardware_info = {}

    # CPU Info
    def get_cpu_info():
        if platform.system() == "Linux":
            try:
                output = subprocess.check_output(["cat", "/proc/cpuinfo"], encoding="utf-8")
                if verbose:
                    print("CPU Info retrieved from /proc/cpuinfo")
                return parse_cpu_info_linux(output)
            except Exception as e:
                if verbose:
                    print(f"Failed to retrieve CPU info: {e}")
        return {}

    def parse_cpu_info_linux(output):
        cpu_model = None
        cpu_count = 0
        cpuinfo_frequencies = {}

        for line in output.splitlines():
            if "model name" in line:
                if cpu_model is None:
                    cpu_model = line.split(":")[1].strip()
                cpu_count += 1

        # Try to read per-core frequencies if available
        freq_files = glob.glob("/sys/devices/system/cpu/cpu[0-9]*/cpufreq/cpuinfo_max_freq")
        if freq_files:
            freqs_mhz = {}
            for path in freq_files:
                try:
                    with open(path, "r") as f:
                        cpu_id = int(path.split("/")[5][3:])  # Extract cpu0, cpu1, etc.
                        freq_khz = int(f.read().strip())
                        freqs_mhz[f"cpu{cpu_id}"] = freq_khz / 1000  # Convert kHz -> MHz
                except Exception:
                    continue
            cpuinfo_frequencies = freqs_mhz

        return {"model_name": cpu_model,
                "physical_cores": cpu_count,
                "frequencies_mhz": cpuinfo_frequencies if cpuinfo_frequencies else "Not Available"}

    hardware_info["cpu"] = get_cpu_info()

    # RAM Info
    def get_ram_info():
        try:
            virtual_mem = psutil.virtual_memory()
            return {"total_memory_gb": round(virtual_mem.total / 1e9, 2),
                    "available_memory_gb": round(virtual_mem.available / 1e9, 2),
                    "used_memory_gb": round(virtual_mem.used / 1e9, 2),
                    "percent_used": virtual_mem.percent}
        except Exception as e:
            if verbose:
                print(f"Failed to retrieve RAM info: {e}")
            return {}

    hardware_info["ram"] = get_ram_info()

    # Disk Info
    def get_disk_info():
        try:
            disks = []
            if platform.system() in ["Linux", "Darwin"]:
                if shutil.which("df"):
                    output = subprocess.check_output(["df", "-h"], encoding="utf-8")
                    lines = output.splitlines()
                    headers = lines[0].split()
                    for line in lines[1:]:
                        if line.strip():
                            parts = line.split()
                            disk_info = dict(zip(headers, parts))
                            disks.append(disk_info)
            elif platform.system() == "Windows":
                for partition in psutil.disk_partitions():
                    usage = psutil.disk_usage(partition.mountpoint)
                    disks.append({"device": partition.device,
                                  "mountpoint": partition.mountpoint,
                                  "fstype": partition.fstype,
                                  "total_gb": round(usage.total / 1e9, 2),
                                  "used_gb": round(usage.used / 1e9, 2),
                                  "free_gb": round(usage.free / 1e9, 2),
                                  "percent_used": usage.percent})
            return disks
        except Exception as e:
            if verbose:
                print(f"Failed to retrieve disk info: {e}")
            return []

    hardware_info["disks"] = get_disk_info()

    # Optional: Print Hardware summary
    if verbose:
        print(f"Hardware Summary: {json.dumps(hardware_info, indent=4)}")

    # Optional: Save Hardware summary to JSON
    if save_path is not None:
        try:
            with open(save_path, "w") as f:
                json.dump(hardware_info, f, indent=4)
            if verbose:
                print(f"Saved hardware info JSON to: {save_path}")
        except Exception as e:
            if verbose:
                print(f"Failed to save hardware info JSON: {e}")

    return hardware_info


# Units Monitoring Functions ----------------------------------------------------------------------------------
cpu_usage_samples = Queue()
cpu_monitoring = threading.Event()
gpu_usage_samples = Queue()
gpu_monitoring = threading.Event()


def monitor_cpu_usage():
    """
    Continuously monitor CPU resources usage and append utilization samples to a thread-safe queue.
    
    :global cpu_usage_samples: A thread-safe queue to store CPU usage percentages.
    :type cpu_usage_samples: Queue
    :global cpu_monitoring: A thread-safe event to control the monitoring loop.
    :type cpu_monitoring: threading.Event
    """
    if not cpu_monitoring.is_set():
        cpu_monitoring.set()

    while cpu_monitoring.is_set():
        cpu_usage_samples.put(psutil.cpu_percent(interval=0.1))


def monitor_gpu_usage():
    """
    Continuously monitor GPU utilization and append samples to a thread-safe queue.
    
    :global gpu_usage_samples: A thread-safe queue to store GPU usage percentages.
    :type gpu_usage_samples: Queue
    :global gpu_monitoring: A thread-safe event to control the monitoring loop.
    :type gpu_monitoring: threading.Event
    """
    if not gpu_monitoring.is_set():
        gpu_monitoring.set()

    while gpu_monitoring.is_set():
        try:
            result = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,
                                    universal_newlines=True)
            if result.returncode == 0:
                utilization = int(result.stdout.strip())
                gpu_usage_samples.put(utilization)
        except Exception as e:
            gpu_usage_samples.put(0)  # Assume 0% usage if query fails

        time.sleep(0.1)


# Model profiling functions -----------------------------------------------------------------------------------
def min_binary_search(model, sample_rate, device, save_path, verbose=True):
    """
    Find the minimum input duration the model can process without error using binary search.
    Save (or append) the result to a specified JSON file.

    :param model: Your PyTorch model.
    :param sample_rate: Audio sample rate (e.g., 32000).
    :param device: Device string ('cpu' or 'cuda').
    :param save_path: Filepath to save JSON results.
    :param verbose: Whether to print progress info.
    """
    def generate_input(duration_samples, device):
        return torch.randn((1, duration_samples), device=device) * 2 - 1.

    max_dur = int(sample_rate * 10)  # 10 seconds max
    low, high = 1, max_dur
    total_iterations = high - low + 1

    model.eval()
    with torch.inference_mode():
        with tqdm(total=total_iterations, desc="MIN Input Size Binary Search") as pbar:
            i = 0
            while low < high and (high - low) > 1:
                mid = (high + low) // 2
                try:
                    set_seeds(42)
                    x = generate_input(mid, device)
                    output = model(x.float())
                    high = mid - 1
                except Exception as e:
                    low = mid
                i += 1
                completed_iterations = total_iterations - (high - low + 1)
                pbar.n = completed_iterations
                pbar.refresh()

    # Final results
    min_samples = high
    min_seconds = min_samples / sample_rate

    results_entry = {"min_input_size": {"samples": int(min_samples),
                                        "seconds": float(min_seconds),
                                        "sample_rate": int(sample_rate),
                                        "binary_search_iterations": int(i)}}
    
    if verbose:
        print(f"Results: {json.dumps(results_entry, indent=4)}")

    # Save results to JSON
    if save_path is not None:
        try:
            if os.path.exists(save_path):
                with open(save_path, "r") as f:
                    existing_data = json.load(f)
                if not isinstance(existing_data, dict):
                    existing_data = {}
            else:
                existing_data = {}

            # Update
            existing_data.update(results_entry)

            with open(save_path, "w") as f:
                json.dump(existing_data, f, indent=4)

            if verbose:
                print(f"Saved (updated) profiling results to: {save_path}")
        except Exception as e:
            if verbose:
                print(f"Failed to save profiling results: {e}")

    return results_entry


def overall_time(model, sample_rate, device, save_path, iterations=100, input_duration_sec=10.0, verbose=True, npz_save_path=None):
    """
    Profile the overall wall clock time for the model inference (sleep included).
    Save (or append) the result to a specified JSON file and optionally save NPZ arrays.

    :param model: Your PyTorch model.
    :param sample_rate: Audio sample rate (e.g., 32000).
    :param device: Device string ('cpu' or 'cuda').
    :param save_path: Filepath to save JSON results.
    :param iterations: Number of iterations to average.
    :param input_duration_sec: Duration of the input (seconds). Default 10s.
    :param verbose: Whether to print progress info.
    :param npz_save_path: Optional path to save .npz compressed timings.
    """
    def generate_input(duration_samples, device):
        return torch.randn((1, duration_samples), device=device) * 2 - 1.

    samples = int(sample_rate * input_duration_sec)

    model.eval()
    model.to(device)

    timings = []

    with torch.inference_mode():
        x = generate_input(samples, device)

        for _ in tqdm(range(iterations), desc="CPU Overall Time Profiling"):
            set_seeds(42)
            start_time = time.perf_counter()
            output = model(x.float())
            torch.cuda.synchronize() if device.startswith("cuda") else None
            elapsed = time.perf_counter() - start_time
            timings.append(elapsed)

    timings = np.array(timings)

    results_entry = {"cpu_overall_time": {"iterations": int(iterations),
                                          "input_duration_sec": float(input_duration_sec),
                                          "max_sec": float(np.max(timings)),
                                          "min_sec": float(np.min(timings)),
                                          "mean_sec": float(np.mean(timings)),
                                          "std_dev_sec": float(np.std(timings, ddof=1)),
                                          "median_sec": float(np.median(timings)),
                                          "percentiles": {"25th_perc": float(np.percentile(timings, 25)),
                                                          "33th_perc": float(np.percentile(timings, 33)),
                                                          "66th_perc": float(np.percentile(timings, 66)),
                                                          "75th_perc": float(np.percentile(timings, 75))},
                                          "iqr_sec": float(iqr(timings)),
                                          "skewness": float(skew(timings)),
                                          "kurtosis": float(kurtosis(timings))}}
    
    if verbose:
        print(f"Results: {json.dumps(results_entry, indent=4)}")

    # Save results to JSON
    if save_path is not None:
        try:
            if os.path.exists(save_path):
                with open(save_path, "r") as f:
                    existing_data = json.load(f)
                if not isinstance(existing_data, dict):
                    existing_data = {}
            else:
                existing_data = {}

            existing_data.update(results_entry)

            with open(save_path, "w") as f:
                json.dump(existing_data, f, indent=4)

            if verbose:
                print(f"Saved (updated) profiling results to: {save_path}")
        except Exception as e:
            if verbose:
                print(f"Failed to save profiling results: {e}")

    # Save results to .NPZ
    if npz_save_path is not None:
        try:
            np.savez_compressed(npz_save_path, values=timings, features=np.array(list(results_entry["cpu_overall_time"].items())))
            if verbose:
                print(f"Saved compressed NPZ data to: {npz_save_path}")
        except Exception as e:
            if verbose:
                print(f"Failed to save NPZ: {e}")

    return results_entry


def process_time(model, sample_rate, device, save_path, iterations=100, input_duration_sec=10.0, verbose=True, npz_save_path=None):
    """
    Profile the CPU process time (excluding sleep) for model inference.
    Save (or append) the result to a specified JSON file and optionally save NPZ arrays.

    :param model: Your PyTorch model.
    :param sample_rate: Audio sample rate (e.g., 32000).
    :param device: Device string ('cpu' or 'cuda').
    :param save_path: Filepath to save JSON results.
    :param iterations: Number of iterations to average.
    :param input_duration_sec: Duration of the input (seconds). Default 10s.
    :param verbose: Whether to print progress info.
    :param npz_save_path: Optional path to save .npz compressed timings.
    """
    def generate_input(duration_samples, device):
        return torch.randn((1, duration_samples), device=device) * 2 - 1.

    samples = int(sample_rate * input_duration_sec)

    model.eval()
    model.to(device)

    timings = []

    with torch.inference_mode():
        x = generate_input(samples, device)

        for _ in tqdm(range(iterations), desc="CPU Process Time Profiling"):
            set_seeds(42)
            start_time = time.process_time()
            output = model(x.float())
            torch.cuda.synchronize() if device.startswith("cuda") else None
            elapsed = time.process_time() - start_time
            timings.append(elapsed)

    timings = np.array(timings)

    results_entry = {"cpu_process_time": {"iterations": int(iterations),
                                          "input_duration_sec": float(input_duration_sec),
                                          "max_sec": float(np.max(timings)),
                                          "min_sec": float(np.min(timings)),
                                          "mean_sec": float(np.mean(timings)),
                                          "std_dev_sec": float(np.std(timings, ddof=1)),
                                          "median_sec": float(np.median(timings)),
                                          "percentiles": {"25th_perc": float(np.percentile(timings, 25)),
                                                          "33th_perc": float(np.percentile(timings, 33)),
                                                          "66th_perc": float(np.percentile(timings, 66)),
                                                          "75th_perc": float(np.percentile(timings, 75))},
                                          "iqr_sec": float(iqr(timings)),
                                          "skewness": float(skew(timings)),
                                          "kurtosis": float(kurtosis(timings))}}

    if verbose:
        print(f"Results: {json.dumps(results_entry, indent=4)}")

    # Save results to JSON
    if save_path is not None:
        try:
            if os.path.exists(save_path):
                with open(save_path, "r") as f:
                    existing_data = json.load(f)
                if not isinstance(existing_data, dict):
                    existing_data = {}
            else:
                existing_data = {}

            existing_data.update(results_entry)

            with open(save_path, "w") as f:
                json.dump(existing_data, f, indent=4)

            if verbose:
                print(f"Saved (updated) profiling results to: {save_path}")
        except Exception as e:
            if verbose:
                print(f"Failed to save profiling results: {e}")

    # Save results to .NPZ
    if npz_save_path is not None:
        try:
            np.savez_compressed(npz_save_path, values=timings, features=np.array(list(results_entry["cpu_process_time"].items())))
            if verbose:
                print(f"Saved compressed NPZ data to: {npz_save_path}")
        except Exception as e:
            if verbose:
                print(f"Failed to save NPZ: {e}")

    return results_entry


def memory_and_cache(model, sample_rate, device, save_path, iterations=100, input_duration_sec=10.0, verbose=True, npz_save_path=None):
    """
    Profile peak RAM memory and CPU cache usage during model inference using tracemalloc and perf.
    Save (or update) the results into a specified JSON file and optionally into compressed NPZ arrays.

    :param model: Your PyTorch model.
    :param sample_rate: Audio sample rate (e.g., 32000).
    :param device: Device string ('cpu' or 'cuda').
    :param save_path: Path to save the JSON results.
    :param iterations: Number of iterations to average.
    :param input_duration_sec: Duration of the input (in seconds). Default is 10s.
    :param verbose: Whether to print progress info.
    :param npz_save_path: Optional path to save .npz compressed results.
    """
    def generate_input(duration_samples, device):
        return torch.randn((1, duration_samples), device=device) * 2 - 1.

    samples = int(sample_rate * input_duration_sec)

    model.eval()
    model.to(device)

    with torch.inference_mode():
        x = generate_input(samples, device)

        # Start tracemalloc for memory profiling
        tracemalloc.start()

        # Start perf stat attached to current Python process
        current_pid = os.getpid()
        perf_cmd = ["perf", "stat",
                    "-e", "cache-references,cache-misses,cycles,instructions,branches,branch-misses",
                    "--pid", str(current_pid)]
        perf_process = subprocess.Popen(perf_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Warm-up
        time.sleep(0.1)

        # Model inference
        for _ in tqdm(range(iterations), desc="Memory and Cache Profiling"):
            _ = model(x.float())

        # Stop tracemalloc
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Stop perf
        perf_process.send_signal(signal.SIGINT)
        perf_output, perf_error = perf_process.communicate()
        perf_text = perf_error  # <- perf writes on stderr

    # Parse perf output
    perf_stats = {}
    if perf_text:
        for line in perf_text.splitlines():
            try:
                parts = line.strip().split()
                if len(parts) >= 2 and parts[0].replace(',', '').isdigit():
                    value = int(parts[0].replace(',', ''))
                    event_name = parts[1]
                    perf_stats[event_name] = value
            except Exception:
                continue

    # Advanced metrics computation
    advanced_metrics = {}
    if 'cache-references' in perf_stats and 'cache-misses' in perf_stats:
        cache_refs = perf_stats['cache-references']
        cache_misses = perf_stats['cache-misses']
        if cache_refs > 0:
            advanced_metrics['cache_miss_percentage'] = (cache_misses / cache_refs) * 100

    if 'instructions' in perf_stats and 'cycles' in perf_stats:
        instructions = perf_stats['instructions']
        cycles = perf_stats['cycles']
        if cycles > 0:
            advanced_metrics['instructions_per_cycle'] = instructions / cycles

    if 'branches' in perf_stats and 'branch-misses' in perf_stats:
        branches = perf_stats['branches']
        branch_misses = perf_stats['branch-misses']
        if branches > 0:
            advanced_metrics['branch_miss_percentage'] = (branch_misses / branches) * 100

    results_entry = {"memory_and_cache_usage": {"iterations": int(iterations),
                                                "input_duration_sec": float(input_duration_sec),
                                                "current_bytes": int(current),
                                                "peak_bytes": int(peak),
                                                "current_megabytes": round(current / (1024 ** 2), 4),
                                                "peak_megabytes": round(peak / (1024 ** 2), 4),
                                                "perf_counters": perf_stats,
                                                "advanced_metrics": advanced_metrics}}

    if verbose:
        print(f"Results: {json.dumps(results_entry, indent=4)}")

    # Save results to JSON
    if save_path is not None:
        try:
            if os.path.exists(save_path):
                with open(save_path, "r") as f:
                    existing_data = json.load(f)
                if not isinstance(existing_data, dict):
                    existing_data = {}
            else:
                existing_data = {}

            existing_data.update(results_entry)

            with open(save_path, "w") as f:
                json.dump(existing_data, f, indent=4)

            if verbose:
                print(f"Saved (updated) profiling results to: {save_path}")
        except Exception as e:
            if verbose:
                print(f"Failed to save profiling results: {e}")

    # Save results to .npz
    if npz_save_path is not None:
        try:
            np.savez_compressed(npz_save_path, peak_memory=np.array([current, peak]), perf_stats=perf_stats, advanced_metrics=advanced_metrics)
            if verbose:
                print(f"Saved compressed NPZ data to: {npz_save_path}")
        except Exception as e:
            if verbose:
                print(f"Failed to save NPZ: {e}")

    return results_entry


def cpu_usage(model, sample_rate, device, save_path, iterations=100, input_duration_sec=10.0, verbose=True, npz_save_path=None):
    """
    Profile CPU usage percentage during model inference.
    Save (or append) the results into a specified JSON file and optionally into compressed NPZ arrays.

    :param model: Your PyTorch model.
    :param sample_rate: Audio sample rate (e.g., 32000).
    :param device: Device string ('cpu' or 'cuda').
    :param save_path: Path to save the JSON results.
    :param iterations: Number of iterations to average.
    :param input_duration_sec: Duration of the input (seconds). Default 10s.
    :param verbose: Whether to print progress info.
    :param npz_save_path: Optional path to save .npz compressed results.
    """
    def generate_input(duration_samples, device):
        return torch.randn((1, duration_samples), device=device) * 2 - 1.

    samples = int(sample_rate * input_duration_sec)

    model.eval()
    model.to(device)

    with torch.inference_mode():
        x = generate_input(samples, device)

        # Start CPU usage monitoring in a separate thread
        try:
            cpu_monitoring.set()
            monitor_thread = threading.Thread(target=monitor_cpu_usage)
            monitor_thread.start()

            # Run model inference
            for _ in tqdm(range(iterations), desc="CPU Usage Profiling"):
                _ = model(x.float())

        except Exception as e:
            if verbose:
                print(f"[ERROR] During CPU usage monitoring: {e}")

        finally:
            cpu_monitoring.clear()
            monitor_thread.join()

    # Process CPU usage samples
    cpu_perc_samples = []
    while not cpu_usage_samples.empty():
        cpu_perc_samples.append(cpu_usage_samples.get())

    avg_cpu_usage = sum(cpu_perc_samples) / len(cpu_perc_samples) if cpu_perc_samples else 0
    peak_cpu_usage = max(cpu_perc_samples) if cpu_perc_samples else 0

    results_entry = {"cpu_usage": {"iterations": int(iterations),
                                   "input_duration_sec": float(input_duration_sec),
                                   "avg_cpu_usage_percent": round(avg_cpu_usage, 2),
                                   "peak_cpu_usage_percent": round(peak_cpu_usage, 2)}}

    if verbose:
        print(f"Results: {json.dumps(results_entry, indent=4)}")

    # Save results to JSON
    if save_path is not None:
        try:
            if os.path.exists(save_path):
                with open(save_path, "r") as f:
                    existing_data = json.load(f)
                if not isinstance(existing_data, dict):
                    existing_data = {}
            else:
                existing_data = {}

            existing_data.update(results_entry)

            with open(save_path, "w") as f:
                json.dump(existing_data, f, indent=4)

            if verbose:
                print(f"Saved (updated) CPU usage results to: {save_path}")
        except Exception as e:
            if verbose:
                print(f"Failed to save CPU usage results: {e}")

    # Save results to NPZ (optional)
    if npz_save_path is not None:
        try:
            np.savez_compressed(npz_save_path,
                                cpu_usage_samples=np.array(cpu_perc_samples),
                                features=np.array(list(results_entry["cpu_usage"].items())))
            if verbose:
                print(f"Saved compressed NPZ CPU usage data to: {npz_save_path}")
        except Exception as e:
            if verbose:
                print(f"Failed to save CPU usage NPZ: {e}")

    return results_entry


def energy_co2(model, sample_rate, device, save_path, iterations=100, input_duration_sec=10.0, verbose=True, npz_save_path=None):
    """
    Profile energy consumption and CO₂ emissions during model inference using CodeCarbon.
    Save (or append) the results into a specified JSON file and optionally into compressed NPZ arrays.
    """
    def generate_input(duration_samples, device):
        return torch.randn((1, duration_samples), device=device) * 2 - 1.

    samples = int(sample_rate * input_duration_sec)
    model.eval()
    model.to(device)

    # Fix: real save directory = folder containing save_path
    save_dir = os.path.dirname(save_path)
    os.makedirs(save_dir, exist_ok=True)  # Ensure it exists

    # Set up logger
    logger = logging.getLogger('energy_emissions_logger')
    while logger.hasHandlers():
        logger.removeHandler(logger.handlers[0])

    formatter = logging.Formatter("%(asctime)s - %(name)-12s: %(levelname)-8s %(message)s")

    log_path = os.path.join(save_dir, "energy_emissions.log")
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.WARNING)
    logger.addHandler(console_handler)

    # Setup CodeCarbon tracker
    energy_tracker = EmissionsTracker(project_name="energy_emissions",
                                      tracking_mode="machine",
                                      save_to_file=True,
                                      save_to_logger=True,
                                      output_dir=save_dir,
                                      output_file="energy_emissions.csv",
                                      logging_logger=logger,
                                      measure_power_secs=0.1)

    with torch.inference_mode():
        x = generate_input(samples, device)

        energy_tracker.start()

        for i in tqdm(range(iterations), desc="Energy/CO₂ Emissions Profiling"):
            energy_tracker.start_task(f"Run-{i+1}")
            _ = model(x.float())
            energy_tracker.stop_task(f"Run-{i+1}")

        energy_tracker.stop()

    # Process results
    emissions_csv_pattern = os.path.join(save_dir, "emissions_base_*.csv")
    csv_files = glob.glob(emissions_csv_pattern)
    if not csv_files:
        raise FileNotFoundError(f"No CSV file found matching the pattern '{emissions_csv_pattern}'.")
    csv_file = csv_files[0]

    emissions_rate_values = []
    cpu_energy_values = []
    ram_energy_values = []

    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            emissions_rate_values.append(float(row['emissions_rate']))
            cpu_energy_values.append(float(row['cpu_energy']))
            ram_energy_values.append(float(row['ram_energy']))

    results_entry = {"energy_consumption": {"iterations": int(iterations),
                                            "input_duration_sec": float(input_duration_sec),
                                            "avg_emission_rate_gCO2eq_per_sec": sum(emissions_rate_values) / len(emissions_rate_values) if emissions_rate_values else 0,
                                            "avg_cpu_energy_kWh": sum(cpu_energy_values) / len(cpu_energy_values) if cpu_energy_values else 0,
                                            "avg_ram_energy_kWh": sum(ram_energy_values) / len(ram_energy_values) if ram_energy_values else 0}}

    if verbose:
        print(f"Results: {json.dumps(results_entry, indent=4)}")

    # Save results to JSON
    if save_path is not None:
        try:
            if os.path.exists(save_path):
                with open(save_path, "r") as f:
                    existing_data = json.load(f)
                if not isinstance(existing_data, dict):
                    existing_data = {}
            else:
                existing_data = {}

            existing_data.update(results_entry)

            with open(save_path, "w") as f:
                json.dump(existing_data, f, indent=4)

            if verbose:
                print(f"Saved (updated) energy profiling results to: {save_path}")
        except Exception as e:
            if verbose:
                print(f"Failed to save energy profiling results: {e}")

    # Save results to NPZ (optional)
    if npz_save_path is not None:
        try:
            np.savez_compressed(npz_save_path,
                                emissions_rate=np.array(emissions_rate_values),
                                cpu_energy=np.array(cpu_energy_values),
                                ram_energy=np.array(ram_energy_values))
            if verbose:
                print(f"Saved compressed NPZ energy data to: {npz_save_path}")
        except Exception as e:
            if verbose:
                print(f"Failed to save energy NPZ: {e}")

    return results_entry


def cuda_time(model, sample_rate, device, save_path, iterations=100, input_duration_sec=10.0, verbose=True, npz_save_path=None):
    """
    Measure CUDA event timing (GPU only) for model inference.

    :param model: Your PyTorch model.
    :param sample_rate: Audio sample rate (e.g., 32000).
    :param device: Device string ('cuda' required).
    :param save_path: Path to save the JSON results.
    :param iterations: Number of iterations to average.
    :param input_duration_sec: Input duration in seconds.
    :param verbose: Print verbose output.
    :param npz_save_path: Optional path to save .npz results.
    """
    assert device.startswith("cuda"), "CUDA Event profiling requires a GPU device."

    def generate_input(duration_samples, device):
        return torch.randn((1, duration_samples), device=device) * 2 - 1.

    samples = int(sample_rate * input_duration_sec)
    model.eval()
    model.to(device)

    timings = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    with torch.inference_mode():
        x = generate_input(samples, device)

        for _ in tqdm(range(iterations), desc="CUDA Event Timing"):
            set_seeds(42)
            start.record()
            _ = model(x.float())
            end.record()
            torch.cuda.synchronize()
            timings.append(start.elapsed_time(end) / 1000.)  # Convert ms -> seconds

    timings = np.array(timings)

    results_entry = {"cuda_time": {"iterations": int(iterations),
                                   "input_duration_sec": float(input_duration_sec),
                                   "max_sec": float(np.max(timings)),
                                   "min_sec": float(np.min(timings)),
                                   "mean_sec": float(np.mean(timings)),
                                   "std_dev_sec": float(np.std(timings, ddof=1)),
                                   "median_sec": float(np.median(timings)),
                                   "percentiles": {f"{p}th_perc": float(np.percentile(timings, p)) for p in [25, 33, 66, 75]},
                                   "iqr_sec": float(iqr(timings)),
                                   "skewness": float(skew(timings)),
                                   "kurtosis": float(kurtosis(timings))}}

    if verbose:
        print(f"Results: {json.dumps(results_entry, indent=4)}")

    # Save JSON
    if save_path:
        try:
            if os.path.exists(save_path):
                with open(save_path, 'r') as f:
                    existing = json.load(f)
                if not isinstance(existing, dict):
                    existing = {}
            else:
                existing = {}
            existing.update(results_entry)
            with open(save_path, 'w') as f:
                json.dump(existing, f, indent=4)
            if verbose:
                print(f"Saved CUDA Timing to: {save_path}")
        except Exception as e:
            if verbose:
                print(f"Failed to save CUDA Timing JSON: {e}")

    # Save NPZ
    if npz_save_path:
        try:
            np.savez_compressed(npz_save_path, timings=timings)
            if verbose:
                print(f"Saved compressed CUDA Timings to: {npz_save_path}")
        except Exception as e:
            if verbose:
                print(f"Failed to save CUDA Times NPZ: {e}")

    return results_entry


def e2e_inference_time(model, sample_rate, device, save_path, iterations=100, input_duration_sec=10.0, verbose=True, npz_save_path=None):
    """
    Measure End-to-End (CPU+GPU) inference timing.

    :param model: Your PyTorch model.
    :param sample_rate: Audio sample rate (e.g., 32000).
    :param device: Device string ('cpu' or 'cuda').
    :param save_path: Path to save the JSON results.
    :param iterations: Number of iterations.
    :param input_duration_sec: Input duration in seconds.
    :param verbose: Print verbose output.
    :param npz_save_path: Optional path to save .npz results.
    """
    def generate_input(duration_samples, device):
        return torch.randn((1, duration_samples), device=device) * 2 - 1.

    samples = int(sample_rate * input_duration_sec)
    model.eval()
    model.to(device)

    timings = []

    with torch.inference_mode():
        x = generate_input(samples, device)

        for _ in tqdm(range(iterations), desc="E2E Inference Timing"):
            set_seeds(42)
            start = time.perf_counter()
            _ = model(x.float())
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            timings.append(time.perf_counter() - start)

    timings = np.array(timings)

    results_entry = {"e2e_inference_time": {"iterations": int(iterations),
                                            "input_duration_sec": float(input_duration_sec),
                                            "max_sec": float(np.max(timings)),
                                            "min_sec": float(np.min(timings)),
                                            "mean_sec": float(np.mean(timings)),
                                            "std_dev_sec": float(np.std(timings, ddof=1)),
                                            "median_sec": float(np.median(timings)),
                                            "percentiles": {f"{p}th_perc": float(np.percentile(timings, p)) for p in [25, 33, 66, 75]},
                                            "iqr_sec": float(iqr(timings)),
                                            "skewness": float(skew(timings)),
                                            "kurtosis": float(kurtosis(timings))}}

    if verbose:
        print(f"Results: {json.dumps(results_entry, indent=4)}")

    if save_path:
        try:
            if os.path.exists(save_path):
                with open(save_path, 'r') as f:
                    existing = json.load(f)
                if not isinstance(existing, dict):
                    existing = {}
            else:
                existing = {}
            existing.update(results_entry)
            with open(save_path, 'w') as f:
                json.dump(existing, f, indent=4)
            if verbose:
                print(f"Saved E2E inference timing to: {save_path}")
        except Exception as e:
            if verbose:
                print(f"Failed to save E2E timing JSON: {e}")

    if npz_save_path:
        try:
            np.savez_compressed(npz_save_path, timings=timings)
            if verbose:
                print(f"Saved compressed E2E timings to: {npz_save_path}")
        except Exception as e:
            if verbose:
                print(f"Failed to save E2E NPZ: {e}")

    return results_entry


def gpu_memory(model, sample_rate, device, save_path, iterations=100, input_duration_sec=10.0, verbose=True):
    """
    Measure peak GPU memory usage during model inference.

    :param model: Your PyTorch model.
    :param sample_rate: Audio sample rate (e.g., 32000).
    :param device: Device string ('cuda' required).
    :param save_path: Path to save the JSON results.
    :param iterations: Number of iterations.
    :param input_duration_sec: Input duration in seconds.
    :param verbose: Print verbose output.
    """
    assert device.startswith("cuda"), "GPU memory profiling requires a CUDA device."

    def generate_input(duration_samples, device):
        return torch.randn((1, duration_samples), device=device) * 2 - 1.

    samples = int(sample_rate * input_duration_sec)
    model.eval()
    model.to(device)

    with torch.inference_mode():
        x = generate_input(samples, device)
        torch.cuda.reset_peak_memory_stats()

        for _ in tqdm(range(iterations), desc="GPU Memory Usage Profiling"):
            _ = model(x.float())
            torch.cuda.synchronize()

    peak_memory_bytes = torch.cuda.max_memory_allocated()

    results_entry = {"gpu_memory_usage": {"iterations": int(iterations),
                                          "input_duration_sec": float(input_duration_sec),
                                          "peak_memory_bytes": int(peak_memory_bytes),
                                          "peak_memory_megabytes": round(peak_memory_bytes / (1024**2), 4)}}

    if verbose:
        print(f"Results: {json.dumps(results_entry, indent=4)}")

    if save_path:
        try:
            if os.path.exists(save_path):
                with open(save_path, 'r') as f:
                    existing = json.load(f)
                if not isinstance(existing, dict):
                    existing = {}
            else:
                existing = {}
            existing.update(results_entry)
            with open(save_path, 'w') as f:
                json.dump(existing, f, indent=4)
            if verbose:
                print(f"Saved GPU memory usage to: {save_path}")
        except Exception as e:
            if verbose:
                print(f"Failed to save GPU memory usage JSON: {e}")

    return results_entry


def gpu_usage(model, sample_rate, device, save_path, iterations=100, input_duration_sec=10.0, verbose=True):
    """
    Measure GPU utilization percentage during model inference.

    :param model: Your PyTorch model.
    :param sample_rate: Audio sample rate (e.g., 32000).
    :param device: Device string ('cuda' required).
    :param save_path: Path to save the JSON results.
    :param iterations: Number of iterations.
    :param input_duration_sec: Input duration in seconds.
    :param verbose: Print verbose output.
    """
    assert device.startswith("cuda"), "GPU utilization monitoring requires a CUDA device."

    def generate_input(duration_samples, device):
        return torch.randn((1, duration_samples), device=device) * 2 - 1.

    samples = int(sample_rate * input_duration_sec)
    model.eval()
    model.to(device)

    gpu_queue = Queue()
    stop_event = threading.Event()

    with torch.inference_mode():
        x = generate_input(samples, device)

        monitor_thread = threading.Thread(target=monitor_gpu_usage, args=(gpu_queue, stop_event))
        monitor_thread.start()

        for _ in tqdm(range(iterations), desc="GPU Utilization Profiling"):
            _ = model(x.float())
            torch.cuda.synchronize()

        stop_event.set()
        monitor_thread.join()

    gpu_samples = []
    while not gpu_queue.empty():
        gpu_samples.append(gpu_queue.get())

    avg_gpu = sum(gpu_samples) / len(gpu_samples) if gpu_samples else 0
    peak_gpu = max(gpu_samples) if gpu_samples else 0

    results_entry = {"gpu_utilization": {"iterations": int(iterations),
                                         "input_duration_sec": float(input_duration_sec),
                                         "avg_utilization_percent": float(avg_gpu),
                                         "peak_utilization_percent": float(peak_gpu)}}

    if verbose:
        print(f"Results: {json.dumps(results_entry, indent=4)}")

    if save_path:
        try:
            if os.path.exists(save_path):
                with open(save_path, 'r') as f:
                    existing = json.load(f)
                if not isinstance(existing, dict):
                    existing = {}
            else:
                existing = {}
            existing.update(results_entry)
            with open(save_path, 'w') as f:
                json.dump(existing, f, indent=4)
            if verbose:
                print(f"Saved GPU utilization to: {save_path}")
        except Exception as e:
            if verbose:
                print(f"Failed to save GPU utilization JSON: {e}")

    return results_entry
