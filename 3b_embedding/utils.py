import time
import logging
import random
import csv
import soundfile as sf
import numpy as np
import torch
import matplotlib.pyplot as plt

from globals import *
from perf_monitor import PerformanceMonitor
from CircularBuffer import CircularBuffer


# Loaders ---------------------------------------------------------------------
def sound_loader(audio_file_path):
    """
    Load an audio file and return it as a NumPy array.

    :param audio_file_path: The path to the audio file to load.
    :return: The audio data as a NumPy array.
    :raises ValueError: If the audio file's sampling rate does not match the model's one.
    """
    audio_data, in_sr = sf.read(audio_file_path) # Load the audio file

    # Ensure the audio is Monophonic
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)  # Stereo2Mono averaging

    if in_sr != SAMPLING_RATE:
        raise ValueError(f"Audio file SR: {in_sr}Hz, does not match model's one: {SAMPLING_RATE}Hz.")

    # Peak Normalization
    audio_data = audio_data / np.max(np.abs(audio_data))
    
    return audio_data


def load_lightning2pt(checkpoint_path, model, device="cpu", verbose=False, validate_updates=True):
    """
    Loads a PyTorch Lightning checkpoint's state_dict into a plain PyTorch model and optionally verifies parameter updates.

    :param checkpoint_path: Absolute Path to the Lightning checkpoint file (.ckpt).
    :param model: The plain PyTorch model instance to load the checkpoint into.
    :param device: Device to load the model onto ('cpu' or 'cuda').
    :param verbose: Whether to print detailed information about the loading process (default: False).
    :param validate_updates: Whether to validate which layers were updated during fine-tuning (default: True).
    :return tuple: 
        model: The plain PyTorch model with weights loaded from the checkpoint, 
        updated_layers: List of updated layers (if validated).
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


# Experiment Utilities --------------------------------------------------------
def init_env():
    """
    Initialize random generators.
    Initialize HousekeepingLogger object and sets the start timestamp for house-keeping functions.
    Initialize and start Performance monitoring.

    :return tuple:
        - HousekeepingLogger: The house-keeping logger object
        - PerformanceMonitor: The performance monitor object
    """
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    # Safe for Non-GPU users
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
    
    plt.rcParams['mathtext.fontset'] = 'cm'
    plt.rcParams['font.family'] = 'STIXGeneral'
    plt.rc('font', size=12)

    torch.set_num_threads(TORCH_NUM_THREADS)

    # Init House-keeping logger
    hk_logger = HousekeepingLogger(base_filename=audio_filename, 
                                   logging_enabled=HK_LOGGING_ENABLED)
    hk_logger.start() # init reference timestamp

    # Init and start Performance monitoring
    monitor = PerformanceMonitor(hk_logger.t_start, 
                                 interval=MONITORING_INTERVAL, 
                                 log_file=MONITORING_FILE)
    if ENABLE_MONITORING:
        monitor.start()

    return hk_logger, monitor


def write_to_buffer(buffer: CircularBuffer, audio_data: np.ndarray, sampling_rate: int, chunk_duration: float):
    """
    Simulates real-time writing of audio data to the circular buffer.

    
    :param buffer: The circular buffer to write to.
    :param audio_data: The input audio data array to simulate input from.
    :param sampling_rate: The sampling rate of the audio (in Hz).
    :param chunk_duration: The duration of each audio chunk (in seconds).
    """
    audio_index = 0  # Tracks the current position in the audio data
    chunk_size = int(sampling_rate * chunk_duration)  # in samples

    # Init function logger
    if 'write_to_buffer' not in logging.Logger.manager.loggerDict:
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')
        #logging.getLogger().setLevel(logging.CRITICAL)
        write_logger = logging.getLogger('write_to_buffer')

    try:
        while audio_index < len(audio_data):
            # Extract the next chunk of audio samples
            chunk = audio_data[audio_index:audio_index + chunk_size]
            
            # Write the chunk to the circular buffer
            buffer.write(chunk)
            write_logger.info(f"Writing chunk: start_index={audio_index}, chunk_size={len(chunk)}, write_pointer={buffer.write_pointer}")
            write_logger.critical(f"current_time (s): {audio_index / sampling_rate} of {len(audio_data)/sampling_rate}")
            
            # Advance the index to the next input chunk
            audio_index += chunk_size
            
            # Simulate real-time delay
            if SIMULATE_REAL_TIME:
                time.sleep(chunk_duration)
    
    except Exception as e:
        write_logger.error(f"Error in writing thread: {e}")


class HousekeepingLogger:
    def __init__(self, base_filename, logging_enabled=True):
        """
        Initialize the HousekeepingLogger for RT-audio simulation experiments.

        :param base_filename: The base filename for the HK-log-file.
        :param logging_enabled: Whether to enable logging (default: True).
        """
        self.base_filepath = HKL_PATH + base_filename
        self.logging_enabled = logging_enabled
        self.loggers = {}

    def start(self):
        """
        Start absolute reference time.
        """
        self.t_start = time.perf_counter()

    def _get_logger(self, aspect):
        """
        Get the logger for a specific aspect of the experiment.

        :param aspect: The feature of the experiment to log.
        :return tuple: The CSV file and the writer object for the logger.
        """
        if aspect not in self.loggers:
            file = open(f"{self.base_filepath}_{aspect}.csv", "w", newline="")
            writer = csv.writer(file)
            
            if aspect == "frame_size":
                writer.writerow(["Frame Request Time (sec.)", 
                                 "Frame Get Duration (sec.)", 
                                 "Frame Size"])
            elif aspect == "frame_timestamps":
                writer.writerow(["Frame Request Time (sec.)", 
                                 "Frame End Time (sec.)", 
                                 "Frame Get Duration (sec.)"])
            elif aspect == "inference_metrics":
                writer.writerow(["Inference Start Time (sec.)", 
                                 "Inference Duration (sec.)", 
                                 "Inference Probability"])
            
            self.loggers[aspect] = (file, writer)
        
        return self.loggers[aspect]

    def log(self, aspect: str, relative_time: float, data: tuple, sampling_rate: int):
        """
        Log monitoring data related to the experiment.

        :param aspect: The features of the experiment to log.
        :param relative_time: The relative time of the event (sec.).
        :param data: Data to log.
        :param sampling_rate: The sampling rate of the system (model & audio).
        """
        if not self.logging_enabled:
            return
        
        file, writer = self._get_logger(aspect)

        # Compute absolute time related to the simulation start
        absolute_time = relative_time - self.t_start

        if aspect == "frame_size":
            frame_get_duration, frame_size = data
            writer.writerow([absolute_time, frame_get_duration, frame_size / sampling_rate])
        elif aspect == "frame_timestamps":
            end_time, frame_get_duration = data
            writer.writerow([absolute_time, end_time - self.t_start, frame_get_duration])
        elif aspect == "inference_metrics":
            inference_duration, result = data
            writer.writerow([absolute_time, inference_duration, result])

    def close(self):
        """
        Close all open logging files.
        """
        for file, _ in self.loggers.values():
            file.close()
