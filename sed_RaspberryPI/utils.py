import time
import soundfile as sf
import numpy as np
import csv
import torch
import logging
from globals import HKL_PATH
from pathlib import Path

def sound_loader(audio_file_path):
  audio_data, sr = sf.read(audio_file_path) # Load the audio file

  # Ensure the audio is mono
  if len(audio_data.shape) > 1:
      audio_data = np.mean(audio_data, axis=1)  # Convert to mono by averaging channels
  
  # Check SR
  target_sr = 32000
  if sr != target_sr:
      raise ValueError(f"Sampling rate of the audio file ({sr} Hz) does not match the target ({target_sr} Hz).")

  # Normalize audio data to range [-1, 1]
  audio_data = audio_data / np.max(np.abs(audio_data))
  return audio_data

def load_lightning2pt(checkpoint_path, model, device="cpu", verbose=False, validate_updates=True):
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

def write_to_buffer(buffer, audio_data, sampling_rate, chunk_duration):
    """
    Simulates real-time writing of audio data to the circular buffer.

    Args:
        buffer (CircularBuffer): The circular buffer to write to.
        audio_data (np.ndarray): The audio data to simulate input from.
        sampling_rate (int): The sampling rate of the audio (samples per second).
        chunk_duration (float): The duration of each chunk in seconds.
    """
    audio_index = 0  # Tracks the current position in the audio data
    chunk_size = int(sampling_rate * chunk_duration)  # Calculate chunk size in samples

    try:
        while audio_index < len(audio_data):
            # Extract the next chunk of audio samples
            chunk = audio_data[audio_index:audio_index + chunk_size]
            
            # Write the chunk to the circular buffer
            buffer.write(chunk)
            logging.info(f"Writing chunk: start_index={audio_index}, chunk_size={len(chunk)}, write_pointer={buffer.write_pointer}")
            logging.critical(f"current time (s): {audio_index / sampling_rate} of {len(audio_data)/sampling_rate}")
            
            # Advance the index in the audio data
            audio_index += chunk_size
            
            # Simulate real-time delay
            time.sleep(chunk_duration)
    except Exception as e:
        logging.error(f"Error in writing thread: {e}")
    finally:
        buffer.writing_done = True
        logging.info("Writing thread finished writing all audio.")

import time
import csv
from pathlib import Path

class HousekeepingLogger:
    def __init__(self, checkpoint_path, audio_path, output_base_dir="outputs", logging_enabled=True):
        self.checkpoint_name = Path(checkpoint_path).name
        self.audio_name = Path(audio_path).stem
        self.output_base_dir = Path(output_base_dir)
        self.logging_enabled = logging_enabled
        self.loggers = {}
        self.t_start = time.perf_counter()

    def start(self):
        self.t_start = time.perf_counter()

    def _get_logger(self, aspect):
        if aspect not in self.loggers:
            # Construct the full output path
            out_dir = self.output_base_dir / self.checkpoint_name / aspect
            out_dir.mkdir(parents=True, exist_ok=True)

            filename = out_dir / f"{self.audio_name}.csv"
            file = open(filename, "w", newline="")
            writer = csv.writer(file)

            # Header
            if aspect == "frame_size":
                writer.writerow(["Frame Request Time (s)", "Frame Get Duration (s)", "Frame Size (s)"])
            elif aspect == "frame_timestamps":
                writer.writerow(["Frame Request Time (s)", "Frame End Time (s)", "Frame Get Duration (s)"])
            elif aspect == "inference_metrics":
                writer.writerow(["Inference Start Time (s)", "Inference Duration (s)", "Inference Result", "Frame Start Time (s)"])
            elif aspect == "perf":
                writer.writerow(["Perf Metric", "Time (s)"])  # Optional, if used

            self.loggers[aspect] = (file, writer)
        return self.loggers[aspect]

    def log(self, aspect, relative_time, data, sampling_rate):
        if not self.logging_enabled:
            return
        file, writer = self._get_logger(aspect)
        absolute_time = relative_time - self.t_start

        if aspect == "frame_size":
            frame_get_duration, frame_size = data
            writer.writerow([absolute_time, frame_get_duration, frame_size / sampling_rate])
        elif aspect == "frame_timestamps":
            end_time, frame_get_duration = data
            writer.writerow([absolute_time, end_time - self.t_start, frame_get_duration])
        elif aspect == "inference_metrics":
            inference_duration, result, frame_start_time = data
            writer.writerow([absolute_time, inference_duration, result, frame_start_time])
        elif aspect == "perf":
            metric_name, timestamp = data
            writer.writerow([metric_name, timestamp])

    def close(self):
        for file, _ in self.loggers.values():
            file.close()
