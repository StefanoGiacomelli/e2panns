import time
from tqdm import tqdm
import os
import numpy as np
import torch
from torch.profiler import profile, record_function, ProfilerActivity


# Results Processors -------------------------------------------------------------------------------------------
def parse_logits_and_labels(npz_file_path: str, csv_file_path: str, report_file_path: str):
    """
    Parse the NPZ file (for logits), the CSV file (for ordered file paths), and the download report (for HR-labels)

    :param npz_file_path: Path to the NPZ file containing the logits matrix.
    :param csv_file_path: Path to the CSV file containing rading ordered file paths.
    :param report_file_path: Path to the associated downloading phase "Success_samples_report.txt" file.

    :return tuple: (ordered_filenames, associated_labels)
    - ordered_filenames: ordered YouTube IDs.
    - associated_labels: list of associated labels for each YouTube ID.
    
    Example:
    >>> filenames, labels = parse_logits_and_labels('results_phase_0/logits_Alarms.npz',
                                                    'results_phase_0/Alarms_read_paths.csv', 
                                                    './tecnojest_datasets/AudioSet_True_Negatives_Alarms/Success_samples_report.txt')

    """
    # Ensure absolute paths for the files
    npz_file_path = os.path.abspath(os.path.join(os.getcwd(), npz_file_path))
    csv_file_path = os.path.abspath(os.path.join(os.getcwd(), csv_file_path))
    report_file_path = os.path.abspath(os.path.join(os.getcwd(), report_file_path))

    # Stage 1: Parse the CSV and NPZ to get ordered YouTube IDs
    with open(csv_file_path, 'r') as f:
        file_paths = [line.strip() for line in f.readlines()]

    # Load the NPZ file
    logits_data = np.load(npz_file_path)
    logits_matrix = logits_data['logits']

    # Verify .CSV/.NPZ correspondence
    if len(file_paths) != logits_matrix.shape[0]:
        raise ValueError("Mismatch between number of file paths and rows in the logits matrix.")

    # Process file paths to remove extensions and keep the YouTube IDs
    ordered_filenames = [path.split('/')[-1].split('.')[0].rsplit('_', 1)[0] for path in file_paths]

    # Stage 2: Parse the report file to get associated labels
    yt_id_to_labels = {}
    with open(report_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue  # Skip empty lines
            try:
                yt_id, labels = line.split(": ", 1)
                labels_list = eval(labels)  # Convert string representation of list to an actual list
                yt_id_to_labels[yt_id] = labels_list
            except ValueError:
                raise ValueError(f"Error parsing line: {line}")

    # Create the output list of lists, based on the ordered filenames
    associated_labels = [yt_id_to_labels.get(yt_id, []) for yt_id in ordered_filenames]

    return ordered_filenames, associated_labels


# Lightning Checkpoint Loader ---------------------------------------------------------------------------------
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


# Audio Synth --------------------------------------------------------------------------------------------------
def white_noise(shape, device='cpu'):
    """
    Generate a synthetic random audio tensor (a.k.a. = white noise).

    :param shape: tuple, synthetic input tensor shape 
                  (usually 'batch_size', 'channels', 'duration' in samps).
    :param device: str, device to store the tensor ('cpu' or 'gpu').
    :return signal_tensor: torch.Tensor, white noise audio tensor.
    """
    signal_tensor = torch.rand(shape, dtype=float, device=device) * 2 - 1
    
    return signal_tensor


# Profilers ----------------------------------------------------------------------------------------------------
def profile_thread_time(model, input, n_iter, results_path, verbose=False):
    """
    Profile CPU thread time (kernel + user_space) of a model forward (inference).

    :param model: torch.nn.Module
    :param input: torch.Tensor, input signal.
    :param n_iter: int, number of experiment iterations.
    :param results_path: str, path to save results.
    :param verbose: bool, debug prints mode.
    
    :return measures: np.ndarray, latency measures (in ms).
    :return throughput: float, throughput (in samp(s)/sec).
    """
    # Set model on CPU and TEST mode
    model.cpu()
    model.eval()

    # Output measures array
    measures = np.zeros(n_iter)

    # Deactivate gradients computation
    with torch.no_grad():
        # Initialize experiment progress bar
        progress_bar = tqdm(desc=f'CPU_thread_profiler', 
							total=100., 
							bar_format='{desc}: {percentage:.2f}%|{bar}| {n:.2f}/{total_fmt} [{elapsed}<{remaining}')
        
        # For throughput calculation (in sec.)
        tot_time = 0.0
        
        for i in range(n_iter):
            start = time.process_time()						        # START system (CPU) process timer
            _ = model(input.float())								# forward pass
            stop = time.process_time()						        # STOP system (CPU) process timer
            thread_time = (stop - start) * 1000.	                # sec. to ms
            
            # (partial) results storage
            measures[i] = thread_time
            tot_time += thread_time / 1000.							# ms to sec.

            progress_bar.update(100. / n_iter)
        progress_bar.close()

        # Torch Memory/Activities profiler (ONE-SHOT)
        with profile(activities=[ProfilerActivity.CPU], 
                     profile_memory=True, 
                     record_shapes=True) as prof:
            with record_function("model_inference"):
                _ = model(input.float())
            
        if verbose:
            print(prof.key_averages().table())   
        prof.export_chrome_trace(f"{results_path}/one-shot_trace_{input.shape}.json")
        
        # Save results
        throughput = n_iter / tot_time                              # samp(s)/sec.      
        np.savez_compressed(results_path + '/thread_times', proc_latency=measures)

        return measures, throughput


def profile_wall_time(model, input, n_iter, results_path, verbose=False):
    """
    Profile CPU benchmark time (also "stop-watch time") of a model forward (inference).

    :param model: torch.nn.Module
    :param input: torch.Tensor, input signal.
    :param n_iter: int, number of profiling iterations.
    :param results_path: str, path to store the results.
    :param verbose: bool, debug prints mode.
    
    :return measures: np.ndarray, latency measures (in ms).
    :return throughput: float, throughput (in samp(s)/sec).
    """
    # Set model on CPU and TEST mode
    model.cpu()
    model.eval()

    # Output measures array
    measures = np.zeros(n_iter)

    # Deactivate gradients computation
    with torch.no_grad():
        # Initialize experiment progress bar
        progress_bar = tqdm(desc=f'Benchmark_profiler', 
							total=100., 
							bar_format='{desc}: {percentage:.2f}%|{bar}| {n:.2f}/{total_fmt} [{elapsed}<{remaining}')
        
        # For throughput calculation (in sec.)
        tot_time = 0.0
        
        for i in range(n_iter):
            start = time.perf_counter()						        # START system (CPU) wall time (benchmark)
            _ = model(input.float())								# forward pass
            stop = time.perf_counter()						        # STOP system (CPU) wall time (benchmark)
            wall_time = (stop - start) * 1000.	                    # sec. to ms
            
            # (partial) results storage
            measures[i] = wall_time
            tot_time += wall_time / 1000.							# ms to sec.

            progress_bar.update(100. / n_iter)
        progress_bar.close()
        
        # Save results
        throughput = n_iter / tot_time                              # samp(s)/sec.      
        np.savez_compressed(results_path + '/wall_times', proc_latency=measures)

        return measures, throughput


def profile_input_dim(model, max_dur, verbose=True):
    """
    Profile the minimum input dimensionality successfully processed by a model

    :param model: torch.nn.Module
    :param max_dur: int, maximum duration to test (in samples)
    :param verbose: bool, debug prints mode
    
    :return duration: int, in samples
    """
    # Set model on CPU and TEST mode
    model.cpu()
    model.eval()
    
    batch_size = 1
    found = False

    # Main routine
    progress_bar = tqdm(desc=f'Input_DIM_profiler', 
                        total=max_dur, 
                        bar_format='{desc}: {percentage:.2f}%|{bar}| {n:.2f}/{total_fmt} [{elapsed}<{remaining}')
    
    for duration in range(1, max_dur + 1):
        try:
            # Synthesize input
            test_input = white_noise(shape=(batch_size, duration), device='cpu')

            # Deactivate gradients computation
            with torch.no_grad():
                output = model(test_input.float())					# forward pass

            # Check if the output is valid (not empty or throwing errors)
            if output is not None:
                found = True
                if verbose:
                    print(f"The minimum input size successfully processed is {duration} samp(s).")
                break
        except Exception as e:
            progress_bar.update(1)
            continue

    if not found:
        raise ValueError("No valid input length found within [1, max_dur] samps.")

    progress_bar.close()
    
    return duration
