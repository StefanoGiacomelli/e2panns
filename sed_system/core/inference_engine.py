"""Inference Engine for Real-Time SED
========================================
Thread-based inference engine for Sound Event Detection.

Integrates with:
- InputFrameProvider for frame extraction
- Model for inference
- Adaptive frame sizing based on confidence

Author: Stefano Giacomelli - Ph.D. candidate in ICT (DISIM dpt. - University of L'Aquila)
"""

import logging
import time
import threading
from typing import List, Dict, Optional

import torch
import numpy as np


def inference_task(stop_event: threading.Event,
                  frame_provider,
                  model: torch.nn.Module,
                  class_index: int,
                  threshold: float = 0.5,
                  device: str = 'cpu',
                  results_list: Optional[List] = None) -> List[Dict]:
    """
    Run inference task in a separate thread.
    
    This function continuously extracts frames from the frame_provider,
    runs inference with the model, and stores results.
    
    Args:
        stop_event: Threading event to signal stop
        frame_provider: InputFrameProvider instance
        model: PyTorch model for inference
        class_index: Index of class to extract probability for
        threshold: Probability threshold for adaptive frame sizing
        device: Device to run inference on
        results_list: Optional external list to store results
    
    Returns:
        List of inference results (dicts with timestamp, probability, duration, frame_size)
    """
    if results_list is None:
        results_list = []
    
    inference_count = 0
    
    try:
        while not stop_event.is_set():
            # Check if buffer writing is done
            if frame_provider.buffer.writing_done:
                # Check if there's still data to process
                with frame_provider.buffer.lock:
                    available = (frame_provider.buffer.write_pointer - frame_provider.start_pos) % frame_provider.buffer.size
                
                if available < frame_provider.frame_size_min:
                    logging.info("Writing done and no more data to process")
                    break
            
            # Determine adaptive width based on previous result
            adapt_width = False
            if len(results_list) > 0:
                last_prob = results_list[-1]['probability']
                adapt_width = (last_prob >= threshold)
            
            # Get frame
            frame, is_valid, frame_timestamp = frame_provider.get_frame(
                adapt_width=adapt_width,
                timeout=2.0
            )
            
            if not is_valid:
                logging.debug("Invalid frame received, checking if done...")
                if frame_provider.buffer.writing_done:
                    break
                continue
            
            # Skip frames that are too short for the model
            # EPANNs/CED/CLAP need at least ~310ms (frame_duration_min)
            if len(frame) < frame_provider.frame_size_min:
                # Check if this is the final partial frame
                is_final = frame_provider.buffer.writing_done
                with frame_provider.buffer.lock:
                    remaining = (frame_provider.buffer.write_pointer - frame_provider.start_pos) % frame_provider.buffer.size
                
                logging.warning(f"Frame too short ({len(frame)} < {frame_provider.frame_size_min}), skipping "
                              f"(writing_done={is_final}, remaining_in_buffer={remaining} samples)")
                
                if is_final:
                    logging.debug("This appears to be the final partial frame at end of stream")
                    break
                continue
            
            # Run inference
            inference_start = time.perf_counter()
            probability = single_inference(model, frame, class_index, device)
            inference_end = time.perf_counter()
            
            inference_duration = inference_end - inference_start
            inference_count += 1
            
            # Store result
            frame_duration_sec = len(frame) / frame_provider.sampling_rate
            frame_end = frame_timestamp + frame_duration_sec
            binarized = 1 if probability >= threshold else 0
            
            result = {
                'timestamp': frame_timestamp,
                'probability': probability,
                'duration': inference_duration,
                'frame_size': len(frame)
            }
            results_list.append(result)
            
            # Frame-by-frame detailed logging (DEBUG level)
            logging.debug(f"Frame #{inference_count:3d} | "
                         f"t=[{frame_timestamp:6.2f}s - {frame_end:6.2f}s] | "
                         f"prob={probability:.3f} (bin={binarized}) | "
                         f"forward_time={inference_duration*1000:5.1f}ms")
    
    except Exception as e:
        logging.error(f"Inference thread error: {e}", exc_info=True)
    
    return results_list


def single_inference(model: torch.nn.Module,
                     frame: np.ndarray,
                     class_index: int,
                     device: str = 'cpu') -> float:
    """
    Run single inference on an audio frame.
    
    Args:
        model: PyTorch model
        frame: Audio frame (numpy array)
        class_index: Index of class to extract
        device: Device to run on
    
    Returns:
        Probability (float) for the specified class
    """
    # Convert to tensor
    frame_tensor = torch.tensor(frame, dtype=torch.float32).unsqueeze(0)  # [1, samples]
    frame_tensor = frame_tensor.to(device)
    
    # Run inference
    with torch.no_grad():
        output = model(frame_tensor)
        
        # Extract clipwise output
        if isinstance(output, dict):
            clipwise = output['clipwise_output']
        else:
            # Assume direct output
            clipwise = output
        
        # Get probability for class_index
        probability = clipwise.squeeze()[class_index].item()
    
    return probability


def run_inference_simulation(audio_data: np.ndarray,
                             model: torch.nn.Module,
                             class_index: int,
                             sampling_rate: int,
                             buffer_duration: float = 20.0,
                             chunk_duration: float = 0.310,
                             frame_duration_min: float = 0.310,
                             frame_duration_max: float = 1.0,
                             threshold: float = 0.5,
                             adapt_width_coeff: float = 0.4,
                             device: str = 'cpu') -> List[Dict]:
    """
    Run complete inference simulation on audio data.
    
    This is a convenience function that sets up all components and runs
    the simulation end-to-end.
    
    Args:
        audio_data: Audio samples (numpy array)
        model: PyTorch model for inference
        class_index: Class index to extract
        sampling_rate: Audio sampling rate
        buffer_duration: Circular buffer duration (seconds)
        chunk_duration: Write chunk duration (seconds)
        frame_duration_min: Minimum frame duration (seconds)
        frame_duration_max: Maximum frame duration (seconds)
        threshold: Probability threshold for adaptive sizing
        adapt_width_coeff: Frame increment coefficient
        device: Device for inference
    
    Returns:
        List of inference results
    """
    from .buffer import CircularBuffer
    from .frame_provider import InputFrameProvider
    from .audio_processor import write_to_buffer
    
    # Setup components
    buffer_size = int(buffer_duration * sampling_rate)
    buffer = CircularBuffer(buffer_size)
    
    frame_provider = InputFrameProvider(
        buffer=buffer,
        frame_duration_min=frame_duration_min,
        frame_duration_max=frame_duration_max,
        sampling_rate=sampling_rate,
        adapt_width_coeff=adapt_width_coeff
    )
    
    # Start threads
    stop_event = threading.Event()
    results = []
    
    write_thread = threading.Thread(
        target=write_to_buffer,
        args=(buffer, audio_data, sampling_rate, chunk_duration),
        daemon=True
    )
    
    inference_thread = threading.Thread(
        target=inference_task,
        args=(stop_event, frame_provider, model, class_index, threshold, device, results),
        daemon=True
    )
    
    write_thread.start()
    inference_thread.start()
    
    # Wait for completion
    audio_duration = len(audio_data) / sampling_rate
    time.sleep(audio_duration + 0.5)  # Add small buffer
    
    # Signal stop and wait
    stop_event.set()
    inference_thread.join(timeout=5.0)
    write_thread.join(timeout=2.0)
    
    return results
