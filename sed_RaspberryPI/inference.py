import torch
import time
import logging
from globals import SAMPLING_RATE, chunk_duration


def inference_task(event, frame_provider, results, model, class_index, hk_logger, output_threshold=0.5):
    """
    Simulates a real-time inference task using frames provided by InputFrameProvider.

    Args:
        event (threading.Event): Signal to stop the inference task.
        frame_provider (InputFrameProvider): Provides frames for inference.
        results (list): Shared list to store inference results.
        model: PyTorch model for inference.
        class_index (int): Index of the class probability to extract.
        output_threshold (float): Threshold to enable adaptive frame sizing.
    """
    try:
        while not (event.is_set() and frame_provider.buffer.writing_done):
            # Determine adapt_width based on the latest inference result
            adapt_width = bool(results and (results[-1] >= output_threshold))

            # Fetch a frame with the determined adapt_width
            frame, is_valid, frame_start_time = frame_provider.get_frame(hk_logger, adapt_width=adapt_width)

            if not is_valid:
                # Handle invalid frame case
                logging.warning("Invalid frame received. Skipping inference.")
                time.sleep(chunk_duration)  # Wait a chunk before retrying
                continue

            # Convert frame to PyTorch tensor
            segment_tensor = torch.tensor(frame, dtype=torch.float32).unsqueeze(0)  # Shape: [1, frame_length]

            start_time = time.perf_counter()
            # Perform inference
            with torch.no_grad():
                output = model(segment_tensor)
                class_probability = output['clipwise_output'].squeeze()[class_index].item()

            end_time = time.perf_counter()
            inference_duration = end_time - start_time
            hk_logger.log("inference_metrics", start_time, [inference_duration, class_probability, frame_start_time], SAMPLING_RATE)

            # Store result
            results.append(class_probability)

            # Debugging output
            logging.info(f"Inference result: {class_probability:.4f}, adapt_width: {adapt_width}, frame size: {len(frame)}")

        if (event.is_set() and frame_provider.buffer.writing_done):
            logging.info("Inference thread stopping: all data processed.")

    except Exception as e:
        logging.error(f"Inference task error: {e}")
