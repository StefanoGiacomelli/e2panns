import threading
import torch
import time
import logging

from globals import SAMPLING_RATE, chunk_duration
from InputFrameProvider import InputFrameProvider


# Set up the module logger
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
#logging.getLogger().setLevel(logging.CRITICAL)
inf_logger = logging.getLogger(__name__)


def inference_task(event: threading.Event, 
                   frame_provider: InputFrameProvider, 
                   results: list, 
                   model: torch.nn.Module, 
                   class_index: int, 
                   hk_logger, 
                   output_threshold: float = 0.5):
    """
    Simulates a real-time inference task using frames provided by InputFrameProvider.

    :param event: Signal to stop the inference task.
    :param frame_provider: Provides frames for inference.
    :param results: Shared list to store inference results.
    :param model: PyTorch model for inference.
    :param class_index: Index of the output class probability to evaluate.
    :param output_threshold: Threshold to enable adaptive frame sizing.
    """
    try:
        while not event.is_set():
            # Determine adapt_width based on the latest inference result
            adapt_width = results and results[-1] >= output_threshold

            # Fetch a frame with the determined adapt_width
            frame, is_valid = frame_provider.get_frame(hk_logger, adapt_width=adapt_width)

            if not is_valid:
                inf_logger.warning("Invalid frame received. Skipping inference.")
                time.sleep(chunk_duration)  # Wait a chunk before retrying
                continue

            # Convert frame to PyTorch tensor
            segment_tensor = torch.tensor(frame, dtype=torch.float32).unsqueeze(0)  # Shape: [batch_size=1, frame_length]

            # Compute inference
            start_time = time.perf_counter()
            with torch.no_grad():
                model.eval()
                output = model(segment_tensor)
                class_probability = output['clipwise_output'].squeeze()[class_index].item()
            end_time = time.perf_counter()
            inference_duration = end_time - start_time
            
            hk_logger.log("inference_metrics", start_time, [inference_duration, class_probability], SAMPLING_RATE)

            # Store result
            results.append(class_probability)

            # Debug output
            inf_logger.info(f"output_prob: {class_probability:.4f}, adapt_width: {adapt_width}, frame_size: {len(frame)}")

    except Exception as e:
        inf_logger.error(f"Inference task error: {e}")
