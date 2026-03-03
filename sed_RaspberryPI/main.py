from config import init_env
from epanns_inference import models
import logging
import threading
import time
from utils import load_lightning2pt, write_to_buffer, sound_loader
from CircularBuffer import CircularBuffer
from InputFrameProvider import InputFrameProvider
from inference import inference_task
from globals import CHECKPOINT_PATH, CHECKPOINT, DEVICE, SAMPLING_RATE, buffer_duration, chunk_duration, AUDIO_FILE_PATH, AUDIO_FILE, frame_duration_min, frame_duration_max, CLASS_INDEX, output_threshold, ENABLE_MONITORING
from perf_monitor import PerformanceMonitor
import argparse
from pathlib import Path 

def main(audio_file=None, checkpoint=None):
    from globals import AUDIO_FILE_PATH, AUDIO_FILE, CHECKPOINT_PATH, CHECKPOINT

    if audio_file is None:
        audio_file = str(Path(AUDIO_FILE_PATH) / AUDIO_FILE)
    if checkpoint is None:
        checkpoint = str(Path(CHECKPOINT_PATH) / CHECKPOINT)

    hk_logger, monitor = init_env(audio_file,checkpoint)

    original_model = models.Cnn14_pruned(pre_trained=False)
    print("\n" * 5, end="")

    model, updated_layers = load_lightning2pt(checkpoint_path=checkpoint,
                                          model=original_model,
                                          device=DEVICE,
                                          verbose=False,
                                          validate_updates=False)
    model.eval()

    # Derived Parameters
    buffer_size = int(buffer_duration * SAMPLING_RATE) # dimension of the overall buffer
    chunk_size = int(chunk_duration * SAMPLING_RATE) # dimension of the buffer used to write to the buffer

    # Load audio file
    audio_data = sound_loader(audio_file)
    total_duration = len(audio_data) / SAMPLING_RATE

    # Ensure audio data is long enough
    #if len(audio_data) < buffer_size:
    #    raise ValueError("Audio data is too short for the specified simulation duration.")

    logging.info(f"Buffer size: {buffer_size} samples")
    logging.info(f"Chunk size: {chunk_size} samples")
    logging.info(f"Total audio samples: {int(total_duration * SAMPLING_RATE)} Total duration: {total_duration} (s)")

    # Initialize components
    circular_buffer = CircularBuffer(buffer_size)
    frame_provider = InputFrameProvider(circular_buffer, frame_duration_min, frame_duration_max, SAMPLING_RATE)
    inference_event = threading.Event() # used to signal production of an inference
    inference_results = []


    # Start threads
    wp_thread = threading.Thread(target=write_to_buffer, args=(circular_buffer, audio_data, SAMPLING_RATE, chunk_duration))
    inference_thread = threading.Thread(target=inference_task, args=(inference_event, frame_provider, inference_results, model, CLASS_INDEX, hk_logger, output_threshold))

    
    logging.info("Starting threads...")

    wp_thread.start()
    inference_thread.start()

    # Run simulation for the total duration
    try:
        logging.info(f"Simulation running for {total_duration} seconds...")
        time.sleep(total_duration)
    except KeyboardInterrupt:
        logging.warning("Simulation interrupted by user.")
    except Exception as e:
        logging.error(f"Unexpected error during simulation: {e}")
    finally:
        # Stop threads
        inference_event.set()
        wp_thread.join()
        for _ in range(len(inference_results)): # Release all the possible remaining signals
          circular_buffer.semaphore.release()
        inference_thread.join()

    # Close the logger
    hk_logger.close()

    if ENABLE_MONITORING:
        monitor.stop()

    logging.warning("Simulation completed.")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Run main with an optional audio file path.")
    parser.add_argument("--audio", type=str, help="Path to the audio file", default=None)
    parser.add_argument("--checkpoint", type=str, help="Path to the current checkpoint", default=None)
    
    args = parser.parse_args()

    main(audio_file=args.audio, checkpoint=args.checkpoint)
