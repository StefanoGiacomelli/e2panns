import time
import logging
import threading
from epanns_inference import models

from globals import *
from CircularBuffer import CircularBuffer
from InputFrameProvider import InputFrameProvider
from utils import load_lightning2pt, write_to_buffer, sound_loader, init_env
from inference import inference_task


def main():
    # Experiment initialization
    hk_logger, monitor = init_env()
    
    # Model init
    original_model = models.Cnn14_pruned(pre_trained=False)
    model, _ = load_lightning2pt(checkpoint_path=CHECKPOINT_PATH,
                                 model=original_model,
                                 device=DEVICE,
                                 verbose=False,
                                 validate_updates=True)

    # Derived Parameters
    buffer_size = int(buffer_duration * SAMPLING_RATE)
    chunk_size = int(chunk_duration * SAMPLING_RATE)

    # Load audio file
    audio_data = sound_loader(audio_file_path)
    total_duration = len(audio_data) / SAMPLING_RATE
    if len(audio_data) < buffer_size:
        raise ValueError("Audio data is too short for the specified simulation duration.")

    logging.info(f"Buffer size: {buffer_size} samples")
    logging.info(f"Chunk size: {chunk_size} samples")
    logging.info(f"Total samples: {int(total_duration * SAMPLING_RATE)} Total duration: {total_duration} (sec.)")

    # Initialize components
    circular_buffer = CircularBuffer(buffer_size)
    frame_provider = InputFrameProvider(circular_buffer, 
                                        frame_duration_min, 
                                        frame_duration_max, 
                                        SAMPLING_RATE)
    inference_event = threading.Event()
    inference_results = []

    # Start writing & inference threads
    wp_thread = threading.Thread(target=write_to_buffer, 
                                 args=(circular_buffer, 
                                       audio_data, 
                                       SAMPLING_RATE, 
                                       chunk_duration))
    inference_thread = threading.Thread(target=inference_task, 
                                        args=(inference_event, 
                                              frame_provider, 
                                              inference_results, 
                                              model, 
                                              CLASS_INDEX, 
                                              hk_logger, 
                                              output_threshold))
    wp_thread.start()
    inference_thread.start()

    # Run (main thread simply counts until specified duration)
    try:
        time.sleep(total_duration)
    except KeyboardInterrupt:
        logging.warning("Simulation interrupted by user.")
    except Exception as e:
        logging.error(f"Unexpected error during simulation: {e}")
    finally:
        inference_event.set()
        wp_thread.join()
        for _ in range(len(inference_results)): # Release all remaining signals
          circular_buffer.semaphore.release()
        inference_thread.join()
    
    hk_logger.close()

    if ENABLE_MONITORING:
        monitor.stop()


if __name__ == "__main__":
    main()
