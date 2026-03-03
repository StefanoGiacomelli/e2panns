import logging
import time
import numpy as np
from globals import SAMPLING_RATE, adapt_width_coeff

class InputFrameProvider:
    def __init__(self, buffer, frame_duration_min, frame_duration_max, sampling_rate):
        """
        Provides frames for inference from a circular buffer.

        Args:
            buffer (CircularBuffer): The circular buffer instance.
            frame_duration_min (float): Minimum frame duration (seconds).
            frame_duration_max (float): Maximum frame duration (seconds).
            SAMPLING_RATE (int): Sampling rate of the audio (samples per second).
        """
        self.buffer = buffer
        self.sampling_rate = sampling_rate
        self.frame_size_min = int(frame_duration_min * sampling_rate)
        self.frame_size_max = int(frame_duration_max * sampling_rate)
        self.start_pos = 0
        self.total_samples_read = 0
        self.last_frame_start_time = 0
        self.frame_size = self.frame_size_min  # the current frame size. Initialized to minimum frame size

    def get_frame(self, hk_logger, adapt_width):
        """
        Retrieves a frame from the circular buffer.

        Args:
            adapt_width (bool): Whether to adaptively increase frame size.

        Returns:
            tuple: A tuple containing:
                - frame (np.ndarray): The retrieved frame or an empty frame if invalid.
                - is_valid (bool): Whether the frame is valid.
        """
        self.buffer.semaphore.acquire()  # Wait for a signal from the writing thread

        write_pointer = self.buffer.write_pointer

        # Compute the actual frame size based on WP and SP, considering wrapping
        act_frame_size = (write_pointer - self.start_pos) % self.buffer.size
        logging.info(f"Computed frame size: {act_frame_size}, current frame size: {self.frame_size}, min size: {self.frame_size_min}")

        # Validate the frame size
        if (self.frame_size > act_frame_size) or (act_frame_size < self.frame_size_min):
            # Invalid frame size, return empty frame and False
            logging.warning(f"Invalid frame: WP={write_pointer}, SP={self.start_pos}, act_frame_size={act_frame_size}, frame_size={self.frame_size}")
            return np.zeros(self.frame_size, dtype=np.float32), False, self.last_frame_start_time

        frame_start_time = time.perf_counter() # track the start time 
        # Read the frame from the circular buffer
        frame = self.buffer.read(self.start_pos, act_frame_size)
        frame_end_time = time.perf_counter() # track the end time
        frame_duration = frame_end_time - frame_start_time # compute the duration
        hk_logger.log("frame_timestamps", frame_start_time, [frame_end_time, frame_duration], SAMPLING_RATE) # log the duration
        hk_logger.log("frame_size", frame_start_time, [frame_duration, act_frame_size], SAMPLING_RATE) # log the size

        # Update start_pos i.e. the read pointer
        self.start_pos = (self.start_pos + act_frame_size) % self.buffer.size

        # Adaptive frame size adjustment
        if adapt_width:
            proposed_size = self.frame_size + int(adapt_width_coeff * self.sampling_rate)
            self.frame_size = min(proposed_size, self.frame_size_max)
        else:
            self.frame_size = self.frame_size_min  # bring back to the minimum if probability goes down

        # Debug logging
        logging.info(f"Valid frame: WP={write_pointer}, SP={self.start_pos}, act_frame_size={act_frame_size}, frame_size={self.frame_size}")
        
        self.last_frame_start_time = self.total_samples_read / self.sampling_rate # convert to time
        self.total_samples_read += act_frame_size # increment the count of samples read so far

        return frame, True, self.last_frame_start_time
