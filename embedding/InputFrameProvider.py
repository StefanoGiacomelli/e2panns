import logging
import time
import numpy as np
from CircularBuffer import CircularBuffer
from globals import SAMPLING_RATE


# Set up the module logger
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
#logging.getLogger().setLevel(logging.CRITICAL)
ifp_logger = logging.getLogger(__name__)


class InputFrameProvider:
    def __init__(self, 
                 buffer: CircularBuffer, 
                 frame_duration_min: float, 
                 frame_duration_max: float, 
                 sampling_rate: int):
        """
        Class to provide frames for NNs inference, from a given CircularBuffer object.

        :param buffer: The CircularBuffer instance.
        :param frame_duration_min: Minimum frame duration (in seconds).
        :param frame_duration_max: Maximum frame duration (in seconds).
        :param sampling_rate: Sampling rate of the audio (in Hz).
        """
        self.buffer = buffer
        self.sampling_rate = sampling_rate
        self.frame_size_min = int(frame_duration_min * sampling_rate)
        self.frame_size_max = int(frame_duration_max * sampling_rate)
        self.start_pos = 0
        self.frame_size = self.frame_size_min  # Current frame size: init to the minimum size

    def get_frame(self, hk_logger, adapt_width: bool) -> tuple:
        """
        Retrieves a frame from the circular buffer.

        :param hk_logger: The logger instance for house-keeping data.
        :param adapt_width: Whether to adaptively increase frame size.

        :return tuple:
                - frame (np.ndarray): The retrieved frame or a null (0s) frame if invalid.
                - is_valid (bool): Whether the frame is valid.
        """
        self.buffer.semaphore.acquire()  # Wait for a signal from the writing thread
        write_pointer = self.buffer.write_pointer

        # Compute the actual frame size based on WP and SP, considering wrapping
        act_frame_size = (write_pointer - self.start_pos) % self.buffer.size
        ifp_logger.info(f"computed_frame_size: {act_frame_size}, 
                          current_frame_size: {self.frame_size}, 
                          min_size: {self.frame_size_min}")

        # Validate the frame size
        if (self.frame_size > act_frame_size) or (act_frame_size < self.frame_size_min):
            ifp_logger.warning(f"INVALID_frame: write_pointer={write_pointer}, 
                                                start_pos={self.start_pos}, 
                                                act_frame_size={act_frame_size}, 
                                                frame_size={self.frame_size}")
            return np.zeros(self.frame_size, dtype=np.float32), False

        # Read the frame from the circular buffer
        frame_start_time = time.perf_counter()
        frame = self.buffer.read(self.start_pos, act_frame_size)
        frame_end_time = time.perf_counter()
        
        frame_duration = frame_end_time - frame_start_time
        hk_logger.log("frame_timestamps", frame_start_time, [frame_end_time, frame_duration], SAMPLING_RATE)
        hk_logger.log("frame_size", frame_start_time, [frame_duration, act_frame_size], SAMPLING_RATE)

        # Update start_pos
        self.start_pos = (self.start_pos + act_frame_size) % self.buffer.size

        # Adaptive frame size adjustment
        if adapt_width:
            self.frame_size = min(self.frame_size + self.sampling_rate // 2, self.frame_size_max)
        else:
            self.frame_size = act_frame_size  # Sync frame size with actual size

        # Debug logging
        ifp_logger.info(f"VALID_frame: write_pointer={write_pointer}, 
                                       start_pos={self.start_pos}, 
                                       act_frame_size={act_frame_size}, 
                                       frame_size={self.frame_size}")
        
        return frame, True
