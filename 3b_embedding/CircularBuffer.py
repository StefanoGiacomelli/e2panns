import threading
import logging
import numpy as np


# Set up the module logger
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
#logging.getLogger().setLevel(logging.CRITICAL)
cb_logger = logging.getLogger(__name__)


class CircularBuffer:
    def __init__(self, size: int):
        """
        Initialize a circular buffer.

        :param size: The size of the circular buffer (in samples).
        """
        self.buffer = np.zeros(size, dtype=np.float32)
        self.size = size
        self.lock = threading.Lock()  # To ensure thread-safe access
        self.write_pointer = 0
        self.semaphore = threading.Semaphore(0)  # Semaphore to signal new writes


    def write(self, data: np.ndarray):
        """
        Write data into the circular buffer.
        Locks the buffer on access.
        Signals through a semaphore a written frame

        :param data: NumPy array of data to write into the buffer.
        """
        with self.lock:
            data_len = len(data)
            if data_len > self.size:
                raise ValueError("Data length exceeds buffer size!")

            end_pos = (self.write_pointer + data_len) % self.size
            if self.write_pointer + data_len <= self.size:
                self.buffer[self.write_pointer:self.write_pointer + data_len] = data
            else:
                split = self.size - self.write_pointer
                self.buffer[self.write_pointer:] = data[:split]
                self.buffer[:end_pos] = data[split:]
            self.write_pointer = end_pos

        self.semaphore.release()  # Signal that new data is available

    
    def read(self, start_pos: int, frame_size: int) -> np.ndarray:
        """
        Read a signal frame from the circular buffer.
        Locks the buffer on access.

        :param start_pos: Starting position of the frame in the buffer.
        :param frame_size: Number of samples to read.
        :return np.ndarray: The extracted frame.
        """
        with self.lock:
            end_pos = (start_pos + frame_size) % self.size
            if start_pos + frame_size <= self.size:
                return self.buffer[start_pos:start_pos + frame_size]
            else:
                split = self.size - start_pos
                return np.concatenate((self.buffer[start_pos:], self.buffer[:end_pos]))
    

    def debug_state(self):
        """
        Print the current state of the buffer for debugging.
        """
        with self.lock:
            cb_logger.info(f"buffer_state: {self.buffer}")
            cb_logger.info(f"write_pointer: {self.write_pointer}")
            cb_logger.info(f"semaphore: {self.semaphore._value}")
