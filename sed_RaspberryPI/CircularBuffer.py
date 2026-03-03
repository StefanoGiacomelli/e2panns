import threading
import numpy as np
import logging

class CircularBuffer:
    def __init__(self, size):
        """
        Initialize a circular buffer.

        Args:
            size (int): The size of the circular buffer in samples.
        """
        self.buffer = np.zeros(size, dtype=np.float32)
        self.size = size
        self.lock = threading.Lock()  # To ensure thread-safe access
        self.write_pointer = 0
        self.semaphore = threading.Semaphore(0)  # Semaphore to signal new writes
        self.writing_done = False # property to signal that writes are done

    def write(self, data):
        """
        Write data into the circular buffer.
        Locks the buffer on access.
        Signals through a semaphore a written frame

        Args:
            data (np.ndarray): Array of data to write into the buffer.
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

    def read(self, start_pos, frame_size):
        """
        Read a frame from the circular buffer.
        Locks the buffer on access.

        Args:
            start_pos (int): Starting position of the frame in the buffer.
            frame_size (int): Number of samples to read.

        Returns:
            np.ndarray: The extracted frame.
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
            logging.info(f"Buffer state: {self.buffer}")
            logging.info(f"Write pointer: {self.write_pointer}")
