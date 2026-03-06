"""Circular Buffer for Audio Streaming
========================================
Thread-safe circular buffer implementation for real-time audio simulation.

Author: Stefano Giacomelli - Ph.D. candidate in ICT (DISIM dpt. - University of L'Aquila)
"""

import threading
import numpy as np
import logging


class CircularBuffer:
    """
    Thread-safe circular buffer for audio data streaming.
    
    Features:
    - Lock-based thread safety for concurrent read/write
    - Semaphore signaling for producer-consumer synchronization
    - Writing completion flag for graceful shutdown
    """
    
    def __init__(self, size: int):
        """
        Initialize circular buffer.
        
        Args:
            size: Buffer size in samples
        """
        self.buffer = np.zeros(size, dtype=np.float32)
        self.size = size
        self.lock = threading.Lock()
        self.write_pointer = 0
        self.semaphore = threading.Semaphore(0)
        self.writing_done = False
    
    def write(self, data: np.ndarray):
        """
        Write data into the circular buffer (thread-safe).
        Signals availability via semaphore release.
        
        Args:
            data: Audio samples to write (np.ndarray)
        
        Raises:
            ValueError: If data length exceeds buffer size
        """
        with self.lock:
            data_len = len(data)
            if data_len > self.size:
                raise ValueError(f"Data length ({data_len}) exceeds buffer size ({self.size})")
            
            end_pos = (self.write_pointer + data_len) % self.size
            
            # Handle wrapping
            if self.write_pointer + data_len <= self.size:
                # No wrapping needed
                self.buffer[self.write_pointer:self.write_pointer + data_len] = data
            else:
                # Wrapping required
                split = self.size - self.write_pointer
                self.buffer[self.write_pointer:] = data[:split]
                self.buffer[:end_pos] = data[split:]
            
            self.write_pointer = end_pos
        
        # Signal that new data is available
        self.semaphore.release()
    
    def read(self, start_pos: int, frame_size: int) -> np.ndarray:
        """
        Read a frame from the circular buffer (thread-safe).
        
        Args:
            start_pos: Starting position in the buffer
            frame_size: Number of samples to read
        
        Returns:
            Extracted frame as np.ndarray
        """
        with self.lock:
            end_pos = (start_pos + frame_size) % self.size
            
            # Handle wrapping
            if start_pos + frame_size <= self.size:
                # No wrapping
                return self.buffer[start_pos:start_pos + frame_size].copy()
            else:
                # Wrapping required
                split = self.size - start_pos
                return np.concatenate((self.buffer[start_pos:], self.buffer[:end_pos]))
    
    def debug_state(self):
        """Print current buffer state for debugging."""
        with self.lock:
            logging.debug(f"Buffer state: write_pointer={self.write_pointer}, writing_done={self.writing_done}")
