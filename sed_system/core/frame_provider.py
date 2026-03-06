"""Input Frame Provider for SED System
=========================================
Adaptive frame extraction from circular buffer with variable frame sizing.

Author: Stefano Giacomelli - Ph.D. candidate in ICT (DISIM dpt. - University of L'Aquila)
"""

import logging
import time
import numpy as np


class InputFrameProvider:
    """
    Provides adaptive-sized frames for inference from a circular buffer.
    
    Frame size adaptation logic:
    - If confidence is low: use minimum frame size (fast inference)
    - If confidence is high: incrementally increase frame size (more context)
    """
    
    def __init__(self, 
                 buffer, 
                 frame_duration_min: float, 
                 frame_duration_max: float, 
                 sampling_rate: int, 
                 adapt_width_coeff: float = 0.4):
        """
        Initialize frame provider.
        
        Args:
            buffer: CircularBuffer instance
            frame_duration_min: Minimum frame duration (seconds)
            frame_duration_max: Maximum frame duration (seconds)
            sampling_rate: Audio sampling rate (Hz)
            adapt_width_coeff: Frame increment coefficient (0=disabled, 1=full SR)
        """
        self.buffer = buffer
        self.sampling_rate = sampling_rate
        self.adapt_width_coeff = adapt_width_coeff
        
        self.frame_size_min = int(frame_duration_min * sampling_rate)
        self.frame_size_max = int(frame_duration_max * sampling_rate)
        self.frame_size = self.frame_size_min
        
        self.start_pos = 0
        self.total_samples_read = 0
        self.last_frame_start_time = 0.0
    
    def get_frame(self, adapt_width: bool = False, timeout: float = 5.0) -> tuple:
        """
        Retrieve a frame from the circular buffer.
        
        Waits until enough data is available to extract a valid frame.
        Never returns invalid frames under normal operation.
        
        Args:
            adapt_width: Whether to increase frame size (True when high confidence)
            timeout: Maximum total time to wait for data (seconds)
        
        Returns:
            Tuple of (frame, is_valid, timestamp):
                - frame: Audio frame (np.ndarray)
                - is_valid: Whether the frame is valid (False only on timeout or shutdown)
                - timestamp: Frame start time in seconds
        """
        start_wait = time.time()
        chunks_acquired = 0
        
        # Acquire semaphores (wait for chunks) until we have enough data
        while True:
            # Check timeout
            if timeout and (time.time() - start_wait) > timeout:
                logging.warning(f"Frame request timeout after {timeout}s")
                return np.zeros(self.frame_size_min, dtype=np.float32), False, self.last_frame_start_time
            
            # Wait for a chunk (with short timeout to check conditions periodically)
            acquired = self.buffer.semaphore.acquire(timeout=0.1)
            
            if acquired:
                chunks_acquired += 1
            
            # Check available data
            with self.buffer.lock:
                write_pointer = self.buffer.write_pointer
                available = (write_pointer - self.start_pos) % self.buffer.size
            
            # If we have enough data, extract frame
            if available >= self.frame_size_min:
                logging.debug(f"Acquired {chunks_acquired} chunks, available data: {available} samples")
                
                # Use available data (bounded by max frame size)
                act_frame_size = min(available, self.frame_size_max)
                
                # Read frame from buffer
                frame = self.buffer.read(self.start_pos, act_frame_size)
                
                # Update read pointer
                self.start_pos = (self.start_pos + act_frame_size) % self.buffer.size
                
                # Adaptive frame size adjustment for NEXT frame
                if adapt_width:
                    proposed_size = self.frame_size + int(self.adapt_width_coeff * self.sampling_rate)
                    self.frame_size = min(proposed_size, self.frame_size_max)
                else:
                    self.frame_size = self.frame_size_min
                
                # Update timing
                self.last_frame_start_time = self.total_samples_read / self.sampling_rate
                self.total_samples_read += act_frame_size
                
                logging.debug(f"Valid frame extracted: size={act_frame_size} ({act_frame_size/self.sampling_rate:.3f}s), "
                            f"timestamp={self.last_frame_start_time:.3f}s")
                
                return frame, True, self.last_frame_start_time
            
            # Not enough data yet - check if writing is done
            if self.buffer.writing_done and available < self.frame_size_min:
                logging.debug(f"Writing done, but only {available} samples available (need {self.frame_size_min})")
                if available == 0:
                    return np.zeros(self.frame_size_min, dtype=np.float32), False, self.last_frame_start_time
                else:
                    # Return whatever is left (partial frame)
                    frame = self.buffer.read(self.start_pos, available)
                    self.total_samples_read += available
                    return frame, True, self.last_frame_start_time
            
            # Continue waiting for more chunks
            logging.debug(f"Acquired {chunks_acquired} chunk(s), have {available} samples, need {self.frame_size_min}, waiting for more...")

