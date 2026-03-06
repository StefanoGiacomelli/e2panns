"""Audio Processing Module
============================
Audio loading, resampling, normalization and real-time buffer streaming.

Author: Stefano Giacomelli - Ph.D. candidate in ICT (DISIM dpt. - University of L'Aquila)
"""

import time
import logging
import numpy as np
import soundfile as sf

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    logging.warning("librosa not available. Resampling will not work if sample rates don't match.")


def load_audio(audio_path: str, target_sr: int) -> np.ndarray:
    """
    Load audio file and resample to target sample rate.
    
    Args:
        audio_path: Path to audio file
        target_sr: Target sampling rate (Hz)
    
    Returns:
        Audio waveform as np.ndarray (mono, normalized to [-1, 1])
    
    Raises:
        FileNotFoundError: If audio file doesn't exist
        ValueError: If resampling needed but librosa not available
    """
    # Load audio
    try:
        audio_data, sr = sf.read(audio_path)
    except Exception as e:
        raise FileNotFoundError(f"Cannot load audio file '{audio_path}': {e}")
    
    # Convert to mono
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)
    
    # Resample if needed
    if sr != target_sr:
        if not LIBROSA_AVAILABLE:
            raise ValueError(f"Sample rate mismatch ({sr} Hz vs {target_sr} Hz) but librosa not available. "
                            "Install with: pip install librosa")
        logging.info(f"Resampling audio from {sr} Hz to {target_sr} Hz")
        audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=target_sr)
    
    # Normalize to [-1, 1]
    max_val = np.max(np.abs(audio_data))
    if max_val > 0:
        audio_data = audio_data / max_val
    
    return audio_data


def write_to_buffer(buffer, audio_data: np.ndarray, sampling_rate: int, chunk_duration: float):
    """
    Simulate real-time audio streaming by writing chunks to circular buffer.
    
    This function is designed to run in a separate thread and simulates
    real-time audio input by sleeping between writes.
    
    Args:
        buffer: CircularBuffer instance
        audio_data: Audio samples to stream
        sampling_rate: Audio sampling rate (Hz)
        chunk_duration: Duration of each chunk to write (seconds)
    """
    audio_index = 0
    chunk_size = int(sampling_rate * chunk_duration)
    total_chunks = int(np.ceil(len(audio_data) / chunk_size))
    
    try:
        chunk_counter = 0
        while audio_index < len(audio_data):
            # Extract chunk
            chunk = audio_data[audio_index:audio_index + chunk_size]
            
           # Write to buffer
            buffer.write(chunk)
            
            chunk_counter += 1
            
            # Advance index
            audio_index += chunk_size
            
            # Simulate real-time delay
            time.sleep(chunk_duration)
    
    except Exception as e:
        logging.error(f"Error in write_to_buffer: {e}")
    
    finally:
        buffer.writing_done = True


def normalize_audio(audio_data: np.ndarray) -> np.ndarray:
    """
    Normalize audio to [-1, 1] range.
    
    Args:
        audio_data: Audio samples
    
    Returns:
        Normalized audio
    """
    max_val = np.max(np.abs(audio_data))
    
    if max_val > 0:
        return audio_data / max_val
    
    return audio_data
