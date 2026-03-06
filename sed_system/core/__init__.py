"""
SED System - Core Components
=============================
Core components for real-time Sound Event Detection simulation.

Components:
- CircularBuffer: Thread-safe circular buffer for audio streaming
- InputFrameProvider: Adaptive frame provider with variable frame size
- Audio processing utilities: load, resample, normalize
- Model loader: Multi-model support (EPANNs, CED, CLAP)
- Inference engine: Real-time inference with adaptive framing
- Monitoring: Metrics logging, SED metrics, performance monitoring
"""

from .buffer import CircularBuffer
from .frame_provider import InputFrameProvider
from .audio_processor import load_audio, write_to_buffer
from .model_loader import load_inference_model, get_model_config
from .inference_engine import inference_task, single_inference, run_inference_simulation

__all__ = ['CircularBuffer',
           'InputFrameProvider',
           'load_audio',
           'write_to_buffer',
           'load_inference_model',
           'get_model_config',
           'inference_task',
           'single_inference',
           'run_inference_simulation']
