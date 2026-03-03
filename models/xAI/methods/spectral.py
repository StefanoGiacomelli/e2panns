"""
Spectral Analysis Methods
==========================
Filterbank analysis and spectrogram extraction.
"""

import torch
import torch.nn as nn
import numpy as np
import torchlibrosa as tl
from typing import Dict, Tuple, Optional


class FilterbankAnalyzer:
    """
    Analyze and compare mel filterbanks.
    For models with learnable filterbanks (EPANNs, CLAP).
    """
    
    def __init__(
        self,
        model: nn.Module,
        sr: int = 32000,
        n_fft: int = 1024,
        n_mels: int = 64,
        fmin: int = 50,
        fmax: int = 14000
    ):
        """
        Args:
            model: Model with logmel_extractor
            sr: Sample rate
            n_fft: FFT size
            n_mels: Number of mel bins
            fmin: Minimum frequency
            fmax: Maximum frequency
        """
        self.model = model
        self.sr = sr
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
    
    def extract_learned_filterbank(self) -> np.ndarray:
        """
        Extract learned mel filterbank from model.
        
        Returns:
            Filterbank matrix (n_mels, n_freqs)
        """
        
        # Access model's logmel extractor
        logmel_extractor = None
        
        # Try different access patterns
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'audio_branch'):
            # CLAP structure: model.model.audio_branch.logmel_extractor
            logmel_extractor = self.model.model.audio_branch.logmel_extractor
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'logmel_extractor'):
            # EPANNs structure: model.model.logmel_extractor
            logmel_extractor = self.model.model.logmel_extractor
        elif hasattr(self.model, 'logmel_extractor'):
            # Direct access
            logmel_extractor = self.model.logmel_extractor
        elif hasattr(self.model, 'audio_branch'):
            # Alternative structure
            logmel_extractor = self.model.audio_branch.logmel_extractor
        elif hasattr(self.model, 'front_end'):
            # CED uses torchaudio MelSpectrogram instead of LogmelFilterBank
            # front_end is Sequ (FrontEnd) with MelSpectrogram at [0]
            if hasattr(self.model.front_end, '__getitem__'):
                mel_transform = self.model.front_end[0]  # Get MelSpectrogram
                if hasattr(mel_transform, 'mel_scale'):
                    with torch.no_grad():
                        filterbank = mel_transform.mel_scale.fb.cpu().numpy()  # (n_freqs, n_mels)
                        filterbank = filterbank.T  # -> (n_mels, n_freqs)
                    return filterbank
                else:
                    raise ValueError("CED MelSpectrogram doesn't have mel_scale attribute")
            else:
                raise ValueError("CED front_end is not indexable")
        else:
            raise ValueError("Cannot find logmel_extractor or front_end in model")
        
        if logmel_extractor is None:
            raise ValueError("Cannot find logmel_extractor in model")
        
        # Extract melW parameter
        with torch.no_grad():
            for name, param in logmel_extractor.named_parameters():
                if 'melW' in name:
                    filterbank = param.cpu().numpy().T  # (n_mels, n_freqs)
                    return filterbank
        
        raise ValueError("No melW parameter found in logmel_extractor")
    
    def create_standard_filterbank(self) -> np.ndarray:
        """
        Create standard mel filterbank using torchlibrosa or torchaudio.
        
        Returns:
            Standard filterbank matrix (n_mels, n_freqs)
        """
        # For CED, use torchaudio
        if hasattr(self.model, 'front_end'):
            import torchaudio.functional as F_audio
            
            # Create mel filterbank using torchaudio functional API
            fb = F_audio.melscale_fbanks(
                n_freqs=self.n_fft // 2 + 1,
                f_min=self.fmin,
                f_max=self.fmax,
                n_mels=self.n_mels,
                sample_rate=self.sr,
                norm='slaney'
            )
            
            return fb.T.cpu().numpy()  # Transpose to (n_mels, n_freqs)
        
        # For EPANNs/CLAP, use torchlibrosa
        with torch.no_grad():
            standard_logmel = tl.LogmelFilterBank(
                sr=self.sr,
                n_fft=self.n_fft,
                n_mels=self.n_mels,
                fmin=self.fmin,
                fmax=self.fmax,
                ref=1.0,
                amin=1e-10,
                top_db=None,
                is_log=True,
                freeze_parameters=True
            )
            
            for name, param in standard_logmel.named_parameters():
                if 'melW' in name:
                    return param.cpu().numpy().T
        
        raise ValueError("Cannot create standard filterbank")
    
    def compute_centroids_and_peaks(
        self,
        filterbank: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute centroid and peak frequencies for each filter.
        
        Args:
            filterbank: Filterbank matrix (n_mels, n_freqs)
            
        Returns:
            centroids: Centroid frequency for each filter
            peaks: Peak frequency for each filter
        """
        n_mels, n_freqs = filterbank.shape
        freqs = np.linspace(0, self.sr / 2, n_freqs)
        
        centroids = np.zeros(n_mels)
        peaks = np.zeros(n_mels)
        
        for i, weights in enumerate(filterbank):
            # Ensure non-negative
            w_pos = np.clip(weights, a_min=0, a_max=None)
            total = w_pos.sum()
            
            if total > 0:
                centroids[i] = (w_pos * freqs).sum() / total
            else:
                centroids[i] = 0.0
            
            peaks[i] = freqs[np.argmax(w_pos)]
        
        return centroids, peaks
    
    def compare_filterbanks(self) -> Dict[str, np.ndarray]:
        """
        Compare learned and standard filterbanks.
        
        Returns:
            Dictionary with comparison metrics
        """
        learned_fb = self.extract_learned_filterbank()
        standard_fb = self.create_standard_filterbank()
        
        learned_centroids, learned_peaks = self.compute_centroids_and_peaks(learned_fb)
        standard_centroids, standard_peaks = self.compute_centroids_and_peaks(standard_fb)
        
        # Compute differences
        centroid_diff = learned_centroids - standard_centroids
        peak_diff = learned_peaks - standard_peaks
        
        # Compute correlation
        correlation = np.corrcoef(
            learned_fb.flatten(),
            standard_fb.flatten()
        )[0, 1]
        
        return {
            'learned_filterbank': learned_fb,
            'standard_filterbank': standard_fb,
            'learned_centroids': learned_centroids,
            'learned_peaks': learned_peaks,
            'standard_centroids': standard_centroids,
            'standard_peaks': standard_peaks,
            'centroid_difference': centroid_diff,
            'peak_difference': peak_diff,
            'correlation': correlation
        }


class SpectrogramExtractor:
    """
    Extract spectrograms from models in a unified way.
    """
    
    def __init__(self, model: nn.Module, model_type: str):
        """
        Args:
            model: Model instance
            model_type: Type of model ('epanns', 'ced', 'clap')
        """
        self.model = model
        self.model_type = model_type
    
    def extract(
        self,
        waveform: torch.Tensor,
        device: str = "cuda"
    ) -> Dict[str, np.ndarray]:
        """
        Extract spectrograms from model.
        
        Args:
            waveform: Input waveform
            device: Device
            
        Returns:
            Dictionary with extracted spectrograms
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        waveform = waveform.to(device)
        
        result = {}
        
        with torch.no_grad():
            if self.model_type == 'epanns':
                # Access CNN model
                if hasattr(self.model, 'model'):
                    cnn_model = self.model.model
                else:
                    cnn_model = self.model
                
                spec = cnn_model.spectrogram_extractor(waveform)
                logmel = cnn_model.logmel_extractor(spec)
                
                result['spectrogram'] = spec.squeeze().cpu().numpy()
                result['logmel'] = logmel.squeeze().cpu().numpy()
            
            elif self.model_type == 'ced':
                # CED uses torchaudio frontend
                # front_end outputs (1, 1, n_mels, time) -> squeeze -> (n_mels, time)
                # Transpose to (time, n_mels) to match EPANNs/CLAP format
                mel_spec = self.model.front_end(waveform)
                mel_spec_np = mel_spec.squeeze().cpu().numpy()  # (n_mels, time)
                result['mel_spectrogram'] = mel_spec_np.T  # (time, n_mels)
            
            elif self.model_type == 'clap':
                # CLAP/HTSAT similar to EPANNs
                if hasattr(self.model, 'model'):
                    audio_model = self.model.model.audio_branch
                elif hasattr(self.model, 'audio_branch'):
                    audio_model = self.model.audio_branch
                else:
                    audio_model = self.model
                
                spec = audio_model.spectrogram_extractor(waveform)
                logmel = audio_model.logmel_extractor(spec)
                
                result['spectrogram'] = spec.squeeze().cpu().numpy()
                result['logmel'] = logmel.squeeze().cpu().numpy()
        
        return result
