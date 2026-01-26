"""Detector de actividad de voz (Voice Activity Detection)."""

import torch
import numpy as np


class VADProcessor:
    """Detector de pausas usando Silero VAD."""
    
    def __init__(self, threshold: float = 0.5):
        # Cargar modelo Silero VAD
        self.model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False
        )
        self.threshold = threshold
        self.get_speech_timestamps = utils[0]
    
    def detect_speech_segments(self, audio_array: np.ndarray, sample_rate: int = 16000):
        """Detecta segmentos de voz en el audio."""
        audio_tensor = torch.from_numpy(audio_array)
        
        speech_timestamps = self.get_speech_timestamps(
            audio_tensor,
            self.model,
            threshold=self.threshold,
            sampling_rate=sample_rate
        )
        
        return speech_timestamps