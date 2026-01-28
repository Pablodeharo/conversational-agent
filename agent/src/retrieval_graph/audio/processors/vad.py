"""Detector de actividad de voz (Voice Activity Detection).

Actualmente no está integrado en el graph (solo STT usa VAD interno)

"""

import torch
import numpy as np


class VADProcessor:
    """
    Voice Activity Detection processor using Silero VAD.
    """

    def __init__(self, threshold: float = 0.5):
        """
        Load the Silero VAD model.
        """
        self.model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
        )
        self.threshold = threshold
        self.get_speech_timestamps = utils[0]

    def detect_speech_segments(
        self,
        audio_array: np.ndarray,
        sample_rate: int = 16000,
    ):
        """
        Detect speech segments in an audio signal.

        Args:
            audio_array (np.ndarray): Audio samples.
            sample_rate (int): Sampling rate.

        Returns:
            list[dict]: Detected speech segments.
        """
        audio_tensor = torch.from_numpy(audio_array)

        return self.get_speech_timestamps(
            audio_tensor,
            self.model,
            threshold=self.threshold,
            sampling_rate=sample_rate,
        )