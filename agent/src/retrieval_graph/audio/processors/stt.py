"""Procesador Speech-to-Text con faster-whisper."""

import numpy as np
from faster_whisper import WhisperModel
from retrieval_graph.audio.config import AudioConfiguration


class STTProcessor:
    """Procesador de Speech-to-Text."""
    
    def __init__(self, config: AudioConfiguration):
        self.model = WhisperModel(
            config.whisper_model,
            device=config.whisper_device,
            compute_type=config.whisper_compute_type
        )
        self.config = config
    
    def transcribe(self, audio_bytes: bytes) -> str:
        """Transcribe audio a texto."""
        # Convertir bytes a array numpy
        audio_array = np.frombuffer(
            audio_bytes, 
            dtype=np.int16
        ).astype(np.float32) / 32768.0
        
        segments, info = self.model.transcribe(
            audio_array,
            language="es",
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=self.config.min_silence_duration_ms
            )
        )
        
        transcription = " ".join([seg.text for seg in segments])
        return transcription.strip()