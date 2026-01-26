"""Configuración para STT/TTS."""

from dataclasses import dataclass, field
from typing import Literal
from retrieval_graph.configuration import Configuration

@dataclass(kw_only=True)
class AudioConfiguration(Configuration):
    """Configuración extendida con parámetros de audio."""
    
    # STT (faster-whisper)
    whisper_model: str = field(
        default="base",
        metadata={"description": "Modelo Whisper: tiny, base, small, medium"}
    )
    whisper_device: str = field(
        default="cuda",
        metadata={"description": "Device: cuda o cpu"}
    )
    whisper_compute_type: str = field(
        default="float16",
        metadata={"description": "float16 o int8"}
    )
    
    # TTS
    tts_backend: Literal["kokoro", "edge", "piper"] = field(
        default="kokoro",
        metadata={"description": "Backend de TTS a usar"}
    )
    
    # Para Kokoro
    kokoro_voice: str = field(
        default="af_bella",
        metadata={"description": "Voz de Kokoro: af_bella, af_sarah"}
    )
    
    # Para Edge-TTS
    edge_voice: str = field(
        default="es-ES-AlvaroNeural",
        metadata={"description": "Voz de Edge-TTS"}
    )
    
    # Para Piper (si lo usas)
    piper_model: str = field(
        default="es_ES-davefx-medium",
        metadata={"description": "Modelo Piper"}
    )
    
    # VAD
    vad_threshold: float = field(
        default=0.5,
        metadata={"description": "Umbral VAD (0-1)"}
    )
    min_silence_duration_ms: int = field(
        default=800,
        metadata={"description": "Milisegundos de silencio para pausas"}
    )