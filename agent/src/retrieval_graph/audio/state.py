from dataclasses import dataclass, field
from typing import Annotated, Literal
from retrieval_graph.state import State

@dataclass(kw_only=True)
class AudioInputState:
    """Input state que acepta texto o audio."""
    
    input_type: Literal["text", "audio"] = "text"
    """Tipo de input: 'text' para mensajes escritos, 'audio' para voz."""
    
    audio_data: bytes | None = None
    """Audio crudo en bytes (si input_type='audio')."""
    
    messages: list = field(default_factory=list)
    """Mensajes de texto (heredado de InputState)."""


@dataclass(kw_only=True)
class AudioState(State):
    """Estado completo del grafo con capacidades de audio."""
    
    input_type: Literal["text", "audio"] = "text"
    audio_data: bytes | None = None
    
    transcribed_text: str | None = None
    """Texto transcrito del audio (output de STT)."""
    
    response_audio: bytes | None = None
    """Audio generado de la respuesta (output de TTS)."""
    
    enable_tts: bool = True
    """Flag para activar/desactivar TTS en la respuesta."""