"""Procesadores de audio (STT, TTS, VAD)."""

from retrieval_graph.audio.processors.stt import STTProcessor
from retrieval_graph.audio.processors.tts import TTSProcessor

__all__ = ["STTProcessor", "TTSProcessor"]