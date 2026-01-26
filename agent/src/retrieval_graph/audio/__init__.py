"""Módulo de procesamiento de audio para el agente socrático."""

from retrieval_graph.audio.state import AudioState, AudioInputState
from retrieval_graph.audio.config import AudioConfiguration
from retrieval_graph.audio.nodes import stt_node, tts_node

__all__ = [
    "AudioState",
    "AudioInputState", 
    "AudioConfiguration",
    "stt_node",
    "tts_node",
]