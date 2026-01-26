"""Nodos de LangGraph para procesamiento de audio."""

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from retrieval_graph.audio.state import AudioState
from retrieval_graph.audio.config import AudioConfiguration
from retrieval_graph.audio.processors import STTProcessor, TTSProcessor


# Singletons para evitar recargar modelos
_stt_processor = None
_tts_processor = None


async def stt_node(state: AudioState, config: RunnableConfig) -> dict:
    """Nodo STT: Convierte audio a texto."""
    global _stt_processor
    
    if state.input_type == "text":
        return {}
    
    if not state.audio_data:
        raise ValueError("input_type='audio' pero no hay audio_data")
    
    # Lazy loading
    if _stt_processor is None:
        audio_config = AudioConfiguration.from_runnable_config(config)
        _stt_processor = STTProcessor(audio_config)
    
    transcribed = _stt_processor.transcribe(state.audio_data)
    
    return {
        "transcribed_text": transcribed,
        "messages": [HumanMessage(content=transcribed)]
    }


async def tts_node(state: AudioState, config: RunnableConfig) -> dict:
    """Nodo TTS: Convierte respuesta a audio."""
    global _tts_processor
    
    if not state.enable_tts:
        return {}
    
    # Buscar última respuesta de Sócrates
    last_ai_message = None
    for msg in reversed(state.messages):
        if isinstance(msg, AIMessage):
            last_ai_message = msg
            break
    
    if not last_ai_message:
        return {}
    
    # Lazy loading
    if _tts_processor is None:
        audio_config = AudioConfiguration.from_runnable_config(config)
        _tts_processor = TTSProcessor(audio_config)
    
    audio_bytes = _tts_processor.synthesize(last_ai_message.content)
    
    return {"response_audio": audio_bytes}