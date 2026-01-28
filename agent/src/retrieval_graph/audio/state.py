from dataclasses import dataclass, field
from typing import Optional, Literal
from langchain_core.messages import AnyMessage


@dataclass
class AudioInputState:
    """
    Input schema for the audio graph.

    Either text or raw audio bytes can be provided.
    """
    input_type: Literal["text", "audio"]
    audio_data: Optional[bytes] = None


@dataclass
class AudioState(AudioInputState):
    """
    Runtime state for the audio graph.

    Extends the input state with:
    - Messages exchanged with the main graph
    - Transcribed text
    - Generated audio response
    """
    messages: list[AnyMessage] = field(default_factory=list)
    transcribed_text: Optional[str] = None
    response_audio: Optional[bytes] = None
    enable_tts: bool = True
