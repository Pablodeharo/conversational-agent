"""a sandwich).

"""

from langgraph.graph import StateGraph
from retrieval_graph.audio import (
    AudioState,
    AudioInputState,
    AudioConfiguration,
    stt_node,
    tts_node,
)
from retrieval_graph.graph import graph as main_graph


# Construir grafo sandwich
builder = StateGraph(
    AudioState,
    input_schema=AudioInputState,
    config_schema=AudioConfiguration
)

builder.add_node("stt", stt_node)
builder.add_node("socrates", main_graph)
builder.add_node("tts", tts_node)

builder.add_edge("__start__", "stt")
builder.add_edge("stt", "socrates")
builder.add_edge("socrates", "tts")
builder.add_edge("tts", "__end__")

audio_graph = builder.compile()
audio_graph.name = "SocratesAudioGraph"