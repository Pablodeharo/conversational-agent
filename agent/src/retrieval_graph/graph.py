"""
graph.py

Main entrypoint for the conversational retrieval graph.

This module defines the execution flow of the conversational agent
using LangGraph. It specifies:
- The global shared state
- The processing nodes
- The execution order (edges)
- The compilation of the graph into an executable object
"""
from langgraph.graph import StateGraph

from retrieval_graph.configuration import Configuration
from retrieval_graph.state import InputState, State
from retrieval_graph.nodes import (
    generate_query,         
    retrieve,
    reflect_on_question,
    call_model,
)


# Graph definition
# Create a StateGraph instance
# - State: shared mutable state passed between nodes
# - input_schema: structure of user input
# - config_schema: runtime configuration options
builder = StateGraph(State, input_schema=InputState, config_schema=Configuration)


builder.add_node("generate_query", generate_query)
builder.add_node("retrieve", retrieve)
builder.add_node("reflect_on_question", reflect_on_question)
builder.add_node("call_model", call_model)


builder.add_edge("__start__", "generate_query")
builder.add_edge("generate_query", "retrieve")
builder.add_edge("retrieve", "reflect_on_question")
builder.add_edge("reflect_on_question", "call_model")
builder.add_edge("call_model", "__end__")

graph = builder.compile(
    interrupt_before=[],
    interrupt_after=[],
)
graph.name = "RetrievalGraph"