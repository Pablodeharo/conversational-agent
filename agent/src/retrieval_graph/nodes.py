"""
nodes.py

Node functions for the Socratic retrieval graph.

Each function in this module represents a node in the LangGraph pipeline.
Nodes:
- Read from the shared State
- Perform a specific transformation or action
- Return partial state updates
"""

from pathlib import Path
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from retrieval_graph import retrieval
from retrieval_graph.configuration import Configuration
from retrieval_graph.state import State
from retrieval_graph.utils import format_docs, get_message_text
from retrieval_graph.tools import Reflection
from retrieval_graph.prompts import build_reflection_prompt, build_socratic_system_prompt
from retrieval_graph.model_manager import ModelManager
from retrieval_graph.Backend.base import GenerationConfig


async def generate_query(
    state: State, *, config: RunnableConfig
) -> dict[str, list[str]]:
    """
    Generate a search query from the most recent user message.

    This node extracts the user's last message and uses it directly
    as a retrieval query.

    Args:
        state (State):
            The current agent state.
        config (RunnableConfig):
            Runtime configuration (unused here).

    Returns:
        dict[str, list[str]]:
            A dictionary containing newly generated queries.
    """
    human_input = get_message_text(state.messages[-1])
    return {"queries": [human_input]}


async def retrieve(
    state: State, *, config: RunnableConfig
) -> dict[str, list[Document]]:
    """
    Retrieve relevant documents based on the latest search query.

    This node:
    - Instantiates a retriever using the runtime configuration
    - Executes an asynchronous retrieval call
    - Returns the retrieved documents

    Args:
        state (State):
            The current agent state.
        config (RunnableConfig):
            Runtime configuration used to build the retriever.

    Returns:
        dict[str, list[Document]]:
            Retrieved documents to be stored in the state.
    """
    with retrieval.make_retriever(config) as retriever:
        response = await retriever.ainvoke(state.queries[-1], config)
        return {"retrieved_docs": response}


async def reflect_on_question(
    state: State, *, config: RunnableConfig
) -> dict[str, Reflection]:
    """
    Perform internal Socratic reflection on the user's question.

    This node:
    - Builds a structured reflection prompt
    - Uses a local LLM backend via ModelManager
    - Produces a validated Reflection object

    The reflection is used internally to guide the response strategy.
    """
    configuration = Configuration.from_runnable_config(config)

    model_manager = ModelManager(
        config_path=Path(__file__).parent / "config" / "models.yaml"
    )
    backend = await model_manager.get_backend(configuration.response_model)

    # Prepare retrieved document context
    retrieved_context = (
        format_docs(state.retrieved_docs)
        if state.retrieved_docs
        else "No hay documentos relevantes."
    )
    # Prepare conversation history
    conversation = "\n".join(
        f"- {get_message_text(m)}" for m in state.messages
    )

    # Build reflection prompt
    prompt = build_reflection_prompt(retrieved_context, conversation)

    # Generate reflection
    response = await backend.generate(
        prompt=prompt,
        config=GenerationConfig(
            max_tokens=400,
            temperature=0.2,
        ),
    )

    # Parse and validate
    try:
        reflection = Reflection.model_validate_json(response.content)
    except Exception as e:
        raise ValueError(
            f"Error parsing Reflection JSON.\nOutput was:\n{response.content}"
        ) from e

    return {"reflection": reflection}


async def call_model(state: State, config: RunnableConfig):
    """
    Generate the final Socratic response to the user.

    This node:
    - Uses the Reflection object to select a response strategy
    - Builds a system prompt enforcing a Socratic style
    - Generates the final answer using the LLM
    - Appends the response as an AIMessage
    """
    configuration = Configuration.from_runnable_config(config)
    reflection = state.reflection

    # Initialize model backend
    model_manager = ModelManager(
        config_path=Path(__file__).parent / "config" / "models.yaml"
    )
    backend = await model_manager.get_backend(configuration.response_model)

    # Extract reflection attributes with safe fallbacks
    strategy = (
        reflection.suggested_strategy
        if reflection
        else "clarify_terms"
    )
    question_type = (
        reflection.question_type
        if reflection
        else "unclear"
    )
    topic = (
        reflection.topic
        if reflection
        else "unknown"
    )
    assumptions = (
        "\n".join(f"- {a}" for a in reflection.assumptions)
        if reflection and reflection.assumptions
        else "- None detected"
    )

    # Format retrieved documents
    retrieved_context = (
        format_docs(state.retrieved_docs)
        if state.retrieved_docs
        else "No relevant documents available."
    )

    # Build system prompt
    system_prompt = build_socratic_system_prompt(
        strategy=strategy,
        question_type=question_type,
        topic=topic,
        assumptions=assumptions,
        retrieved_context=retrieved_context,
    )

    # Build conversation transcript
    conversation = []
    for msg in state.messages:
        role = "Socrates" if isinstance(msg, AIMessage) else "Interlocutor"
        conversation.append(f"{role}: {msg.content}")

    # Assemble full prompt
    full_prompt = (
        f"{system_prompt}\n\n"
        f"CONVERSATION:\n{chr(10).join(conversation)}\n\n"
        f"Socrates:"
    )

    # Generate final response
    response = await backend.generate(
        prompt=full_prompt,
        config=GenerationConfig(
            max_tokens=512,
            temperature=0.7,
        ),
    )

    # Return AI message to be appended to the conversation
    return {
        "messages": [
            AIMessage(content=response.content)
        ]
    }