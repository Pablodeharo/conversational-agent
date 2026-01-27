"""Main entrypoint for the conversational retrieval graph.

This module defines the core structure and functionality of the conversational
retrieval graph. It includes the main graph definition, state management,
and key functions for processing user inputs, generating queries, retrieving
relevant documents, and formulating responses.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import cast

from pydantic import BaseModel
from langgraph.graph import StateGraph
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig

from retrieval_graph import retrieval
from retrieval_graph.configuration import Configuration
from retrieval_graph.state import InputState, State
from retrieval_graph.utils import format_docs, get_message_text, load_chat_model
from retrieval_graph.tools import Reflection
from retrieval_graph.prompts import RESPONSE_SYSTEM_PROMPT, QUERY_SYSTEM_PROMPT, REFLECTION_SYSTEM_PROMPT
from retrieval_graph.model_manager import ModelManager
from retrieval_graph.Backend.base import GenerationConfig



async def call_model(state: State, config: RunnableConfig):
    """
    Nodo para generar respuestas socráticas usando ModelManager con Mixtral.
    
    Este nodo:
    1. Obtiene la reflexión socrática del estado
    2. Construye un prompt socrático en español
    3. Incluye los documentos de Platón recuperados
    4. Usa Mixtral (modelo local) para generar la respuesta
    """
    configuration = Configuration.from_runnable_config(config)
    reflection = state.reflection

    # Inicializar ModelManager
    model_manager = ModelManager(
        config_path=Path(__file__).parent / "config" / "models.yaml"
    )
    backend = await model_manager.get_backend(configuration.response_model)

    # Extraer información de la reflexión
    strategy = reflection.suggested_strategy if reflection else "clarify_terms"
    question_type = reflection.question_type if reflection else "unclear"
    topic = reflection.topic if reflection else "unknown"
    assumptions = (
        "\n".join(f"- {a}" for a in reflection.assumptions)
        if reflection and reflection.assumptions
        else "- Ninguno detectado"
    )
    
    # Formatear documentos recuperados
    retrieved_context = format_docs(state.retrieved_docs) if state.retrieved_docs else "No hay documentos relevantes."
    
    # Construir system prompt socrático en español
    system_prompt = f"""Eres Sócrates, el filósofo griego, en un diálogo con tu interlocutor.
Debes responder siguiendo el método socrático, guiando a través de preguntas y reflexiones.

CONTEXTO DE LA CONVERSACIÓN:
Estrategia socrática recomendada: {strategy}
Tipo de pregunta del interlocutor: {question_type}
Tema principal: {topic}
Supuestos implícitos detectados:
{assumptions}

DOCUMENTOS RELEVANTES DE PLATÓN:
{retrieved_context}

REGLAS DEL MÉTODO SOCRÁTICO:
1. Prefiere hacer preguntas reflexivas en lugar de dar respuestas directas
2. No proporciones respuestas finales o conclusiones definitivas
3. Guía al interlocutor a examinar sus propias creencias y contradicciones
4. Usa los textos de Platón para fundamentar tus cuestionamientos
5. Sé conciso, claro y preciso en tus intervenciones
6. Mantén un tono respetuoso pero desafiante intelectualmente

Responde en español de forma breve y socrática."""

    # Construir el historial de conversación
    conversation = []
    for msg in state.messages:
        if isinstance(msg, AIMessage):
            role = "Sócrates"
        elif isinstance(msg, ToolMessage):
            role = "Herramienta"
        else:  # HumanMessage u otros
            role = "Interlocutor"
        
        conversation.append(f"{role}: {msg.content}")

    # Prompt completo
    full_prompt = f"{system_prompt}\n\nCONVERSACIÓN:\n" + "\n".join(conversation) + "\n\nSócrates:"

    # Generar respuesta usando el backend (Mixtral)
    response = await backend.generate(
        prompt=full_prompt,
        config=GenerationConfig(
            max_tokens=512, 
            temperature=0.7,
            # Añade aquí otros parámetros si tu GenerationConfig los soporta:
            # top_p=0.9,
            # repetition_penalty=1.1,
        )
    )

    # Retornar la respuesta como AIMessage
    return {"messages": [AIMessage(content=response.content)]}


class SearchQuery(BaseModel):
    """Search the indexed documents for a query."""

    query: str


async def generate_query(
    state: State, *, config: RunnableConfig
) -> dict[str, list[str]]:
    """
    Generate a search query based on the latest user message.
    Fully local, no LLM call.
    """
    # Usamos directamente el último mensaje humano
    human_input = get_message_text(state.messages[-1])
    return {"queries": [human_input]}


async def retrieve(
    state: State, *, config: RunnableConfig
) -> dict[str, list[Document]]:
    """Retrieve documents based on the latest query in the state.

    This function takes the current state and configuration, uses the latest query
    from the state to retrieve relevant documents using the retriever, and returns
    the retrieved documents.

    Args:
        state (State): The current state containing queries and the retriever.
        config (RunnableConfig | None, optional): Configuration for the retrieval process.

    Returns:
        dict[str, list[Document]]: A dictionary with a single key "retrieved_docs"
        containing a list of retrieved Document objects.
    """
    with retrieval.make_retriever(config) as retriever:
        response = await retriever.ainvoke(state.queries[-1], config)
        return {"retrieved_docs": response}

async def reflect_on_question(
    state: State, *, config: RunnableConfig
) -> dict[str, Reflection]:
    configuration = Configuration.from_runnable_config(config)

    model_manager = ModelManager(
        config_path=Path(__file__).parent / "config" / "models.yaml"
    )
    backend = await model_manager.get_backend(configuration.response_model)

    retrieved_docs = (
        format_docs(state.retrieved_docs)
        if state.retrieved_docs
        else "No hay documentos relevantes."
    )

    conversation = "\n".join(
        f"- {get_message_text(m)}" for m in state.messages
    )

    prompt = prompts.REFLECTION_SYSTEM_PROMPT.format(
        retrieved_docs=retrieved_docs,
        conversation=conversation,
    )

    response = await backend.generate(
        prompt=prompt,
        config=GenerationConfig(
            max_tokens=400,
            temperature=0.2,
        ),
    )

    reflection = Reflection.model_validate_json(response.content)

    return {"reflection": reflection}

async def respond(
    state: State, *, config: RunnableConfig
) -> dict[str, list[BaseMessage]]:
    """Call the LLM powering our "agent"."""
    configuration = Configuration.from_runnable_config(config)
    reflection = state.reflection
    # Feel free to customize the prompt, model, and other logic!
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are Socrates in dialogue.\n"
                    "You must respond according to the given Socratic strategy.\n\n"
                    "Socratic strategy: {strategy}\n"
                    "Question type: {question_type}\n"
                    "Main topic: {topic}\n"
                    "Implicit assumptions:\n{assumptions}\n\n"
                    "Rules:\n"
                    "- Prefer questions over assertions.\n"
                    "- Do not give final answers.\n"
                    "- Guide the interlocutor to examine their own beliefs.\n"
                    "- Be concise and precise.\n"
                ),
            ),
            ("placeholder", "{messages}"),
        ]
    )

    model = load_chat_model(configuration.response_model)

    retrieved_docs = format_docs(state.retrieved_docs)
    message_value = await prompt.ainvoke(
        {
            "messages": state.messages,
            "retrieved_docs": retrieved_docs,
            "strategy": reflection.suggested_strategy if reflection else "clarify_terms",
            "question_type": reflection.question_type if reflection else "unclear",
            "topic": reflection.topic if reflection else "unknown",
            "assumptions": (
                "\n".join(f"- {a}" for a in reflection.assumptions)
                if reflection and reflection.assumptions
                else "- None detected"
            ),
        },
        config,
    )
    response = await model.ainvoke(message_value, config)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


# Define a new graph (It's just a pipe)


builder = StateGraph(State, input_schema=InputState, config_schema=Configuration)

builder.add_node(call_model)
builder.add_node(generate_query)  # type: ignore[arg-type]
builder.add_node(retrieve)  # type: ignore[arg-type]
builder.add_node(reflect_on_question)  # type: ignore[arg-type]
#builder.add_node(respond)  # type: ignore[arg-type]

builder.add_edge("__start__", "generate_query")
builder.add_edge("generate_query", "retrieve")
builder.add_edge("retrieve", "reflect_on_question")
builder.add_edge("reflect_on_question", "call_model")
builder.add_edge("call_model", "__end__")
#builder.add_edge("reflect_on_question", "respond")

# Finally, we compile it!
# This compiles it into a graph you can invoke and deploy.
graph = builder.compile(
    interrupt_before=[],  # if you want to update the state before calling the tools
    interrupt_after=[],
)
graph.name = "RetrievalGraph"
