"""
state.py

State management for the retrieval graph.

This module defines:
- State data structures used across the graph
- Reduction functions that control how state fields are updated
- Input and internal state separation

The state is the backbone of the agent: all nodes read from and write to it.
"""

import uuid
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal, Sequence, Union

from langchain_core.documents import Document
from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from retrieval_graph.tools import Reflection

# Document Indexing State

def reduce_docs(
    existing: Sequence[Document] | None,
    new: Union[
        Sequence[Document],
        Sequence[dict[str, Any]],
        Sequence[str],
        str,
        Literal["delete"],
    ],
) -> Sequence[Document]:
    """
    Reduce and normalize document inputs into a sequence of Document objects.

    This reducer function is responsible for handling multiple input formats
    and converting them into a consistent internal representation.

    Supported behaviors:
    - "delete": clears all existing documents
    - str: creates a single Document from a string
    - list[str]: creates Documents from strings
    - list[dict]: creates Documents from dictionaries
    - list[Document]: passes through unchanged

    Args:
        existing (Sequence[Document] | None):
            The current documents stored in the state.
        new (Union[...] ):
            New document input in various supported formats.

    Returns:
        Sequence[Document]:
            The updated list of Document objects.
    """
    if new == "delete":
        return []
    if isinstance(new, str):
        return [Document(page_content=new, metadata={"id": str(uuid.uuid4())})]
    if isinstance(new, list):
        coerced = []
        for item in new:
            if isinstance(item, str):
                coerced.append(
                    Document(page_content=item, metadata={"id": str(uuid.uuid4())})
                )
            elif isinstance(item, dict):
                coerced.append(Document(**item))
            else:
                coerced.append(item)
        return coerced
    return existing or []



@dataclass(kw_only=True)
class IndexState:
    """
    Represents the state used for document indexing operations.

    This state is typically used in a separate indexing graph,
    not the main conversational agent graph.
    """

    docs: Annotated[Sequence[Document], reduce_docs]
    """A list of documents that the agent can index."""


#############################  Agent State  ###################################


# Optional, the InputState is a restricted version of the State that is used to
# define a narrower interface to the outside world vs. what is maintained
# internally.
@dataclass(kw_only=True)
class InputState:
    """Represents the input state for the agent.

    This class defines the structure of the input state, which includes
    the messages exchanged between the user and the agent. It serves as
    a restricted version of the full State, providing a narrower interface
    to the outside world compared to what is maintained internally.
    """

    messages: Annotated[Sequence[AnyMessage], add_messages]
    """Messages track the primary execution state of the agent.

    Typically accumulates a pattern of Human/AI/Human/AI messages; if
    you were to combine this template with a tool-calling ReAct agent pattern,
    it may look like this:

    1. HumanMessage - user input
    2. AIMessage with .tool_calls - agent picking tool(s) to use to collect
         information
    3. ToolMessage(s) - the responses (or errors) from the executed tools
    
        (... repeat steps 2 and 3 as needed ...)
    4. AIMessage without .tool_calls - agent responding in unstructured
        format to the user.

    5. HumanMessage - user responds with the next conversational turn.

        (... repeat steps 2-5 as needed ... )
    
    Merges two lists of messages, updating existing messages by ID.

    By default, this ensures the state is "append-only", unless the
    new message has the same ID as an existing message.

    Returns:
        A new list of messages with the messages from `right` merged into `left`.
        If a message in `right` has the same ID as a message in `left`, the
        message from `right` will replace the message from `left`."""


# This is the primary state of your agent, where you can store any information
def reduce_reflection(
        existing: Reflection | None,
        new: Reflection,
) -> Reflection:
    """
    Reducer for the Reflection object.

    Since reflection represents the most recent internal analysis,
    the new value always replaces the existing one.
    """
    return new

def add_queries(existing: Sequence[str], new: Sequence[str]) -> Sequence[str]:
    """
    Combine existing search queries with newly generated ones.

    Args:
        existing (Sequence[str]):
            Queries already stored in the state.
        new (Sequence[str]):
            New queries generated by the agent.

    Returns:
        Sequence[str]:
            A concatenated list of queries.
    """
    return list(existing) + list(new)

@dataclass(kw_only=True)
class State(InputState):
    """
    The primary internal state of the conversational agent.

    This state extends InputState and includes all internal variables
    needed across the graph execution.
    """
    queries: Annotated[list[str], add_queries] = field(default_factory=list)
    """A list of search queries that the agent has generated."""

    retrieved_docs: list[Document] = field(default_factory=list)
    """
    Documents retrieved from the vector store or retriever.
    These are used as context for reflection and response generation.
    """

    reflection: Annotated[Reflection | None, reduce_reflection] = None
    """
    Internal philosophical reflection about the user's question.
    Used to guide the Socratic response strategy.
    """
