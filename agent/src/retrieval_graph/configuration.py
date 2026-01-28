"""
configuration.py

Defines the configurable parameters for the conversational agent.

This module centralizes all runtime configuration, including:
- Retrieval and indexing settings
- Model selection
- Prompt selection

Configurations are injected into the graph via RunnableConfig.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Annotated, Any, Literal, Type, TypeVar

from langchain_core.runnables import RunnableConfig, ensure_config

from retrieval_graph import prompts


@dataclass(kw_only=True)
class IndexConfiguration:
    """
    Configuration class for indexing and retrieval operations.

    This configuration controls:
    - User identity
    - Embedding model
    - Vector store provider
    - Search parameters
    """
    user_id: str = field(
        default="default_user",
        metadata={"description": "Unique identifier for the user."})

    embedding_model: Annotated[
        str,
        {"__template_metadata__": {"kind": "embeddings"}},
    ] = field(
        default="openai/text-embedding-3-small",
        metadata={
            "description": "Name of the embedding model to use. Must be a valid embedding model name."
        },
    )

    retriever_provider: Annotated[
        Literal[
            "elastic",
            "elastic-local", 
            "pinecone", 
            "mongodb", 
            "faiss-local",
        ],
        {"__template_metadata__": {"kind": "retriever"}},
    ] = field(
        default="faiss-local",
        metadata={
            "description": "The vector store provider to use for retrieval. Options are 'elastic', 'pinecone', 'mongodb' and 'faiss-local'."
        },
    )

    index_path: str = field(
        default="/home/pablo/Documentos/conversational-agent/agent/src/retrieval_graph/data/faiss",
        metadata={"description": "Path to local FAISS vectorestore."},
    )

    search_kwargs: dict[str, Any] = field(
        default_factory=dict,
        metadata={
            "description": "Additional keyword arguments to pass to the search function of the retriever."
        },
    )

    @classmethod
    def from_runnable_config(
        cls: Type[T], config: RunnableConfig | None = None
    ) -> T:
        """
        Create an IndexConfiguration from a RunnableConfig object.

        This method extracts only the fields relevant to this class
        from the 'configurable' section of the RunnableConfig.
        """
        config = ensure_config(config)
        configurable = config.get("configurable") or {}
        _fields = {f.name for f in fields(cls) if f.init}
        return cls(**{k: v for k, v in configurable.items() if k in _fields})


T = TypeVar("T", bound=IndexConfiguration)


@dataclass(kw_only=True)
class Configuration(IndexConfiguration):
    """
    Full configuration for the conversational agent.

    Extends IndexConfiguration by adding:
    - Prompt configuration
    - Language model selection
    """

    response_system_prompt: str = field(
        default=prompts.RESPONSE_SYSTEM_PROMPT,
        metadata={"description": "The system prompt used for generating responses."},
    )

    response_model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="mistral",
        metadata={
            "description": "LLM used for generating final responses."
        },
    )

    query_system_prompt: str = field(
        default=prompts.QUERY_SYSTEM_PROMPT,
        metadata={
            "description": "System prompt used for query generation."
        },
    )

    query_model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="anthropic/claude-3-haiku-20240307",
        metadata={
            "description": "LLM used for query processing."
        },
    )
