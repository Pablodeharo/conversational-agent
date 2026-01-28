"""
tools.py

Cognitive tools for the Socratic retrieval agent.

This module defines structured data models used for internal reasoning.
These tools are NOT responsible for generating user-facing responses.
Instead, they guide the agent's reasoning strategy.
"""

from typing import List, Literal
from pydantic import BaseModel, Field

class Reflection(BaseModel):
    """
    Represents an internal reflection about the user's question.

    This model captures the agent's analysis of the question, including:
    - The philosophical topic
    - Implicit assumptions
    - The type of question being asked
    - The recommended Socratic strategy

    This structure is intended for internal use only and is not shown
    directly to the user.
    """

    topic: str = Field(
        ...,
        description="The main philosophical topic of the user's question."
    )

    assumptions: List[str] = Field(
        default_factory=list,
        description="Implicit assumptions detected in the user's question."
    )

    question_type: Literal[
        "definition",
        "ethical",
        "epistemological",
        "practical",
        "metaphysical",
        "contradiction",
        "unclear",
    ] = Field(
        ...,
        description="The philosophical type of the question."
    )

    suggested_strategy: Literal[
        "ask_for_definition",
        "challenge_assumption",
        "use_counterexample",
        "lead_to_aporia",
        "clarify_terms",
    ] = Field(
        ...,
        description="Recommended Socratic strategy for responding."
    )



