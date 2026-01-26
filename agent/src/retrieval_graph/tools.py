"""
Cognitive tools for the Socratic retrieval agent.

This module defines structured reasoning tools used by the agent
to reflect on user questions before generating a response.
These tools do NOT directly respond to the user.
"""

from typing import List, Literal

from pydantic import BaseModel, Field


class Reflection(BaseModel):
    """
    Represents an internal reflection about the user's question.

    This structure captures the agent's analysis of the question,
    including assumptions, topic, and the recommended Socratic strategy.
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


def reflection_system_prompt() -> str:
    """
    System prompt used to guide the reflection process.

    This prompt instructs the language model to analyze the user's
    question in a Socratic and philosophical manner.
    """
    return (
        "You are Socrates engaged in internal reflection.\n"
        "Do NOT answer the user.\n"
        "Analyze the user's last question carefully.\n\n"
        "Your task:\n"
        "- Identify the main philosophical topic.\n"
        "- Detect implicit assumptions.\n"
        "- Classify the type of question.\n"
        "- Decide the best Socratic strategy to continue the dialogue.\n\n"
        "Be precise, concise, and philosophical.\n"
        "Do not include rhetorical flourishes or explanations.\n"
        "Return only structured data."
    )
