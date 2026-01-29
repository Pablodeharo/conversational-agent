"""
prompts.py

Prompt construction utilities for the Socratic retrieval agent.

This module is responsible for:
- Defining static system prompts
- Building dynamic prompts using retrieved context and conversation history

It contains NO model calls and NO state mutation.
"""

# ---------------------------------------------------------------------
# Base system prompts
# ---------------------------------------------------------------------

# System prompt used when generating the final Socratic response
RESPONSE_SYSTEM_PROMPT = """
You are Socrates.
You do not provide direct answers.
You guide the interlocutor through questioning, clarification,
and examination of assumptions.
"""

# System prompt used when analyzing or refining user queries
QUERY_SYSTEM_PROMPT = """
You are an expert at transforming natural language questions
into effective retrieval queries.
"""


# ---------------------------------------------------------------------
# Reflection prompt
# ---------------------------------------------------------------------

def build_reflection_prompt(
    retrieved_context: str,
    conversation: str
) -> str:
    return f"""
You are an internal reflection engine for a Socratic assistant.

Your job is to analyze the user's question and produce a JSON object
that conforms EXACTLY to the Reflection schema.

Rules:
- Output ONLY valid JSON.
- No explanations, no markdown, no comments.
- Use concise values.
- Do not repeat the retrieved text verbatim.
- If unsure, use empty lists or nulls where allowed.
- Ensure the JSON is complete and well-formed.

RETRIEVED CONTEXT (summarize internally, do not quote):
{retrieved_context}

CONVERSATION:
{conversation}

Begin JSON output now.
"""


# ---------------------------------------------------------------------
# Socratic response prompt
# ---------------------------------------------------------------------

def build_socratic_system_prompt(
    *,
    strategy: str,
    question_type: str,
    topic: str,
    assumptions: str,
    retrieved_context: str,
) -> str:
    """
    Build the system prompt used to generate the final Socratic response.

    This prompt conditions the language model to:
    - Follow a specific Socratic strategy
    - Respect the philosophical type of the question
    - Take into account detected assumptions
    - Use retrieved documents as background knowledge

    Args:
        strategy (str):
            The Socratic strategy to apply.
        question_type (str):
            The philosophical category of the question.
        topic (str):
            The main philosophical topic.
        assumptions (str):
            Formatted list of detected assumptions.
        retrieved_context (str):
            Relevant background documents.

    Returns:
        str:
            A system prompt enforcing a Socratic dialogue style.
    """
    return f"""
You are Socrates.

PHILOSOPHICAL TOPIC:
{topic}

QUESTION TYPE:
{question_type}

SOCRATIC STRATEGY:
{strategy}

DETECTED ASSUMPTIONS:
{assumptions}

RETRIEVED CONTEXT:
{retrieved_context}

Engage the interlocutor using questions.
Do not provide final answers.
Guide them toward deeper understanding.
"""
