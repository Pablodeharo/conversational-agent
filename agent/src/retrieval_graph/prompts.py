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
    """
    Build a prompt for internal philosophical reflection.

    This prompt instructs the model to analyze the user's question
    and produce a structured Reflection object in JSON format.

    Args:
        retrieved_context (str):
            Relevant documents retrieved from the knowledge base.
        conversation (str):
            The conversation history so far.

    Returns:
        str:
            A formatted prompt instructing the model to output
            a Reflection JSON object.
    """
    return f"""
You are a philosophical analyst.

Using the retrieved context and the conversation below,
analyze the user's question and output a JSON object
matching the Reflection schema.

RETRIEVED CONTEXT:
{retrieved_context}

CONVERSATION:
{conversation}

Return ONLY valid JSON.
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
