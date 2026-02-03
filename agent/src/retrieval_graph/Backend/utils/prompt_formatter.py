"""
Backend/utils/prompt_formatter.py

Formateadores de prompts estructurados para diferentes tareas.

Funciones:
    - format_json_prompt: Genera prompts que fuerzan JSON
    - format_sql_prompt: Genera prompts para SQL
    - format_react_prompt: Genera prompts estilo ReAct

Uso:
    from Backend.utils.prompt_formatter import format_json_prompt
    
    prompt = format_json_prompt(
        question="¿Cuál es la capital de Francia?",
        schema={"answer": "string", "confidence": "number"}
    )
"""

import json
from typing import Dict, List, Optional, Any


def format_json_prompt(
    question: str,
    schema: Dict[str, Any],
    examples: Optional[List[Dict]] = None,
    system_instruction: str = "Eres un asistente útil."
) -> str:
    """
    Genera un prompt que fuerza salida en JSON.
    
    Args:
        question: Pregunta del usuario
        schema: Esquema JSON esperado (dict simple o JSON Schema)
        examples: Lista opcional de ejemplos few-shot
        system_instruction: Instrucción del sistema
        
    Returns:
        Prompt formateado
        
    Ejemplo:
        >>> schema = {"answer": "string", "reasoning": "string"}
        >>> prompt = format_json_prompt("¿Qué es 2+2?", schema)
    """
    prompt = f"{system_instruction}\n"
    prompt += "Responde SOLO en formato JSON válido.\n\n"
    
    # Mostrar esquema esperado
    prompt += "### Formato Esperado\n\n"
    prompt += "```json\n"
    
    # Si schema es simple (dict de tipos)
    if all(isinstance(v, str) for v in schema.values()):
        example_schema = {k: f"<{v}>" for k, v in schema.items()}
        prompt += json.dumps(example_schema, indent=2, ensure_ascii=False)
    else:
        # JSON Schema completo
        prompt += json.dumps(schema, indent=2, ensure_ascii=False)
    
    prompt += "\n```\n\n"
    
    # Añadir ejemplos si se proporcionan
    if examples:
        prompt += "### Ejemplos\n\n"
        for i, ex in enumerate(examples, 1):
            prompt += f"**Ejemplo {i}:**\n"
            prompt += f"Pregunta: {ex['question']}\n"
            prompt += "Respuesta:\n```json\n"
            prompt += json.dumps(ex['response'], indent=2, ensure_ascii=False)
            prompt += "\n```\n\n"
    
    # Añadir la pregunta
    prompt += "### Tu Tarea\n\n"
    prompt += f"Pregunta: {question}\n\n"
    prompt += "Responde en formato JSON válido siguiendo el esquema.\n"
    prompt += "NO incluyas explicaciones fuera del JSON.\n\n"
    prompt += "Respuesta:\n"
    
    return prompt


def format_sql_prompt(
    question: str,
    schema: Dict[str, List[Dict]],
    examples: Optional[List[Dict]] = None,
    dialect: str = "PostgreSQL"
) -> str:
    """
    Genera un prompt para generación de SQL.
    
    Args:
        question: Pregunta en lenguaje natural
        schema: Esquema de BD {tabla: [columnas]}
        examples: Ejemplos opcionales de few-shot
        dialect: Dialecto SQL (PostgreSQL, MySQL, SQLite)
        
    Returns:
        Prompt formateado
        
    Ejemplo:
        >>> schema = {
        ...     "users": [
        ...         {"name": "id", "type": "INTEGER"},
        ...         {"name": "username", "type": "TEXT"}
        ...     ]
        ... }
        >>> prompt = format_sql_prompt("¿Cuántos usuarios hay?", schema)
    """
    prompt = f"Eres un experto en {dialect}. Genera queries SQL válidas.\n\n"
    
    # Mostrar esquema
    prompt += "### Esquema de Base de Datos\n\n"
    
    for table, columns in schema.items():
        prompt += f"**Tabla: {table}**\n"
        for col in columns:
            col_name = col.get('name', col.get('column', ''))
            col_type = col.get('type', col.get('data_type', ''))
            prompt += f"  - {col_name} ({col_type})\n"
        prompt += "\n"
    
    # Añadir ejemplos
    if examples:
        prompt += "### Ejemplos\n\n"
        for i, ex in enumerate(examples, 1):
            prompt += f"**Ejemplo {i}:**\n"
            prompt += f"Pregunta: {ex['question']}\n"
            prompt += f"SQL:\n```sql\n{ex['sql']}\n```\n\n"
    
    # Añadir pregunta
    prompt += "### Tarea\n\n"
    prompt += f"Pregunta: {question}\n\n"
    prompt += "Genera una query SQL para responder esta pregunta.\n"
    prompt += "Devuelve SOLO el SQL, sin explicaciones.\n\n"
    prompt += "SQL:\n"
    
    return prompt


def format_react_prompt(
    question: str,
    tools: List[Dict],
    history: Optional[List[Dict]] = None
) -> str:
    """
    Genera un prompt estilo ReAct (Reasoning + Acting).
    
    Args:
        question: Pregunta del usuario
        tools: Lista de herramientas disponibles
        history: Historial previo de acciones/observaciones
        
    Returns:
        Prompt ReAct formateado
        
    Ejemplo:
        >>> tools = [
        ...     {"name": "query_db", "description": "Consulta base de datos"}
        ... ]
        >>> prompt = format_react_prompt("¿Cuántos usuarios hay?", tools)
    """
    prompt = "Eres un agente de IA con acceso a herramientas.\n\n"
    
    # Mostrar herramientas
    prompt += "### Herramientas Disponibles\n\n"
    for tool in tools:
        name = tool.get('name', '')
        desc = tool.get('description', '')
        params = tool.get('parameters', {})
        
        prompt += f"**{name}**: {desc}\n"
        if params:
            prompt += f"  Parámetros: {json.dumps(params, ensure_ascii=False)}\n"
        prompt += "\n"
    
    # Añadir historial si existe
    if history:
        prompt += "### Acciones Previas\n\n"
        for item in history:
            if 'thought' in item:
                prompt += f"Pensamiento: {item['thought']}\n"
            if 'action' in item:
                prompt += f"Acción: {item['action']}\n"
            if 'observation' in item:
                prompt += f"Observación: {item['observation']}\n"
            prompt += "\n"
    
    # Añadir pregunta
    prompt += "### Pregunta\n\n"
    prompt += f"{question}\n\n"
    
    # Formato de respuesta
    prompt += "### Formato de Respuesta\n\n"
    prompt += "Responde con:\n"
    prompt += "```\n"
    prompt += "Pensamiento: [tu razonamiento]\n"
    prompt += "Acción: [nombre_herramienta]\n"
    prompt += "Entrada de Acción: {\"parametro\": \"valor\"}\n"
    prompt += "```\n\n"
    prompt += "Tu respuesta:\n"
    
    return prompt


def format_chat_prompt(
    messages: List[Dict[str, str]],
    system_prompt: Optional[str] = None
) -> str:
    """
    Formatea mensajes de chat en un prompt unificado.
    
    Args:
        messages: Lista de mensajes [{"role": "user/assistant", "content": "..."}]
        system_prompt: Prompt del sistema opcional
        
    Returns:
        Prompt formateado
        
    Ejemplo:
        >>> messages = [
        ...     {"role": "user", "content": "Hola"},
        ...     {"role": "assistant", "content": "¡Hola! ¿Cómo estás?"},
        ...     {"role": "user", "content": "Bien, ¿y tú?"}
        ... ]
        >>> prompt = format_chat_prompt(messages)
    """
    prompt = ""
    
    if system_prompt:
        prompt += f"<|system|>\n{system_prompt}\n\n"
    
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        
        if role == "system":
            prompt += f"<|system|>\n{content}\n\n"
        elif role == "user":
            prompt += f"<|user|>\n{content}\n\n"
        elif role == "assistant":
            prompt += f"<|assistant|>\n{content}\n\n"
    
    # Añadir tag de asistente al final para que el modelo continúe
    if not messages or messages[-1]["role"] != "assistant":
        prompt += "<|assistant|>\n"
    
    return prompt

