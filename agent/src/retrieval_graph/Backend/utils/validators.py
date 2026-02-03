"""
Backend/utils/validators.py

Validadores para diferentes tipos de salidas y configuraciones.

Funciones:
    - validate_sql_query: Valida SQL (seguridad básica)
    - validate_json_schema: Valida campos requeridos en JSON
    - validate_model_config: Valida configuración YAML

Uso:
    from Backend.utils.validators import validate_sql_query
    
    is_valid, error = validate_sql_query("SELECT * FROM users")
    if not is_valid:
        print(f"Query inválida: {error}")
"""

from typing import Dict, List, Tuple, Any, Optional
import re


def validate_sql_query(query: str, allow_writes: bool = False) -> Tuple[bool, Optional[str]]:
    """
    Validación básica de queries SQL.
    
    Chequea:
    - Query no vacía
    - Operaciones peligrosas (DROP, DELETE, etc.)
    - Sintaxis básica
    - Paréntesis balanceados
    
    Args:
        query: Query SQL a validar
        allow_writes: Si True, permite INSERT/UPDATE/DELETE
        
    Returns:
        (is_valid, error_message)
        
    Ejemplo:
        >>> is_valid, err = validate_sql_query("DROP TABLE users;")
        >>> print(is_valid)  # → False
        >>> print(err)       # → "Operación peligrosa: DROP"
    """
    query_stripped = query.strip()
    query_upper = query_stripped.upper()
    
    # Chequeo 1: No vacío
    if not query_stripped:
        return False, "Query vacía"
    
    # Chequeo 2: Operaciones peligrosas
    dangerous_ops = ['DROP', 'TRUNCATE', 'ALTER', 'CREATE']
    for op in dangerous_ops:
        if op in query_upper:
            return False, f"Operación peligrosa: {op}"
    
    # Chequeo 3: Escrituras (si no están permitidas)
    if not allow_writes:
        write_ops = ['DELETE', 'UPDATE', 'INSERT']
        for op in write_ops:
            if op in query_upper and 'SELECT' not in query_upper:
                return False, f"Operación de escritura no permitida: {op}"
    
    # Chequeo 4: Debe tener palabra clave SQL válida
    valid_keywords = ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'WITH']
    if not any(kw in query_upper for kw in valid_keywords):
        return False, "No se encontró palabra clave SQL válida"
    
    # Chequeo 5: Paréntesis balanceados
    if query.count('(') != query.count(')'):
        return False, "Paréntesis desbalanceados"
    
    # Chequeo 6: Comillas balanceadas
    single_quotes = query.count("'")
    if single_quotes % 2 != 0:
        return False, "Comillas simples desbalanceadas"
    
    return True, None


def validate_json_schema(
    data: Dict[str, Any],
    required_fields: List[str],
    optional_fields: Optional[List[str]] = None,
    strict: bool = False
) -> Tuple[bool, List[str]]:
    """
    Valida que un dict JSON tenga los campos requeridos.
    
    Args:
        data: Dict a validar
        required_fields: Lista de campos obligatorios
        optional_fields: Lista de campos opcionales
        strict: Si True, no permite campos extra
        
    Returns:
        (is_valid, list_of_errors)
        
    Ejemplo:
        >>> data = {"answer": "Paris", "extra": 123}
        >>> is_valid, errors = validate_json_schema(
        ...     data,
        ...     required_fields=["answer"],
        ...     strict=True
        ... )
        >>> print(errors)  # → ["Campo no esperado: extra"]
    """
    errors = []
    
    # Chequear campos requeridos
    for field in required_fields:
        if field not in data:
            errors.append(f"Campo requerido faltante: {field}")
    
    # Si es strict, chequear campos extra
    if strict:
        allowed_fields = set(required_fields)
        if optional_fields:
            allowed_fields.update(optional_fields)
        
        for field in data.keys():
            if field not in allowed_fields:
                errors.append(f"Campo no esperado: {field}")
    
    return len(errors) == 0, errors


def validate_model_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Valida configuración de modelo desde YAML.
    
    Args:
        config: Dict de configuración
        
    Returns:
        (is_valid, list_of_errors)
        
    Ejemplo:
        >>> config = {"type": "llamacpp", "model_path": "/path/to/model"}
        >>> is_valid, errors = validate_model_config(config)
    """
    errors = []
    
    # Campo obligatorio: type
    if "type" not in config:
        errors.append("Campo 'type' es obligatorio")
    else:
        valid_types = ["llamacpp", "transformers"]
        if config["type"] not in valid_types:
            errors.append(f"Tipo inválido: {config['type']}. Válidos: {valid_types}")
    
    # Validación según tipo
    if config.get("type") == "llamacpp":
        if "model_path" not in config:
            errors.append("Backend 'llamacpp' requiere campo 'model_path'")
        
        # Validar n_gpu_layers si existe
        if "n_gpu_layers" in config:
            layers = config["n_gpu_layers"]
            if not isinstance(layers, int) or layers < 0 or layers > 100:
                errors.append(f"n_gpu_layers inválido: {layers}. Debe ser int entre 0-100")
    
    elif config.get("type") == "transformers":
        if "repo" not in config:
            errors.append("Backend 'transformers' requiere campo 'repo'")
    
    # Validar context_length si existe
    if "context_length" in config:
        ctx = config["context_length"]
        if not isinstance(ctx, int) or ctx < 512:
            errors.append(f"context_length inválido: {ctx}. Debe ser int >= 512")
    
    # Validar temperature si existe
    if "temperature" in config:
        temp = config["temperature"]
        if not isinstance(temp, (int, float)) or temp < 0 or temp > 2:
            errors.append(f"temperature inválida: {temp}. Debe ser float entre 0-2")
    
    return len(errors) == 0, errors


def validate_prompt_length(
    prompt: str,
    max_tokens: int,
    context_length: int
) -> Tuple[bool, Optional[str]]:
    """
    Valida que un prompt no exceda el contexto del modelo.
    
    Args:
        prompt: Texto del prompt
        max_tokens: Tokens máximos para generación
        context_length: Tamaño del contexto del modelo
        
    Returns:
        (is_valid, warning_message)
        
    Ejemplo:
        >>> is_valid, warn = validate_prompt_length(
        ...     "texto muy largo...",
        ...     max_tokens=512,
        ...     context_length=4096
        ... )
    """
    # Estimación: 1 token ≈ 4 caracteres
    estimated_tokens = len(prompt) // 4
    
    # Chequear si prompt + max_tokens excede contexto
    total_needed = estimated_tokens + max_tokens
    
    if total_needed > context_length:
        return False, (
            f"Prompt demasiado largo. "
            f"Estimado: {estimated_tokens} tokens + {max_tokens} generación = {total_needed} tokens. "
            f"Contexto máximo: {context_length}"
        )
    
    # Warning si usa >80% del contexto
    if total_needed > context_length * 0.8:
        return True, (
            f"Advertencia: usando {total_needed}/{context_length} tokens "
            f"({total_needed/context_length*100:.0f}% del contexto)"
        )
    
    return True, None


def sanitize_filename(filename: str) -> str:
    """
    Sanitiza un nombre de archivo para evitar problemas.
    
    Args:
        filename: Nombre de archivo original
        
    Returns:
        Nombre sanitizado
        
    Ejemplo:
        >>> sanitize_filename("my file/test.txt")
        'my_file_test.txt'
    """
    # Reemplazar caracteres problemáticos
    sanitized = re.sub(r'[^\w\-_\.]', '_', filename)
    
    # Limitar longitud
    if len(sanitized) > 255:
        # Mantener extensión
        name, ext = sanitized.rsplit('.', 1) if '.' in sanitized else (sanitized, '')
        sanitized = name[:250] + ('.' + ext if ext else '')
    
    return sanitized


def validate_file_path(path: str, must_exist: bool = True) -> Tuple[bool, Optional[str]]:
    """
    Valida que un path sea válido y opcionalmente exista.
    
    Args:
        path: Path a validar
        must_exist: Si True, el archivo debe existir
        
    Returns:
        (is_valid, error_message)
    """
    from pathlib import Path
    
    try:
        p = Path(path)
        
        # Chequear caracteres inválidos
        if any(c in str(p) for c in ['<', '>', '|', '\0']):
            return False, "Path contiene caracteres inválidos"
        
        # Chequear si debe existir
        if must_exist and not p.exists():
            return False, f"Archivo no encontrado: {path}"
        
        # Chequear permisos de lectura
        if must_exist and not p.is_file():
            return False, f"No es un archivo: {path}"
        
        return True, None
        
    except Exception as e:
        return False, f"Path inválido: {e}"
