"""
Backend/utils/__init__.py

Utilidades modulares para el backend de modelos.

Exports:
    - JSONHandler: Manejo robusto de JSON
    - format_json_prompt, format_sql_prompt: Formateo de prompts
    - get_optimal_layers_for_gpu: Cálculo de layers
    - validate_sql_query, validate_json_schema: Validadores
"""

from .json_handler import JSONHandler, parse_json_safe, ensure_json_complete
from .prompt_formatter import (
    format_json_prompt,
    format_sql_prompt,
    format_react_prompt
)
from .quantization import (
    get_optimal_layers_for_gpu,
    estimate_vram_usage,
    detect_quantization_from_filename
)
from .validators import (
    validate_sql_query,
    validate_json_schema,
    validate_model_config
)

__all__ = [
    # JSON handling
    "JSONHandler",
    "parse_json_safe",
    "ensure_json_complete",
    
    # Prompt formatting
    "format_json_prompt",
    "format_sql_prompt",
    "format_react_prompt",
    
    # Quantization
    "get_optimal_layers_for_gpu",
    "estimate_vram_usage",
    "detect_quantization_from_filename",
    
    # Validators
    "validate_sql_query",
    "validate_json_schema",
    "validate_model_config",
]