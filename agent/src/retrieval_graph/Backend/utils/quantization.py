"""
Backend/utils/quantization.py

Utilidades para cuantización y cálculo de VRAM.

Funciones principales:
    - get_optimal_layers_for_gpu: Calcula layers GPU óptimas
    - estimate_vram_usage: Estima VRAM necesaria
    - detect_quantization_from_filename: Detecta tipo de cuantización

Casos de uso:
    1. Saber cuántas layers poner en GPU
    2. Verificar si un modelo cabe en VRAM
    3. Detectar cuantización de archivos GGUF

Optimizado para RTX 3050 (4GB VRAM)
"""

import re
from typing import Optional, Dict, Tuple
from pathlib import Path


# ==========================================
# TABLAS DE REFERENCIA
# ==========================================

# VRAM por layer para modelos comunes (en MB)
# Formato: (modelo, cuantización) -> MB por layer
VRAM_PER_LAYER = {
    ("mistral-7b", "Q4_K_M"): 145,   # ~145 MB por layer
    ("mistral-7b", "Q5_K_M"): 175,   # ~175 MB por layer
    ("mistral-7b", "Q8_0"): 280,     # ~280 MB por layer
    ("llama-3-8b", "Q4_K_M"): 165,
    ("llama-3-8b", "Q5_K_M"): 200,
    ("llama-3-8b", "Q8_0"): 320,
}

# Configuraciones recomendadas para RTX 3050 (4GB)
RTX_3050_CONFIGS = {
    "mistral-7b": {
        "Q4_K_M": {"layers": 24, "expected_vram_mb": 3480},
        "Q5_K_M": {"layers": 20, "expected_vram_mb": 3500},
        "Q8_0": {"layers": 0, "expected_vram_mb": 0, "note": "No cabe, usar CPU"},
    },
    "llama-3-8b": {
        "Q4_K_M": {"layers": 22, "expected_vram_mb": 3630},
        "Q5_K_M": {"layers": 18, "expected_vram_mb": 3600},
        "Q8_0": {"layers": 0, "expected_vram_mb": 0, "note": "No cabe, usar CPU"},
    }
}


def get_optimal_layers_for_gpu(
    model_name: str,
    quantization: str,
    vram_gb: float = 4.0,
    safety_margin_mb: float = 500
) -> Dict[str, any]:
    """
    Calcula el número óptimo de layers para GPU.
    
    Args:
        model_name: Nombre del modelo (mistral-7b, llama-3-8b, etc.)
        quantization: Tipo de cuantización (Q4_K_M, Q5_K_M, Q8_0)
        vram_gb: VRAM disponible en GB
        safety_margin_mb: Margen de seguridad en MB (para overhead)
        
    Returns:
        Dict con:
            - optimal_layers: Número de layers recomendadas
            - expected_vram_mb: VRAM que usará
            - fits_in_gpu: Si cabe completamente
            - note: Notas adicionales
            
    Ejemplo:
        >>> result = get_optimal_layers_for_gpu("mistral-7b", "Q4_K_M", vram_gb=4.0)
        >>> print(result["optimal_layers"])  # → 24
    """
    model_lower = model_name.lower()
    quant_upper = quantization.upper()
    vram_mb = vram_gb * 1024
    
    # Buscar config pre-calculada
    if vram_gb == 4.0:  # RTX 3050
        for model_key in RTX_3050_CONFIGS:
            if model_key in model_lower:
                if quant_upper in RTX_3050_CONFIGS[model_key]:
                    config = RTX_3050_CONFIGS[model_key][quant_upper]
                    return {
                        "optimal_layers": config["layers"],
                        "expected_vram_mb": config["expected_vram_mb"],
                        "fits_in_gpu": config["layers"] > 0,
                        "note": config.get("note", "Configuración óptima para RTX 3050")
                    }
    
    # Cálculo dinámico si no hay config pre-calculada
    # Buscar MB por layer
    mb_per_layer = None
    for (model_key, quant_key), mb in VRAM_PER_LAYER.items():
        if model_key in model_lower and quant_key == quant_upper:
            mb_per_layer = mb
            break
    
    if mb_per_layer is None:
        # Estimación genérica basada en cuantización
        if "Q4" in quant_upper:
            mb_per_layer = 150
        elif "Q5" in quant_upper:
            mb_per_layer = 180
        elif "Q8" in quant_upper:
            mb_per_layer = 300
        else:
            mb_per_layer = 200  # Fallback
    
    # Calcular layers que caben
    available_vram = vram_mb - safety_margin_mb
    optimal_layers = int(available_vram / mb_per_layer)
    
    # Limitar a 32 layers (máximo común para Mistral/Llama)
    optimal_layers = min(optimal_layers, 32)
    
    expected_vram = optimal_layers * mb_per_layer + safety_margin_mb
    
    return {
        "optimal_layers": optimal_layers,
        "expected_vram_mb": expected_vram,
        "fits_in_gpu": optimal_layers >= 32,  # Si caben todas
        "note": f"Estimación: {mb_per_layer}MB por layer"
    }


def estimate_vram_usage(
    model_name: str,
    quantization: str,
    n_gpu_layers: int
) -> Dict[str, float]:
    """
    Estima VRAM que usará una configuración.
    
    Args:
        model_name: Nombre del modelo
        quantization: Tipo de cuantización
        n_gpu_layers: Número de layers en GPU
        
    Returns:
        Dict con:
            - vram_mb: VRAM estimada
            - vram_gb: VRAM en GB
            - safe_for_4gb: Si es seguro para 4GB
            
    Ejemplo:
        >>> usage = estimate_vram_usage("mistral-7b", "Q4_K_M", 24)
        >>> print(f"{usage['vram_gb']:.1f} GB")  # → 3.4 GB
    """
    model_lower = model_name.lower()
    quant_upper = quantization.upper()
    
    # Buscar MB por layer
    mb_per_layer = None
    for (model_key, quant_key), mb in VRAM_PER_LAYER.items():
        if model_key in model_lower and quant_key == quant_upper:
            mb_per_layer = mb
            break
    
    if mb_per_layer is None:
        # Estimación genérica
        if "Q4" in quant_upper:
            mb_per_layer = 150
        elif "Q5" in quant_upper:
            mb_per_layer = 180
        elif "Q8" in quant_upper:
            mb_per_layer = 300
        else:
            mb_per_layer = 200
    
    # Calcular VRAM
    vram_mb = (n_gpu_layers * mb_per_layer) + 500  # +500MB overhead
    vram_gb = vram_mb / 1024
    
    return {
        "vram_mb": vram_mb,
        "vram_gb": vram_gb,
        "safe_for_4gb": vram_gb < 3.8  # Margen de seguridad
    }


def detect_quantization_from_filename(filename: str) -> Optional[str]:
    """
    Detecta tipo de cuantización desde nombre de archivo GGUF.
    
    Args:
        filename: Nombre del archivo (ej: mistral-7b-q4_k_m.gguf)
        
    Returns:
        Tipo de cuantización (Q4_K_M, Q5_K_M, etc.) o None
        
    Ejemplos:
        >>> detect_quantization_from_filename("mistral-7b-Q4_K_M.gguf")
        'Q4_K_M'
        
        >>> detect_quantization_from_filename("model.q5_k_s.gguf")
        'Q5_K_S'
    """
    patterns = [
        r'Q([2-8])_K_M',
        r'Q([2-8])_K_S',
        r'Q([2-8])_K_L',
        r'Q([2-8])_0',
        r'Q([2-8])',
    ]
    
    filename_upper = filename.upper()
    
    for pattern in patterns:
        match = re.search(pattern, filename_upper)
        if match:
            return match.group(0)
    
    return None


def get_model_info_from_path(model_path: str) -> Dict[str, any]:
    """
    Extrae información del modelo desde su path.
    
    Args:
        model_path: Path al archivo GGUF
        
    Returns:
        Dict con info detectada
        
    Ejemplo:
        >>> info = get_model_info_from_path("/models/mistral-7b-q4_k_m.gguf")
        >>> print(info)
        {
            'quantization': 'Q4_K_M',
            'size_mb': 4100,
            'model_family': 'mistral-7b'
        }
    """
    path = Path(model_path)
    filename = path.name.lower()
    
    info = {
        "quantization": detect_quantization_from_filename(filename),
        "size_mb": None,
        "model_family": None
    }
    
    # Detectar familia del modelo
    if "mistral" in filename:
        if "7b" in filename:
            info["model_family"] = "mistral-7b"
    elif "llama" in filename:
        if "8b" in filename or "7b" in filename:
            info["model_family"] = "llama-3-8b"
    
    # Tamaño del archivo
    try:
        if path.exists():
            info["size_mb"] = path.stat().st_size / (1024 ** 2)
    except Exception:
        pass
    
    return info


def recommend_config_for_rtx3050(model_path: str) -> Dict[str, any]:
    """
    Recomienda configuración óptima para RTX 3050.
    
    Args:
        model_path: Path al modelo GGUF
        
    Returns:
        Configuración recomendada completa
        
    Ejemplo:
        >>> config = recommend_config_for_rtx3050("/models/mistral-7b-q4.gguf")
        >>> print(config["n_gpu_layers"])  # → 24
    """
    info = get_model_info_from_path(model_path)
    
    if not info["quantization"] or not info["model_family"]:
        return {
            "n_gpu_layers": 0,
            "note": "No se pudo detectar modelo/cuantización, usar CPU",
            "confidence": "low"
        }
    
    result = get_optimal_layers_for_gpu(
        model_name=info["model_family"],
        quantization=info["quantization"],
        vram_gb=4.0
    )
    
    return {
        "n_gpu_layers": result["optimal_layers"],
        "n_threads": 8,
        "expected_vram_gb": result["expected_vram_mb"] / 1024,
        "note": result["note"],
        "confidence": "high"
    }
