"""
Backend/transformers_v2.py

Backend mejorado usando HuggingFace Transformers con soporte para:
- GPTQ (más rápido que BitsAndBytes)
- AWQ (el más rápido para GPU)
- BitsAndBytes 4bit/8bit (fallback)
- FP16/FP32
- JSON mode integrado

Optimizado para RTX 3050 (4GB VRAM)
"""

import asyncio
import time
import torch
import json
from typing import Dict, List, Optional, Any
from pathlib import Path

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    GPTQConfig
)

from .base import (
    ModelBackend,
    GenerationConfig,
    ModelResponse,
    ModelInfo,
    ToolCall
)

# Importar utils
try:
    from .utils.json_handler import JSONHandler
    HAS_JSON_HANDLER = True
except ImportError:
    HAS_JSON_HANDLER = False
    print("⚠️  utils.json_handler no encontrado. JSONHandler deshabilitado.")


class TransformersBackend(ModelBackend):
    """
    Backend optimizado usando HuggingFace Transformers.
    
    Soporta múltiples métodos de cuantización:
    - GPTQ: Rápido, buena calidad (GPU)
    - AWQ: Más rápido, mejor calidad (GPU)
    - BitsAndBytes: Compatible, más lento (GPU)
    - FP16: Sin cuantización (GPU/CPU)
    
    Características:
    - Auto device_map para manejo de memoria
    - JSON mode con validación
    - Detección automática de capacidades del modelo
    - Error handling robusto
    """
    
    SUPPORTED_QUANTIZATIONS = ["gptq", "awq", "4bit", "8bit", None]
    
    def __init__(self, model_config: Dict[str, Any]):
        """
        Args:
            model_config: Dict con configuración del modelo
                Required:
                    - repo: HuggingFace repo ID
                Optional:
                    - quantization: tipo de cuantización
                    - context_length: tamaño contexto
                    - trust_remote_code: si permite código custom
                    - verbose: modo debug
        """
        super().__init__(model_config)
        
        self.repo_id = model_config.get("repo")
        if not self.repo_id:
            raise ValueError(f"'repo' no especificado para {self.model_name}")
        
        self.quantization = model_config.get("quantization")
        self.device = "cpu"
        self.trust_remote_code = model_config.get("trust_remote_code", True)
        self.verbose = model_config.get("verbose", False)
        
        # Validar cuantización
        if self.quantization not in self.SUPPORTED_QUANTIZATIONS:
            raise ValueError(
                f"Cuantización no soportada: {self.quantization}. "
                f"Soportadas: {self.SUPPORTED_QUANTIZATIONS}"
            )
        
        # Flags internos
        self._use_awq = False
        self._use_gptq = False
        self._use_bnb = False
        
        # JSON handler
        self.json_handler = JSONHandler(verbose=self.verbose) if HAS_JSON_HANDLER else None
        
        self.context_length = model_config.get("context_length", 4096)
    
    async def load(self) -> None:
        """
        Carga el modelo con la cuantización especificada.
        
        Proceso:
        1. Detecta GPU disponible
        2. Carga tokenizer
        3. Configura cuantización según tipo
        4. Carga modelo con optimizaciones
        5. Mueve a device si es necesario
        """
        if self.is_loaded:
            if self.verbose:
                print(f"Modelo {self.model_name} ya cargado")
            return
        
        print(f"⏳ Cargando {self.model_name} con Transformers...")
        
        def _load():
            # 1. Detectar device
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            if self.verbose:
                if self.device == "cuda":
                    print(f"  ✓ GPU detectada: {torch.cuda.get_device_name(0)}")
                    print(f"  ✓ VRAM disponible: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
                else:
                    print("  ⚠️  GPU no disponible, usando CPU")
            
            # 2. Cargar tokenizer
            if self.verbose:
                print(f"  📥 Cargando tokenizer desde {self.repo_id}...")
            
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.repo_id,
                trust_remote_code=self.trust_remote_code
            )
            
            # Asegurar pad_token
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            
            if self.verbose:
                print("  ✓ Tokenizer cargado")
            
            # 3. Configurar cuantización
            model_kwargs = {
                "trust_remote_code": self.trust_remote_code,
                "low_cpu_mem_usage": True,
            }
            
            # ============================================
            # GPTQ (recomendado para RTX 3050)
            # ============================================
            if self.quantization == "gptq":
                if self.device != "cuda":
                    print("  ⚠️  GPTQ requiere GPU, cambiando a BitsAndBytes 4bit")
                    self.quantization = "4bit"
                else:
                    if self.verbose:
                        print("  🔧 Configurando GPTQ 4-bit...")
                    
                    try:
                        gptq_config = GPTQConfig(
                            bits=4,
                            use_exllama=True,  # Acelera inferencia
                            exllama_config={"version": 2},  # v2 es más rápido
                        )
                        model_kwargs["quantization_config"] = gptq_config
                        model_kwargs["device_map"] = "auto"
                        self._use_gptq = True
                        
                        if self.verbose:
                            print("  ✓ GPTQ configurado (ExLlamaV2 enabled)")
                        
                    except ImportError:
                        print("  ⚠️  AutoGPTQ no instalado. Instalar: pip install auto-gptq")
                        print("  ⚠️  Fallback a BitsAndBytes 4bit")
                        self.quantization = "4bit"
            
            # ============================================
            # AWQ (el más rápido, pero menos modelos)
            # ============================================
            if self.quantization == "awq":
                if self.device != "cuda":
                    print("  ⚠️  AWQ requiere GPU, cambiando a BitsAndBytes 4bit")
                    self.quantization = "4bit"
                else:
                    if self.verbose:
                        print("  🔧 Configurando AWQ 4-bit...")
                    
                    try:
                        from awq import AutoAWQForCausalLM
                        
                        # AWQ usa su propio loader
                        if self.verbose:
                            print("  📥 Cargando modelo AWQ...")
                        
                        self._model = AutoAWQForCausalLM.from_quantized(
                            self.repo_id,
                            fuse_layers=True,  # Fusiona capas para más velocidad
                            device_map="auto"
                        )
                        self._use_awq = True
                        
                        # Obtener context_length del config
                        self.context_length = getattr(
                            self._model.config,
                            "max_position_embeddings",
                            4096
                        )
                        
                        if self.verbose:
                            print("  ✓ AWQ modelo cargado (fused layers enabled)")
                        
                        return self._tokenizer, self._model
                        
                    except ImportError:
                        print("  ⚠️  AutoAWQ no instalado. Instalar: pip install autoawq")
                        print("  ⚠️  Fallback a BitsAndBytes 4bit")
                        self.quantization = "4bit"
                    except Exception as e:
                        print(f"  ⚠️  Error cargando AWQ: {e}")
                        print("  ⚠️  Fallback a BitsAndBytes 4bit")
                        self.quantization = "4bit"
            
            # ============================================
            # BitsAndBytes 4-bit (fallback compatible)
            # ============================================
            if self.quantization == "4bit" and self.device == "cuda":
                if self.verbose:
                    print("  🔧 Configurando BitsAndBytes 4-bit (NF4)...")
                
                try:
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,  # Ahorra memoria
                        bnb_4bit_quant_type="nf4"  # Normal Float 4
                    )
                    model_kwargs["quantization_config"] = quantization_config
                    model_kwargs["device_map"] = "auto"
                    self._use_bnb = True
                    
                    if self.verbose:
                        print("  ✓ BitsAndBytes 4bit configurado")
                    
                except ImportError:
                    print("  ⚠️  BitsAndBytes no instalado. Instalar: pip install bitsandbytes")
                    print("  ⚠️  Fallback a FP16")
                    self.quantization = None
            
            # ============================================
            # BitsAndBytes 8-bit
            # ============================================
            elif self.quantization == "8bit" and self.device == "cuda":
                if self.verbose:
                    print("  🔧 Configurando BitsAndBytes 8-bit...")
                
                try:
                    model_kwargs["load_in_8bit"] = True
                    model_kwargs["device_map"] = "auto"
                    self._use_bnb = True
                    
                    if self.verbose:
                        print("  ✓ BitsAndBytes 8bit configurado")
                    
                except ImportError:
                    print("  ⚠️  BitsAndBytes no instalado")
                    print("  ⚠️  Fallback a FP16")
                    self.quantization = None
            
            # ============================================
            # Sin cuantización (FP16/FP32)
            # ============================================
            if self.quantization is None:
                if self.device == "cuda":
                    model_kwargs["torch_dtype"] = torch.float16
                    model_kwargs["device_map"] = "auto"
                    if self.verbose:
                        print("  🔧 Usando FP16 (sin cuantización)")
                else:
                    model_kwargs["torch_dtype"] = torch.float32
                    if self.verbose:
                        print("  🔧 Usando FP32 (CPU)")
            
            # 4. Cargar modelo (si no es AWQ, que ya se cargó)
            if not self._use_awq:
                if self.verbose:
                    print(f"  📥 Cargando modelo desde {self.repo_id}...")
                    print(f"  📦 Config: {model_kwargs}")
                
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.repo_id,
                    **model_kwargs
                )
                
                # Obtener context_length
                self.context_length = getattr(
                    self._model.config,
                    "max_position_embeddings",
                    4096
                )
                
                # Mover a device si no usa device_map
                if "device_map" not in model_kwargs:
                    self._model.to(self.device)
                    if self.verbose:
                        print(f"  ✓ Modelo movido a {self.device}")
            
            if self.verbose:
                print(f"  ✓ Modelo cargado en {self.device}")
                print(f"  ✓ Context length: {self.context_length} tokens")
            
            return self._tokenizer, self._model
        
        # Cargar en thread para no bloquear
        self._tokenizer, self._model = await asyncio.to_thread(_load)
        self.is_loaded = True
        
        quant_str = self.quantization or "fp16"
        print(f"✅ {self.model_name} listo ({self.device}, {quant_str})")
    
    async def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        tools: Optional[List[Dict]] = None,
        json_mode: bool = False  # ← NUEVO parámetro
    ) -> ModelResponse:
        """
        Genera texto usando el modelo cargado.
        
        Args:
            prompt: Texto de entrada
            config: Configuración de generación
            tools: Herramientas disponibles (para tool calling)
            json_mode: Si True, fuerza salida JSON válida
            
        Returns:
            ModelResponse con texto generado y metadata
        """
        if not self.is_loaded:
            raise RuntimeError(f"Modelo {self.model_name} no cargado. Llamar load() primero.")
        
        if config is None:
            config = GenerationConfig()
        
        start_time = time.time()
        
        def _generate():
            # Tokenizar
            inputs = self._tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.context_length
            )
            
            # Mover a device
            if self.device == "cuda":
                inputs = inputs.to(self.device)
            
            # Parámetros de generación
            gen_kwargs = {
                "max_new_tokens": config.max_tokens,
                "temperature": config.temperature,
                "top_p": config.top_p,
                "top_k": config.top_k,
                "do_sample": config.temperature > 0,
                "pad_token_id": self._tokenizer.eos_token_id,
                "eos_token_id": self._tokenizer.eos_token_id,
            }
            
            # ✅ JSON mode: forzar determinismo
            if json_mode:
                gen_kwargs["temperature"] = 0.0
                gen_kwargs["top_p"] = 1.0
                gen_kwargs["top_k"] = 1
                gen_kwargs["do_sample"] = False
                
                if self.verbose:
                    print("  🔒 JSON mode activado (generación determinista)")
            
            # Generar
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    **gen_kwargs
                )
            
            # Decodificar solo nuevos tokens
            input_length = inputs.input_ids.shape[1]
            generated_tokens = outputs[0][input_length:]
            generated_text = self._tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True
            )
            
            # Contar tokens
            prompt_tokens = inputs.input_ids.shape[1]
            completion_tokens = len(generated_tokens)
            
            return generated_text, prompt_tokens, completion_tokens
        
        # Ejecutar generación en thread
        generated_text, prompt_tokens, completion_tokens = await asyncio.to_thread(_generate)
        
        latency_ms = (time.time() - start_time) * 1000
        
        # ✅ Validar/reparar JSON si es modo JSON
        if json_mode and self.json_handler:
            result = self.json_handler.parse(generated_text)
            if result.success:
                generated_text = result.repaired
                if self.verbose and result.method != "none":
                    print(f"  ✓ JSON reparado (método: {result.method})")
            else:
                if self.verbose:
                    print(f"  ⚠️  JSON inválido: {result.error}")
        
        # Parsear tool calls si se proporcionaron tools
        tool_calls = []
        if tools:
            tool_calls = self._parse_tool_calls(generated_text, tools)
        
        return ModelResponse(
            content=generated_text,
            tool_calls=tool_calls,
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            },
            latency_ms=latency_ms,
            backend_name=f"transformers-{self.quantization or 'fp16'}",
            raw_response=None
        )
    
    def unload(self) -> None:
        """
        Libera memoria GPU/CPU.
        
        Importante llamar esto cuando termines de usar el modelo
        para liberar VRAM.
        """
        if self._model is not None:
            del self._model
            self._model = None
        
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        
        # Limpiar cache de GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if self.verbose:
                print("  🗑️  Cache GPU limpiado")
        
        self.is_loaded = False
        print(f"🗑️  {self.model_name} descargado de memoria")
    
    def get_info(self) -> ModelInfo:
        """
        Obtiene información sobre el modelo cargado.
        
        Returns:
            ModelInfo con detalles del modelo
        """
        memory_mb = 0.0
        
        if self._model is not None:
            # Intentar obtener memoria real
            if hasattr(self._model, "get_memory_footprint"):
                memory_mb = self._model.get_memory_footprint() / (1024 ** 2)
            elif torch.cuda.is_available():
                memory_mb = torch.cuda.memory_allocated() / (1024 ** 2)
        
        return ModelInfo(
            name=self.model_name,
            backend="transformers-v2",
            device=self.device,
            memory_usage_mb=memory_mb,
            quantization=self.quantization,
            context_length=self.context_length
        )
    
    def supports_tool_calling(self) -> bool:
        """
        Verifica si el modelo soporta tool calling.
        
        Chequea:
        - Nombre del modelo contiene "tool", "function", "agent"
        - Config del modelo tiene tool calling habilitado
        
        Returns:
            True si soporta tool calling
        """
        # Check básico por nombre
        model_name_lower = self.model_name.lower()
        if any(kw in model_name_lower for kw in ["tool", "function", "agent"]):
            return True
        
        # Check por config si el modelo está cargado
        if self._model is not None and hasattr(self._model, "config"):
            config = self._model.config
            # Algunos modelos tienen este flag
            if hasattr(config, "tool_use") and config.tool_use:
                return True
        
        return False
    
    def _count_tokens(self, text: str) -> int:
        """
        Cuenta tokens usando el tokenizer real.
        
        Args:
            text: Texto a contar
            
        Returns:
            Número de tokens
        """
        if self._tokenizer:
            return len(self._tokenizer.encode(text))
        # Fallback: estimación (1 token ≈ 4 chars)
        return len(text) // 4


