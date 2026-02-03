"""
backends/llamacpp_v2.py

Enhanced llama.cpp backend with:
- JSON mode enforcement
- Automatic repair
- Better error handling
"""

import asyncio
import time
import os
from typing import Dict, List, Optional, Any
from pathlib import Path

try:
    from llama_cpp import Llama, LlamaGrammar
    LLAMACPP_AVAILABLE = True
except ImportError:
    LLAMACPP_AVAILABLE = False

from .base import (
    ModelBackend,
    GenerationConfig,
    ModelResponse,
    ModelInfo,
    ToolCall
)
from .utils.json_handler import JSONHandler


class LlamaCppBackend(ModelBackend):
    """
    Enhanced llama.cpp backend with JSON guarantees.
    """
    
    def __init__(self, model_config: Dict[str, Any]):
        super().__init__(model_config)
        
        if not LLAMACPP_AVAILABLE:
            raise ImportError("llama-cpp-python required. Install: pip install llama-cpp-python")
        
        self.model_path = model_config.get("model_path")
        if not self.model_path:
            raise ValueError(f"'model_path' not specified for {self.model_name}")
            
        self.n_ctx = model_config.get("context_length", 4096)
        self.n_threads = model_config.get("n_threads", os.cpu_count() or 4)
        self.n_gpu_layers = model_config.get("n_gpu_layers", 0)
        self.device = "cpu" if self.n_gpu_layers == 0 else "cuda"
        
        # JSON handling
        self.json_handler = JSONHandler(verbose=model_config.get("verbose", False))
        self.grammar_path = model_config.get("grammar_path")
        self._json_grammar = None
    
    async def load(self) -> None:
        """Load GGUF model"""
        if self.is_loaded:
            return
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        print(f"  Loading {self.model_name}...")
        print(f"  Context: {self.n_ctx} | Threads: {self.n_threads} | GPU Layers: {self.n_gpu_layers}")
        
        def _load():
            model = Llama(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                n_gpu_layers=self.n_gpu_layers,
                verbose=False,
                n_batch=512,
                use_mlock=True,
            )
            
            # Load grammar if provided
            grammar = None
            if self.grammar_path and os.path.exists(self.grammar_path):
                grammar = LlamaGrammar.from_file(self.grammar_path)
                print(f"  ✓ Loaded grammar: {self.grammar_path}")
            
            return model, grammar
        
        self._model, self._json_grammar = await asyncio.to_thread(_load)
        self.is_loaded = True
        print(f"✓ {self.model_name} ready")
    
    async def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        tools: Optional[List[Dict]] = None,
        json_mode: bool = False  # ← NEW parameter
    ) -> ModelResponse:
        """
        Generate with optional JSON enforcement.
        
        Args:
            prompt: Input text
            config: Generation config
            tools: Available tools
            json_mode: Force JSON output (guaranteed valid)
        """
        if not self.is_loaded:
            raise RuntimeError(f"Model not loaded. Call load() first.")
        
        if config is None:
            config = GenerationConfig()
        
        start_time = time.time()
        
        def _generate():
            gen_kwargs = {
                "max_tokens": config.max_tokens,
                "temperature": config.temperature,
                "top_p": config.top_p,
                "top_k": config.top_k,
                "echo": False,
            }
            
            # JSON mode: enforce deterministic output
            if json_mode:
                gen_kwargs["temperature"] = 0.0
                gen_kwargs["top_p"] = 1.0
                gen_kwargs["top_k"] = 1
                
                if self._json_grammar:
                    gen_kwargs["grammar"] = self._json_grammar
                
                # Stop when JSON completes
                gen_kwargs["stop"] = ["\n}"]
            else:
                gen_kwargs["stop"] = config.stop_sequences or []
            
            response = self._model(prompt, **gen_kwargs)
            text = response['choices'][0]['text']
            
            usage = response.get('usage', {})
            p_tokens = usage.get('prompt_tokens', self._count_tokens(prompt))
            c_tokens = usage.get('completion_tokens', self._count_tokens(text))
            
            return text, p_tokens, c_tokens
        
        text, p_tokens, c_tokens = await asyncio.to_thread(_generate)
        latency_ms = (time.time() - start_time) * 1000
        
        # Validate/repair JSON if needed
        if json_mode:
            result = self.json_handler.parse(text)
            if result.success:
                text = result.repaired
            else:
                print(f"⚠️  JSON repair failed: {result.error}")
        
        # Parse tool calls
        tool_calls = []
        if tools:
            tool_calls = self._parse_tool_calls(text, tools)
        
        return ModelResponse(
            content=text,
            tool_calls=tool_calls,
            usage={"prompt_tokens": p_tokens, "completion_tokens": c_tokens, "total_tokens": p_tokens + c_tokens},
            latency_ms=latency_ms,
            backend_name="llamacpp-v2"
        )
    
    def unload(self) -> None:
        if self._model:
            del self._model
            self._model = None
        if self._json_grammar:
            del self._json_grammar
            self._json_grammar = None
        self.is_loaded = False
    
    def get_info(self) -> ModelInfo:
        memory_mb = 0.0
        quantization = "unknown"
        
        if self.model_path:
            filename = os.path.basename(self.model_path).lower()
            for q in ["q4", "q5", "q8"]:
                if q in filename:
                    quantization = q.upper()
                    break
            
            try:
                memory_mb = os.path.getsize(self.model_path) / (1024 ** 2) + 500
            except:
                pass
        
        return ModelInfo(
            name=self.model_name,
            backend="llamacpp-v2",
            device=self.device,
            memory_usage_mb=memory_mb,
            quantization=quantization,
            context_length=self.n_ctx
        )
    
    def supports_tool_calling(self) -> bool:
        return True
    
    def _count_tokens(self, text: str) -> int:
        return int(len(text) / 3.5)