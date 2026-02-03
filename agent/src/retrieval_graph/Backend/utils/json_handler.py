import json
import re
from typing import Any, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class JSONRepairResult:
    """Resultado del intento de reparación de JSON"""
    success: bool
    data: Optional[Dict[str, Any]]
    original: str
    repaired: str
    method: str  
    error: Optional[str] = None


class JSONHandler:
    """
    Maneja parsing, validación y reparación de JSON.
    """
    
    def __init__(self, verbose: bool = False):
        """
        Args:
            verbose: Si True, imprime info de debug
        """
        self.verbose = verbose
        
        # Intentar importar json-repair
        try:
            from json_repair import repair_json
            self._repair_json = repair_json
            self._has_repair_lib = True
        except ImportError:
            self._repair_json = None
            self._has_repair_lib = False
            if verbose:
                print("⚠️  json-repair no instalado. Instalar: pip install json-repair")
    
    def parse(self, text: str) -> JSONRepairResult:
        """
        Parsea JSON con reparación automática si es necesario.
        
        Estrategias en orden:
        1. Parse directo
        2. Limpiar espacios y reintentar
        3. Reparación manual
        4. Reparación con librería
        
        Args:
            text: Texto que debería contener JSON
            
        Returns:
            JSONRepairResult con data parseada o error
        """
        original = text
        
        # Estrategia 1: Parse directo
        try:
            data = json.loads(text)
            return JSONRepairResult(
                success=True,
                data=data,
                original=original,
                repaired=text,
                method="none"
            )
        except json.JSONDecodeError as e:
            if self.verbose:
                print(f"  Parse inicial falló: {e}")
        
        # Estrategia 2: Limpiar espacios
        cleaned = text.strip()
        try:
            data = json.loads(cleaned)
            if self.verbose:
                print("  ✓ Parseado después de limpiar espacios")
            return JSONRepairResult(
                success=True,
                data=data,
                original=original,
                repaired=cleaned,
                method="cleanup"
            )
        except json.JSONDecodeError:
            pass
        
        # Estrategia 3: Reparación manual
        manually_repaired = self._repair_manual(cleaned)
        try:
            data = json.loads(manually_repaired)
            if self.verbose:
                print("  ✓ Reparado manualmente")
            return JSONRepairResult(
                success=True,
                data=data,
                original=original,
                repaired=manually_repaired,
                method="manual"
            )
        except json.JSONDecodeError:
            pass
        
        # Estrategia 4: Reparación con librería
        if self._has_repair_lib:
            try:
                lib_repaired = self._repair_json(cleaned)
                data = json.loads(lib_repaired)
                if self.verbose:
                    print("  ✓ Reparado con librería")
                return JSONRepairResult(
                    success=True,
                    data=data,
                    original=original,
                    repaired=lib_repaired,
                    method="library"
                )
            except Exception as e:
                if self.verbose:
                    print(f"  Reparación con librería falló: {e}")
        
        # Todas las estrategias fallaron
        return JSONRepairResult(
            success=False,
            data=None,
            original=original,
            repaired=manually_repaired,
            method="failed",
            error="No se pudo parsear ni reparar el JSON"
        )
    
    def _repair_manual(self, text: str) -> str:
        """
        Reparación manual para problemas comunes de truncamiento.
        
        Arregla:
        - Strings sin cerrar
        - Llaves/corchetes desbalanceados
        - Comas finales
        
        Args:
            text: JSON potencialmente roto
            
        Returns:
            JSON reparado (no garantiza validez)
        """
        result = text
        
        # 1. Balancear comillas
        quote_count = result.count('"')
        if quote_count % 2 != 0:
            # Añadir comilla de cierre
            result = result.rstrip() + '"'
        
        # 2. Balancear llaves
        open_braces = result.count('{')
        close_braces = result.count('}')
        if open_braces > close_braces:
            result = result + ('}' * (open_braces - close_braces))
        
        # 3. Balancear corchetes
        open_brackets = result.count('[')
        close_brackets = result.count(']')
        if open_brackets > close_brackets:
            result = result + (']' * (open_brackets - close_brackets))
        
        # 4. Eliminar comas finales antes de cerrar
        result = re.sub(r',(\s*[}\]])', r'\1', result)
        
        # 5. Asegurar que empieza con { o [
        result = result.lstrip()
        if not result.startswith(('{', '[')):
            for i, char in enumerate(result):
                if char in '{[':
                    result = result[i:]
                    break
        
        # 6. Cerrar strings incompletos al final
        # Ejemplo: "answer": "Paris → "answer": "Paris"
        result = re.sub(r':\s*"([^"]*?)$', r': "\1"', result)
        
        return result
    
    def extract_json_from_text(self, text: str) -> Optional[str]:
        """
        Extrae JSON de texto que puede contener otra cosa.
        
        Casos:
        - "La respuesta es: {...}"
        - "```json\n{...}\n```"
        - "El resultado: {...} Espero ayude!"
        
        Args:
            text: Texto raw
            
        Returns:
            String JSON extraído o None
        """
        # Patrón 1: JSON en code blocks
        code_block_pattern = r'```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```'
        match = re.search(code_block_pattern, text, re.DOTALL)
        if match:
            return match.group(1)
        
        # Patrón 2: Buscar primer JSON completo
        start_idx = text.find('{')
        if start_idx == -1:
            start_idx = text.find('[')
        
        if start_idx == -1:
            return None
        
        # Contar llaves para encontrar cierre
        balance = 0
        in_string = False
        escape_next = False
        
        for i in range(start_idx, len(text)):
            char = text[i]
            
            if escape_next:
                escape_next = False
                continue
            
            if char == '\\':
                escape_next = True
                continue
            
            if char == '"':
                in_string = not in_string
                continue
            
            if not in_string:
                if char in '{[':
                    balance += 1
                elif char in '}]':
                    balance -= 1
                    if balance == 0:
                        return text[start_idx:i+1]
        
        # No encontró JSON balanceado
        return text[start_idx:]


# ==========================================
# FUNCIONES DE CONVENIENCIA
# ==========================================

def parse_json_safe(text: str, verbose: bool = False) -> Tuple[bool, Optional[Dict], str]:
    """
    Función de conveniencia para parseo seguro.
    
    Args:
        text: Texto raw
        verbose: Imprimir debug
        
    Returns:
        (success, parsed_data, error_message)
    
    Ejemplo:
        success, data, error = parse_json_safe('{"answer": "Paris')
        if success:
            print(data["answer"])
    """
    handler = JSONHandler(verbose=verbose)
    result = handler.parse(text)
    
    if result.success:
        return True, result.data, ""
    else:
        return False, None, result.error or "Error desconocido"


def ensure_json_complete(text: str) -> str:
    """
    Reparación rápida de JSON incompleto.
    
    Args:
        text: JSON posiblemente incompleto
        
    Returns:
        JSON reparado (no garantiza validez 100%)
    
    Ejemplo:
        repaired = ensure_json_complete('{"answer": "Paris')
        # → '{"answer": "Paris"}'
    """
    handler = JSONHandler(verbose=False)
    return handler._repair_manual(text)


# ==========================================
# TESTS / EJEMPLOS
# ==========================================

if __name__ == "__main__":
    # Casos de prueba
    test_cases = [
        ('{"answer": "Paris", "confidence": 0.95}', "JSON válido"),
        ('{"answer": "The capital is Paris', "String truncado"),
        ('{"answer": "Paris", "confidence": 0.95', "Falta }"),
        ('{"answer": "Paris", "confidence": 0.95,}', "Coma final"),
        ('```json\n{"answer": "Paris"}\n```', "En code block"),
        ('Respuesta: {"answer": "Paris"} ¡Espero ayude!', "Con texto alrededor"),
    ]
    
    handler = JSONHandler(verbose=True)
    
    for i, (test, desc) in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"Test {i}: {desc}")
        print(f"Input: {test[:50]}...")
        print('='*60)
        
        result = handler.parse(test)
        
        if result.success:
            print(f"✅ ÉXITO (método: {result.method})")
            print(f"Data: {result.data}")
        else:
            print(f"❌ FALLO")
            print(f"Error: {result.error}")