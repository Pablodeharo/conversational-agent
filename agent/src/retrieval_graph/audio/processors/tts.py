"""Procesador Text-to-Speech con Piper."""

import subprocess
import tempfile
from pathlib import Path
from retrieval_graph.audio.config import AudioConfiguration


class TTSProcessor:
    """Procesador de Text-to-Speech usando Piper CLI."""
    
    def __init__(self, config: AudioConfiguration):
        self.model_name = config.piper_model
        self.speaker = config.piper_speaker
        
        # Directorio donde se guardan los modelos de Piper
        self.models_dir = Path.home() / ".local" / "share" / "piper-voices"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Verificar si piper está instalado
        try:
            subprocess.run(
                ["piper", "--version"],
                check=True,
                capture_output=True
            )
        except FileNotFoundError:
            raise RuntimeError(
                "Piper no está instalado. Instálalo con:\n"
                "wget https://github.com/rhasspy/piper/releases/download/v1.2.0/piper_amd64.tar.gz\n"
                "tar -xvzf piper_amd64.tar.gz\n"
                "sudo cp piper/piper /usr/local/bin/"
            )
    
    def synthesize(self, text: str) -> bytes:
        """Sintetiza texto a audio WAV usando Piper CLI."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            output_path = temp_audio.name
        
        try:
            # Ejecutar Piper
            process = subprocess.Popen(
                [
                    "piper",
                    "--model", self.model_name,
                    "--output_file", output_path,
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            stdout, stderr = process.communicate(input=text)
            
            if process.returncode != 0:
                raise RuntimeError(f"Piper falló: {stderr}")
            
            # Leer el audio generado
            with open(output_path, "rb") as f:
                audio_bytes = f.read()
            
            return audio_bytes
            
        finally:
            # Limpiar archivo temporal
            Path(output_path).unlink(missing_ok=True)