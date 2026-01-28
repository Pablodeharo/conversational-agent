"""Procesador Text-to-Speech con Piper.
Bug: config.piper_speaker NO existe en AudioConfiguration
→ debes eliminarlo o añadirlo al config

"""

import subprocess
import tempfile
from pathlib import Path
from retrieval_graph.audio.config import AudioConfiguration


class TTSProcessor:
    """
    Text-to-Speech processor using the Piper CLI.
    """

    def __init__(self, config: AudioConfiguration):
        """
        Initialize Piper TTS and verify that the CLI is available.
        """
        self.model_name = config.piper_model

        # Directory where Piper voice models are stored
        self.models_dir = Path.home() / ".local" / "share" / "piper-voices"
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Ensure Piper is installed
        try:
            subprocess.run(
                ["piper", "--version"],
                check=True,
                capture_output=True,
            )
        except FileNotFoundError:
            raise RuntimeError(
                "Piper is not installed. Install it from:\n"
                "https://github.com/rhasspy/piper"
            )

    def synthesize(self, text: str) -> bytes:
        """
        Convert text into speech audio using Piper.

        Args:
            text (str): Input text.

        Returns:
            bytes: Generated WAV audio.
        """
        with tempfile.NamedTemporaryFile(
            suffix=".wav", delete=False
        ) as temp_audio:
            output_path = temp_audio.name

        try:
            process = subprocess.Popen(
                [
                    "piper",
                    "--model", self.model_name,
                    "--output_file", output_path,
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            _, stderr = process.communicate(input=text)

            if process.returncode != 0:
                raise RuntimeError(f"Piper failed: {stderr}")

            with open(output_path, "rb") as f:
                return f.read()

        finally:
            Path(output_path).unlink(missing_ok=True)