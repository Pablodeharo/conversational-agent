import wave
import io
import numpy as np


def bytes_to_wav(
    audio_bytes: bytes,
    sample_rate: int = 22050,
    channels: int = 1
) -> bytes:
    """
    Convert raw PCM audio bytes into a WAV container.

    Args:
        audio_bytes (bytes): Raw PCM audio data (16-bit).
        sample_rate (int): Audio sample rate.
        channels (int): Number of audio channels.

    Returns:
        bytes: WAV-formatted audio.
    """
    wav_buffer = io.BytesIO()

    with wave.open(wav_buffer, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)  # 16-bit PCM
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_bytes)

    return wav_buffer.getvalue()


def normalize_audio(audio_array: np.ndarray) -> np.ndarray:
    """
    Normalize audio amplitude to the range [-1, 1].

    Args:
        audio_array (np.ndarray): Audio samples.

    Returns:
        np.ndarray: Normalized audio.
    """
    max_val = np.abs(audio_array).max()
    if max_val > 0:
        return audio_array / max_val
    return audio_array