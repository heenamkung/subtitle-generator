from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    openai_api_key: str = ""
    transcription_model: str = "whisper-1"
    output_root: Path = Path("output")
    audio_bitrate: str = "64k"
    audio_sample_rate: int = 16000
    audio_channels: int = 1
    # WhisperX settings
    whisperx_model: str = "large-v2"
    whisperx_device: str = "cpu"
    whisperx_compute_type: str = "int8"

    @staticmethod
    def load() -> "Settings":
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        return Settings(openai_api_key=api_key)
