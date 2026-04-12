from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import List


class AudioSkill:
    def __init__(self) -> None:
        for binary in ("ffmpeg", "ffprobe"):
            if shutil.which(binary) is None:
                raise RuntimeError(
                    f"{binary} is not installed or not on PATH. Install ffmpeg first."
                )

    def extract_audio(
        self,
        input_video: Path,
        output_audio: Path,
        bitrate: str,
        sample_rate: int,
        channels: int,
    ) -> Path:
        output_audio.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(input_video),
            "-vn",
            "-ac",
            str(channels),
            "-ar",
            str(sample_rate),
            "-b:a",
            bitrate,
            str(output_audio),
        ]
        self._run(cmd, "Audio extraction failed")
        return output_audio

    def split_audio(self, input_audio: Path, chunks_dir: Path, chunk_seconds: int) -> List[Path]:
        chunks_dir.mkdir(parents=True, exist_ok=True)
        output_pattern = chunks_dir / "chunk_%04d.mp3"
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(input_audio),
            "-f",
            "segment",
            "-segment_time",
            str(chunk_seconds),
            "-c",
            "copy",
            str(output_pattern),
        ]
        self._run(cmd, "Audio splitting failed")
        chunks = sorted(chunks_dir.glob("chunk_*.mp3"))
        if not chunks:
            raise RuntimeError("No audio chunks were created.")
        return chunks

    def get_duration_seconds(self, media_path: Path) -> float:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(media_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        payload = json.loads(result.stdout)
        duration = float(payload["format"]["duration"])
        return duration

    @staticmethod
    def _run(cmd: list[str], error_message: str) -> None:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            stderr = result.stderr.strip()
            # Extract just the last meaningful line from ffmpeg's verbose output
            last_lines = [l for l in stderr.splitlines() if l.strip()]
            short_err = last_lines[-1] if last_lines else "Unknown error"
            raise RuntimeError(
                f"{error_message}: {short_err}\n\n"
                f"(Full ffmpeg output: {stderr[-500:] if len(stderr) > 500 else stderr})"
            )
