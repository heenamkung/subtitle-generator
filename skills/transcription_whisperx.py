from __future__ import annotations

import threading
from pathlib import Path
from typing import Callable, List, Optional

from models import TranscriptSegment


# Approximate on-disk sizes (MB) for faster-whisper int8 model snapshots.
# Used only to show a rough percentage during the first download — actual
# completion is determined by load_model() returning, not by these numbers.
_MODEL_SIZES_MB = {
    "tiny": 75,
    "base": 145,
    "small": 490,
    "medium": 1500,
    "large-v2": 3100,
    "large-v3": 3100,
}


def _dir_size_mb(path: Path) -> float:
    """Sum of all file sizes under `path`, in MB. Safe against missing files.

    Skips symlinks: HF hub stores each file as a blob under `blobs/<sha>` and
    a symlink at `snapshots/<commit>/<name>` pointing to it. `stat()` follows
    symlinks, so counting both would double-count every file.
    """
    total = 0
    for f in path.rglob("*"):
        try:
            if f.is_symlink():
                continue
            if f.is_file():
                total += f.stat().st_size
        except (OSError, FileNotFoundError):
            # File may have been renamed/deleted between rglob and stat — fine.
            continue
    return total / (1024 * 1024)


def _watch_download(cache_dir: Path, expected_mb: float, stop: threading.Event) -> None:
    """Log download progress by polling cache-dir size every 5 seconds.

    Runs on a daemon thread. Exits when `stop` is set. Intentionally produces
    one log line per tick (no \\r tricks) so Docker's log driver captures it.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    while not stop.wait(5.0):
        mb = _dir_size_mb(cache_dir)
        pct = min(99, (mb / expected_mb) * 100) if expected_mb else 0
        print(f"    Downloaded {mb:.0f} MB / ~{expected_mb:.0f} MB ({pct:.0f}%)...", flush=True)


class WhisperXTranscriptionSkill:
    """Transcribe audio locally using WhisperX with forced alignment.

    WhisperX runs Whisper locally and adds phoneme-level forced alignment
    for much more accurate word-level timestamps than the OpenAI API.
    Handles its own internal chunking — no need to split audio beforehand.
    """

    def __init__(
        self,
        model_name: str = "large-v2",
        device: str = "cpu",
        compute_type: str = "int8",
        initial_prompt: str = "",
    ) -> None:
        import whisperx

        self._whisperx = whisperx
        self.device = device
        self.model_name = model_name
        self.compute_type = compute_type
        self.initial_prompt = initial_prompt

        # Model loaded lazily on first transcribe() call so the constructor
        # stays fast for UI startup. Set to None here; loaded in _ensure_model().
        self._model = None
        self._align_model = None
        self._align_metadata = None

    def _ensure_model(self) -> None:
        """Load the Whisper model if not already loaded."""
        if self._model is not None:
            return

        # Check if model is already downloaded by looking for a complete snapshot
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub" / f"models--Systran--faster-whisper-{self.model_name}"
        has_cache = cache_dir.exists() and not any(cache_dir.rglob("*.incomplete"))

        progress_thread: Optional[threading.Thread] = None
        stop_event: Optional[threading.Event] = None

        if has_cache:
            print(f"  Loading WhisperX model '{self.model_name}' from cache...", flush=True)
        else:
            expected_mb = _MODEL_SIZES_MB.get(self.model_name, 1500)
            print(
                f"  Downloading WhisperX model '{self.model_name}' "
                f"(~{expected_mb} MB, one-time)...",
                flush=True,
            )
            stop_event = threading.Event()
            progress_thread = threading.Thread(
                target=_watch_download,
                args=(cache_dir, expected_mb, stop_event),
                daemon=True,
            )
            progress_thread.start()

        asr_options = {}
        if self.initial_prompt:
            asr_options["initial_prompt"] = self.initial_prompt

        try:
            self._model = self._whisperx.load_model(
                self.model_name,
                self.device,
                compute_type=self.compute_type,
                asr_options=asr_options if asr_options else None,
            )
        finally:
            # Stop the progress watcher regardless of success/failure.
            if stop_event is not None:
                stop_event.set()
            if progress_thread is not None:
                progress_thread.join(timeout=1.0)
                final_mb = _dir_size_mb(cache_dir)
                print(f"  Model ready ({final_mb:.0f} MB on disk).", flush=True)

    def _ensure_align_model(self, language_code: str) -> None:
        """Load the alignment model for the detected language."""
        if self._align_model is not None:
            return
        print(f"  Loading alignment model for '{language_code}'...")
        self._align_model, self._align_metadata = self._whisperx.load_align_model(
            language_code=language_code,
            device=self.device,
        )

    def transcribe(
        self,
        audio_path: Path,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> List[TranscriptSegment]:
        """Transcribe a full audio file with word-level aligned timestamps.

        Args:
            audio_path: Path to the audio file (mp3, wav, etc.)
            progress_callback: Called with (stage_name, percent 0-100).
                               Stages: "transcribing", "aligning".

        Returns:
            List of TranscriptSegments with accurate timestamps.
        """
        self._ensure_model()

        # WhisperX has its own audio loader (resamples to 16kHz internally)
        audio = self._whisperx.load_audio(str(audio_path))

        # Transcribe — WhisperX handles internal 30-sec batching.
        # print_progress=True makes WhisperX emit per-chunk progress to stdout,
        # which is the only way to see transcription progress in Docker logs.
        transcribe_kwargs: dict = {
            "batch_size": 16,
            "language": "en",
            "print_progress": True,
        }
        if progress_callback:
            transcribe_kwargs["progress_callback"] = lambda p: progress_callback("transcribing", p)

        result = self._model.transcribe(audio, **transcribe_kwargs)

        raw_segments = result.get("segments", [])
        language = result.get("language", "en")

        if not raw_segments:
            return []

        # Run forced alignment for precise word-level timestamps
        self._ensure_align_model(language)

        align_kwargs: dict = dict(
            return_char_alignments=False,
            print_progress=True,
        )
        if progress_callback:
            align_kwargs["progress_callback"] = lambda p: progress_callback("aligning", p)

        aligned = self._whisperx.align(
            raw_segments,
            self._align_model,
            self._align_metadata,
            audio,
            self.device,
            **align_kwargs,
        )
        aligned_segments = aligned.get("segments", raw_segments)

        # Map to our TranscriptSegment model, preserving word timestamps
        segments: List[TranscriptSegment] = []
        for idx, seg in enumerate(aligned_segments, start=1):
            text = (seg.get("text") or "").strip()
            if not text:
                continue

            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", start))

            # Preserve word-level timestamps for downstream splitting
            word_timings = []
            raw_words = seg.get("words") or []
            for w in raw_words:
                wt: dict = {"word": w.get("word", "")}
                if w.get("start") is not None:
                    wt["start"] = float(w["start"])
                if w.get("end") is not None:
                    wt["end"] = float(w["end"])
                word_timings.append(wt)

            # Use first word's start for precise timing
            if word_timings and "start" in word_timings[0]:
                start = word_timings[0]["start"]

            segments.append(TranscriptSegment(
                index=idx,
                start=start,
                end=end,
                text=text,
                words=word_timings,
            ))

        return segments
