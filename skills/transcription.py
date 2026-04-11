from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, List

from openai import OpenAI

from models import TranscriptSegment


def _extract_field(obj: Any, name: str) -> Any:
    """Get a field from an API response, checking direct attrs and model_extra."""
    val = getattr(obj, name, None)
    if val is not None:
        return val
    extra = getattr(obj, "model_extra", None)
    if isinstance(extra, dict):
        return extra.get(name)
    if isinstance(obj, dict):
        return obj.get(name)
    return None


def _get_float(obj: Any, name: str, default: float = 0.0) -> float:
    val = _extract_field(obj, name)
    return float(val) if val is not None else default


class TranscriptionSkill:
    """Transcribe audio chunks using OpenAI's Whisper API.

    Note: only whisper-1 supports timestamped output (verbose_json).
    gpt-4o-transcribe models return text only — no timestamps — so they
    cannot be used for subtitle generation.
    """

    def __init__(self, api_key: str, model: str = "whisper-1") -> None:
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def transcribe_chunk(
        self,
        audio_path: Path,
        time_offset: float = 0.0,
        prompt: str = "",
    ) -> List[TranscriptSegment]:
        with audio_path.open("rb") as audio_file:
            kwargs: dict = dict(
                model=self.model,
                file=audio_file,
                response_format="verbose_json",
                timestamp_granularities=["word", "segment"],
            )
            if prompt:
                kwargs["prompt"] = prompt
            response = self.client.audio.transcriptions.create(**kwargs)

        raw_segments = _extract_field(response, "segments") or []
        if not raw_segments:
            text = (_extract_field(response, "text") or "").strip()
            if not text:
                return []
            return [
                TranscriptSegment(
                    index=1,
                    start=time_offset,
                    end=time_offset + 3.0,
                    text=text,
                )
            ]

        # Word timestamps live at the response level or inside each segment.
        response_words = _extract_field(response, "words") or []

        segments: List[TranscriptSegment] = []
        for idx, seg in enumerate(raw_segments, start=1):
            text = (_extract_field(seg, "text") or "").strip()
            if not text:
                continue

            seg_start_raw = _get_float(seg, "start")
            seg_end_raw = _get_float(seg, "end", seg_start_raw)

            # Use first word's start time to avoid leading silence.
            seg_words = _extract_field(seg, "words")
            if seg_words:
                first_start = _get_float(seg_words[0], "start", seg_start_raw)
            elif response_words:
                in_seg = [
                    w for w in response_words
                    if seg_start_raw - 0.1 <= _get_float(w, "start") <= seg_end_raw + 0.1
                ]
                first_start = _get_float(in_seg[0], "start", seg_start_raw) if in_seg else seg_start_raw
            else:
                first_start = seg_start_raw

            segments.append(TranscriptSegment(
                index=idx,
                start=first_start + time_offset,
                end=seg_end_raw + time_offset,
                text=text,
            ))
        return segments

    @staticmethod
    def merge_segment_lists(segment_lists: Iterable[List[TranscriptSegment]]) -> List[TranscriptSegment]:
        merged: List[TranscriptSegment] = []
        for chunk_segments in segment_lists:
            merged.extend(chunk_segments)

        for i, segment in enumerate(merged, start=1):
            segment.index = i
        return merged
