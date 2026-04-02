from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from openai import OpenAI

from models import TranscriptSegment


class TranscriptionSkill:
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

        raw_segments = getattr(response, "segments", None)
        if not raw_segments:
            text = getattr(response, "text", "").strip()
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

        segments: List[TranscriptSegment] = []
        for idx, seg in enumerate(raw_segments, start=1):
            text = (getattr(seg, "text", "") or "").strip()
            if not text:
                continue
            # Use the first word's start time to avoid leading silence
            words = getattr(seg, "words", None)
            if words:
                start = float(getattr(words[0], "start", getattr(seg, "start", 0.0))) + time_offset
            else:
                start = float(getattr(seg, "start", 0.0)) + time_offset
            end = float(getattr(seg, "end", start)) + time_offset
            segments.append(
                TranscriptSegment(index=idx, start=start, end=end, text=text)
            )
        return segments

    @staticmethod
    def merge_segment_lists(segment_lists: Iterable[List[TranscriptSegment]]) -> List[TranscriptSegment]:
        merged: List[TranscriptSegment] = []
        for chunk_segments in segment_lists:
            merged.extend(chunk_segments)

        for i, segment in enumerate(merged, start=1):
            segment.index = i
        return merged
