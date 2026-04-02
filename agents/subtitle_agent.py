from __future__ import annotations

import re
from typing import Iterable, List

from models import TranscriptSegment
from utils.timecode import seconds_to_itt_timestamp, seconds_to_srt_timestamp

_SENTENCE_BOUNDARY = re.compile(r'(?<=[.!?])\s+')


class SubtitleAgent:
    def __init__(self, max_chars_per_line: int = 42, max_lines_per_block: int = 2) -> None:
        self.max_chars_per_line = max_chars_per_line
        self.max_lines_per_block = max_lines_per_block

    def reformat_as_sentences(self, segments: List[TranscriptSegment]) -> List[TranscriptSegment]:
        """Merge/split segments so each subtitle unit contains exactly one sentence."""
        max_chars = self.max_chars_per_line * self.max_lines_per_block

        # Pass 1: split any segment that contains multiple sentences
        split: List[TranscriptSegment] = []
        for seg in segments:
            parts = [p.strip() for p in _SENTENCE_BOUNDARY.split(seg.text.strip()) if p.strip()]
            if len(parts) <= 1:
                split.append(seg)
                continue
            duration = seg.end - seg.start
            total_chars = sum(len(p) for p in parts)
            cursor = seg.start
            for part in parts:
                part_dur = duration * len(part) / max(total_chars, 1)
                split.append(TranscriptSegment(index=0, start=cursor, end=cursor + part_dur, text=part))
                cursor += part_dur

        # Pass 2: merge segments that don't end a sentence; force-break if too long
        result: List[TranscriptSegment] = []
        buf_text = ""
        buf_start = 0.0
        buf_end = 0.0

        for seg in split:
            if not buf_text:
                buf_start = seg.start
            buf_text = f"{buf_text} {seg.text}".strip() if buf_text else seg.text
            buf_end = seg.end

            ends_sentence = bool(re.search(r'[.!?]$', buf_text))
            if ends_sentence or len(buf_text) > max_chars:
                result.append(TranscriptSegment(index=0, start=buf_start, end=buf_end, text=buf_text))
                buf_text = ""

        if buf_text:
            result.append(TranscriptSegment(index=0, start=buf_start, end=buf_end, text=buf_text))

        for i, seg in enumerate(result, start=1):
            seg.index = i

        return result

    def to_srt(self, segments: Iterable[TranscriptSegment]) -> str:
        blocks: List[str] = []
        for index, segment in enumerate(segments, start=1):
            start = seconds_to_srt_timestamp(segment.start)
            end = seconds_to_srt_timestamp(segment.end)
            text = self._wrap_text(segment.text)
            blocks.append(f"{index}\n{start} --> {end}\n{text}")
        return "\n\n".join(blocks).strip() + "\n"

    def to_itt(self, segments: Iterable[TranscriptSegment], lang: str = "en") -> str:
        lines: List[str] = []
        lines.append('<?xml version="1.0" encoding="UTF-8"?>')
        lines.append(
            f'<tt xml:lang="{lang}"'
            ' xmlns="http://www.w3.org/ns/ttml"'
            ' xmlns:tts="http://www.w3.org/ns/ttml#styling">'
        )
        lines.append("  <head>")
        lines.append("    <styling>")
        lines.append(
            '      <style xml:id="s1"'
            ' tts:color="white"'
            ' tts:textAlign="center"/>'
        )
        lines.append("    </styling>")
        lines.append("    <layout>")
        lines.append(
            '      <region xml:id="r1" tts:displayAlign="after" tts:textAlign="center"/>'
        )
        lines.append("    </layout>")
        lines.append("  </head>")
        lines.append("  <body>")
        lines.append("    <div>")
        for segment in segments:
            begin = seconds_to_itt_timestamp(segment.start)
            end = seconds_to_itt_timestamp(segment.end)
            text = self._escape_xml(self._wrap_text(segment.text).replace("\n", "<br/>"))
            lines.append(f'      <p begin="{begin}" end="{end}" style="s1" region="r1">{text}</p>')
        lines.append("    </div>")
        lines.append("  </body>")
        lines.append("</tt>")
        return "\n".join(lines) + "\n"

    def _escape_xml(self, text: str) -> str:
        return text.replace("&", "&amp;").replace("<br/>", "\uffff").replace("<", "&lt;").replace(">", "&gt;").replace("\uffff", "<br/>")

    def _wrap_text(self, text: str) -> str:
        words = text.split()
        if not words:
            return ""

        lines: List[str] = []
        current = words[0]
        for word in words[1:]:
            candidate = f"{current} {word}"
            if len(candidate) <= self.max_chars_per_line:
                current = candidate
            else:
                lines.append(current)
                current = word

        lines.append(current)

        if len(lines) <= self.max_lines_per_block:
            return "\n".join(lines)

        collapsed: List[str] = []
        buffer = []
        for line in lines:
            buffer.append(line)
            if len(buffer) == self.max_lines_per_block:
                collapsed.append(" ".join(buffer))
                buffer = []

        if buffer:
            collapsed.append(" ".join(buffer))

        return "\n".join(collapsed)
