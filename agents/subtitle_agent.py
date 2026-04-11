from __future__ import annotations

import json
import re
from typing import Iterable, List, Optional

from openai import OpenAI

from models import TranscriptSegment, WordTiming
from utils.timecode import seconds_to_itt_timestamp, seconds_to_srt_timestamp

_SENTENCE_BOUNDARY = re.compile(r'(?<=[.!?])\s+')

_REFORMAT_SYSTEM_PROMPT = """\
You are a subtitle formatter. You receive a raw transcript with pause markers.
Your job is to split it into subtitle lines that:
1. Each contain exactly one sentence or natural phrase.
2. Are at most 84 characters long.
3. Break at natural speech pauses (marked with [PAUSE]) when possible.
4. Add proper punctuation (periods, commas, question marks) where missing.
5. Do NOT change, add, or remove any words — only add punctuation and decide where to split lines.

Return a JSON array of strings, where each string is one subtitle line.
Example: ["Hey guys, this is HeeRin.", "Welcome back to another video.", "Today we have some merch to show."]
Return ONLY the JSON array, no other text."""


class SubtitleAgent:
    def __init__(self, max_chars_per_line: int = 42, max_lines_per_block: int = 2) -> None:
        self.max_chars_per_line = max_chars_per_line
        self.max_lines_per_block = max_lines_per_block

    def reformat_with_ai(
        self,
        segments: List[TranscriptSegment],
        api_key: str,
        model: str = "gpt-4o-mini",
    ) -> List[TranscriptSegment]:
        """Use an LLM to intelligently split the transcript into subtitle lines,
        then map each line back to real word-level timestamps.

        Args:
            segments: Raw segments with word-level timestamps from WhisperX.
            api_key: OpenAI API key.
            model: LLM model to use (default: gpt-4o-mini, very cheap).

        Returns:
            Re-grouped TranscriptSegments with accurate timestamps.
        """
        # Step 1: flatten to a word stream
        all_words = self._flatten_to_words(segments)
        if not all_words:
            return segments

        # Step 2: build transcript text with [PAUSE] markers at silence gaps
        pause_threshold = 0.5  # seconds
        text_parts: List[str] = []
        for i, w in enumerate(all_words):
            word_text = w.get("word", "").strip()
            if not word_text:
                continue
            # Insert [PAUSE] marker if there's a significant gap before this word
            if i > 0 and "start" in w and "end" in all_words[i - 1]:
                gap = w["start"] - all_words[i - 1]["end"]
                if gap >= pause_threshold:
                    text_parts.append("[PAUSE]")
            text_parts.append(word_text)

        transcript_with_pauses = " ".join(text_parts)

        # Step 3: ask the LLM to split into subtitle lines
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _REFORMAT_SYSTEM_PROMPT},
                {"role": "user", "content": transcript_with_pauses},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )

        raw_json = response.choices[0].message.content or "[]"
        parsed = json.loads(raw_json)
        # Handle both {"lines": [...]} and [...] formats
        if isinstance(parsed, dict):
            subtitle_lines = parsed.get("lines") or parsed.get("subtitles") or list(parsed.values())[0]
        else:
            subtitle_lines = parsed

        if not subtitle_lines or not isinstance(subtitle_lines, list):
            # Fallback to basic reformatting if AI response is bad
            print("  [warning] AI reformat failed, falling back to basic splitting")
            return self.reformat_as_sentences(segments)

        # Step 4: map each subtitle line back to word timestamps
        # Strip punctuation for matching since AI may have added/changed it
        clean = re.compile(r'[^\w\s]', re.UNICODE)
        word_index = 0  # pointer into all_words
        result: List[TranscriptSegment] = []

        for line in subtitle_lines:
            line = line.strip()
            if not line:
                continue
            line_tokens = line.split()
            matched_words: List[WordTiming] = []

            for token in line_tokens:
                token_clean = clean.sub("", token).lower()
                if not token_clean:
                    continue
                # Find the next matching word in the stream
                search_start = word_index
                found = False
                for j in range(search_start, min(search_start + 10, len(all_words))):
                    candidate = clean.sub("", all_words[j].get("word", "")).lower()
                    if candidate == token_clean:
                        matched_words.append(all_words[j])
                        word_index = j + 1
                        found = True
                        break
                if not found and word_index < len(all_words):
                    # Fuzzy fallback: just take the next word
                    matched_words.append(all_words[word_index])
                    word_index += 1

            if matched_words:
                start = self._first_start(matched_words)
                end = self._last_end(matched_words)
                result.append(TranscriptSegment(
                    index=0, start=start, end=end, text=line, words=matched_words,
                ))

        # Re-index
        for i, seg in enumerate(result, start=1):
            seg.index = i

        return result if result else self.reformat_as_sentences(segments)

    def reformat_as_sentences(self, segments: List[TranscriptSegment]) -> List[TranscriptSegment]:
        """Fallback: split segments using punctuation and max_chars only.

        Uses word-level timestamps (when available) for accurate timing.
        """
        max_chars = self.max_chars_per_line * self.max_lines_per_block
        all_words = self._flatten_to_words(segments)

        if not all_words:
            return segments

        result: List[TranscriptSegment] = []
        buf_words: List[WordTiming] = []
        buf_text = ""

        for i, w in enumerate(all_words):
            word_text = w.get("word", "").strip()
            if not word_text:
                continue

            buf_words.append(w)
            buf_text = f"{buf_text} {word_text}".strip() if buf_text else word_text

            # Check for natural break points
            ends_sentence = bool(re.search(r'[.!?]$', word_text))
            has_pause = False
            if i + 1 < len(all_words) and "end" in w and "start" in all_words[i + 1]:
                gap = all_words[i + 1]["start"] - w["end"]
                has_pause = gap >= 0.5

            if ends_sentence or len(buf_text) > max_chars or (has_pause and len(buf_text) > 20):
                self._flush_buf(result, buf_words, buf_text)
                buf_words = []
                buf_text = ""

        if buf_words:
            self._flush_buf(result, buf_words, buf_text)

        for i, seg in enumerate(result, start=1):
            seg.index = i

        return result

    @staticmethod
    def _flatten_to_words(segments: List[TranscriptSegment]) -> List[WordTiming]:
        """Flatten segments into a single word stream with timestamps."""
        all_words: List[WordTiming] = []
        for seg in segments:
            if seg.words:
                all_words.extend(seg.words)
            else:
                tokens = seg.text.split()
                if not tokens:
                    continue
                duration = seg.end - seg.start
                total_chars = max(sum(len(t) for t in tokens), 1)
                cursor = seg.start
                for token in tokens:
                    t_dur = duration * len(token) / total_chars
                    all_words.append({"word": token, "start": cursor, "end": cursor + t_dur})
                    cursor += t_dur
        return all_words

    @staticmethod
    def _first_start(words: List[WordTiming]) -> float:
        for w in words:
            if "start" in w:
                return w["start"]
        return 0.0

    @staticmethod
    def _last_end(words: List[WordTiming]) -> float:
        for w in reversed(words):
            if "end" in w:
                return w["end"]
        return 0.0

    @staticmethod
    def _flush_buf(
        result: List[TranscriptSegment],
        buf_words: List[WordTiming],
        buf_text: str,
    ) -> None:
        """Create a TranscriptSegment from buffered words."""
        start = 0.0
        end = 0.0
        for w in buf_words:
            if "start" in w:
                start = w["start"]
                break
        for w in reversed(buf_words):
            if "end" in w:
                end = w["end"]
                break
        if end <= start and buf_words:
            end = start + 0.5

        result.append(TranscriptSegment(
            index=0, start=start, end=end, text=buf_text, words=list(buf_words),
        ))

    def to_srt(self, segments: Iterable[TranscriptSegment], end_padding: float = 0.3) -> str:
        """Convert segments to SRT format.

        Args:
            segments: Subtitle segments.
            end_padding: Seconds to extend each subtitle's end time so it
                         doesn't vanish the instant the last word is spoken.
                         Capped so it never overlaps the next subtitle's start.
        """
        seg_list = list(segments)
        blocks: List[str] = []
        for i, segment in enumerate(seg_list):
            padded_end = segment.end + end_padding
            # Don't overlap the next subtitle
            if i + 1 < len(seg_list):
                padded_end = min(padded_end, seg_list[i + 1].start)

            start = seconds_to_srt_timestamp(segment.start)
            end = seconds_to_srt_timestamp(padded_end)
            text = self._wrap_text(segment.text)
            blocks.append(f"{i + 1}\n{start} --> {end}\n{text}")
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
