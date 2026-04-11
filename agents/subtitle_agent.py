from __future__ import annotations

import json
import re
from typing import Iterable, List, Optional

from openai import OpenAI

from models import TranscriptSegment, WordTiming
from utils.timecode import seconds_to_itt_timestamp, seconds_to_srt_timestamp

_SENTENCE_BOUNDARY = re.compile(r'(?<=[.!?])\s+')

_PUNCTUATION_SYSTEM_PROMPT = """\
You are a punctuation corrector. You receive a raw speech transcript with [PAUSE] markers.
Your ONLY job is to add proper punctuation (periods, commas, question marks, exclamation marks) where missing.

Rules:
1. Do NOT change, add, or remove any words — only add punctuation marks.
2. Remove [PAUSE] markers from the output.
3. Every sentence must end with a period, question mark, or exclamation mark.
4. Use commas for natural pauses within sentences.
5. Return the full punctuated text as a single string.

Return a JSON object: {"text": "The punctuated transcript here."}"""


class SubtitleAgent:
    def __init__(self, max_chars_per_line: int = 35, max_lines_per_block: int = 1) -> None:
        self.max_chars_per_line = max_chars_per_line
        self.max_lines_per_block = max_lines_per_block

    def reformat_with_ai(
        self,
        segments: List[TranscriptSegment],
        api_key: str,
        model: str = "gpt-4o-mini",
    ) -> List[TranscriptSegment]:
        """Two-step AI formatting:
        1. AI adds punctuation to the raw transcript (simple, reliable task).
        2. Our code splits at sentence boundaries using real word timestamps.

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
            if i > 0 and "start" in w and "end" in all_words[i - 1]:
                gap = w["start"] - all_words[i - 1]["end"]
                if gap >= pause_threshold:
                    text_parts.append("[PAUSE]")
            text_parts.append(word_text)

        transcript_with_pauses = " ".join(text_parts)

        # Step 3: ask AI to add punctuation only (chunked for long transcripts)
        chunks = self._split_for_api(transcript_with_pauses, max_chunk_chars=2000)
        client = OpenAI(api_key=api_key)
        punctuated_parts: List[str] = []

        for chunk_i, chunk_text in enumerate(chunks, start=1):
            print(f"  AI adding punctuation chunk {chunk_i}/{len(chunks)}...")
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": _PUNCTUATION_SYSTEM_PROMPT},
                        {"role": "user", "content": chunk_text},
                    ],
                    temperature=0.0,
                    response_format={"type": "json_object"},
                    timeout=60,
                )

                raw_json = response.choices[0].message.content or "{}"
                parsed = json.loads(raw_json)
                text = parsed.get("text", "") if isinstance(parsed, dict) else ""
                if text:
                    punctuated_parts.append(text)
                else:
                    print(f"  [warning] AI returned no text for chunk {chunk_i}")
                    # Use original chunk without [PAUSE] markers as fallback
                    punctuated_parts.append(chunk_text.replace("[PAUSE]", ""))
            except Exception as e:
                print(f"  [warning] AI punctuation failed for chunk {chunk_i}: {e}")
                punctuated_parts.append(chunk_text.replace("[PAUSE]", ""))

        punctuated_text = " ".join(punctuated_parts)

        # Step 4: map punctuated words back to word timestamps.
        # Walk both lists in parallel. If the AI changed a word, skip it
        # rather than cascading the error through every subsequent word.
        print(f"  Mapping punctuated text to {len(all_words)} word timestamps...")
        punctuated_words = punctuated_text.split()
        clean = re.compile(r'[^\w\s]', re.UNICODE)
        p_idx = 0
        w_idx = 0

        while p_idx < len(punctuated_words) and w_idx < len(all_words):
            p_word = punctuated_words[p_idx]
            p_clean = clean.sub("", p_word).lower()

            if not p_clean:
                p_idx += 1
                continue

            w_clean = clean.sub("", all_words[w_idx].get("word", "")).lower()

            if p_clean == w_clean:
                # Match — copy punctuated version to the word stream
                all_words[w_idx]["word"] = p_word
                p_idx += 1
                w_idx += 1
            else:
                # Mismatch — try to re-sync by looking ahead in both lists
                found = False

                # Look ahead in original words (AI might have skipped a word)
                for look_w in range(w_idx + 1, min(w_idx + 5, len(all_words))):
                    if clean.sub("", all_words[look_w].get("word", "")).lower() == p_clean:
                        # AI skipped some original words — advance original to match
                        w_idx = look_w
                        all_words[w_idx]["word"] = p_word
                        p_idx += 1
                        w_idx += 1
                        found = True
                        break

                if not found:
                    # Look ahead in punctuated words (AI might have inserted a word)
                    for look_p in range(p_idx + 1, min(p_idx + 5, len(punctuated_words))):
                        lp_clean = clean.sub("", punctuated_words[look_p]).lower()
                        if lp_clean == w_clean:
                            # AI inserted extra words — skip them
                            p_idx = look_p
                            all_words[w_idx]["word"] = punctuated_words[p_idx]
                            p_idx += 1
                            w_idx += 1
                            found = True
                            break

                if not found:
                    # Can't re-sync — keep original word, skip AI word, move on
                    p_idx += 1

        # Step 5: rebuild segments from the now-punctuated word stream
        # and split at sentence boundaries using reformat_as_sentences
        full_text = " ".join(w.get("word", "") for w in all_words).strip()
        combined = TranscriptSegment(
            index=1,
            start=self._first_start(all_words),
            end=self._last_end(all_words),
            text=full_text,
            words=list(all_words),
        )

        return self.reformat_as_sentences([combined])

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
    def _split_for_api(text: str, max_chunk_chars: int = 2000) -> List[str]:
        """Split transcript text into chunks at [PAUSE] markers."""
        if len(text) <= max_chunk_chars:
            return [text]

        chunks: List[str] = []
        current = ""
        # Split on [PAUSE] markers as natural break points
        parts = text.split("[PAUSE]")
        for part in parts:
            part = part.strip()
            if not part:
                continue
            candidate = f"{current} [PAUSE] {part}".strip() if current else part
            if len(candidate) > max_chunk_chars and current:
                chunks.append(current.strip())
                current = part
            else:
                current = candidate
        if current.strip():
            chunks.append(current.strip())

        return chunks if chunks else [text]

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

    def merge_orphans(
        self, segments: List[TranscriptSegment], max_words: int = 2, max_gap: float = 0.5
    ) -> List[TranscriptSegment]:
        """Merge very short subtitles (≤max_words) into their neighbor.

        Tries the previous subtitle first, then next. Only merges if:
        - Combined text stays under max_chars_per_line
        - Time gap between them is ≤max_gap seconds
        """
        if len(segments) < 2:
            return segments

        max_chars = self.max_chars_per_line
        merged: List[TranscriptSegment] = list(segments)
        changed = True

        # Keep merging until no more changes (handles consecutive orphans)
        while changed:
            changed = False
            new_list: List[TranscriptSegment] = []
            skip_next = False

            for i, seg in enumerate(merged):
                if skip_next:
                    skip_next = False
                    continue

                word_count = len(seg.text.split())
                if word_count > max_words:
                    new_list.append(seg)
                    continue

                # Try merging with previous
                if new_list:
                    prev = new_list[-1]
                    gap = seg.start - prev.end
                    combined = f"{prev.text} {seg.text}"
                    if gap <= max_gap and len(combined) <= max_chars:
                        new_list[-1] = TranscriptSegment(
                            index=0,
                            start=prev.start,
                            end=seg.end,
                            text=combined,
                            words=prev.words + seg.words,
                        )
                        changed = True
                        continue

                # Try merging with next
                if i + 1 < len(merged):
                    nxt = merged[i + 1]
                    gap = nxt.start - seg.end
                    combined = f"{seg.text} {nxt.text}"
                    if gap <= max_gap and len(combined) <= max_chars:
                        new_list.append(TranscriptSegment(
                            index=0,
                            start=seg.start,
                            end=nxt.end,
                            text=combined,
                            words=seg.words + nxt.words,
                        ))
                        skip_next = True
                        changed = True
                        continue

                # Can't merge — keep as-is
                new_list.append(seg)

            merged = new_list

        for i, seg in enumerate(merged, start=1):
            seg.index = i

        return merged

    def enforce_max_duration(
        self, segments: List[TranscriptSegment], max_duration: float = 3.0
    ) -> List[TranscriptSegment]:
        """Split any subtitle that lasts longer than max_duration seconds.

        Uses word-level timestamps to find the best split point.
        Prevents subtitles from lingering on screen when speech is slow.
        """
        result: List[TranscriptSegment] = []

        for seg in segments:
            duration = seg.end - seg.start
            if duration <= max_duration or not seg.words or len(seg.words) < 2:
                result.append(seg)
                continue

            # Split at word boundary closest to max_duration intervals
            current_words: List[WordTiming] = []
            current_start: float = seg.start

            for w in seg.words:
                current_words.append(w)
                w_end = w.get("end", seg.end)
                elapsed = w_end - current_start

                if elapsed >= max_duration and len(current_words) >= 1:
                    text = " ".join(cw.get("word", "") for cw in current_words).strip()
                    if text:
                        result.append(TranscriptSegment(
                            index=0,
                            start=current_start,
                            end=w_end,
                            text=text,
                            words=list(current_words),
                        ))
                    current_words = []
                    # Next subtitle starts at the next word
                    current_start = w_end

            # Flush remaining words
            if current_words:
                text = " ".join(cw.get("word", "") for cw in current_words).strip()
                if text:
                    result.append(TranscriptSegment(
                        index=0,
                        start=current_start,
                        end=seg.end,
                        text=text,
                        words=list(current_words),
                    ))

        for i, seg in enumerate(result, start=1):
            seg.index = i

        return result

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
            # Single line only — no wrapping
            text = segment.text.strip()
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
