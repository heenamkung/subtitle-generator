from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, TypedDict


class WordTiming(TypedDict, total=False):
    word: str
    start: float
    end: float


@dataclass
class TranscriptSegment:
    index: int
    start: float
    end: float
    text: str
    words: List[WordTiming] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "start": self.start,
            "end": self.end,
            "text": self.text,
        }
