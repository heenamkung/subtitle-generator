from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class TranscriptSegment:
    index: int
    start: float
    end: float
    text: str

    def to_dict(self) -> dict:
        return asdict(self)
