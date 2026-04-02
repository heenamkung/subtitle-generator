from __future__ import annotations


def seconds_to_srt_timestamp(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0

    total_ms = int(round(seconds * 1000))
    hours = total_ms // 3_600_000
    remainder = total_ms % 3_600_000
    minutes = remainder // 60_000
    remainder %= 60_000
    secs = remainder // 1000
    millis = remainder % 1000
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"


def seconds_to_itt_timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS.mmm for iTT (iTunes Timed Text / Final Cut Pro)."""
    if seconds < 0:
        seconds = 0.0

    total_ms = int(round(seconds * 1000))
    hours = total_ms // 3_600_000
    remainder = total_ms % 3_600_000
    minutes = remainder // 60_000
    remainder %= 60_000
    secs = remainder // 1000
    millis = remainder % 1000
    return f"{hours:02}:{minutes:02}:{secs:02}.{millis:03}"
