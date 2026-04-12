"""SRT → FCPXML converter for Final Cut Pro.

Generates FCPXML files with styled title elements that import directly
into Final Cut Pro via File > Import > XML.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, NamedTuple
from urllib.parse import quote as url_quote
from xml.sax.saxutils import escape


class SubtitleEntry(NamedTuple):
    index: int
    start: float  # seconds
    end: float    # seconds
    text: str


# ---------------------------------------------------------------------------
# Frame rate lookup tables
# ---------------------------------------------------------------------------
_FPS_INFO = {
    "23.976": (1001, 24000),
    "24":     (100,  2400),
    "25":     (100,  2500),
    "29.97":  (1001, 30000),
    "30":     (100,  3000),
    "50":     (100,  5000),
    "59.94":  (1001, 60000),
    "60":     (100,  6000),
}

# FCP uses format names like FFVideoFormat3840x2160p5994
_FORMAT_NAME = {
    ("1080p", "23.976"): "FFVideoFormat1920x1080p2398",
    ("1080p", "24"):     "FFVideoFormat1920x1080p24",
    ("1080p", "25"):     "FFVideoFormat1920x1080p25",
    ("1080p", "29.97"):  "FFVideoFormat1920x1080p2997",
    ("1080p", "30"):     "FFVideoFormat1920x1080p30",
    ("1080p", "50"):     "FFVideoFormat1920x1080p50",
    ("1080p", "59.94"):  "FFVideoFormat1920x1080p5994",
    ("1080p", "60"):     "FFVideoFormat1920x1080p60",
    ("4K", "23.976"):    "FFVideoFormat3840x2160p2398",
    ("4K", "24"):        "FFVideoFormat3840x2160p24",
    ("4K", "25"):        "FFVideoFormat3840x2160p25",
    ("4K", "29.97"):     "FFVideoFormat3840x2160p2997",
    ("4K", "30"):        "FFVideoFormat3840x2160p30",
    ("4K", "50"):        "FFVideoFormat3840x2160p50",
    ("4K", "59.94"):     "FFVideoFormat3840x2160p5994",
    ("4K", "60"):        "FFVideoFormat3840x2160p60",
}

_RESOLUTION = {
    "1080p": (1920, 1080),
    "4K":    (3840, 2160),
}

_POSITION_Y = {
    "bottom": -950,
    "center": 0,
    "top": 900,
}


def parse_srt(srt_text: str) -> List[SubtitleEntry]:
    """Parse an SRT string into a list of SubtitleEntry objects."""
    entries: List[SubtitleEntry] = []
    blocks = re.split(r"\n\n+", srt_text.strip())

    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) < 3:
            continue

        match = re.match(
            r"(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})",
            lines[1],
        )
        if not match:
            continue

        start = _srt_ts_to_seconds(match.group(1))
        end = _srt_ts_to_seconds(match.group(2))
        text = " ".join(l.strip() for l in lines[2:]).strip()

        if text:
            entries.append(SubtitleEntry(
                index=len(entries) + 1,
                start=start,
                end=end,
                text=text,
            ))

    return entries


def _srt_ts_to_seconds(ts: str) -> float:
    h, m, rest = ts.split(":")
    s, ms = rest.split(",")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def _snap_to_frame(seconds: float, fd_num: int, fd_den: int) -> str:
    """Convert seconds to FCPXML rational time, snapped to frame boundary."""
    frame_count = round(seconds * fd_den / fd_num)
    numerator = frame_count * fd_num
    return f"{numerator}/{fd_den}s"


def _color_to_rgba(color_hex: str) -> str:
    color_hex = color_hex.lstrip("#")
    if len(color_hex) == 6:
        r = int(color_hex[0:2], 16) / 255.0
        g = int(color_hex[2:4], 16) / 255.0
        b = int(color_hex[4:6], 16) / 255.0
        return f"{r:.4f} {g:.4f} {b:.4f} 1"
    return "1 1 1 1"


def generate_fcpxml(
    subtitles: List[SubtitleEntry],
    fps: str = "59.94",
    resolution: str = "4K",
    font: str = "Helvetica Neue",
    font_size: int = 100,
    font_color: str = "#FFFFFF",
    position: str = "bottom",
) -> str:
    if not subtitles:
        raise ValueError("No subtitles to convert.")

    width, height = _RESOLUTION.get(resolution, (3840, 2160))
    fd_num, fd_den = _FPS_INFO.get(fps, (1001, 60000))
    frame_dur = f"{fd_num}/{fd_den}s"
    format_name = _FORMAT_NAME.get((resolution, fps), f"FFVideoFormat{width}x{height}p5994")
    font_rgba = _color_to_rgba(font_color)
    pos_y = _POSITION_Y.get(position, -900)

    total_seconds = max(s.end for s in subtitles) + 1.0
    total_dur = _snap_to_frame(total_seconds, fd_num, fd_den)

    xml = []
    xml.append('<?xml version="1.0" encoding="UTF-8"?>')
    xml.append("<!DOCTYPE fcpxml>")
    xml.append("")
    xml.append('<fcpxml version="1.12">')
    xml.append("    <resources>")
    xml.append(
        f'        <format id="r1" name="{format_name}"'
        f' frameDuration="{frame_dur}"'
        f' width="{width}" height="{height}"'
        f' colorSpace="1-1-1 (Rec. 709)"/>'
    )
    # Build Motion template path from user's home directory (not hardcoded)
    _moti_rel = "Movies/Motion Templates.localized/Titles.localized/Tap5a/Tap5a Multiline Text Backgr. 2.moti"
    _moti_abs = Path.home() / _moti_rel
    _moti_url = "file:///" + url_quote(str(_moti_abs), safe="/:")
    xml.append(
        '        <effect id="r2" name="Tap5a Multiline Text Backgr. 2"'
        ' uid="~/Titles.localized/Tap5a/Tap5a Multiline Text Backgr. 2.moti"'
        f' src="{_moti_url}"/>'
    )
    xml.append("    </resources>")
    xml.append("    <library>")
    xml.append('        <event name="Subtitles">')
    xml.append('            <project name="Subtitles">')
    xml.append(
        f'                <sequence format="r1"'
        f' duration="{total_dur}"'
        f' tcStart="0s" tcFormat="NDF"'
        f' audioLayout="stereo" audioRate="48k">'
    )
    xml.append("                    <spine>")

    # Single gap as the base layer — titles overlay on lane 1
    xml.append(f'                        <gap offset="0s" duration="{total_dur}" name="Gap">')

    for sub in subtitles:
        offset = _snap_to_frame(sub.start, fd_num, fd_den)
        duration = _snap_to_frame(sub.end - sub.start, fd_num, fd_den)
        ts_id = f"ts{sub.index}"
        t = escape(sub.text)

        xml.append(
            f'                            <title ref="r2" lane="1"'
            f' offset="{offset}"'
            f' name="{t} - Tap5a Multiline Text Backgr. 2"'
            f' start="216216000/60000s"'
            f' duration="{duration}">'
        )
        # Tap5a template params (matched from exported FCP project)
        xml.append('                                <param name="Margin" key="9999/1825823367/100/1825839365/2/100" value="0.16388"/>')
        xml.append('                                <param name="Style" key="9999/1825823367/100/1825849146/2/100" value="1 (Multiline Background)"/>')
        xml.append('                                <param name="Roundness" key="9999/1825838212/1825838597/3/1825838620/1" value="0"/>')
        xml.append(f'                                <param name="Position" key="9999/1825870248/10044/10045/1/100/101" value="0 {pos_y}"/>')
        xml.append('                                <param name="Scale" key="9999/1825870248/10044/10045/1/100/105" value="0 0"/>')
        xml.append('                                <param name="Opacity" key="9999/1825870248/1825841471/10043/1825838143/1825838107/1/200/202" value="0.9"/>')
        xml.append('                                <param name="Color" key="9999/1825870248/1825841471/10043/1825838143/1825838107/10085/3/1825838231/2" value="0 0 0"/>')
        xml.append('                                <param name="Offset Vertical" key="9999/1825870248/1825841471/10043/1825838143/1825838107/10085/4/10380/10384" value="5"/>')
        xml.append(f"                                <text>")
        xml.append(f'                                    <text-style ref="{ts_id}">{t}</text-style>')
        xml.append(f"                                </text>")
        xml.append(f'                                <text-style-def id="{ts_id}">')
        xml.append(
            f'                                    <text-style font="{escape(font)}"'
            f' fontSize="{font_size}"'
            f' fontFace="Medium"'
            f' fontColor="{font_rgba}"'
            f' alignment="center"'
            f' lineSpacing="20"/>'
        )
        xml.append(f"                                </text-style-def>")
        xml.append(f"                            </title>")

    xml.append("                        </gap>")
    xml.append("                    </spine>")
    xml.append("                </sequence>")
    xml.append("            </project>")
    xml.append("        </event>")
    xml.append("    </library>")
    xml.append("</fcpxml>")

    return "\n".join(xml) + "\n"
