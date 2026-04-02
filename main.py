from __future__ import annotations

import argparse
from pathlib import Path

from agents.subtitle_agent import SubtitleAgent
from config import Settings
from skills.audio import AudioSkill
from skills.files import FileSkill
from skills.transcription import TranscriptionSkill


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract audio from a video and generate an SRT subtitle file."
    )
    parser.add_argument("input_video", type=Path, help="Path to the source video file")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output root directory (defaults to config output/)",
    )
    parser.add_argument(
        "--chunk-seconds",
        type=int,
        default=600,
        help="Audio chunk length in seconds. Lower this for very long files.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="",
        help="Vocabulary hints for Whisper (proper nouns, channel name, show titles, etc.).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_video = args.input_video.expanduser().resolve()
    if not input_video.exists():
        raise FileNotFoundError(f"Input video not found: {input_video}")

    settings = Settings.load()
    output_root = args.output_dir.expanduser().resolve() if args.output_dir else settings.output_root.resolve()

    project_dir = output_root / input_video.stem
    audio_dir = project_dir / "audio"
    chunks_dir = project_dir / "chunks"

    files = FileSkill()
    files.ensure_dir(project_dir)
    files.ensure_dir(audio_dir)
    files.ensure_dir(chunks_dir)

    audio_skill = AudioSkill()
    transcription_skill = TranscriptionSkill(
        api_key=settings.openai_api_key,
        model=settings.transcription_model,
    )
    subtitle_agent = SubtitleAgent()

    print(f"[1/5] Extracting audio from: {input_video.name}")
    source_audio = audio_skill.extract_audio(
        input_video=input_video,
        output_audio=audio_dir / "source.mp3",
        bitrate=settings.audio_bitrate,
        sample_rate=settings.audio_sample_rate,
        channels=settings.audio_channels,
    )

    print(f"[2/5] Splitting audio into {args.chunk_seconds}s chunks")
    chunk_paths = audio_skill.split_audio(source_audio, chunks_dir, args.chunk_seconds)

    print(f"[3/5] Transcribing {len(chunk_paths)} chunk(s)")
    all_chunk_segments = []
    current_offset = 0.0

    for chunk_path in chunk_paths:
        print(f"  - {chunk_path.name}")
        chunk_segments = transcription_skill.transcribe_chunk(
            audio_path=chunk_path,
            time_offset=current_offset,
            prompt=args.prompt,
        )
        all_chunk_segments.append(chunk_segments)
        current_offset += audio_skill.get_duration_seconds(chunk_path)

    merged_segments = transcription_skill.merge_segment_lists(all_chunk_segments)
    if not merged_segments:
        raise RuntimeError("No transcript segments were produced.")

    print("[4/5] Building SRT file")
    sentence_segments = subtitle_agent.reformat_as_sentences(merged_segments)
    srt_text = subtitle_agent.to_srt(sentence_segments)

    print("[5/5] Saving output files")
    segments_path = project_dir / "segments.json"
    srt_path = project_dir / f"{input_video.stem}.srt"

    files.save_json(segments_path, [segment.to_dict() for segment in merged_segments])
    files.save_text(srt_path, srt_text)

    print("Done")
    print(f"SRT: {srt_path}")
    print(f"Segments JSON: {segments_path}")


if __name__ == "__main__":
    main()
