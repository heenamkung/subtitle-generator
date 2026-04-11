from __future__ import annotations

import argparse
from pathlib import Path

from agents.subtitle_agent import SubtitleAgent
from config import Settings
from skills.audio import AudioSkill
from skills.files import FileSkill


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
        "--engine",
        type=str,
        default="whisperx",
        choices=["whisperx", "openai"],
        help="Transcription engine: whisperx (local, default) or openai (API).",
    )
    parser.add_argument(
        "--whisperx-model",
        type=str,
        default="large-v2",
        choices=["tiny", "base", "small", "medium", "large-v2"],
        help="WhisperX model size (default: large-v2).",
    )
    parser.add_argument(
        "--chunk-seconds",
        type=int,
        default=600,
        help="Audio chunk length in seconds (only used with --engine openai).",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="",
        help="Vocabulary hints for Whisper (proper nouns, channel name, show titles, etc.).",
    )
    return parser.parse_args()


def _run_whisperx(
    source_audio: Path,
    prompt: str,
    model_name: str,
    settings: Settings,
) -> list:
    """Transcribe using WhisperX (local, no API key needed)."""
    from skills.transcription_whisperx import WhisperXTranscriptionSkill

    skill = WhisperXTranscriptionSkill(
        model_name=model_name,
        device=settings.whisperx_device,
        compute_type=settings.whisperx_compute_type,
        initial_prompt=prompt,
    )
    print("[2/4] Transcribing with WhisperX (this may take a while on first run)...")
    return skill.transcribe(audio_path=source_audio)


def _run_openai(
    source_audio: Path,
    chunks_dir: Path,
    prompt: str,
    chunk_seconds: int,
    settings: Settings,
    audio_skill: AudioSkill,
) -> list:
    """Transcribe using OpenAI Whisper API (requires API key)."""
    from skills.transcription import TranscriptionSkill

    if not settings.openai_api_key:
        raise RuntimeError(
            "OpenAI API key is required for --engine openai. "
            "Set OPENAI_API_KEY in your .env file."
        )

    skill = TranscriptionSkill(
        api_key=settings.openai_api_key,
        model=settings.transcription_model,
    )

    print(f"[2/4] Splitting audio into {chunk_seconds}s chunks")
    chunk_paths = audio_skill.split_audio(source_audio, chunks_dir, chunk_seconds)

    print(f"[3/4] Transcribing {len(chunk_paths)} chunk(s) via OpenAI API")
    all_chunk_segments = []
    current_offset = 0.0
    for chunk_path in chunk_paths:
        print(f"  - {chunk_path.name}")
        chunk_segments = skill.transcribe_chunk(
            audio_path=chunk_path,
            time_offset=current_offset,
            prompt=prompt,
        )
        all_chunk_segments.append(chunk_segments)
        current_offset += audio_skill.get_duration_seconds(chunk_path)

    return skill.merge_segment_lists(all_chunk_segments)


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
    subtitle_agent = SubtitleAgent()

    print(f"[1/4] Extracting audio from: {input_video.name}")
    source_audio = audio_skill.extract_audio(
        input_video=input_video,
        output_audio=audio_dir / "source.mp3",
        bitrate=settings.audio_bitrate,
        sample_rate=settings.audio_sample_rate,
        channels=settings.audio_channels,
    )

    # Transcribe using the selected engine
    if args.engine == "whisperx":
        segments = _run_whisperx(source_audio, args.prompt, args.whisperx_model, settings)
    else:
        segments = _run_openai(
            source_audio, chunks_dir, args.prompt,
            args.chunk_seconds, settings, audio_skill,
        )

    if not segments:
        raise RuntimeError("No transcript segments were produced.")

    step = "3/4" if args.engine == "whisperx" else "4/4"
    if settings.openai_api_key:
        print(f"[{step}] AI is formatting subtitles...")
        sentence_segments = subtitle_agent.reformat_with_ai(segments, api_key=settings.openai_api_key)
    else:
        print(f"[{step}] Building SRT file (no API key — using basic splitting)")
        sentence_segments = subtitle_agent.reformat_as_sentences(segments)
    duration_segments = subtitle_agent.enforce_max_duration(sentence_segments, max_duration=3.0)
    final_segments = subtitle_agent.merge_orphans(duration_segments)
    srt_text = subtitle_agent.to_srt(final_segments)

    final_step = "4/4" if args.engine == "whisperx" else "4/4"
    print(f"[{final_step}] Saving output files")
    segments_path = project_dir / "segments.json"
    srt_path = project_dir / f"{input_video.stem}.srt"

    files.save_json(segments_path, [segment.to_dict() for segment in segments])
    files.save_text(srt_path, srt_text)

    print("Done")
    print(f"SRT: {srt_path}")
    print(f"Segments JSON: {segments_path}")


if __name__ == "__main__":
    main()
