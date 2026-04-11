from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import gradio as gr

from agents.subtitle_agent import SubtitleAgent
from skills.audio import AudioSkill
from skills.files import FileSkill

_ENV_PATH = Path(__file__).parent / ".env"


def _load_saved_key() -> str:
    if not _ENV_PATH.exists():
        return ""
    for line in _ENV_PATH.read_text(encoding="utf-8").splitlines():
        if line.startswith("OPENAI_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


def _save_key(api_key: str) -> None:
    key = api_key.strip()
    if _ENV_PATH.exists():
        lines = [l for l in _ENV_PATH.read_text(encoding="utf-8").splitlines()
                 if not l.startswith("OPENAI_API_KEY=")]
    else:
        lines = []
    lines.append(f"OPENAI_API_KEY={key}")
    _ENV_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _key_from_file(file_path: str | None) -> str:
    if not file_path:
        return ""
    content = Path(file_path).read_text(encoding="utf-8").strip()
    for line in content.splitlines():
        if line.startswith("OPENAI_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    return content.splitlines()[0].strip() if content else ""


# ---------------------------------------------------------------------------
# WhisperX pipeline (local, no API key)
# ---------------------------------------------------------------------------
def _generate_whisperx(
    video_file: str, prompt: str, whisperx_model: str,
    progress: gr.Progress = gr.Progress(),
):
    if not video_file:
        raise gr.Error("Please upload a video file.")

    input_video = Path(video_file)
    work_dir = Path(tempfile.mkdtemp())

    def on_progress(stage: str, pct: float) -> None:
        label = "Transcribing" if stage == "transcribing" else "Aligning timestamps"
        progress(pct / 100, desc=f"{label} — {pct:.0f}%")

    try:
        from skills.transcription_whisperx import WhisperXTranscriptionSkill

        audio_dir = work_dir / "audio"
        audio_dir.mkdir()

        files = FileSkill()
        audio_skill = AudioSkill()
        subtitle_agent = SubtitleAgent()

        progress(0, desc="Extracting audio...")

        source_audio = audio_skill.extract_audio(
            input_video=input_video,
            output_audio=audio_dir / "source.mp3",
            bitrate="64k",
            sample_rate=16000,
            channels=1,
        )

        # Check if model needs downloading or just loading from cache
        from pathlib import Path as _P
        _cache = _P.home() / ".cache" / "huggingface" / "hub" / f"models--Systran--faster-whisper-{whisperx_model}"
        if _cache.exists() and not any(_cache.rglob("*.incomplete")):
            progress(0, desc="Loading WhisperX model from cache...")
        else:
            progress(0, desc=f"Downloading WhisperX model '{whisperx_model}' (this only happens once)...")

        skill = WhisperXTranscriptionSkill(
            model_name=whisperx_model,
            device="cpu",
            compute_type="int8",
            initial_prompt=prompt or "",
        )

        progress(0, desc="Transcribing — 0%...")

        segments = skill.transcribe(
            audio_path=source_audio,
            progress_callback=on_progress,
        )

        if not segments:
            raise gr.Error("No transcript segments were produced. Check your video has audio.")

        # Use AI to intelligently split into readable subtitles
        saved_key = _load_saved_key()
        if saved_key:
            progress(0.90, desc="AI is formatting subtitles...")
            sentence_segments = subtitle_agent.reformat_with_ai(segments, api_key=saved_key)
        else:
            progress(0.90, desc="Building SRT file (no API key — using basic splitting)...")
            sentence_segments = subtitle_agent.reformat_as_sentences(segments)

        progress(0.95, desc="Building SRT file...")
        srt_text = subtitle_agent.to_srt(sentence_segments)

        stable_dir = Path(tempfile.mkdtemp())
        stable_srt = stable_dir / "subtitle.srt"
        files.save_text(stable_srt, srt_text)
        shutil.rmtree(work_dir, ignore_errors=True)

        progress(1.0, desc="Done!")
        return str(stable_srt)

    except gr.Error:
        shutil.rmtree(work_dir, ignore_errors=True)
        raise
    except Exception as e:
        shutil.rmtree(work_dir, ignore_errors=True)
        raise gr.Error(str(e))


# ---------------------------------------------------------------------------
# OpenAI API pipeline (requires API key)
# ---------------------------------------------------------------------------
def _generate_openai(api_key: str, video_file: str, chunk_seconds: int, prompt: str):
    if not api_key or not api_key.strip():
        raise gr.Error("Please enter your OpenAI API key.")
    api_key = api_key.strip()
    if not api_key.startswith("sk-"):
        raise gr.Error("Invalid API key — OpenAI keys must start with 'sk-'.")
    if len(api_key) < 40:
        raise gr.Error("Invalid API key — key is too short.")
    if not video_file:
        raise gr.Error("Please upload a video file.")

    from skills.transcription import TranscriptionSkill

    input_video = Path(video_file)
    work_dir = Path(tempfile.mkdtemp())

    try:
        audio_dir = work_dir / "audio"
        chunks_dir = work_dir / "chunks"
        audio_dir.mkdir()
        chunks_dir.mkdir()

        files = FileSkill()
        audio_skill = AudioSkill()
        transcription_skill = TranscriptionSkill(api_key=api_key)
        subtitle_agent = SubtitleAgent()

        yield "Extracting audio...", None

        source_audio = audio_skill.extract_audio(
            input_video=input_video,
            output_audio=audio_dir / "source.mp3",
            bitrate="64k",
            sample_rate=16000,
            channels=1,
        )

        yield "Splitting audio into chunks...", None

        chunk_paths = audio_skill.split_audio(source_audio, chunks_dir, chunk_seconds)

        all_chunk_segments = []
        current_offset = 0.0
        for i, chunk_path in enumerate(chunk_paths, start=1):
            yield f"Transcribing chunk {i} of {len(chunk_paths)}...", None
            chunk_segments = transcription_skill.transcribe_chunk(
                audio_path=chunk_path,
                time_offset=current_offset,
                prompt=prompt or "",
            )
            all_chunk_segments.append(chunk_segments)
            current_offset += audio_skill.get_duration_seconds(chunk_path)

        merged_segments = transcription_skill.merge_segment_lists(all_chunk_segments)
        if not merged_segments:
            raise gr.Error("No transcript segments were produced. Check your video has audio.")

        yield "Building SRT file...", None

        sentence_segments = subtitle_agent.reformat_as_sentences(merged_segments)
        srt_text = subtitle_agent.to_srt(sentence_segments)

        stable_dir = Path(tempfile.mkdtemp())
        stable_srt = stable_dir / "subtitle.srt"
        files.save_text(stable_srt, srt_text)
        shutil.rmtree(work_dir, ignore_errors=True)

        yield "Done!", str(stable_srt)

    except gr.Error:
        shutil.rmtree(work_dir, ignore_errors=True)
        raise
    except Exception as e:
        shutil.rmtree(work_dir, ignore_errors=True)
        raise gr.Error(str(e))


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
css = """
.container { max-width: 760px; margin: auto; }
.download-btn { min-height: 60px; }
.download-btn a { font-size: 1.1em; }
@keyframes spin { to { transform: rotate(360deg); } }
.spinner::before {
    content: "";
    display: inline-block;
    width: 14px; height: 14px;
    border: 2px solid #888;
    border-top-color: transparent;
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
    margin-right: 8px;
    vertical-align: middle;
}
"""

with gr.Blocks(title="Subtitle Generator") as demo:
    with gr.Column(elem_classes="container"):
        gr.Markdown(
            "# Subtitle Generator\n"
            "Generate `.srt` subtitle files from any video.\n\n"
            "**WhisperX** runs locally (no API key, best timestamps). "
            "**OpenAI API** is available as a fallback."
        )

        engine_toggle = gr.Radio(
            label="Transcription engine",
            choices=["WhisperX (local)", "OpenAI API"],
            value="WhisperX (local)",
        )

        # --- OpenAI API key section (hidden by default) ---
        with gr.Column(visible=False) as openai_section:
            api_key = gr.Textbox(
                label="OpenAI API Key",
                placeholder="sk-...",
                type="password",
                value=_load_saved_key(),
                info="Don't have one? Get it at platform.openai.com/api-keys",
            )
            with gr.Row():
                key_file = gr.File(
                    label="Or drag & drop a key file (.txt or .env)",
                    file_types=[".txt", ".env"],
                    scale=3,
                )
                save_key_btn = gr.Button("Save key for next time", scale=1)

        video_input = gr.File(
            label="Video File",
            file_types=["video"],
        )
        prompt_input = gr.Textbox(
            label="Vocabulary hints",
            placeholder="Names, brands, technical terms, show titles...",
            info="Helps Whisper correctly spell proper nouns and uncommon words. Separate with commas.",
        )

        with gr.Accordion("Advanced options", open=False):
            whisperx_model = gr.Dropdown(
                label="WhisperX model",
                choices=["large-v2", "medium", "small", "base", "tiny"],
                value="large-v2",
                info="large-v2: best accuracy. smaller = faster but less accurate.",
            )
            chunk_seconds = gr.Slider(
                label="Chunk size (seconds) — OpenAI only",
                minimum=60,
                maximum=1200,
                value=600,
                step=60,
                info="Only used with OpenAI API engine.",
            )

        generate_btn = gr.Button("Generate Subtitles", variant="primary")
        status = gr.HTML(visible=False)
        srt_output = gr.File(label="Download SRT", interactive=False, visible=False, elem_classes="download-btn")

    # --- Toggle OpenAI section visibility ---
    def toggle_engine(engine):
        return gr.update(visible=(engine == "OpenAI API"))

    engine_toggle.change(
        fn=toggle_engine,
        inputs=[engine_toggle],
        outputs=[openai_section],
    )

    # --- Key file helpers ---
    key_file.change(fn=_key_from_file, inputs=[key_file], outputs=[api_key])
    save_key_btn.click(fn=_save_key, inputs=[api_key], outputs=[])

    # --- Main generation ---
    def run(engine, api_key, video_file, chunk_seconds, prompt, whisperx_model, progress=gr.Progress()):
        if engine == "WhisperX (local)":
            # WhisperX uses gr.Progress directly for real-time updates
            srt_path = _generate_whisperx(video_file, prompt, whisperx_model, progress)
            yield (
                gr.update(value="", visible=False),
                gr.update(value=srt_path, visible=True),
            )
        else:
            # OpenAI API uses the yield/generator pattern
            for status_text, srt_path in _generate_openai(api_key, video_file, chunk_seconds, prompt):
                if srt_path is not None:
                    yield (
                        gr.update(value="", visible=False),
                        gr.update(value=srt_path, visible=True),
                    )
                else:
                    html = f'<div class="spinner" style="padding:12px 0;color:#ccc;font-size:1em;">{status_text}</div>'
                    yield (
                        gr.update(value=html, visible=True),
                        gr.update(visible=False),
                    )

    generate_btn.click(
        fn=run,
        inputs=[engine_toggle, api_key, video_input, chunk_seconds, prompt_input, whisperx_model],
        outputs=[status, srt_output],
    )


if __name__ == "__main__":
    demo.launch(inbrowser=True, css=css)
