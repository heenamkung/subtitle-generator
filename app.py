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


def _status_html(text: str) -> str:
    return (
        f'<div style="padding:12px 0;color:#ccc;font-size:1em;">'
        f'<span class="spinner"></span>{text}</div>'
    )


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
            "Generate `.srt` subtitle files from any video using WhisperX.\n\n"
            "Transcription runs **locally** on your machine. "
            "An OpenAI API key is used for AI-powered subtitle formatting."
        )

        video_input = gr.File(
            label="Video File",
            file_types=["video"],
        )
        whisperx_model = gr.Dropdown(
            label="Whisper model",
            choices=["large-v2", "medium", "small", "base", "tiny"],
            value="medium",
            info="medium: good balance of speed and accuracy. large-v2: best accuracy but slower.",
        )
        prompt_input = gr.Textbox(
            label="Vocabulary hints",
            placeholder="Names, brands, technical terms, show titles...",
            info="Helps Whisper correctly spell proper nouns and uncommon words. Separate with commas.",
        )

        with gr.Accordion("OpenAI API Key (for subtitle formatting)", open=False):
            gr.Markdown(
                "The API key is used by GPT-4o-mini to add punctuation and split subtitles into "
                "natural sentences. Without it, basic splitting is used instead."
            )
            api_key = gr.Textbox(
                label="API Key",
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

        generate_btn = gr.Button("Generate Subtitles", variant="primary")
        status = gr.HTML(visible=False)
        srt_output = gr.File(label="Download SRT", interactive=False, elem_classes="download-btn")

    # --- Key file helpers ---
    key_file.change(fn=_key_from_file, inputs=[key_file], outputs=[api_key])
    save_key_btn.click(fn=_save_key, inputs=[api_key], outputs=[])

    # --- Main generation (generator for real-time UI updates) ---
    def run(video_file, whisperx_model_name, prompt):
        if not video_file:
            raise gr.Error("Please upload a video file.")

        input_video = Path(video_file)
        work_dir = Path(tempfile.mkdtemp())

        try:
            from skills.transcription_whisperx import WhisperXTranscriptionSkill

            audio_dir = work_dir / "audio"
            audio_dir.mkdir()

            files = FileSkill()
            audio_skill = AudioSkill()
            subtitle_agent = SubtitleAgent()

            # --- Extracting audio ---
            yield (
                gr.update(value=_status_html("Extracting audio..."), visible=True),
                gr.update(),
            )

            source_audio = audio_skill.extract_audio(
                input_video=input_video,
                output_audio=audio_dir / "source.mp3",
                bitrate="64k",
                sample_rate=16000,
                channels=1,
            )

            # --- Loading model ---
            _cache = Path.home() / ".cache" / "huggingface" / "hub" / f"models--Systran--faster-whisper-{whisperx_model_name}"
            if _cache.exists() and not any(_cache.rglob("*.incomplete")):
                msg = "Loading WhisperX model from cache..."
            else:
                msg = f"Downloading WhisperX model '{whisperx_model_name}' (this only happens once)..."

            yield (
                gr.update(value=_status_html(msg), visible=True),
                gr.update(),
            )

            skill = WhisperXTranscriptionSkill(
                model_name=whisperx_model_name,
                device="cpu",
                compute_type="int8",
                initial_prompt=prompt or "",
            )

            # --- Transcribing (VAD + Whisper + alignment) ---
            yield (
                gr.update(value=_status_html("Detecting speech and transcribing..."), visible=True),
                gr.update(),
            )

            segments = skill.transcribe(audio_path=source_audio)

            if not segments:
                raise gr.Error("No transcript segments were produced. Check your video has audio.")

            # --- AI formatting ---
            saved_key = _load_saved_key()
            if saved_key:
                yield (
                    gr.update(value=_status_html("AI is formatting subtitles..."), visible=True),
                    gr.update(),
                )
                sentence_segments = subtitle_agent.reformat_with_ai(segments, api_key=saved_key)
            else:
                yield (
                    gr.update(value=_status_html("Formatting subtitles (no API key — basic splitting)..."), visible=True),
                    gr.update(),
                )
                sentence_segments = subtitle_agent.reformat_as_sentences(segments)

            # --- Building SRT ---
            yield (
                gr.update(value=_status_html("Building SRT file..."), visible=True),
                gr.update(),
            )

            duration_segments = subtitle_agent.enforce_max_duration(sentence_segments, max_duration=3.0)
            final_segments = subtitle_agent.merge_orphans(duration_segments)
            srt_text = subtitle_agent.to_srt(final_segments)

            stable_dir = Path(tempfile.mkdtemp())
            stable_srt = stable_dir / "subtitle.srt"
            files.save_text(stable_srt, srt_text)
            shutil.rmtree(work_dir, ignore_errors=True)
            print(f"  Done! SRT saved to: {stable_srt}")

            yield (
                gr.update(value="", visible=False),
                gr.update(value=str(stable_srt)),
            )

        except gr.Error:
            shutil.rmtree(work_dir, ignore_errors=True)
            raise
        except Exception as e:
            shutil.rmtree(work_dir, ignore_errors=True)
            raise gr.Error(str(e))

    generate_btn.click(
        fn=run,
        inputs=[video_input, whisperx_model, prompt_input],
        outputs=[status, srt_output],
    )


if __name__ == "__main__":
    demo.launch(inbrowser=True, css=css)
