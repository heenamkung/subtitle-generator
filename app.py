from __future__ import annotations

import os
import shutil
import tempfile
import threading
import time
from pathlib import Path

import gradio as gr

# When running in Docker we skip the macOS-only Motion template install and
# let Gradio bind to 0.0.0.0 via env vars (set in the Dockerfile).
_IN_DOCKER = os.getenv("IN_DOCKER") == "1"

from agents.subtitle_agent import SubtitleAgent
from skills.audio import AudioSkill
from skills.fcpxml import parse_srt, generate_fcpxml
from skills.files import FileSkill

_ENV_PATH = Path(__file__).parent / ".env"

# Motion template: bundled in repo, auto-installed to FCP's template folder on startup
_TEMPLATE_NAME = "Tap5a Multiline Text Backgr. 2.moti"
_TEMPLATE_SRC = Path(__file__).parent / "templates" / _TEMPLATE_NAME
_TEMPLATE_DST = (
    Path.home() / "Movies" / "Motion Templates.localized"
    / "Titles.localized" / "Tap5a" / _TEMPLATE_NAME
)


def _install_motion_template() -> None:
    """Copy the bundled Motion template to FCP's template folder if not already there."""
    if _IN_DOCKER:
        # Containers can't write to the host's ~/Movies. The user must copy
        # templates/Tap5a Multiline Text Backgr. 2.moti manually on the Mac host.
        print("  Skipping Motion template install (running in Docker).")
        return
    if _TEMPLATE_DST.exists():
        return
    _TEMPLATE_DST.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(_TEMPLATE_SRC, _TEMPLATE_DST)
    print(f"  Installed Motion template → {_TEMPLATE_DST}")


_install_motion_template()


def _load_saved_key() -> str:
    if not _ENV_PATH.exists():
        return ""
    for line in _ENV_PATH.read_text(encoding="utf-8").splitlines():
        if line.startswith("OPENAI_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


def _save_key(api_key: str) -> None:
    key = api_key.strip()
    if not key:
        return  # Don't overwrite saved key with empty string
    if _ENV_PATH.exists():
        lines = [l for l in _ENV_PATH.read_text(encoding="utf-8").splitlines()
                 if not l.startswith("OPENAI_API_KEY=")]
    else:
        lines = []
    lines.append(f"OPENAI_API_KEY={key}")
    _ENV_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")




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

with gr.Blocks(title="Subtitle Generator", css=css) as demo:
    with gr.Column(elem_classes="container"):
        gr.Markdown("# Subtitle Generator")
        gr.Markdown(
            "Generate `.srt` and `.fcpxml` subtitle files from any video using WhisperX.\n\n"
            "Transcription runs **locally** — no audio is uploaded. "
            "An optional OpenAI API key enables AI-powered punctuation and natural sentence splitting. "
            "Only the text transcript is sent to OpenAI — never your audio or video."
        )

        video_input = gr.File(
            label="Video File",
            file_types=["video"],
        )
        whisperx_model = gr.Dropdown(
            label="Whisper model",
            choices=["large-v2", "medium", "small", "base", "tiny"],
            value="large-v2",
            info="large-v2: best accuracy but slower. medium: good balance of speed and accuracy.",
        )
        prompt_input = gr.Textbox(
            label="Vocabulary hints",
            placeholder="Anime names, character names, brands, technical terms...",
            info="Helps Whisper correctly spell proper nouns and uncommon words. Separate with commas.",
        )

        gr.Markdown(
            "### OpenAI API Key (highly recommended)\n"
            "Adding an API key **dramatically improves subtitle quality** — GPT-4o-mini adds "
            "natural punctuation, sentence breaks, and commas for far more readable subtitles. "
            "Without it, you'll get basic splitting that may feel rough."
        )
        api_key = gr.Textbox(
            label="API Key",
            placeholder="sk-...",
            type="password",
            value=_load_saved_key(),
            info="Don't have one? Get it at platform.openai.com/api-keys",
        )

        generate_btn = gr.Button("Generate Subtitles", variant="primary", visible=False)
        gen_status = gr.HTML(visible=False)
        with gr.Row():
            srt_output = gr.File(label="Download SRT", interactive=False, elem_classes="download-btn", visible=False)
            fcpxml_gen_output = gr.File(label="Download FCPXML", interactive=False, elem_classes="download-btn", visible=False)

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------
    # Auto-save the key whenever the user changes it (no separate Save button needed)
    api_key.change(fn=_save_key, inputs=[api_key], outputs=[])

    # Only show the Generate button once a video file has been uploaded.
    video_input.change(
        fn=lambda f: gr.update(visible=f is not None),
        inputs=[video_input],
        outputs=[generate_btn],
    )

    def run_generate(video_file, whisperx_model_name, prompt, openai_key):
        if not video_file:
            raise gr.Error("Please upload a video file.")

        input_video = Path(video_file)
        work_dir = Path(tempfile.mkdtemp())

        try:
            # Lazy imports: heavy ML deps — don't load at module import time.
            from skills.transcription_whisperx import (
                WhisperXTranscriptionSkill,
                _MODEL_SIZES_MB,
                _dir_size_mb,
            )

            audio_dir = work_dir / "audio"
            audio_dir.mkdir()

            files = FileSkill()
            audio_skill = AudioSkill()
            subtitle_agent = SubtitleAgent()

            yield (
                gr.update(value=_status_html("Extracting audio..."), visible=True),
                gr.update(value=None, visible=False),
                gr.update(value=None, visible=False),
            )

            source_audio = audio_skill.extract_audio(
                input_video=input_video,
                output_audio=audio_dir / "source.mp3",
                bitrate="64k",
                sample_rate=16000,
                channels=1,
            )

            # ----------------------------------------------------------
            # Model load (may download ~3 GB on first run)
            # ----------------------------------------------------------
            cache_dir = (
                Path.home() / ".cache" / "huggingface" / "hub"
                / f"models--Systran--faster-whisper-{whisperx_model_name}"
            )
            has_cache = cache_dir.exists() and not any(cache_dir.rglob("*.incomplete"))
            expected_mb = _MODEL_SIZES_MB.get(whisperx_model_name, 1500)

            skill = WhisperXTranscriptionSkill(
                model_name=whisperx_model_name,
                device="cpu",
                compute_type="int8",
                initial_prompt=prompt or "",
            )

            # Run the blocking load on a thread so the generator can keep
            # yielding UI updates while the download / load happens.
            load_err: dict = {}

            def _load() -> None:
                try:
                    skill._ensure_model()
                except Exception as exc:
                    load_err["e"] = exc

            load_thread = threading.Thread(target=_load, daemon=True)
            load_thread.start()

            while load_thread.is_alive():
                load_thread.join(timeout=2.0)
                if not load_thread.is_alive():
                    break
                if has_cache:
                    msg = "Loading WhisperX model from cache..."
                else:
                    mb = _dir_size_mb(cache_dir)
                    pct = min(99, int((mb / expected_mb) * 100))
                    msg = (
                        f"Downloading WhisperX model '{whisperx_model_name}' — "
                        f"{mb:.0f} / ~{expected_mb} MB ({pct}%). One-time download."
                    )
                yield (
                    gr.update(value=_status_html(msg), visible=True),
                    gr.update(),
                    gr.update(),
                )

            if "e" in load_err:
                raise gr.Error(f"Failed to load model: {load_err['e']}")

            # ----------------------------------------------------------
            # Transcription + forced alignment (1-3 min on CPU, typically)
            # ----------------------------------------------------------
            # We can't get true % progress without patching WhisperX, but
            # showing elapsed time reassures the user nothing is stuck.
            transcribe_result: dict = {}
            transcribe_err: dict = {}

            def _transcribe() -> None:
                try:
                    transcribe_result["segments"] = skill.transcribe(audio_path=source_audio)
                except Exception as exc:
                    transcribe_err["e"] = exc

            tr_start = time.monotonic()
            tr_thread = threading.Thread(target=_transcribe, daemon=True)
            tr_thread.start()

            while tr_thread.is_alive():
                tr_thread.join(timeout=2.0)
                if not tr_thread.is_alive():
                    break
                elapsed = int(time.monotonic() - tr_start)
                mins, secs = divmod(elapsed, 60)
                time_str = f"{mins}m {secs:02d}s" if mins else f"{secs}s"
                yield (
                    gr.update(
                        value=_status_html(
                            f"Transcribing audio and aligning word timestamps... "
                            f"({time_str} elapsed)"
                        ),
                        visible=True,
                    ),
                    gr.update(),
                    gr.update(),
                )

            if "e" in transcribe_err:
                raise gr.Error(str(transcribe_err["e"]))

            segments = transcribe_result["segments"]

            if not segments:
                raise gr.Error("No transcript segments were produced. Check your video has audio.")

            # Use the key from UI input (falls back to saved key if UI is empty)
            effective_key = (openai_key or "").strip() or _load_saved_key()
            if effective_key:
                yield (
                    gr.update(value=_status_html("AI is formatting subtitles..."), visible=True),
                    gr.update(),
                    gr.update(),
                )
                sentence_segments = subtitle_agent.reformat_with_ai(segments, api_key=effective_key)
            else:
                yield (
                    gr.update(value=_status_html("Formatting subtitles (no API key — basic splitting)..."), visible=True),
                    gr.update(),
                    gr.update(),
                )
                sentence_segments = subtitle_agent.reformat_as_sentences(segments)

            yield (
                gr.update(value=_status_html("Building subtitle files..."), visible=True),
                gr.update(),
                gr.update(),
            )

            duration_segments = subtitle_agent.enforce_max_duration(sentence_segments, max_duration=3.0)
            final_segments = subtitle_agent.merge_orphans(duration_segments)
            srt_text = subtitle_agent.to_srt(final_segments)

            # Also generate FCPXML from the SRT
            fcpxml_entries = parse_srt(srt_text)
            fcpxml_text = generate_fcpxml(subtitles=fcpxml_entries)

            stable_dir = Path(tempfile.mkdtemp())
            stable_srt = stable_dir / "subtitle.srt"
            stable_fcpxml = stable_dir / "subtitle.fcpxml"
            files.save_text(stable_srt, srt_text)
            files.save_text(stable_fcpxml, fcpxml_text)
            shutil.rmtree(work_dir, ignore_errors=True)
            print(f"  Done! SRT: {stable_srt}, FCPXML: {stable_fcpxml}")

            yield (
                gr.update(value="", visible=False),
                gr.update(value=str(stable_srt), visible=True),
                gr.update(value=str(stable_fcpxml), visible=True),
            )

        except gr.Error:
            shutil.rmtree(work_dir, ignore_errors=True)
            raise
        except Exception as e:
            shutil.rmtree(work_dir, ignore_errors=True)
            raise gr.Error(str(e))

    generate_btn.click(
        fn=run_generate,
        inputs=[video_input, whisperx_model, prompt_input, api_key],
        outputs=[gen_status, srt_output, fcpxml_gen_output],
    )



if __name__ == "__main__":
    # In Docker there's no browser to open, and GRADIO_SERVER_NAME is already
    # set to 0.0.0.0 via the Dockerfile env. On the Mac host, open a browser tab.
    demo.launch(inbrowser=not _IN_DOCKER)
