from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import gradio as gr

from agents.subtitle_agent import SubtitleAgent
from skills.audio import AudioSkill
from skills.files import FileSkill
from skills.transcription import TranscriptionSkill


def generate_subtitles(api_key: str, video_file: str, chunk_seconds: int, prompt: str):
    if not api_key or not api_key.strip():
        raise gr.Error("Please enter your OpenAI API key.")
    api_key = api_key.strip()
    if not api_key.startswith("sk-"):
        raise gr.Error("Invalid API key — OpenAI keys must start with 'sk-'.")
    if len(api_key) < 40:
        raise gr.Error("Invalid API key — key is too short.")
    if not video_file:
        raise gr.Error("Please upload a video file.")

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

        yield f"Transcribing {len(chunk_paths)} chunk(s)...", None

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

        srt_path = work_dir / f"{input_video.stem}.srt"
        files.save_text(srt_path, srt_text)

        # Copy SRT to a stable temp file so we can clean up the work dir
        import shutil as _shutil
        stable_srt = Path(tempfile.mktemp(suffix=".srt"))
        _shutil.copy2(srt_path, stable_srt)
        shutil.rmtree(work_dir, ignore_errors=True)

        yield "Done!", str(stable_srt)

    except gr.Error:
        shutil.rmtree(work_dir, ignore_errors=True)
        raise
    except Exception as e:
        shutil.rmtree(work_dir, ignore_errors=True)
        raise gr.Error(str(e))


css = """
.container { max-width: 760px; margin: auto; }
"""

with gr.Blocks(title="Subtitle Generator") as demo:
    with gr.Column(elem_classes="container"):
        gr.Markdown("# Subtitle Generator\nGenerate `.srt` subtitle files from any video using OpenAI Whisper.")
        api_key = gr.Textbox(
            label="OpenAI API Key",
            placeholder="sk-...",
            type="password",
        )
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
            chunk_seconds = gr.Slider(
                label="Chunk size (seconds)",
                minimum=60,
                maximum=1200,
                value=600,
                step=60,
                info="Reduce this for very long videos.",
            )
        generate_btn = gr.Button("Generate Subtitles", variant="primary")
        status = gr.Textbox(label="Status", interactive=False, visible=False)
        srt_output = gr.File(label="Download SRT", interactive=False, visible=False)

    def run(api_key, video_file, chunk_seconds, prompt):
        for status_text, srt_path in generate_subtitles(api_key, video_file, chunk_seconds, prompt):
            is_done = srt_path is not None
            yield (
                gr.update(value=status_text, visible=True),
                gr.update(value=srt_path, visible=is_done),
            )

    generate_btn.click(
        fn=run,
        inputs=[api_key, video_input, chunk_seconds, prompt_input],
        outputs=[status, srt_output],
    )


if __name__ == "__main__":
    demo.launch(inbrowser=True, css=css)
