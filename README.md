# Subtitle Generator

Generate `.srt` and `.fcpxml` subtitle files from any video using local WhisperX transcription with word-level timestamps.

**Key features:**
- **Runs locally** — no audio is uploaded anywhere. WhisperX transcribes on your machine.
- **Word-level timestamps** — WhisperX uses forced alignment for precise timing, not just segment-level guesses.
- **AI punctuation** — an optional OpenAI API key lets GPT-4o-mini add natural punctuation and sentence splitting.
- **FCPXML export** — import subtitles directly into Final Cut Pro with styled backgrounds (Tap5a Motion template, auto-installed).
- **Vocabulary hints** — feed names, brands, and technical terms so Whisper spells them correctly.

---

## Requirements

- macOS (Apple Silicon or Intel)
- Python 3.10+
- ffmpeg

---

## Setup

### 1. Install ffmpeg

```bash
brew install ffmpeg
```

### 2. Clone and install

```bash
git clone https://github.com/heenamkung/subtitle-generator.git
cd subtitle-generator
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Launch

```bash
python app.py
```

The app opens automatically in your browser.

On first run:
- The Tap5a Motion template is auto-installed to `~/Movies/Motion Templates.localized/` for FCP compatibility.
- The selected WhisperX model is downloaded once and cached at `~/.cache/huggingface/`.

---

## Usage

1. Upload your video file
2. Select a Whisper model (`large-v2` for best accuracy, `medium` for faster)
3. Add **vocabulary hints** — names, brands, anime characters, etc.
4. Click **Generate Subtitles**
5. Download the `.srt` and/or `.fcpxml` file

### FCPXML in Final Cut Pro

1. Download the `.fcpxml` file
2. In FCP: **File > Import > XML**
3. Subtitles appear as styled title clips on the timeline

### OpenAI API Key (optional)

The API key enables AI-powered punctuation via GPT-4o-mini. Without it, basic rule-based splitting is used instead.

- Enter the key once in the accordion — it auto-saves to `.env` for next time.
- Costs are minimal (GPT-4o-mini processes text only, not audio).
- Get a key at [platform.openai.com/api-keys](https://platform.openai.com/api-keys)

---

## CLI

```bash
# WhisperX (default, local)
python main.py /path/to/video.mp4

# Specify model
python main.py /path/to/video.mp4 --whisperx-model medium

# With vocabulary hints
python main.py /path/to/video.mp4 --prompt "Lala, Rito, Haruna"

# OpenAI Whisper API (requires API key in .env)
python main.py /path/to/video.mp4 --engine openai
```

Run `python main.py --help` for all options.

---

## Project Structure

```
app.py                          # Gradio web UI
main.py                         # CLI entry point
config.py                       # Settings (env vars, defaults)
models.py                       # TranscriptSegment, WordTiming
agents/subtitle_agent.py        # AI punctuation + subtitle formatting
skills/
  audio.py                      # ffmpeg audio extraction
  transcription_whisperx.py     # WhisperX local transcription
  transcription.py              # OpenAI Whisper API (fallback)
  fcpxml.py                     # SRT -> FCPXML converter
  files.py                      # File I/O utilities
templates/
  Tap5a Multiline Text Backgr. 2.moti   # FCP Motion template
utils/
  timecode.py                   # Timestamp formatting
```
