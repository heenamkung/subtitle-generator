# Subtitle Generator

Generate `.srt` and `.fcpxml` subtitle files from any video using local WhisperX transcription with word-level timestamps.

**Key features:**
- **Runs locally** — no audio is uploaded anywhere. WhisperX transcribes on your machine.
- **Word-level timestamps** — WhisperX uses forced alignment for precise timing, not just segment-level guesses.
- **AI punctuation** — GPT-4o-mini adds natural punctuation and sentence splitting for dramatically more readable subtitles (OpenAI API key required; only the text transcript is sent — never your audio).
- **FCPXML export** — import subtitles directly into Final Cut Pro with styled backgrounds (Tap5a Motion template, auto-installed).
- **Vocabulary hints** — feed names, brands, and technical terms so Whisper spells them correctly.

---

## Requirements

- macOS, Windows, or Linux
- Docker Desktop (recommended) **or** Python 3.10+ and ffmpeg (native setup)

---

## Setup

### Option A — Docker (recommended, works on macOS / Windows / Linux)

The easiest way. No Python, no ffmpeg, no dependency conflicts.

**macOS / Linux:**

```bash
git clone https://github.com/heenamkung/subtitle-generator.git
cd subtitle-generator
touch .env           # so compose can mount it for API-key persistence
docker compose up --build
```

**Windows (PowerShell):**

```powershell
git clone https://github.com/heenamkung/subtitle-generator.git
cd subtitle-generator
New-Item -ItemType File -Path .env -Force | Out-Null   # create empty .env
New-Item -ItemType Directory -Path output -Force | Out-Null
docker compose up --build
```

> Windows note: the `.env` file and `output/` directory must exist **before**
> `docker compose up`. If they don't, Docker Desktop silently creates them as
> directories instead of files, which breaks the bind-mount.

Then open http://localhost:7860 in your browser.

> **First run takes ~10 minutes** — Docker downloads and installs PyTorch,
> WhisperX, FFmpeg, and the Python runtime. This is a one-time cost.
> The first time you transcribe, WhisperX also downloads the ~3GB `large-v2`
> model. Both the image and the model are cached, so every subsequent run
> starts in seconds. Download progress shows in the web UI.

**Note for Final Cut Pro users (macOS only):** the container can't install the
Motion template to your Mac's `~/Movies/` folder. Copy it manually once:

```bash
mkdir -p "$HOME/Movies/Motion Templates.localized/Titles.localized/Tap5a"
cp "templates/Tap5a Multiline Text Backgr. 2.moti" \
   "$HOME/Movies/Motion Templates.localized/Titles.localized/Tap5a/"
```

### Option B — Native (macOS)

```bash
brew install ffmpeg
git clone https://github.com/heenamkung/subtitle-generator.git
cd subtitle-generator
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
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

### OpenAI API Key (highly recommended)

The key enables GPT-4o-mini to punctuate the transcript and split it into natural sentences. Without it, you'll get basic pause-based splitting that may feel rough. Quality difference is significant.

- **Privacy:** only the text transcript is sent to OpenAI — never your audio or video.
- **Cost:** ~$0.001 per video (less than a tenth of a cent).
- **Persistence:** enter the key once in the UI — it auto-saves to `.env` for next time.
- **Get one:** [platform.openai.com/api-keys](https://platform.openai.com/api-keys)

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
