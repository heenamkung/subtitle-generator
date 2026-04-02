# Subtitle Generator

Generate high-quality `.srt` subtitle files from any video using OpenAI's transcription API.

**video file → extracted audio → transcription → `.srt` subtitle file**

---

## Requirements

- macOS
- Python 3.8+
- ffmpeg
- An OpenAI API key

---

## Setup

### 1. Install ffmpeg

```bash
brew install ffmpeg
```

### 2. Create a virtual environment and install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Add your OpenAI API key

```bash
cp .env.example .env
```

Open `.env` and set:

```env
OPENAI_API_KEY=your_api_key_here
```

---

## Usage

```bash
python main.py /path/to/your/video.mp4
```

Choose output folder:

```bash
python main.py /path/to/your/video.mp4 --output-dir ./output
```

Use smaller chunks for long videos:

```bash
python main.py /path/to/your/video.mp4 --chunk-seconds 480
```

---

## Output

Inside `output/<video-name>/`:

| File | Description |
|------|-------------|
| `audio/source.mp3` | Extracted audio |
| `chunks/*.mp3` | Chunked audio files |
| `segments.json` | Transcript segments with timestamps |
| `<video-name>.srt` | Final subtitle file |

---

## Notes

- Uses OpenAI's `whisper-1` model via the transcription API
- Files are split into chunks to stay within the 25 MB API limit
- For best results, use clean audio source
- If the video is very long, reduce `--chunk-seconds`

---

## API Key & Costs

This tool uses your own OpenAI API key. You are responsible for any charges incurred on your OpenAI account.

- Transcription is billed per minute of audio. Check [OpenAI's pricing page](https://openai.com/pricing) for current rates.
- A one-hour video will use approximately 60 minutes of audio transcription.
- Your API key is stored locally in your `.env` file and is never sent anywhere other than directly to OpenAI.
- Set usage limits on your OpenAI account at [platform.openai.com/account/limits](https://platform.openai.com/account/limits) to avoid unexpected charges.
