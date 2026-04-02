# Subtitle Generator

Generate high-quality `.srt` subtitle files from any video using OpenAI's Whisper transcription API.

---

## Requirements

- macOS (Apple Silicon or Intel)
- Python 3.8+
- ffmpeg
- An OpenAI API key

---

## Setup

### 1. Install ffmpeg

```bash
brew install ffmpeg
```

### 2. Clone the repo and install dependencies

```bash
git clone https://github.com/heenamkung/subtitle-generator.git
cd subtitle-generator
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Launch the app

```bash
python app.py
```

The app will open automatically in your browser.

---

## Usage

1. Enter your OpenAI API key
2. Upload your video file
3. Click **Generate Subtitles**
4. Download the `.srt` file when done

---

## API Key & Costs

This tool uses your own OpenAI API key. You are responsible for any charges incurred on your OpenAI account.

- Transcription is billed per minute of audio. Check [OpenAI's pricing page](https://openai.com/pricing) for current rates.
- A one-hour video will use approximately 60 minutes of audio transcription.
- Your API key is stored locally in your `.env` file and is never sent anywhere other than directly to OpenAI.
- Set usage limits on your OpenAI account at [platform.openai.com/account/limits](https://platform.openai.com/account/limits) to avoid unexpected charges.
