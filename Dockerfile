# Subtitle Generator — CPU-only image
# WhisperX + PyTorch CPU wheels. No GPU support; that's fine since the app
# already runs on CPU (see app.py line 184).

FROM python:3.11-slim

# ffmpeg is required by WhisperX / the AudioSkill. git is needed because some
# pip dependencies (occasionally) resolve to VCS installs.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first so the layer caches independently of source edits.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app.
COPY . .

# Gradio binds to 127.0.0.1 by default — expose it on all interfaces so the
# host can reach it. app.py also checks IN_DOCKER to skip the Motion template
# install (macOS-only path) and to disable the inbrowser=True flag.
ENV IN_DOCKER=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    HF_HOME=/root/.cache/huggingface \
    PYTHONUNBUFFERED=1

EXPOSE 7860

CMD ["python", "app.py"]
