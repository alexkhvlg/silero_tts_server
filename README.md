# Silero TTS Server

OpenAI API-compatible Text-to-Speech server using Silero models.

## Features

- **OpenAI API compatible** — drop-in replacement for `openai.audio.speech.create()`
- **Russian** — model `v5_5_ru` (voices: aidar, baya, kseniya, eugene, xenia)
- **English** — model `v3_en` (voices: en_0, en_1, en_2, ...)
- **Output formats** — WAV (native), MP3, OGG, FLAC, AAC, OPUS (via pydub + ffmpeg)
- **CORS enabled** — works from browser clients
- **Multiline text** — input with `\n` is split and synthesized line by line
- **Detailed error logging** — all failures logged to console + rotating file (`logs/server.log`)

## Quick Start

### Option 1: Windows shortcut

```bash
start_server.cmd
```

### Option 2: Manual

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate   # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Start server (loads both models at startup)
python server.py --host 0.0.0.0 --port 5000 --device cpu

# With GPU
python server.py --device cuda
```

## API Endpoints

### POST /v1/audio/speech

Synthesize speech from text.

```bash
curl -X POST http://localhost:5000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "silero-tts",
    "input": "Привет мир! Это тест русскоязычного синтеза.",
    "voice": "ru_xenia",
    "response_format": "wav"
  }' \
  --output speech.wav
```

```bash
# English
curl -X POST http://localhost:5000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "silero-tts",
    "input": "Hello world! This is a test of English speech synthesis.",
    "voice": "en_2",
    "response_format": "mp3"
  }' \
  --output speech.mp3
```

```bash
# OPUS format (good for voice messaging)
curl -X POST http://localhost:5000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "silero-tts",
    "input": "Привет! Это тест формата OPUS.",
    "voice": "ru_xenia",
    "response_format": "opus"
  }' \
  --output speech.opus
```

### GET /v1/audio/speech/voices

List all available voices.

```bash
curl http://localhost:5000/v1/audio/speech/voices
```

### GET /v1/models

List available TTS models.

```bash
curl http://localhost:5000/v1/models
```

### GET /

Health check and server info.

```bash
curl http://localhost:5000/
```

## Python Client (OpenAI SDK)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:5000/v1",
    api_key="not-needed"
)

response = client.audio.speech.create(
    model="silero-tts",
    voice="ru_xenia",
    input="Привет! Как дела?"
)
response.stream_to_file("output.mp3")
```

## Available Voices

| Language | Model    | Voices                    |
|----------|----------|---------------------------|
| Russian  | v5_5_ru  | aidar, baya, kseniya, eugene, xenia |
| English  | v3_en    | en_0, en_1, en_2, ...     |

Voice names can be used as `ru_xenia`, `en_2`, or just `xenia` for Russian.

## Server Options

| Option           | Default   | Description                                    |
|------------------|-----------|------------------------------------------------|
| `--host`         | 0.0.0.0   | Listen address                                 |
| `--port`         | 5000      | Listen port                                    |
| `--device`       | cpu       | cpu / cuda / auto                              |
| `--sample-rate`  | 48000     | 8000 / 24000 / 48000                          |
| `--workers`      | 1         | Uvicorn worker count (1 for GPU)               |
| `--debug-run`    | disabled  | Auto-shutdown after N seconds (testing only)   |

## Environment Variables

| Variable             | Default | Description                              |
|----------------------|---------|------------------------------------------|
| `SILERO_DEVICE`      | cpu     | Overrides `--device` if CLI arg not set  |
| `SILERO_SAMPLE_RATE` | 48000   | Overrides `--sample-rate` if not set     |

## Logging

All server activity is logged to both console and rotating file:

- **Location:** `logs/server.log` (project root)
- **Rotation:** 5 MB per file, 3 backups retained
- **Includes:** model loading, synthesis requests, errors, warnings

## Dependencies

- **silero-tts** — Silero TTS library (models downloaded automatically)
- **torch** — PyTorch for model inference
- **fastapi + uvicorn** — ASGI web framework
- **pydub** — Audio format conversion (optional, for MP3/OGG/FLAC)
- **ffmpeg** — Required for non-WAV formats (install separately)

## License

MIT
