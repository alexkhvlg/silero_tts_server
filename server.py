"""Silero TTS Server - OpenAI API Compatible TTS endpoint.

Supports Russian (v5_5_ru) and English (v3_en) models with
multiple speakers. Both models are loaded at startup for fast response.

Usage:
    python server.py --host 0.0.0.0 --port 5000
"""

import argparse
import io
import logging
import logging.handlers
import os
import sys
import wave
import warnings
from contextlib import asynccontextmanager
from typing import Literal

import fastapi
from fastapi import HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Silence loguru from silero_tts submodule
import loguru

loguru.logger.remove()
loguru.logger.add(sys.stderr, level="WARNING")

# Suppress SyntaxWarning from torch/silero packages only (e.g. invalid escape sequences)
warnings.filterwarnings("ignore", category=SyntaxWarning, module=".*torch.*")
warnings.filterwarnings("ignore", category=SyntaxWarning, module=".*silero.*")

# ---------------------------------------------------------------------------
# Logging — console + rotating file in logs/
# ---------------------------------------------------------------------------

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
log_file = os.path.join(LOG_DIR, "server.log")

LOG_FMT = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)

# Console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(LOG_FMT)
console_handler.setLevel(logging.INFO)

# File handler (rotating)
file_handler = logging.handlers.RotatingFileHandler(
    log_file,
    maxBytes=5 * 1024 * 1024,  # 5 MB
    backupCount=3,
    encoding="utf-8",
)
file_handler.setFormatter(LOG_FMT)
file_handler.setLevel(logging.INFO)

# Module logger — handlers attached directly, no propagation to root
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(console_handler)
logger.addHandler(file_handler)
logger.propagate = False

# Redirect Python warnings into our logger
logging.captureWarnings(True)
warnings_logger = logging.getLogger("py.warnings")
warnings_logger.addHandler(console_handler)
warnings_logger.addHandler(file_handler)
warnings_logger.propagate = False

# ---------------------------------------------------------------------------
# Model cache - loaded once at startup
# ---------------------------------------------------------------------------

tts_models: dict = {}  # language -> SileroTTS instance

# Known speakers per model (populated after loading)
SPEAKERS: dict = {}  # language -> list[str]

# Mapping: OpenAI voice name -> (language, speaker)
# e.g. "ru_xenia" -> ("ru", "xenia"), "en_2" -> ("en", "en_2")
VOICE_MAP: dict = {}  # voice_name -> (language, speaker)
# Global reference for debug shutdown
_debug_server = None


def _check_model_file(cfg: dict) -> str:
    """Check if model file already exists; return status message."""
    from silero_tts import silero_tts as _mod
    base = os.path.join(os.path.dirname(_mod.__file__), "silero_models")
    path = os.path.join(base, f"{cfg['model_id']}_{cfg['language']}.pt")
    if os.path.exists(path):
        size_mb = os.path.getsize(path) / (1024 * 1024)
        return f"  Model file exists ({size_mb:.1f} MB)"
    return "  Model file not found — will download on first load (~30-100 MB)"


def load_models(device: str = "cpu", sample_rate: int = 48000) -> None:
    """Load Russian and English Silero TTS models into memory."""
    global tts_models, SPEAKERS, VOICE_MAP

    from silero_tts.silero_tts import SileroTTS

    model_configs = [
        {"model_id": "v5_5_ru", "language": "ru", "speaker": "xenia"},
        {"model_id": "v3_en", "language": "en", "speaker": "en_2"},
    ]

    # Pre-check model files
    for cfg in model_configs:
        status = _check_model_file(cfg)
        logger.info("Checking %s model %s: %s", cfg["language"].upper(), cfg["model_id"], status)

    for cfg in model_configs:
        lang = cfg["language"]
        logger.info(
            "Loading %s model: %s (default speaker: %s) — this may download the model on first run...",
            lang.upper(),
            cfg["model_id"],
            cfg["speaker"],
        )
        try:
            tts_models[lang] = SileroTTS(
                model_id=cfg["model_id"],
                language=lang,
                speaker=cfg["speaker"],
                sample_rate=sample_rate,
                device=device,
            )
            SPEAKERS[lang] = tts_models[lang].get_available_speakers()
            logger.info(
                "  %s model loaded OK — %d speakers",
                lang.upper(),
                len(SPEAKERS[lang])
            )
        except Exception as exc:
            logger.error("Failed to load %s model %s: %s", lang.upper(), cfg["model_id"], exc)
            raise

    # Build voice map
    for lang, speakers_list in SPEAKERS.items():
        for sp in speakers_list:
            # Avoid double prefix like "en_en_0" if speaker already has lang prefix
            if sp.startswith(f"{lang}_"):
                voice_name = sp
            else:
                voice_name = f"{lang}_{sp}"
            VOICE_MAP[voice_name] = (lang, sp)
            # Also allow bare speaker name for backward compat
            VOICE_MAP[sp] = (lang, sp)

    logger.info("All models loaded: %d total voices across %d languages", len(VOICE_MAP), len(tts_models))


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: fastapi.FastAPI):
    """Startup / shutdown lifecycle."""
    device = app.state.device
    sample_rate = app.state.sample_rate
    try:
        load_models(device=device, sample_rate=sample_rate)
    except Exception as exc:
        logger.error("Failed to load TTS models: %s", exc)
        raise
    logger.info("Silero TTS Server ready — device=%s, sample_rate=%s", device, sample_rate)

    # Auto-shutdown for debug mode is handled in __main__ via server.should_exit

    yield
    logger.info("Shutting down Silero TTS Server")


app = fastapi.FastAPI(
    title="Silero TTS Server",
    description="OpenAI API-compatible Text-to-Speech server using Silero models "
                "(Russian v5_5_ru + English v3_en).",
    version="1.0.0",
    lifespan=lifespan,
)

# Defaults — overridden by CLI args in __main__ before uvicorn.run()
app.state.device = os.getenv("SILERO_DEVICE", "cpu")
app.state.sample_rate = int(os.getenv("SILERO_SAMPLE_RATE", "48000"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class SpeechRequest(BaseModel):
    """POST /v1/audio/speech request body."""
    model: str = Field(
        default="silero-tts",
        description="Model identifier (any value accepted; language is inferred from voice).",
    )
    input: str = Field(
        ..., min_length=1, max_length=5000,
        description="Text to synthesize (max 5000 characters).",
    )
    voice: str = Field(
        ...,
        description="Speaker voice name. Use GET /v1/audio/speech/voices to list available voices.",
    )
    response_format: Literal["mp3", "wav", "flac", "ogg", "aac", "opus"] = Field(
        default="wav",
        description="Output audio format. wav is native; mp3, ogg, opus require optional pydub/ffmpeg.",
    )
    speed: float = Field(
        default=1.0, ge=0.25, le=4.0,
        description="Speech speed multiplier (0.25 – 4.0). Currently ignored by Silero models.",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def synthesize_to_wav(text: str, language: str, speaker: str) -> bytes:
    """Run Silero TTS and return raw WAV bytes in memory."""
    tts = tts_models[language]
    tts.change_speaker(speaker)

    buf = io.BytesIO()
    wf = wave.open(buf, "wb")
    wf.setnchannels(1)
    wf.setsampwidth(2)  # 16-bit
    wf.setframerate(tts.sample_rate)

    lines = [line.strip() for line in text.split("\n") if line.strip()]
    for line in lines:
        try:
            audio = tts.tts_model.apply_tts(
                text=line,
                speaker=speaker,
                sample_rate=tts.sample_rate,
                put_accent=tts.put_accent,
                put_yo=tts.put_yo,
            )
        except Exception as exc:
            logger.error(
                "TTS model apply_tts failed (lang=%s speaker=%s text=%r): %s",
                language, speaker, line[:100], exc,
            )
            raise
        wf.writeframes((audio * 32767).numpy().astype("int16"))
    wf.close()

    buf.seek(0)
    return buf.read()


def convert_wav(wav_bytes: bytes, fmt: str) -> tuple[bytes, str]:
    """Convert WAV bytes to the requested format.

    Returns (audio_bytes, content_type).
    """
    if fmt == "wav":
        return wav_bytes, "audio/wav"

    # Try pydub + ffmpeg for other formats
    try:
        from pydub import AudioSegment

        audio = AudioSegment.from_wav(io.BytesIO(wav_bytes))
        out_buf = io.BytesIO()

        export_params = {
            "mp3": {"format": "mp3", "codec": "libmp3lame"},
            "ogg": {"format": "ogg", "codec": "libvorbis"},
            "flac": {"format": "flac", "codec": "flac"},
            "aac": {"format": "adts", "codec": "aac"},
            "opus": {"format": "opus", "codec": "libopus"},
        }

        params = export_params.get(fmt, {"format": fmt})
        audio.export(out_buf, **params)
        out_buf.seek(0)

        content_types = {
            "mp3": "audio/mpeg",
            "ogg": "audio/ogg",
            "flac": "audio/flac",
            "aac": "audio/aac",
            "opus": "audio/opus",
        }
        return out_buf.read(), content_types.get(fmt, f"audio/{fmt}")

    except ImportError:
        logger.warning(
            "pydub not installed — cannot convert to %s. Returning WAV instead. "
            "Install with: pip install pydub  (and ffmpeg on PATH)",
            fmt,
        )
        return wav_bytes, "audio/wav"
    except Exception as exc:
        logger.error("Failed to convert WAV to %s: %s", fmt, exc)
        raise


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    """Health check / info endpoint."""
    return {
        "name": "Silero TTS Server",
        "version": "1.0.0",
        "status": "ok",
        "models_loaded": list(tts_models.keys()),
        "voices": {lang: SPEAKERS.get(lang, []) for lang in tts_models},
    }


@app.get("/v1/audio/speech/voices")
async def list_voices():
    """List all available voices (OpenAI-compatible)."""
    voices = []
    for lang, speakers_list in SPEAKERS.items():
        for sp in speakers_list:
            if sp.startswith(f"{lang}_"):
                vid = sp
            else:
                vid = f"{lang}_{sp}"
            voices.append({
                "id": vid,
                "name": sp,
                "language": lang,
                "model": f"silero-{lang}",
            })
            # Also add bare speaker name
            voices.append({
                "id": sp,
                "name": sp,
                "language": lang,
                "model": f"silero-{lang}",
            })
    return {"voices": voices}


@app.get("/v1/models")
async def list_models():
    """List available TTS models (OpenAI-compatible)."""
    models = []
    for lang in tts_models:
        models.append({
            "id": f"silero-{lang}",
            "object": "model",
            "created": 0,
            "owned_by": "silero",
        })
    return {"data": models, "object": "list"}


@app.post("/v1/audio/speech")
async def create_speech(request: SpeechRequest):
    """Synthesize speech from text (OpenAI-compatible endpoint)."""
    # Resolve voice
    voice_key = request.voice
    if voice_key not in VOICE_MAP:
        available = list(VOICE_MAP.keys())
        logger.error("Unknown voice requested: '%s'. Available: %s", voice_key, available)
        raise HTTPException(
            status_code=400,
            detail=f"Unknown voice '{voice_key}'. Available voices: {available}",
        )

    language, speaker = VOICE_MAP[voice_key]

    # Validate text length
    if len(request.input) > 5000:
        logger.error("Input text too long: %d chars (max 5000)", len(request.input))
        raise HTTPException(status_code=400, detail="Input text too long (max 5000 characters)")

    if not request.input.strip():
        logger.error("Input text is empty after stripping whitespace")
        raise HTTPException(status_code=400, detail="Input text is empty")

    logger.info(
        "Synthesizing: lang=%s speaker=%s format=%s chars=%d",
        language, speaker, request.response_format, len(request.input),
    )
    logger.info("Input text[%s]: %s", speaker, request.input)

    try:
        wav_bytes = synthesize_to_wav(request.input, language, speaker)
    except Exception as exc:
        logger.error("Synthesis failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Synthesis error: {exc}")

    audio_bytes, content_type = convert_wav(wav_bytes, request.response_format)

    return Response(
        content=audio_bytes,
        media_type=content_type,
        headers={"Content-Disposition": 'attachment; filename="speech.wav"'},
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    """Parse command-line arguments for the Silero TTS Server.

    Returns:
        argparse.Namespace: Parsed arguments containing host, port, device,
        sample_rate, workers, and debug_run.
    """
    parser = argparse.ArgumentParser(description="Silero TTS Server (OpenAI API compatible)")
    parser.add_argument("--host", default="0.0.0.0", help="Listen address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=5000, help="Listen port (default: 5000)")
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "auto"],
        default="cpu",
        help="Device for model inference (default: cpu)",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        choices=[8000, 24000, 48000],
        default=48000,
        help="Audio sample rate (default: 48000)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of Uvicorn worker processes (default: 1; use 1 for GPU)",
    )
    parser.add_argument(
        "--debug-run",
        type=int,
        default=None,
        help="Auto-shutdown after N seconds (for testing, default: disabled)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Attach config to app state for lifespan callback
    app.state.device = args.device
    app.state.sample_rate = args.sample_rate
    if args.debug_run:
        app.state.debug_run = args.debug_run

    import uvicorn

    config = uvicorn.Config(
        "server:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level="info",
    )
    server = uvicorn.Server(config)

    # Store server reference for debug shutdown
    import server as _server_mod
    _server_mod._debug_server = server

    if args.debug_run:
        import threading
        logger.info("DEBUG mode: auto-shutdown in %ds after server starts", args.debug_run)
        def _shutdown():
            # Wait a moment for server to fully start
            import time
            time.sleep(0.5)
            logger.info("DEBUG: shutting down")
            server.should_exit = True
        threading.Timer(args.debug_run, _shutdown).start()

    try:
        server.run()
    except Exception as exc:
        logger.error("Server failed to start: %s", exc)
        raise
