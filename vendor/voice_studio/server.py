"""
Minimal HTTP bridge to OmniVoice (k2-fsa/OmniVoice).

Exposes a small JSON-over-HTTP surface so MoneyPrinterTurbo can use
OmniVoice as a TTS provider without importing torch / transformers in its
own image. Each request loads the model lazily on first call and reuses it
for subsequent calls; concurrent requests are serialized through a single
lock because the heavy OmniVoice model is loaded once per process.

Endpoints:
    GET  /health           -> liveness probe
    GET  /voices           -> list bundled voice-design presets
    POST /generate         -> synthesize WAV (JSON: text, voice, speed?)
    POST /transcribe       -> transcribe raw audio bytes (Metal/MLX whisper)

The legacy profile-management endpoints (/profiles, /tts) used to live
here; they are gone by design. Voice cloning is still possible via the
omnivoice Python API directly when callers need it, but MoneyPrinterTurbo
only consumes the bundled presets here.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import uuid
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8780

app = FastAPI(title="OmniVoice HTTP API", version="2.0.0")

# OmniVoice is a single heavy model attached to one device. Serialize TTS
# so concurrent video tasks don't contend on the GPU / model-load path.
_tts_lock = threading.Lock()
_model = None
_model_lock = threading.Lock()

# Whisper runs on the same GPU as TTS, so it gets its own lock rather than
# sharing _tts_lock: a transcription and a synthesis are independent requests
# and only need to avoid running *concurrently on the GPU*, not to queue
# behind each other's model loads.
_whisper_lock = threading.Lock()

# Bundled voice-design presets. Each entry maps a stable name (kept as the
# original character name the user expects to see in the dropdown) to the
# `instruct` string OmniVoice consumes. Names are short, lowercase, ASCII so
# they round-trip cleanly through MPT's voice id parser
# (voicestudio:<name>) and through TOML config keys.
VOICE_PRESETS: dict[str, str] = {
    "cr7": "male, middle-aged, low pitch, portuguese accent",
    "goku": "male, young adult, moderate pitch",
    "narrator": "male, middle-aged, moderate pitch",
    "casual": "female, young adult, moderate pitch",
    "energetic": "male, young adult, high pitch",
}


def _load_model():
    """Lazy-load the OmniVoice model on first use; serializes concurrent loads."""
    global _model
    if _model is not None:
        return _model
    with _model_lock:
        if _model is not None:
            return _model
        import torch
        from omnivoice.models.omnivoice import OmniVoice

        # ROCm builds of torch expose AMD GPUs through the same torch.cuda
        # API, so this branch covers NVIDIA and AMD alike. mps matches only
        # when this runs natively on a Mac; a Linux container cannot reach
        # Metal, so a containerised Apple host lands on cpu.
        if torch.cuda.is_available():
            device, dtype = "cuda", torch.float16
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device, dtype = "mps", torch.float32
        else:
            device, dtype = "cpu", torch.float32

        model_id = os.environ.get("OMNIVOICE_MODEL_ID", "k2-fsa/OmniVoice")
        # uvicorn.error is the logger uvicorn actually configures, so this
        # line shows up in the server output and you can see which device
        # (cuda / mps / cpu) the model actually landed on.
        import logging
        logging.getLogger("uvicorn.error").info(
            "loading OmniVoice model '%s' on %s ...", model_id, device
        )
        _model = OmniVoice.from_pretrained(model_id, device_map=device, dtype=dtype)
    return _model


class GenerateRequest(BaseModel):
    text: str
    voice: str = "narrator"
    speed: Optional[float] = None


@app.get("/health")
def health() -> dict:
    return {"ok": True, "voices": list(VOICE_PRESETS.keys())}


@app.get("/voices")
def list_voices() -> dict:
    """Return every bundled voice-design preset as ``name -> instruct``.

    MoneyPrinterTurbo's WebUI shows the name as the option label; the
    instruct string stays on the server so callers never need to know how
    OmniVoice phrases voice descriptions internally.
    """
    return {
        "voices": [
            {"name": name, "instruct": instruct}
            for name, instruct in VOICE_PRESETS.items()
        ]
    }


@app.post("/generate")
def generate_audio(request: GenerateRequest):
    text = (request.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text cannot be empty")
    voice = (request.voice or "").strip().lower()
    instruct = VOICE_PRESETS.get(voice)
    if not instruct:
        raise HTTPException(
            status_code=404,
            detail=(
                f"unknown voice '{voice}'; available: "
                + ", ".join(sorted(VOICE_PRESETS.keys()))
            ),
        )

    try:
        with _tts_lock:
            model = _load_model()
            generate_kwargs = {"text": text, "instruct": instruct}
            if request.speed is not None and request.speed > 0:
                generate_kwargs["speed"] = float(request.speed)
            waveforms = model.generate(**generate_kwargs)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"audio generation failed: {exc}"
        ) from exc

    # OmniVoice returns one or more waveform tensors at the model's native
    # 24 kHz. Persist them as a temporary WAV the response can stream, then
    # clean up. Sampling rate is fixed by the model and not user-tunable.
    filename = f"{uuid.uuid4().hex}.wav"
    out_path = os.path.join("/tmp", filename)
    try:
        import torch
        import soundfile as sf

        wav = waveforms[0]
        if hasattr(wav, "detach"):
            wav = wav.detach().cpu().float().numpy()
        sf.write(out_path, wav, 24000)
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"audio write failed: {exc}"
        ) from exc

    return FileResponse(
        out_path,
        media_type="audio/wav",
        filename=filename,
        background=None,
    )


# faster-whisper's CTranslate2 backend has no Metal support (cpu and cuda
# only), so on Apple Silicon the only way to put whisper on the GPU is MLX.
# MoneyPrinterTurbo posts audio here instead of transcribing in-container;
# app/services/subtitle.py falls back to its own faster-whisper when this
# endpoint is unreachable or unavailable.
# large-v3-turbo rather than large-v3: roughly a third of the weights for
# near-identical accuracy on clean TTS audio. Size matters more than usual
# here because OmniVoice is already resident on the same unified memory, and
# Docker Desktop reserves a large slice of it -- two full-size models thrash.
MLX_WHISPER_REPO = os.environ.get(
    "MLX_WHISPER_REPO", "mlx-community/whisper-large-v3-turbo"
)


@app.post("/transcribe")
async def transcribe(request: Request, word_timestamps: bool = True):
    audio = await request.body()
    if not audio:
        raise HTTPException(status_code=400, detail="empty audio body")

    # mlx_whisper decodes via ffmpeg, which needs a real file on disk.
    tmp_path = os.path.join("/tmp", f"{uuid.uuid4().hex}.audio")
    worker = os.path.join(os.path.dirname(os.path.abspath(__file__)), "whisper_worker.py")
    try:
        with open(tmp_path, "wb") as fh:
            fh.write(audio)
        with _whisper_lock:
            proc = subprocess.run(
                [sys.executable, worker, tmp_path]
                + ([] if word_timestamps else ["--no-word-timestamps"]),
                capture_output=True,
                timeout=1800,
            )
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"transcription failed: {exc}"
        ) from exc
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    if proc.returncode != 0:
        detail = proc.stderr.decode("utf-8", "replace")[-500:]
        raise HTTPException(status_code=500, detail=f"whisper worker failed: {detail}")

    return json.loads(proc.stdout)


def main() -> None:
    host = os.environ.get("VOICESTUDIO_HOST", DEFAULT_HOST)
    port = int(os.environ.get("VOICESTUDIO_PORT", DEFAULT_PORT))
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
