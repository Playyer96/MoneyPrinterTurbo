"""
FastAPI HTTP server for the headless VoiceStudio.

Exposes :mod:`core`'s two operations over HTTP so external tools (e.g.
MoneyPrinterTurbo) can use VoiceStudio as a TTS provider without importing
the OmniVoice stack in-process.

Endpoints:
    GET  /health              -> healthy check
    GET  /profiles            -> list of available voice profiles
    POST /profiles            -> create a profile (multipart: profile_name + audio file)
    POST /tts                 -> synthesize audio (JSON: text + profile_name) -> WAV

Run (from the voice_studio directory):
    python server.py
or:
    uvicorn server:app --host 127.0.0.1 --port 8780

Host/port can be overridden via the ``VOICESTUDIO_HOST`` / ``VOICESTUDIO_PORT``
environment variables.
"""

from __future__ import annotations

import os
import tempfile
import threading
import uuid
from typing import Optional

import core
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8780

app = FastAPI(title="VoiceStudio HTTP API", version="1.0.0")

# OmniVoice is a single heavy model attached to one device. Serialize TTS so
# concurrent video tasks don't contend on the GPU / model-load path.
_tts_lock = threading.Lock()


class TtsRequest(BaseModel):
    text: str
    profile_name: str
    voice_rate: Optional[float] = None
    voice_volume: Optional[float] = None


def _clean_profile_name(profile_name: str) -> str:
    name = (profile_name or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="profile_name cannot be empty")
    return name


def _list_profile_names() -> list[str]:
    if not os.path.isdir(core.PROFILES_DIR):
        return []
    names = []
    for filename in sorted(os.listdir(core.PROFILES_DIR)):
        if filename.endswith(".json"):
            names.append(filename[:-5])
    return names


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.get("/profiles")
def list_profiles() -> dict:
    return {"profiles": _list_profile_names()}


@app.post("/profiles")
async def create_profile(
    profile_name: str = Form(...),
    file: UploadFile = File(...),
) -> dict:
    name = _clean_profile_name(profile_name)
    if name in _list_profile_names():
        raise HTTPException(
            status_code=409, detail=f"voice profile '{name}' already exists"
        )

    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="uploaded audio is empty")

    ext = os.path.splitext(file.filename or "")[1].lower() or ".wav"
    tmp_audio = None
    try:
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_audio = tmp.name
        saved_path = core.create_profile(audio_path=tmp_audio, profile_name=name)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"profile creation failed: {exc}"
        ) from exc
    finally:
        if tmp_audio:
            try:
                os.remove(tmp_audio)
            except OSError:
                pass
    return {"profile_name": name, "path": saved_path}


@app.post("/tts")
def generate_audio(request: TtsRequest):
    text = (request.text or "").strip()
    name = _clean_profile_name(request.profile_name)
    if not text:
        raise HTTPException(status_code=400, detail="text cannot be empty")

    filename = f"{uuid.uuid4().hex}.wav"
    try:
        with _tts_lock:
            out_path = core.generate_audio(
                text=text,
                profile_name=name,
                output_filename=filename,
            )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"audio generation failed: {exc}"
        ) from exc
    return FileResponse(out_path, media_type="audio/wav", filename=filename)


def main() -> None:
    host = os.environ.get("VOICESTUDIO_HOST", DEFAULT_HOST)
    port = int(os.environ.get("VOICESTUDIO_PORT", DEFAULT_PORT))
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()