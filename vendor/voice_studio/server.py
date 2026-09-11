"""
Minimal HTTP bridge to OmniVoice (k2-fsa/OmniVoice).

Exposes a small JSON-over-HTTP surface so MoneyPrinterTurbo can use
OmniVoice as a TTS provider without importing torch / transformers in its
own image. Each request loads the model lazily on first call and reuses it
for subsequent calls; concurrent requests are serialized through a single
lock because the heavy OmniVoice model is loaded once per process.

Endpoints:
    GET  /health           -> liveness probe
    GET  /voices           -> list bundled voice-design presets + cloned profiles
    POST /profiles         -> clone a voice from an uploaded sample (multipart)
    POST /generate         -> synthesize WAV (JSON: text, voice, speed?)
    POST /transcribe       -> transcribe raw audio bytes (Metal/MLX whisper)

Voice cloning works by turning a reference sample into a reusable
``VoiceClonePrompt`` (``*.pt``) stored under ``voice_profiles/``; a profile
name then behaves like a preset in ``/generate`` and ``/voices``.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from typing import Optional

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
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

# Cloned voices live as `<name>.pt` (a saved VoiceClonePrompt), an optional
# `<name>.<ext>` audio sample, and a `<name>.json` metadata file in the
# profiles directory. The directory doubles as the source of truth for
# profile names -- every `<name>.pt` is a usable voice. Pointed at the
# `voicestudio_profiles` named volume in docker-compose so clones survive
# container rebuilds.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROFILES_DIR = os.environ.get(
    "VOICESTUDIO_PROFILES_DIR", os.path.join(BASE_DIR, "voice_profiles")
)

# Profiles show up in the same drop-down as the presets, so names follow the
# same lowercase-ASCII convention and stay safe as `voicestudio:<name>` ids
# and as filenames. Spaces become underscores; anything else is dropped.
_MAX_PROFILE_NAME_LEN = 48


def _normalize_profile_name(name: str) -> str:
    """Reduce a user-supplied profile name to a safe, case-folded slug."""
    normalized = re.sub(r"[^a-z0-9_\-]+", "_", (name or "").strip().lower())
    return normalized.strip("_")[:_MAX_PROFILE_NAME_LEN]


def _list_profiles() -> list[str]:
    """Return sorted profile names found in the profiles directory."""
    if not os.path.isdir(PROFILES_DIR):
        return []
    return sorted(
        name[: -len(".pt")]
        for name in os.listdir(PROFILES_DIR)
        if name.endswith(".pt")
    )


def _profile_prompt_path(voice: str) -> Optional[str]:
    """Return the saved prompt path for a profile voice, or None if absent."""
    path = os.path.join(PROFILES_DIR, f"{voice}.pt")
    return path if os.path.isfile(path) else None


def _pick_device(torch):
    """Return (device, dtype) for OmniVoice, refusing a silent CPU fallback.

    ROCm builds of torch expose AMD GPUs through the same torch.cuda API, so
    that branch covers NVIDIA and AMD alike. mps matches only when this runs
    natively on a Mac; a Linux container cannot reach Metal, so a containerised
    Apple host would otherwise land on cpu.
    """
    if torch.cuda.is_available():
        return "cuda", torch.float16
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps", torch.float32
    # CPU inference is ~15x slower and looks like a hang rather than a failure,
    # so it is refused instead of silently accepted. On a Mac this fires when
    # the server is containerised -- the fix is the host server, not CPU.
    if os.environ.get("VOICESTUDIO_ALLOW_CPU") != "1":
        raise RuntimeError(
            "no GPU available to OmniVoice (no cuda/rocm, no mps). "
            "On Apple Silicon, run the host server: `make mac-setup`, and "
            "bring the stack up with docker-compose.mac.yml. "
            "Set VOICESTUDIO_ALLOW_CPU=1 to accept CPU inference."
        )
    return "cpu", torch.float32


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

        device, dtype = _pick_device(torch)

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
    """Return every bundled voice-design preset plus any cloned profiles.

    Each entry is ``{name, instruct}`` where ``instruct`` is empty for
    profile voices.  MoneyPrinterTurbo's WebUI shows the name as the option
    label; the instruct string stays on the server so callers never need to
    know how OmniVoice phrases voice descriptions internally.
    """
    entries = [
        {"name": name, "instruct": instruct}
        for name, instruct in VOICE_PRESETS.items()
    ]
    for name in _list_profiles():
        entries.append({"name": name, "instruct": ""})
    return {"voices": entries}


@app.post("/profiles", status_code=201)
async def create_profile(
    name: str = Form(...),
    audio: UploadFile = File(...),
    ref_text: Optional[str] = Form(None),
):
    """Clone a voice from an uploaded audio sample and persist it as a profile.

    The sample is stored next to its ``VoiceClonePrompt`` (``<name>.pt``) in
    ``PROFILES_DIR`` so it survives restarts (the compose service mounts a
    named volume there). ``ref_text`` is optional: when omitted the prompt is
    auto-transcribed with OmniVoice's ASR model. Returns the normalized
    ``name`` so callers can immediately reference ``voicestudio:<name>``.
    """
    profile_name = _normalize_profile_name(name)
    if not profile_name:
        raise HTTPException(
            status_code=400, detail="profile name must be non-empty"
        )
    if _profile_prompt_path(profile_name) is not None:
        raise HTTPException(
            status_code=409,
            detail=f"voice profile '{profile_name}' already exists",
        )

    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="audio cannot be empty")
    if len(audio_bytes) > 200 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="audio file too large")

    os.makedirs(PROFILES_DIR, exist_ok=True)
    ext = os.path.splitext(audio.filename or "")[1].lower() or ".wav"
    if not re.match(r"^\.[a-z0-9]{1,5}$", ext):
        ext = ".wav"
    audio_path = os.path.join(PROFILES_DIR, f"{profile_name}{ext}")
    prompt_path = os.path.join(PROFILES_DIR, f"{profile_name}.pt")
    meta_path = os.path.join(PROFILES_DIR, f"{profile_name}.json")

    try:
        with open(audio_path, "wb") as fh:
            fh.write(audio_bytes)
        with _tts_lock:
            model = _load_model()
            prompt = model.create_voice_clone_prompt(
                audio_path, ref_text=ref_text or None
            )
            prompt.save(prompt_path)
        metadata = {
            "name": profile_name,
            "audio_file": audio_path,
            "prompt_file": prompt_path,
            "created_at": time.time(),
        }
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2)
    except HTTPException:
        raise
    except Exception as exc:
        for path in (audio_path, prompt_path, meta_path):
            try:
                os.remove(path)
            except OSError:
                pass
        raise HTTPException(
            status_code=500, detail=f"voice profile creation failed: {exc}"
        ) from exc

    return {"ok": True, "name": profile_name, "voices": _list_profiles()}


@app.post("/generate")
def generate_audio(request: GenerateRequest):
    text = (request.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text cannot be empty")
    voice = (request.voice or "").strip().lower()
    instruct = VOICE_PRESETS.get(voice)
    profile_prompt = None
    if not instruct:
        profile_path = _profile_prompt_path(voice)
        if profile_path is None:
            available = ", ".join(sorted(VOICE_PRESETS.keys()) + _list_profiles())
            raise HTTPException(
                status_code=404,
                detail=f"unknown voice '{voice}'; available: {available}",
            )
        from omnivoice.models.omnivoice import VoiceClonePrompt

        profile_prompt = VoiceClonePrompt.load(profile_path)

    try:
        with _tts_lock:
            model = _load_model()
            generate_kwargs: dict = {"text": text}
            if instruct:
                generate_kwargs["instruct"] = instruct
            else:
                generate_kwargs["voice_clone_prompt"] = profile_prompt
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
    out_path = os.path.join(tempfile.gettempdir(), filename)
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
    tmp_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4().hex}.audio")
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
