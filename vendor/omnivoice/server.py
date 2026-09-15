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
    GET  /profiles         -> list cloned profile names
    POST /profiles         -> clone a voice from an uploaded sample (multipart)
    DELETE /profiles/<name> -> delete a cloned profile and all its files
    POST /generate         -> synthesize WAV (JSON: text, voice, speed?)
    POST /transcribe       -> transcribe raw audio bytes (Metal/MLX whisper)

Voice cloning works by turning a reference sample into a reusable
``VoiceClonePrompt`` (``*.pt``) stored under ``voice_profiles/``; a profile
name then behaves like a preset in ``/generate`` and ``/voices``.
"""

from __future__ import annotations

import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from typing import Optional

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8780
# Default to 16 diffusion steps for noticeably better output quality.
# 8 is the OmniVoice default; 16 gives a clear quality bump at ~2x render
# time, still real-time on modern GPUs. Operators who need to trade quality
# for throughput can override with OMNIVOICE_NUM_STEPS=8.
DEFAULT_GENERATION_STEPS = int(os.environ.get("OMNIVOICE_NUM_STEPS", "32"))
# Position temperature of 0 is OmniVoice's greedy default but tends to
# sound flat on cloned voices. 0.45 makes questions rise and emphatic
# statements punch without losing the speaker's identity. Higher values
# add variation but risk breaking prosody consistency across long
# generations. Override with OMNIVOICE_POSITION_TEMPERATURE if you
# need a flatter (lower) or more expressive (higher) default.
DEFAULT_POSITION_TEMPERATURE = float(
    os.environ.get("OMNIVOICE_POSITION_TEMPERATURE", "0.45")
)
# Cloned voices must be deterministic. With the default 0.45 the diffusion
# sampler drifts away from the cloned prompt's timbre from one call to the
# next — the user picks a clone because they want THAT speaker, not a
# stochastic variant that sometimes sounds like the prompt and sometimes
# sounds like the instruct. Greedy sampling on the cloned path keeps the
# voice stable across every generation.
CLONED_VOICE_POSITION_TEMPERATURE = 0.0

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

# Bundled voice-design presets. Each entry maps a stable name to the English
# `instruct` string OmniVoice consumes.
VOICE_PRESETS: dict[str, str] = {
    "narrator": "male, middle-aged, low pitch",
}

# Cloned voices live as `<name>.pt` (a saved VoiceClonePrompt), an optional
# `<name>.<ext>` audio sample, and a `<name>.json` metadata file in the
# profiles directory. The directory doubles as the source of truth for
# profile names -- every `<name>.pt` is a usable voice. Pointed at the
# `omnivoice_profiles` named volume in docker-compose so clones survive
# container rebuilds.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROFILES_DIR = os.environ.get(
    "OMNIVOICE_PROFILES_DIR", os.path.join(BASE_DIR, "voice_profiles")
)

# Profiles show up in the same drop-down as the presets, so names follow the
# same lowercase-ASCII convention and stay safe as `omnivoice:<name>` ids
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


def _probe_audio_seconds(path: str) -> Optional[float]:
    """Return the audio duration in seconds via ffprobe, or None on failure.

    Cloned profiles persist the sample duration so the WebUI can warn
    operators when a short reference (under 20 seconds) is likely to
    produce a flat, under-conditioned clone.
    """
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None
    try:
        result = subprocess.run(
            [
                ffprobe,
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                path,
            ],
            capture_output=True, text=True, check=False, timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return None
    try:
        return float(result.stdout.strip())
    except ValueError:
        return None


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
    raise RuntimeError(
        "no GPU available to OmniVoice (no cuda/rocm, no mps). "
        "On Apple Silicon, use docker-compose.mac.yml so Compose starts "
        "the host Metal service automatically. CPU inference is disabled."
    )


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
    voice: Optional[str] = None
    speed: Optional[float] = None
    style: Optional[str] = None


@app.get("/health")
def health() -> dict:
    return {"ok": True, "voices": list(VOICE_PRESETS.keys())}


@app.post("/warmup")
def warmup() -> dict:
    """Load the TTS model before the first request."""
    _load_model()
    return {"ok": True}


@app.get("/voices")
def list_voices() -> dict:
    """Return every bundled voice-design preset plus any cloned profiles.

    Each entry is ``{name, instruct}`` where ``instruct`` is empty for
    profile voices (the clone's saved metadata ``instruct_override`` is
    surfaced as ``profile_instruct`` so the WebUI can preview and edit it).
    MoneyPrinterTurbo's WebUI shows the name as the option label; the
    instruct string stays on the server so callers never need to know how
    OmniVoice phrases voice descriptions internally.
    """
    entries = [
        {"name": name, "instruct": instruct}
        for name, instruct in VOICE_PRESETS.items()
    ]
    for name in _list_profiles():
        meta = _load_profile_metadata(name)
        entries.append(
            {
                "name": name,
                "instruct": "",
                "profile_instruct": meta.get("instruct_override", "") or "",
                "sample_duration_seconds": meta.get("sample_duration_seconds"),
                "has_ref_text": bool(meta.get("ref_text")),
            }
        )
    return {"voices": entries}


@app.post("/profiles", status_code=201)
async def create_profile(
    name: str = Form(...),
    audio: UploadFile = File(...),
    ref_text: Optional[str] = Form(None),
    instruct_override: Optional[str] = Form(None),
):
    """Clone a voice from an uploaded audio sample and persist it as a profile.

    The sample is stored next to its ``VoiceClonePrompt`` (``<name>.pt``) in
    ``PROFILES_DIR`` so it survives restarts (the compose service mounts a
    named volume there). ``ref_text`` is optional: when omitted the prompt is
    auto-transcribed with OmniVoice's ASR model.

    ``instruct_override`` is the profile-level emotion/delivery descriptor
    (e.g. "enthusiastic, energetic, mid-pitch"). When set, every future
    generation that picks this voice inherits the instruct string, so a
    single clone can carry the speaker's timbre AND a consistent emotional
    register without re-typing it on every call.

    Returns the normalized ``name`` so callers can immediately reference
    ``omnivoice:<name>``.
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

    sample_duration_seconds: Optional[float] = None
    try:
        with open(audio_path, "wb") as fh:
            fh.write(audio_bytes)
        sample_duration_seconds = _probe_audio_seconds(audio_path)
        with _tts_lock:
            model = _load_model()
            prompt = model.create_voice_clone_prompt(
                audio_path, ref_text=ref_text or None
            )
            prompt.save(prompt_path)
        cleaned_instruct = (
            instruct_override.strip()[:240]
            if isinstance(instruct_override, str) and instruct_override.strip()
            else ""
        )
        metadata = {
            "name": profile_name,
            "audio_file": audio_path,
            "prompt_file": prompt_path,
            "created_at": time.time(),
            "sample_duration_seconds": sample_duration_seconds,
            "ref_text": (ref_text or "").strip() or None,
            "instruct_override": cleaned_instruct or None,
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


@app.get("/profiles")
def list_profiles_endpoint() -> dict:
    """Return every cloned profile with its metadata, for management UIs.

    Presets live in code (``VOICE_PRESETS``) and are deliberately excluded:
    this endpoint answers \"what can I delete / manage\", not \"what can I
    synthesize with\" (that is ``/voices``). The full metadata lets the
    WebUI surface the sample duration and current ``instruct_override``
    without a second round-trip per profile.
    """
    profiles = []
    for name in _list_profiles():
        meta = _load_profile_metadata(name)
        profiles.append(
            {
                "name": name,
                "instruct_override": meta.get("instruct_override", "") or "",
                "sample_duration_seconds": meta.get("sample_duration_seconds"),
                "ref_text": meta.get("ref_text") or "",
                "created_at": meta.get("created_at"),
            }
        )
    return {"profiles": profiles}


@app.delete("/profiles/{name}")
def delete_profile(name: str) -> dict:
    """Delete a cloned profile and every file that belongs to it.

    All ``<name>.*`` files in ``PROFILES_DIR`` are removed (the saved
    ``VoiceClonePrompt`` ``.pt``, the ``.json`` metadata, and the reference
    audio sample). Presets are never touchable through this endpoint: they
    live in ``VOICE_PRESETS``, not on disk, so a preset name has no prompt
    file and simply reports 404. Returns the remaining profile list so
    callers can refresh without a second round-trip.
    """
    profile_name = _normalize_profile_name(name)
    if not profile_name:
        raise HTTPException(
            status_code=400, detail="profile name must be non-empty"
        )
    if _profile_prompt_path(profile_name) is None:
        raise HTTPException(
            status_code=404,
            detail=f"voice profile '{profile_name}' not found",
        )

    removed_files = 0
    prefix = f"{profile_name}."
    if os.path.isdir(PROFILES_DIR):
        for entry in os.listdir(PROFILES_DIR):
            if entry.startswith(prefix):
                try:
                    os.remove(os.path.join(PROFILES_DIR, entry))
                    removed_files += 1
                except OSError:
                    pass

    return {
        "ok": True,
        "name": profile_name,
        "removed_files": removed_files,
        "voices": _list_profiles(),
    }


def _load_profile_metadata(profile_name: str) -> dict:
    """Read the optional ``<name>.json`` metadata sidecar, or return ``{}``.

    The metadata is purely advisory (instruct_override for emotional
    conditioning, ref_text for transcripts, sample_duration_seconds for
    quality diagnostics). Missing fields fall back to empty values so
    older metadata files keep working.
    """
    meta_path = os.path.join(PROFILES_DIR, f"{profile_name}.json")
    if not os.path.isfile(meta_path):
        return {}
    try:
        with open(meta_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _compose_instruct(parts: list[str]) -> str:
    """Join non-empty instruct fragments with a comma and trim the result."""
    cleaned = [item.strip() for item in parts if isinstance(item, str) and item.strip()]
    return ", ".join(cleaned)


@app.post("/generate")
def generate_audio(request: GenerateRequest):
    text = (request.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text cannot be empty")
    voice = (request.voice or "").strip().lower()
    style = (request.style or "").strip()
    if len(style) > 240:
        raise HTTPException(status_code=400, detail="style instruction is too long")
    instruct = VOICE_PRESETS.get(voice)
    profile_prompt = None
    profile_metadata: dict = {}
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
        profile_metadata = _load_profile_metadata(voice)

    # Profile-level emotion conditioning. Saved on clone so a user can make
    # their favourite clone sound excited, calm, or any other "instruct"
    # string once and apply it to every future generation.
    profile_instruct = ""
    if profile_metadata:
        raw = profile_metadata.get("instruct_override", "")
        if isinstance(raw, str):
            profile_instruct = raw.strip()[:240]

    try:
        with _tts_lock:
            model = _load_model()
            generate_kwargs: dict = {"text": text}
            # Position temperature drives how much the diffusion sampler can
            # wander from the conditioning. For cloned voices the conditioning
            # IS the speaker identity, so any wandering changes the voice
            # timbre from one call to the next — the bug that made the user
            # hear a different voice on every render. Force greedy sampling
            # on the cloned path; preset voices still get the expressiveness
            # of ``DEFAULT_POSITION_TEMPERATURE`` since there is no prompt to
            # anchor the timbre.
            if profile_prompt is not None:
                # Pass the prompt AND any merged instruct so a cloned voice
                # can carry emotion on top of its timbre. OmniVoice's
                # generate() accepts both kwargs together; older builds
                # that reject ``instruct`` alongside the prompt silently
                # fall back to prompt-only delivery (still the speaker's
                # voice, but the style override is ignored).
                generate_kwargs["voice_clone_prompt"] = profile_prompt
                merged_instruct = _compose_instruct(
                    [profile_instruct, style]
                )
                if merged_instruct:
                    generate_kwargs["instruct"] = merged_instruct
                generate_kwargs["position_temperature"] = (
                    CLONED_VOICE_POSITION_TEMPERATURE
                )
            else:
                if instruct or style:
                    generate_kwargs["instruct"] = _compose_instruct(
                        [instruct or "", style]
                    )
                generate_kwargs["position_temperature"] = (
                    DEFAULT_POSITION_TEMPERATURE
                )
            if request.speed is not None and request.speed > 0:
                generate_kwargs["speed"] = float(request.speed)
            generate_kwargs["num_step"] = DEFAULT_GENERATION_STEPS
            waveforms = model.generate(**generate_kwargs)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"audio generation failed: {exc}"
        ) from exc

    # OmniVoice returns one or more waveform tensors at the model's native
    # 24 kHz. Encode the response in memory; the previous temporary file was
    # never deleted by FileResponse and added disk I/O to every request.
    try:
        import soundfile as sf

        wav = waveforms[0]
        if hasattr(wav, "detach"):
            wav = wav.detach().cpu().float().numpy()
        output = io.BytesIO()
        sf.write(output, wav, 24000, format="WAV")
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"audio write failed: {exc}"
        ) from exc

    return Response(
        content=output.getvalue(),
        media_type="audio/wav",
        headers={"Content-Disposition": 'attachment; filename="speech.wav"'},
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
    host = os.environ.get("OMNIVOICE_HOST", DEFAULT_HOST)
    port = int(os.environ.get("OMNIVOICE_PORT", DEFAULT_PORT))
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
