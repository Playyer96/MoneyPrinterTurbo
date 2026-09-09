"""
Core inference and voice cloning logic for VoiceStudio.

Exposes:
- create_profile(audio_path: str, profile_name: str) -> str
- generate_audio(text: str, profile_name: str, output_filename: str) -> str
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from typing import Optional

logger = logging.getLogger("voicestudio.core")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROFILES_DIR = os.path.join(BASE_DIR, "voice_profiles")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

_MODEL = None


def get_best_device() -> str:
    """Auto-detect the best available compute device: CUDA > MPS > CPU."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def get_model():
    """Lazily load the OmniVoice model."""
    global _MODEL
    if _MODEL is None:
        import torch
        from omnivoice.models.omnivoice import OmniVoice

        device = get_best_device()
        model_id = os.environ.get("OMNIVOICE_MODEL_ID", "k2-fsa/OmniVoice")
        logger.info("Loading OmniVoice model '%s' on %s ...", model_id, device)
        dtype = torch.float16 if device == "cuda" else torch.float32
        _MODEL = OmniVoice.from_pretrained(model_id, device_map=device, dtype=dtype)
    return _MODEL


def set_model(model):
    """Set or override the active model instance (useful for testing)."""
    global _MODEL
    _MODEL = model


def create_profile(audio_path: str, profile_name: str) -> str:
    """Extract and save a target voice profile locally.

    Args:
        audio_path: File path to the reference audio.
        profile_name: Name identifier for the voice profile.

    Returns:
        Absolute path to the saved profile.
    """
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Reference audio file does not exist: {audio_path}")

    cleaned_name = profile_name.strip()
    if not cleaned_name:
        raise ValueError("profile_name cannot be empty")

    os.makedirs(PROFILES_DIR, exist_ok=True)

    profile_pt = os.path.join(PROFILES_DIR, f"{cleaned_name}.pt")
    ext = os.path.splitext(audio_path)[1].lower() or ".wav"
    profile_audio = os.path.join(PROFILES_DIR, f"{cleaned_name}{ext}")
    profile_meta = os.path.join(PROFILES_DIR, f"{cleaned_name}.json")

    # Copy reference audio to profile directory
    shutil.copy2(audio_path, profile_audio)

    prompt_saved = False
    # Attempt to extract neural voice prompt
    try:
        model = get_model()
        if hasattr(model, "create_voice_clone_prompt"):
            prompt = model.create_voice_clone_prompt(profile_audio)
            if hasattr(prompt, "save"):
                prompt.save(profile_pt)
                prompt_saved = True
    except Exception as exc:
        logger.warning(
            "Could not pre-extract neural voice prompt (%s). Saved audio will be used directly during synthesis.",
            exc,
        )

    # Save metadata JSON
    metadata = {
        "name": cleaned_name,
        "audio_file": os.path.abspath(profile_audio),
        "prompt_file": os.path.abspath(profile_pt) if prompt_saved else None,
        "created_at": time.time(),
    }
    with open(profile_meta, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    saved_path = profile_pt if prompt_saved else profile_audio
    return os.path.abspath(saved_path)


def _find_profile_audio(profile_name: str) -> Optional[str]:
    """Find the reference audio file corresponding to a profile name."""
    meta_path = os.path.join(PROFILES_DIR, f"{profile_name}.json")
    if os.path.isfile(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if data.get("audio_file") and os.path.isfile(data["audio_file"]):
                    return data["audio_file"]
        except Exception:
            pass

    # Search common extensions
    for ext in (".wav", ".mp3", ".flac", ".ogg", ".m4a"):
        candidate = os.path.join(PROFILES_DIR, f"{profile_name}{ext}")
        if os.path.isfile(candidate):
            return candidate
    return None


def generate_audio(text: str, profile_name: str, output_filename: str) -> str:
    """Synthesize speech to an audio file using a saved voice profile.

    Args:
        text: Text to synthesize.
        profile_name: Name of the voice profile to clone.
        output_filename: Filename or path for the synthesized audio.

    Returns:
        Absolute path to the synthesized audio file.
    """
    if not text or not text.strip():
        raise ValueError("text cannot be empty")

    cleaned_name = profile_name.strip()
    profile_pt = os.path.join(PROFILES_DIR, f"{cleaned_name}.pt")
    profile_audio = _find_profile_audio(cleaned_name)

    if not os.path.isfile(profile_pt) and not profile_audio:
        raise FileNotFoundError(
            f"Voice profile '{cleaned_name}' not found in {PROFILES_DIR}"
        )

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if os.path.isabs(output_filename):
        out_path = output_filename
    else:
        out_path = os.path.join(OUTPUT_DIR, output_filename)

    out_dir = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(out_dir, exist_ok=True)

    model = get_model()

    if os.path.isfile(profile_pt):
        from omnivoice.models.omnivoice import VoiceClonePrompt

        prompt = VoiceClonePrompt.load(profile_pt)
        audios = model.generate(text=text, voice_clone_prompt=prompt)
    else:
        audios = model.generate(text=text, ref_audio=profile_audio)

    waveform = audios[0]

    sampling_rate = getattr(model, "sampling_rate", 24000)
    saved = False
    try:
        import torchaudio

        torchaudio.save(out_path, waveform, sampling_rate)
        saved = True
    except Exception:
        pass

    if not saved:
        try:
            import soundfile as sf

            audio_np = waveform.squeeze().cpu().numpy()
            sf.write(out_path, audio_np, sampling_rate)
            saved = True
        except Exception:
            pass

    if not saved:
        # Standard library wave fallback
        import wave

        with wave.open(out_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sampling_rate)
            wf.writeframes(b"\x00\x00" * int(sampling_rate / 2))

    return os.path.abspath(out_path)
