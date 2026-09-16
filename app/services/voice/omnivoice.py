"""HTTP client for the self-hosted OmniVoice service."""

from __future__ import annotations

import os
import shutil
import subprocess
from time import perf_counter
from typing import Union

import requests
from edge_tts import SubMaker
from loguru import logger
from moviepy.audio.io.AudioFileClip import AudioFileClip

from app.config import config
from app.services.audio_postprocess import soften_audio_transitions
from app.services.voice._shared import (
    ensure_file_path_exists,
    ensure_legacy_submaker_fields,
    get_omnivoice_base_url,
    get_omnivoice_profile_metadata,
    populate_legacy_submaker_with_full_text,
)
from app.utils import utils


def omnivoice_tts(
    text: str,
    voice_preset: str,
    voice_file: str,
    voice_rate: float = 1.0,
    voice_volume: float = 1.0,
    voice_style: str = "",
) -> Union[SubMaker, None]:
    """Synthesize speech with the self-hosted OmniVoice service.

    ``voice_style`` is the per-call emotion / delivery instruction (e.g.
    "enthusiastic, energetic, mid-pitch"). It is merged on top of the
    server-side ``instruct_override`` saved with the cloned profile (if
    any) and the global ``config.omnivoice.style`` so the operator can
    stack all three sources: profile register > per-call override >
    global default.
    """
    text = (text or "").strip()
    if not text:
        logger.error("OmniVoice TTS text is empty")
        return None
    if not any(character.isalnum() for character in text):
        logger.error("OmniVoice TTS text contains no speakable characters")
        return None

    payload = {"text": text, "voice": voice_preset}

    # Order matters: per-call voice_style wins over the global config
    # default and the profile's saved instruct_override, so the WebUI
    # slider / pipeline override can shadow the saved value when the
    # user wants something specific. Profile-level overrides win over the
    # global default because they describe the speaker's intended register.
    style_parts: list[str] = []
    global_style = str(config.omnivoice.get("style", "") or "").strip()
    if global_style:
        style_parts.append(global_style)
    profile_meta = get_omnivoice_profile_metadata(voice_preset)
    profile_instruct = ""
    if profile_meta:
        raw = profile_meta.get("instruct_override", "")
        if isinstance(raw, str):
            profile_instruct = raw.strip()
    if profile_instruct:
        style_parts.append(profile_instruct)
    if voice_style and voice_style.strip():
        style_parts.append(voice_style.strip())
    merged_style = ", ".join(style_parts)[:240]
    if merged_style:
        payload["style"] = merged_style
    if voice_rate not in (None, 1.0) and float(voice_rate) > 0:
        payload["speed"] = float(voice_rate)

    base_url = get_omnivoice_base_url()
    started = perf_counter()
    for attempt in range(3):
        try:
            logger.info(
                f"start omnivoice tts, voice preset: {voice_preset}, "
                f"text length: {len(text)}, try: {attempt + 1}"
            )
            ensure_file_path_exists(voice_file)
            response = requests.post(
                f"{base_url}/generate", json=payload, timeout=1800
            )
            if response.status_code == 404:
                logger.error(
                    f"OmniVoice voice preset not found: {voice_preset!r} "
                    f"(server response: {response.text[:200]!r})"
                )
                return None
            if response.status_code != 200:
                logger.error(
                    f"omnivoice tts failed with status {response.status_code}: "
                    f"{response.text[:200]}"
                )
                continue
            if not response.content or len(response.content) < 1024:
                logger.error(
                    "OmniVoice TTS returned empty or invalid audio data "
                    f"({len(response.content) if response.content else 0} bytes)"
                )
                continue

            with open(voice_file, "wb") as output:
                output.write(response.content)
            _apply_volume(voice_file, voice_volume)

            audio_clip = AudioFileClip(voice_file)
            try:
                audio_duration = audio_clip.duration
            finally:
                audio_clip.close()
            if audio_duration <= 0:
                logger.error(
                    "omnivoice tts produced audio with zero/negative "
                    f"duration: {audio_duration:.3f}s"
                )
                continue

            # Soften the silence boundaries so the natural sentence pauses
            # don't read as abrupt digital clicks between sentences. The
            # helper no-ops when the file has no detectable silences or
            # isn't a PCM WAV, so legacy volume paths stay untouched.
            try:
                soften_audio_transitions(voice_file)
            except Exception as exc:
                logger.warning(
                    f"omnivoice silence softening skipped: "
                    f"{type(exc).__name__}: {exc}"
                )

            elapsed = perf_counter() - started
            logger.success(
                f"omnivoice tts succeeded: {voice_file}, "
                f"duration={audio_duration:.2f}s, elapsed={elapsed:.2f}s"
            )
            return populate_legacy_submaker_with_full_text(
                sub_maker=ensure_legacy_submaker_fields(SubMaker()),
                text=text,
                audio_duration_seconds=audio_duration,
            )
        except Exception as exc:
            logger.error(f"omnivoice tts failed: {type(exc).__name__}: {exc}")
    return None


def _apply_volume(voice_file: str, voice_volume: float) -> None:
    """Apply non-default gain locally because OmniVoice has no gain setting."""
    normalized_volume = max(0.0, float(voice_volume or 1.0))
    if abs(normalized_volume - 1.0) <= 0.001:
        return
    adjusted_file = f"{voice_file}.vol.wav"
    try:
        result = subprocess.run(
            [
                utils.get_ffmpeg_binary(),
                "-y",
                "-i",
                voice_file,
                "-af",
                f"volume={normalized_volume:.3f}",
                "-acodec",
                "pcm_s16le",
                adjusted_file,
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
        if result.returncode == 0:
            shutil.move(adjusted_file, voice_file)
        else:
            logger.warning(
                "omnivoice volume adjustment failed; output will use unchanged "
                f"audio. stderr: {(result.stderr or '')[:200]}"
            )
            if os.path.exists(adjusted_file):
                os.remove(adjusted_file)
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning(
            "omnivoice volume adjustment skipped due to "
            f"{type(exc).__name__}: {exc}"
        )
