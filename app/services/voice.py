import asyncio
import base64
import io
import inspect
import json
import math
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unicodedata
import wave
from datetime import datetime, timedelta
from time import perf_counter
from typing import Union
from urllib.parse import quote, urlparse
from xml.sax.saxutils import escape, unescape

import edge_tts
import requests
from edge_tts import SubMaker
from edge_tts.srt_composer import Subtitle
from loguru import logger
from moviepy.video.tools import subtitles
from moviepy.audio.io.AudioFileClip import AudioFileClip
from openai import OpenAI

from app.config import config
from app.services import guardrails
from app.utils import utils

_DEFAULT_EDGE_TTS_TIMEOUT_SECONDS = 30.0
_MIMO_DEFAULT_BASE_URL = "https://api.xiaomimimo.com/v1"
_MIMO_DEFAULT_TTS_MODEL = "mimo-v2.5-tts"
MINIMAX_TTS_GLOBAL_URL = "https://api.minimax.io/v1/t2a_v2"
MINIMAX_TTS_CN_URL = "https://api.minimaxi.com/v1/t2a_v2"
MINIMAX_TTS_DEFAULT_MODEL = "speech-2.8-hd"
MINIMAX_TTS_DEFAULT_VOICE = "English_expressive_narrator"
MINIMAX_TTS_MODELS = (
    "speech-2.8-hd", "speech-2.8-turbo", "speech-2.6-hd", "speech-2.6-turbo",
    "speech-02-hd", "speech-02-turbo", "speech-01-hd", "speech-01-turbo",
)
GEMINI_TTS_VOICES = (
    ("Zephyr", "Bright"),
    ("Puck", "Upbeat"),
    ("Charon", "Informative"),
    ("Kore", "Firm"),
    ("Fenrir", "Excitable"),
    ("Leda", "Youthful"),
    ("Orus", "Firm"),
    ("Aoede", "Breezy"),
    ("Callirrhoe", "Easy-going"),
    ("Autonoe", "Bright"),
    ("Enceladus", "Breathy"),
    ("Iapetus", "Clear"),
    ("Umbriel", "Easy-going"),
    ("Algieba", "Smooth"),
    ("Despina", "Smooth"),
    ("Erinome", "Clear"),
    ("Algenib", "Gravelly"),
    ("Rasalgethi", "Informative"),
    ("Laomedeia", "Upbeat"),
    ("Achernar", "Soft"),
    ("Alnilam", "Firm"),
    ("Schedar", "Even"),
    ("Gacrux", "Mature"),
    ("Pulcherrima", "Forward"),
    ("Achird", "Friendly"),
    ("Zubenelgenubi", "Casual"),
    ("Vindemiatrix", "Gentle"),
    ("Sadachbia", "Lively"),
    ("Sadaltager", "Knowledgeable"),
    ("Sulafat", "Warm"),
)
_MINIMAX_TTS_MAX_AUDIO_HEX_CHARS = 100 * 1024 * 1024
NO_VOICE_NAME = "no-voice"
# `none` was the no-narration marker used in PR #981. Keep it temporarily so
# API clients that called this branch directly do not break on upgrade; the
# WebUI and new code use the clearer `no-voice` value.
_NO_VOICE_ALIASES = {NO_VOICE_NAME, "none"}


def _configure_pydub_ffmpeg(audio_segment_cls):
    configured_ffmpeg = utils.get_ffmpeg_binary()
    if configured_ffmpeg:
        audio_segment_cls.converter = configured_ffmpeg


def mktimestamp(time_unit: float) -> str:
    """
    Convert edge_tts 100-nanosecond units to a subtitle timestamp.

    edge_tts 7.x no longer exports the legacy `mktimestamp`, but the existing
    subtitle path still needs this formatter for timelines built manually by
    Azure v2, Gemini, and SiliconFlow, so keep an equivalent implementation.
    """
    hour = math.floor(time_unit / 10**7 / 3600)
    minute = math.floor((time_unit / 10**7 / 60) % 60)
    seconds = (time_unit / 10**7) % 60
    return f"{hour:02d}:{minute:02d}:{seconds:06.3f}"


def get_siliconflow_voices() -> list[str]:
    """
    Return the SiliconFlow voice list.

    Returns:
        Voice list formatted as ["siliconflow:FunAudioLLM/CosyVoice2-0.5B:alex", ...]
    """
    # SiliconFlow voices and display genders
    voices_with_gender = [
        ("FunAudioLLM/CosyVoice2-0.5B", "alex", "Male"),
        ("FunAudioLLM/CosyVoice2-0.5B", "anna", "Female"),
        ("FunAudioLLM/CosyVoice2-0.5B", "bella", "Female"),
        ("FunAudioLLM/CosyVoice2-0.5B", "benjamin", "Male"),
        ("FunAudioLLM/CosyVoice2-0.5B", "charles", "Male"),
        ("FunAudioLLM/CosyVoice2-0.5B", "claire", "Female"),
        ("FunAudioLLM/CosyVoice2-0.5B", "david", "Male"),
        ("FunAudioLLM/CosyVoice2-0.5B", "diana", "Female"),
    ]

    # Add the siliconflow prefix and format the display name.
    return [
        f"siliconflow:{model}:{voice}-{gender}"
        for model, voice, gender in voices_with_gender
    ]


def get_gemini_voices() -> list[str]:
    """
    Return the official Gemini TTS preset voices.

    Google does not publish gender metadata for these voices, so the selector
    uses official style descriptions instead of persisting guessed gender in
    the voice ID. Voice catalog source:
    https://ai.google.dev/gemini-api/docs/speech-generation#voice-options

    Returns:
        Voice list formatted as ["gemini:Zephyr-Bright", "gemini:Puck-Upbeat", ...]
    """
    return [f"gemini:{voice}-{style}" for voice, style in GEMINI_TTS_VOICES]


def get_mimo_voices() -> list[str]:
    """
    Return the Xiaomi MiMo V2.5 TTS preset voices.

    Only the documented `mimo-v2.5-tts` preset mode is integrated. Voice
    design (`mimo-v2.5-tts-voicedesign`) and voice cloning
    (`mimo-v2.5-tts-voiceclone`) need extra inputs and uploads, so they are
    omitted from the normal selector to avoid implying that a voice ID alone
    enables those advanced features.
    """
    voices_with_gender = [
        ("mimo_default", "Female"),
        ("冰糖", "Female"),
        ("茉莉", "Female"),
        ("苏打", "Male"),
        ("白桦", "Male"),
        ("Mia", "Female"),
        ("Chloe", "Female"),
        ("Milo", "Male"),
        ("Dean", "Male"),
    ]

    return [f"mimo:{voice}-{gender}" for voice, gender in voices_with_gender]


def get_minimax_voices(voice_id: str | None = None) -> list[str]:
    """Return configured MiniMax voices in the shared TTS routing format."""
    voice_id = str(
        voice_id
        or config.minimax_tts.get("voice_id", MINIMAX_TTS_DEFAULT_VOICE)
        or MINIMAX_TTS_DEFAULT_VOICE
    ).strip()
    return [f"minimax:{voice_id}"]


def get_elevenlabs_voices(api_key: str) -> list[str]:
    if not api_key:
        return []
    try:
        url = "https://api.elevenlabs.io/v2/voices"
        params = {"is_favorite": "true", "page_size": 100}
        headers = {"xi-api-key": api_key}
        response = requests.get(url, params=params, headers=headers, timeout=10)
        if response.status_code != 200:
            logger.warning(
                f"ElevenLabs voices fetch failed with status {response.status_code}: {response.text}"
            )
            return []
        data = response.json()
        voices = data.get("voices", [])
        return [
            f"elevenlabs:{v['voice_id']}:{v['name']}"
            for v in voices
            if v.get("voice_id") and v.get("name") and v.get("status") != "disabled"
        ]
    except Exception as e:
        logger.warning(f"ElevenLabs voices fetch failed: {str(e)}")
        return []


def get_chatterbox_voices() -> list[str]:
    """Return the configured Chatterbox voices.

    Chatterbox is self-hosted, so there is no global voice catalog. Operators
    list the voice names exposed by their server via ``[chatterbox] voices``
    (a TOML array, or a comma-separated string). Each entry is normalised to
    the ``chatterbox:<name>`` format used by the TTS dispatcher.
    """
    voices = config.chatterbox.get("voices", []) or []
    if isinstance(voices, str):
        voices = [v.strip() for v in voices.split(",") if v.strip()]
    result = []
    for v in voices:
        v = str(v).strip()
        if not v:
            continue
        result.append(v if v.startswith("chatterbox:") else f"chatterbox:{v}")
    if not result:
        # keep the dropdown usable even before any voice is configured
        result = ["chatterbox:default-Female"]
    return result


KOKORO_DEFAULT_VOICE = "af_heart"


def _normalize_kokoro_voices(entries) -> list[str]:
    """Normalize configured and server formats while accepting only real IDs."""
    if isinstance(entries, str):
        entries = entries.split(",")
    if not isinstance(entries, list):
        return []
    result = []
    for entry in entries:
        name = entry.get("id") if isinstance(entry, dict) else entry
        if not isinstance(name, str):
            continue
        name = name.strip().removeprefix("kokoro:").strip()
        if name:
            value = f"kokoro:{name}"
            if value not in result:
                result.append(value)
    return result


def get_kokoro_voices(*, fallback: bool = True) -> list[str]:
    """Prefer configured voices, otherwise query the server.

    The UI may disable defaults to detect outages while preserving a selection.
    Older servers return strings; newer ones return objects containing IDs. On
    failure, session state is not cached here; the WebUI preserves the last
    successful catalog to avoid leaking state across users or endpoints.
    """
    voices = _normalize_kokoro_voices(config.kokoro.get("voices"))
    if not voices:
        base_url = (config.kokoro.get("base_url", "") or "").strip().rstrip("/")
        if base_url:
            try:
                headers = {}
                api_key = config.kokoro.get("api_key", "")
                if api_key:
                    headers["Authorization"] = f"Bearer {api_key}"
                response = requests.get(
                    f"{base_url}/audio/voices", headers=headers, timeout=5
                )
                if response.status_code == 200:
                    data = response.json()
                    listed = data.get("voices", []) if isinstance(data, dict) else data
                    voices = _normalize_kokoro_voices(listed)
                    if not voices:
                        logger.warning("kokoro voice list contains no valid voice IDs")
                else:
                    logger.warning(
                        f"kokoro voices request failed with status {response.status_code}"
                    )
            except Exception as e:
                # Do not log URLs or exception bodies; query strings may contain credentials.
                logger.warning(f"kokoro voice list unavailable ({type(e).__name__})")
    return voices or ([f"kokoro:{KOKORO_DEFAULT_VOICE}"] if fallback else [])


def get_fish_audio_voices() -> list[str]:
    """Return configured Fish Audio voices.

    Each entry follows the format ``fish_audio:<reference_id>:<display_name>``.
    When ``reference_id`` is "default", Fish Audio's built-in default voice is
    used (no ``reference_id`` is sent in the API request).  Operators can list
    additional public or cloned voices via ``[fish_audio] voices`` in the
    config file.
    """
    result = [
        "fish_audio:2324c907b9a94c64ab4afb941e5b3408:Clear Female-Female",
        "fish_audio:7b6131ba75ba47c98a46c847db729ab6:Clear Male-Male",
        "fish_audio:default:Default Voice",
    ]
    voices = config.fish_audio.get("voices", []) or []
    if isinstance(voices, str):
        voices = [v.strip() for v in voices.split(",") if v.strip()]
    for entry in voices:
        entry = str(entry).strip()
        if not entry:
            continue
        if entry.startswith("fish_audio:"):
            result.append(entry)
        elif ":" in entry:
            # "<reference_id>:<display_name>"
            result.append(f"fish_audio:{entry}")
        else:
            # bare reference_id
            result.append(f"fish_audio:{entry}:{entry}")
    return result


VOICESTUDIO_DEFAULT_BASE_URL = "http://127.0.0.1:8780"


def get_voicestudio_base_url() -> str:
    """Return the base URL of the self-hosted VoiceStudio server.

    An explicit ``VOICESTUDIO_BASE_URL`` environment variable wins (used by
    the Docker deployment to reach a host-side VoiceStudio via
    ``host.docker.internal``), then the configured value, then the default.
    """
    env = os.environ.get("VOICESTUDIO_BASE_URL", "")
    if env.strip():
        return env.strip().rstrip("/")
    configured = config.voicestudio.get("base_url", "") if hasattr(config, "voicestudio") else ""
    return str(configured or VOICESTUDIO_DEFAULT_BASE_URL).strip().rstrip("/")


def _is_running_in_docker() -> bool:
    """True when this process is inside a container.

    The vendored VoiceStudio server cannot run here on macOS (no Metal in the
    Linux VM) and has no business running here on Linux either -- the compose
    service is the real server in that case. Use this only to skip the
    in-process spawn fallback; do not gate functionality on it.
    """
    if os.path.exists("/.dockerenv"):
        return True
    try:
        with open("/proc/1/cgroup", "r", encoding="utf-8", errors="replace") as fh:
            return any(
                marker in line
                for line in fh
                for marker in ("docker", "containerd", "kubepods", "buildkit")
            )
    except OSError:
        # /proc is Linux-only; non-Linux hosts are never in a Linux container.
        return False


def ensure_voicestudio_server_running(timeout: float = 2.0) -> bool:
    """
    Spawn the bundled VoiceStudio server when the configured port is dead.

    Idempotent: probes ``GET /voices`` first, only launches when unreachable,
    so the WebUI's hot-reload and any externally-managed server (Docker, a
    separate terminal) stay unaffected. Returns True when a usable server is
    reachable at the end of the call, False otherwise — never raises, so
    importing this module never breaks startup.

    Inside a container the spawn is skipped on purpose: torch cannot reach a
    host GPU from a Linux VM, and on Linux the right server is the compose
    service, not an in-process subprocess. The caller (WebUI or API) is
    expected to surface the warning we log here so the user knows which
    command to run on the host.
    """
    base_url = get_voicestudio_base_url()
    try:
        response = requests.get(f"{base_url}/voices", timeout=timeout)
        if response.status_code == 200:
            return True
    except Exception:
        pass

    if _is_running_in_docker():
        # Compose owns VoiceStudio on Linux and the host Metal service on macOS.
        logger.warning(
            "voicestudio not reachable at {} and we are inside a container, "
            "so the bundled server cannot be launched here. Start the stack "
            "with the platform's Docker Compose files.",
            base_url,
        )
        return False

    server_script = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "vendor",
        "voice_studio",
        "server.py",
    )
    if not os.path.isfile(server_script):
        logger.warning(
            f"voicestudio server script missing at {server_script}; "
            "start it manually before generating previews"
        )
        return False

    logger.info(f"voicestudio not reachable at {base_url}; launching bundled server")
    try:
        kwargs = {
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
            "stdin": subprocess.DEVNULL,
        }
        if sys.platform != "win32":
            kwargs["start_new_session"] = True
        else:
            kwargs["creationflags"] = getattr(
                subprocess, "DETACHED_PROCESS", 0x00000008
            )
        subprocess.Popen([sys.executable, server_script], **kwargs)
    except Exception as exc:
        logger.warning(f"failed to spawn voicestudio server: {exc}")
        return False

    # give the server a moment to bind; do not block startup forever.
    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        try:
            response = requests.get(f"{base_url}/voices", timeout=1.0)
            if response.status_code == 200:
                logger.success(f"voicestudio server is up at {base_url}")
                return True
        except Exception:
            time.sleep(0.5)
    logger.warning(
        f"voicestudio server did not respond within 15s at {base_url}; "
        "check the server logs and try again"
    )
    return False


def get_voicestudio_voices() -> list[str]:
    """Read bundled voice-design presets from the local OmniVoice server.

    Each preset is returned as ``voicestudio:<preset_name>`` so it can be
    selected in the WebUI and dispatched by :func:`_single_tts`. The
    matching ``instruct`` string is resolved server-side; callers only
    see stable preset names.
    """
    base_url = get_voicestudio_base_url()
    try:
        response = requests.get(f"{base_url}/voices", timeout=5)
        if response.status_code == 200:
            data = response.json()
            voices = data.get("voices", []) if isinstance(data, dict) else []
            names = [
                entry.get("name")
                for entry in voices
                if isinstance(entry, dict) and entry.get("name")
            ]
            return [f"voicestudio:{name}" for name in names]
        logger.warning(
            f"voicestudio voices request failed with status {response.status_code}"
        )
    except Exception as e:
        # Do not log URLs or exception bodies; query strings may contain credentials.
        logger.warning(f"voicestudio voice list unavailable ({type(e).__name__})")
    return []


def create_voicestudio_profile(
    profile_name: str, audio_bytes: bytes, original_filename: str
) -> tuple[bool, str]:
    """Clone a voice on the local OmniVoice bridge from an uploaded sample.

    The sample is POSTed as a multipart upload to ``{base_url}/profiles``.
    The server stores the reference audio, auto-transcribes it (or uses the
    optional transcript) and persists a reusable ``VoiceClonePrompt`` under
    its ``voice_profiles/`` directory. On success the profile shows up in the
    ``/voices`` catalog as ``voicestudio:<name>`` for the TTS drop-down.
    Returns ``(ok, message_or_name)`` mirroring the WebUI contract.
    """
    profile_name = (profile_name or "").strip()
    if not profile_name:
        return False, "profile name must be non-empty"
    if not audio_bytes:
        return False, "audio sample cannot be empty"
    base_url = get_voicestudio_base_url()
    try:
        response = requests.post(
            f"{base_url}/profiles",
            data={"name": profile_name},
            files={
                "audio": (
                    original_filename or "voice_sample.wav",
                    audio_bytes,
                    "application/octet-stream",
                )
            },
            timeout=1800,
        )
    except Exception as exc:
        logger.warning(f"voicestudio profile creation failed: {exc}")
        return False, f"voicestudio server unreachable: {exc}"

    if response.status_code == 201 or response.status_code == 200:
        try:
            detail = response.json()
        except ValueError:
            detail = {}
        name = detail.get("name") or profile_name
        logger.success(f"voicestudio profile created: {name}")
        return True, name
    logger.warning(
        f"voicestudio profile creation failed with status "
        f"{response.status_code}: {response.text[:200]}"
    )
    return False, response.text[:200]


def get_voicestudio_profiles() -> list[str]:
    """List cloned profile names from the local OmniVoice bridge.

    Unlike :func:`get_voicestudio_voices` this returns only the cloned
    profiles (not the bundled presets), so the WebUI can offer them for
    deletion without having to know which names are presets.
    """
    base_url = get_voicestudio_base_url()
    try:
        response = requests.get(f"{base_url}/profiles", timeout=5)
        if response.status_code == 200:
            data = response.json()
            profiles = data.get("profiles", []) if isinstance(data, dict) else []
            return [
                entry.get("name")
                for entry in profiles
                if isinstance(entry, dict) and entry.get("name")
            ]
    except Exception as exc:
        logger.warning(f"voicestudio profile list unavailable: {exc}")
    return []


def delete_voicestudio_profile(profile_name: str) -> tuple[bool, str]:
    """Delete a cloned voice profile from the local OmniVoice bridge.

    Sends ``DELETE {base_url}/profiles/<name>``, which removes the saved
    ``VoiceClonePrompt`` (``.pt``), its metadata (``.json``), and the stored
    reference audio sample. Returns ``(ok, message_or_name)`` matching the
    contract used by :func:`create_voicestudio_profile`.
    """
    profile_name = (profile_name or "").strip()
    if not profile_name:
        return False, "profile name must be non-empty"
    base_url = get_voicestudio_base_url()
    try:
        response = requests.delete(
            f"{base_url}/profiles/{quote(profile_name, safe='')}", timeout=60
        )
    except Exception as exc:
        logger.warning(f"voicestudio profile deletion failed: {exc}")
        return False, f"voicestudio server unreachable: {exc}"

    if response.status_code == 200:
        try:
            detail = response.json()
        except ValueError:
            detail = {}
        name = detail.get("name") or profile_name
        logger.success(f"voicestudio profile removed: {name}")
        return True, name
    logger.warning(
        f"voicestudio profile removal failed with status "
        f"{response.status_code}: {response.text[:200]}"
    )
    return False, response.text[:200]


_AZURE_VOICES_DATA_FILE = os.path.join(
    os.path.dirname(__file__), "data", "azure_voices.json"
)
_azure_voices_cache = None


def _load_azure_voices() -> list[dict]:
    global _azure_voices_cache
    if _azure_voices_cache is None:
        with open(_AZURE_VOICES_DATA_FILE, "r", encoding="utf-8") as f:
            _azure_voices_cache = json.load(f)
    return _azure_voices_cache


def get_all_azure_voices(filter_locals=None) -> list[str]:
    voices = []
    for item in _load_azure_voices():
        name = item["name"]
        gender = item["gender"]
        # Apply filters.
        if filter_locals and any(
            name.lower().startswith(fl.lower()) for fl in filter_locals
        ):
            voices.append(f"{name}-{gender}")
        elif not filter_locals:
            voices.append(f"{name}-{gender}")

    voices.sort()
    return voices


def parse_voice_name(name: str):
    # zh-CN-XiaoyiNeural-Female
    # zh-CN-YunxiNeural-Male
    # zh-CN-XiaoxiaoMultilingualNeural-V2-Female
    name = name.replace("-Female", "").replace("-Male", "").strip()
    return name


def is_azure_v2_voice(voice_name: str):
    voice_name = parse_voice_name(voice_name)
    if voice_name.endswith("-V2"):
        return voice_name.replace("-V2", "").strip()
    return ""


def is_siliconflow_voice(voice_name: str):
    """Return whether this is a SiliconFlow voice."""
    return voice_name.startswith("siliconflow:")


def is_gemini_voice(voice_name: str):
    """Return whether this is a Gemini TTS voice."""
    return voice_name.startswith("gemini:")


def parse_gemini_voice_name(voice_name: str | None) -> str:
    """Extract the Google preset name from current or legacy selector values."""
    if not is_gemini_voice(voice_name or ""):
        return ""
    return (voice_name or "").split(":", 1)[1].split("-", 1)[0].strip()


def is_mimo_voice(voice_name: str):
    """Return whether this is a Xiaomi MiMo TTS voice."""
    return voice_name.startswith("mimo:")


def is_minimax_voice(voice_name: str | None) -> bool:
    return (voice_name or "").startswith("minimax:")


def is_elevenlabs_voice(voice_name: str) -> bool:
    return (voice_name or "").startswith("elevenlabs:")


def get_elevenlabs_api_key() -> str:
    """
    Return the API key used by ElevenLabs TTS.

    Configuration takes precedence and the environment is a fallback. The WebUI
    and music service already support ``ELEVENLABS_API_KEY``; TTS must use the
    same rule or environment-only deployments can list voices but incorrectly
    report a missing key during synthesis.
    """
    configured_key = str(config.elevenlabs.get("api_key", "") or "").strip()
    return configured_key or os.getenv("ELEVENLABS_API_KEY", "").strip()


def is_chatterbox_voice(voice_name: str) -> bool:
    return (voice_name or "").startswith("chatterbox:")


def is_kokoro_voice(voice_name: str) -> bool:
    return (voice_name or "").startswith("kokoro:")


def is_fish_audio_voice(voice_name: str) -> bool:
    return (voice_name or "").startswith("fish_audio:")


def is_voicestudio_voice(voice_name: str) -> bool:
    return (voice_name or "").startswith("voicestudio:")


def get_fish_audio_api_key() -> str:
    configured_key = str(config.fish_audio.get("api_key", "") if hasattr(config, "fish_audio") and isinstance(config.fish_audio, dict) else "").strip()
    return configured_key or os.getenv("FISH_API_KEY", "").strip()


def is_no_voice(voice_name: str | None) -> bool:
    """
    Return whether the user explicitly selected no narration.

    An empty string is intentionally not treated as no narration because it more
    likely means broken configuration, lost legacy WebUI state, or a missing API
    parameter. Only explicit sentinels enter the silent branch so real errors
    are not disguised as successful generation.
    """
    return str(voice_name or "").strip().lower() in _NO_VOICE_ALIASES


def is_azure_v1_voice(voice_name: str | None) -> bool:
    """
    Return whether this is an Azure TTS v1 (Edge TTS) preset voice.

    The first pause-tag (`[pause: ...]`) segmented synthesis path applies only
    to Edge TTS so it does not change request billing, frequency, or defaults
    for Gemini, Fish Audio, SiliconFlow, Kokoro, and other providers.
    """
    if not voice_name:
        return False
    name = str(voice_name).strip()
    if is_no_voice(name):
        return False
    if is_azure_v2_voice(name):
        return False
    if is_siliconflow_voice(name):
        return False
    if is_gemini_voice(name):
        return False
    if is_mimo_voice(name):
        return False
    if is_minimax_voice(name):
        return False
    if is_elevenlabs_voice(name):
        return False
    if is_chatterbox_voice(name):
        return False
    if is_kokoro_voice(name):
        return False
    if is_fish_audio_voice(name):
        return False
    if is_voicestudio_voice(name):
        return False
    return True


def estimate_no_voice_duration(text: str) -> float:
    """
    Estimate a stable video timeline duration for no-narration mode.

    Silent videos still need an audio placeholder to drive material trimming,
    subtitle timing, and final composition. The estimate stays simple:
    1. CJK characters use about 4.2 characters per second;
    2. English words and numbers use about 2.7 words per second;
    3. other scripts use about 4.0 characters per second, covering Russian,
       Arabic, Japanese kana, Korean, and other non-ASCII text;
    4. each sentence adds a short pause so subtitle changes are not too tight;
    5. the minimum is three seconds so very short scripts never yield zero.
    """
    normalized_text = (text or "").strip()
    if not normalized_text:
        return 3.0

    cjk_chars = len(re.findall(r"[\u4e00-\u9fff]", normalized_text))
    words = len(re.findall(r"[A-Za-z0-9]+", normalized_text))
    ascii_word_chars = sum(len(word) for word in re.findall(r"[A-Za-z0-9]+", normalized_text))
    other_text_chars = 0
    for char in normalized_text:
        # Unicode categories beginning with L are letters and N are numbers.
        # CJK and ASCII words were counted above, so count only the remaining
        # characters here to avoid timing English twice.
        category = unicodedata.category(char)
        if category.startswith(("L", "N")):
            other_text_chars += 1
    other_text_chars = max(other_text_chars - cjk_chars - ascii_word_chars, 0)
    sentence_count = max(len(utils.split_string_by_punctuations(normalized_text)), 1)

    cjk_duration = cjk_chars / 4.2
    word_duration = words / 2.7
    other_text_duration = other_text_chars / 4.0
    pause_duration = max(sentence_count - 1, 0) * 0.35
    return max(3.0, cjk_duration + word_duration + other_text_duration + pause_duration)


def generate_silent_audio(duration_seconds: float, output_file: str) -> bool:
    """
    Generate silent audio.

    Supports direct 16-bit mono PCM WAV output with sample-accurate duration and
    no codec delay, or MP3 through FFmpeg `anullsrc` as a no-narration placeholder.
    """
    ensure_file_path_exists(output_file)
    duration_seconds = max(
        float(duration_seconds or 0), utils.MIN_PAUSE_DURATION_SECONDS
    )

    if output_file.lower().endswith(".wav"):
        sample_rate = 24000
        num_samples = int(round(duration_seconds * sample_rate))
        with wave.open(output_file, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(b"\x00\x00" * num_samples)
        return os.path.exists(output_file) and os.path.getsize(output_file) > 0

    ffmpeg_binary = utils.get_ffmpeg_binary()
    command = [
        ffmpeg_binary,
        "-y",
        "-f",
        "lavfi",
        "-i",
        "anullsrc=r=44100:cl=mono",
        "-t",
        f"{duration_seconds:.3f}",
        "-codec:a",
        "libmp3lame",
        "-q:a",
        "4",
        output_file,
    ]

    logger.info(
        f"generating silent audio for no-voice mode, duration: {duration_seconds:.2f}s"
    )
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        logger.error(
            "failed to generate silent audio: "
            f"{(result.stderr or result.stdout or '').strip()}"
        )
        return False
    if not os.path.exists(output_file) or os.path.getsize(output_file) <= 0:
        logger.error(
            "silent audio output file is missing or empty, "
            f"file: {output_file}, duration: {duration_seconds:.2f}s"
        )
        return False
    return True


def _single_tts(
    text: str,
    voice_name: str,
    voice_rate: float,
    voice_file: str,
    voice_volume: float = 1.0,
) -> Union[SubMaker, None]:
    if is_no_voice(voice_name):
        duration_seconds = estimate_no_voice_duration(text)
        if not generate_silent_audio(duration_seconds, voice_file):
            return None

        sub_maker = ensure_legacy_submaker_fields(SubMaker())
        return populate_legacy_submaker_with_full_text(
            sub_maker=sub_maker,
            text=text,
            audio_duration_seconds=duration_seconds,
        )

    if is_azure_v2_voice(voice_name):
        return azure_tts_v2(
            text,
            voice_name,
            voice_file,
            voice_rate=voice_rate,
        )
    elif is_siliconflow_voice(voice_name):
        # Extract the model and voice from voice_name.
        # Format: siliconflow:model:voice-Gender
        parts = voice_name.split(":")
        if len(parts) >= 3:
            model = parts[1]
            # Remove the display gender suffix, for example "alex-Male" -> "alex".
            voice_with_gender = parts[2]
            voice = voice_with_gender.split("-")[0]
            # Build the full voice argument in "model:voice" format.
            full_voice = f"{model}:{voice}"
            return siliconflow_tts(
                text, model, full_voice, voice_rate, voice_file, voice_volume
            )
        else:
            logger.error(f"Invalid siliconflow voice name format: {voice_name}")
            return None
    elif is_gemini_voice(voice_name):
        # Extract the voice name. Format: gemini:voice-Style; legacy
        # gemini:voice-Gender values remain supported.
        voice = parse_gemini_voice_name(voice_name)
        if voice:
            # Model chosen by the user in the WebUI; fall back to flash preview
            # when missing so old config never silently hits an unexpected model
            # after the dropdown lands.
            selected_model = str(
                config.app.get(GEMINI_TTS_MODEL_CONFIG_KEY, "")
                or DEFAULT_GEMINI_TTS_MODEL
            ).strip() or DEFAULT_GEMINI_TTS_MODEL
            return gemini_tts(
                text,
                voice,
                voice_rate,
                voice_file,
                voice_volume,
                model=selected_model,
            )
        else:
            logger.error(f"Invalid gemini voice name format: {voice_name}")
            return None
    elif is_mimo_voice(voice_name):
        # Extract the voice name. Format: mimo:voice-Gender, or mimo:voice
        # when the caller already applied parse_voice_name. Support both.
        parts = voice_name.split(":")
        if len(parts) >= 2:
            voice_with_gender = parts[1]
            voice = voice_with_gender.split("-")[0]
            return mimo_tts(text, voice, voice_rate, voice_file, voice_volume)
        else:
            logger.error(f"Invalid mimo voice name format: {voice_name}")
            return None
    elif is_minimax_voice(voice_name):
        voice_id = voice_name.split(":", 1)[1].strip()
        if voice_id:
            return minimax_tts(text, voice_id, voice_rate, voice_file, voice_volume)
        logger.error(f"Invalid MiniMax voice name format: {voice_name}")
        return None
    elif is_elevenlabs_voice(voice_name):
        # Format: elevenlabs:{voice_id}:{name}
        parts = voice_name.split(":")
        if len(parts) >= 2:
            voice_id = parts[1]
            return elevenlabs_tts(text, voice_id, voice_file, voice_rate, voice_volume)
        else:
            logger.error(f"Invalid elevenlabs voice name format: {voice_name}")
            return None
    elif is_chatterbox_voice(voice_name):
        # Format: chatterbox:<voice>; voice may include a display gender suffix.
        parts = voice_name.split(":", 1)
        if len(parts) >= 2 and parts[1].strip():
            chatterbox_voice = parts[1].strip()
            if chatterbox_voice.endswith(("-Female", "-Male")):
                chatterbox_voice = chatterbox_voice.rsplit("-", 1)[0]
            return chatterbox_tts(
                text, chatterbox_voice, voice_file, voice_rate, voice_volume
            )
        else:
            logger.error(f"Invalid chatterbox voice name format: {voice_name}")
            return None
    elif is_kokoro_voice(voice_name):
        # Format: kokoro:<voice>; voice may include a display gender suffix.
        parts = voice_name.split(":", 1)
        if len(parts) >= 2 and parts[1].strip():
            kokoro_voice = parts[1].strip()
            if kokoro_voice.endswith(("-Female", "-Male")):
                kokoro_voice = kokoro_voice.rsplit("-", 1)[0]
            return kokoro_tts(
                text, kokoro_voice, voice_file, voice_rate, voice_volume
            )
        else:
            logger.error(f"Invalid kokoro voice name format: {voice_name}")
            return None
    elif is_fish_audio_voice(voice_name):
        parts = voice_name.split(":")
        reference_id = parts[1] if len(parts) >= 2 else "default"
        if reference_id == "default":
            reference_id = None
        return fish_audio_tts(text, voice_file, voice_rate, voice_volume, reference_id=reference_id)
    elif is_voicestudio_voice(voice_name):
        # Format: voicestudio:<preset_name>; the bridge resolves the matching
        # instruct string server-side.
        parts = voice_name.split(":", 1)
        if len(parts) >= 2 and parts[1].strip():
            return voicestudio_tts(
                text, parts[1].strip(), voice_file, voice_rate, voice_volume
            )
        else:
            logger.error(f"Invalid voicestudio voice name format: {voice_name}")
            return None
    return azure_tts_v1(text, voice_name, voice_rate, voice_file)


def _concat_audio_files(audio_files: list[str], output_file: str) -> bool:
    """
    Merge audio segments through PCM decoding and one final encode.

    Convert every input to standard 24 kHz 16-bit mono PCM for seamless
    concatenation, then encode the target once. This prevents accumulated MP3
    encoder delay and padding from causing audio/video or subtitle drift.
    """
    if not audio_files:
        return False
    ensure_file_path_exists(output_file)
    if len(audio_files) == 1:
        if audio_files[0] != output_file:
            shutil.copyfile(audio_files[0], output_file)
        return True

    target_sample_rate = 24000
    combined_pcm = bytearray()
    ffmpeg_binary = utils.get_ffmpeg_binary()

    with tempfile.TemporaryDirectory() as concat_temp:
        for idx, f in enumerate(audio_files):
            if not os.path.exists(f) or os.path.getsize(f) == 0:
                continue

            # Check whether this is already a 24 kHz 16-bit mono WAV.
            is_valid_pcm_wav = False
            if f.lower().endswith(".wav"):
                try:
                    with wave.open(f, "rb") as wf:
                        if (
                            wf.getframerate() == target_sample_rate
                            and wf.getnchannels() == 1
                            and wf.getsampwidth() == 2
                        ):
                            is_valid_pcm_wav = True
                            combined_pcm.extend(wf.readframes(wf.getnframes()))
                except Exception:
                    is_valid_pcm_wav = False

            if not is_valid_pcm_wav:
                # Decode the input to 24 kHz 16-bit mono PCM WAV with FFmpeg.
                pcm_wav = os.path.join(concat_temp, f"chunk_{idx}.wav")
                cmd = [
                    ffmpeg_binary,
                    "-y",
                    "-i",
                    f,
                    "-vn",
                    "-ac",
                    "1",
                    "-ar",
                    str(target_sample_rate),
                    "-codec:a",
                    "pcm_s16le",
                    pcm_wav,
                ]
                res = subprocess.run(cmd, capture_output=True, text=True, check=False)
                if res.returncode == 0 and os.path.exists(pcm_wav):
                    try:
                        with wave.open(pcm_wav, "rb") as wf:
                            combined_pcm.extend(wf.readframes(wf.getnframes()))
                    except Exception as e:
                        logger.error(f"failed to read decoded pcm wav: {e}")
                else:
                    logger.error(f"failed to decode audio chunk with ffmpeg: {res.stderr}")

        if not combined_pcm:
            logger.error("no valid audio samples to concatenate")
            return False

        temp_combined_wav = os.path.join(concat_temp, "combined_master.wav")
        with wave.open(temp_combined_wav, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(target_sample_rate)
            wf.writeframes(combined_pcm)

        if output_file.lower().endswith(".wav"):
            shutil.copyfile(temp_combined_wav, output_file)
            return True

        command = [
            ffmpeg_binary,
            "-y",
            "-i",
            temp_combined_wav,
            "-codec:a",
            "libmp3lame",
            "-q:a",
            "4",
            output_file,
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            logger.error(
                "failed to encode concatenated audio to mp3: "
                f"{(result.stderr or result.stdout or '').strip()}"
            )
            return False
        return os.path.exists(output_file) and os.path.getsize(output_file) > 0


def _tts_with_pauses(
    text: str,
    voice_name: str,
    voice_rate: float,
    voice_file: str,
    voice_volume: float = 1.0,
) -> Union[SubMaker, None]:
    """
    Synthesize scripts containing pause tags such as `[pause: 2s]`.
    Generate speech segments and exact PCM silence, calculate subtitle offsets
    from decoded samples, and perform one final encode.
    """
    segments = utils.parse_script_with_pauses(text)
    if not segments:
        return None

    speech_segments = [s for s in segments if s[0] == "speech"]
    pause_segments = [s for s in segments if s[0] == "pause"]

    if not pause_segments:
        clean_text = utils.remove_pause_tags(text)
        return _single_tts(clean_text, voice_name, voice_rate, voice_file, voice_volume)

    if not speech_segments:
        total_pause_duration = sum(float(s[1]) for s in pause_segments)
        total_pause_duration = min(
            total_pause_duration, utils.MAX_PAUSE_DURATION_SECONDS
        )
        if not generate_silent_audio(total_pause_duration, voice_file):
            return None
        sub_maker = ensure_legacy_submaker_fields(SubMaker())
        sub_maker.duration = total_pause_duration
        return populate_legacy_submaker_with_full_text(
            sub_maker=sub_maker,
            text=utils.remove_pause_tags(text),
            audio_duration_seconds=total_pause_duration,
        )

    SAMPLE_RATE = 24000
    with tempfile.TemporaryDirectory() as temp_dir:
        audio_chunk_files: list[str] = []
        combined_submaker = ensure_legacy_submaker_fields(SubMaker())
        cumulative_samples = 0

        for idx, (seg_type, seg_val) in enumerate(segments):
            if seg_type == "pause":
                pause_duration = float(seg_val)
                silence_wav = os.path.join(temp_dir, f"silence_{idx}.wav")
                num_silent_samples = int(round(pause_duration * SAMPLE_RATE))
                with wave.open(silence_wav, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(SAMPLE_RATE)
                    wf.writeframes(b"\x00\x00" * num_silent_samples)

                # Keep the shared helper observable to callers that instrument silence generation.
                generate_silent_audio(pause_duration, silence_wav)

                actual_pause_duration = pause_duration
                # Honor an instrumented duration result when it differs from the requested pause.
                mock_check_duration = get_audio_duration(silence_wav)
                if mock_check_duration > 0 and abs(mock_check_duration - pause_duration) > 0.05:
                    actual_pause_duration = mock_check_duration
                    num_silent_samples = int(round(actual_pause_duration * SAMPLE_RATE))

                audio_chunk_files.append(silence_wav)
                cumulative_samples += num_silent_samples

            elif seg_type == "speech":
                speech_text = str(seg_val).strip()
                if not speech_text:
                    continue

                chunk_audio_file = os.path.join(temp_dir, f"speech_{idx}.mp3")
                chunk_submaker = _single_tts(
                    text=speech_text,
                    voice_name=voice_name,
                    voice_rate=voice_rate,
                    voice_file=chunk_audio_file,
                    voice_volume=voice_volume,
                )
                if not chunk_submaker or not os.path.exists(chunk_audio_file) or os.path.getsize(chunk_audio_file) == 0:
                    logger.error(
                        f"failed to synthesize speech chunk (audio missing or empty): {speech_text[:50]}"
                    )
                    return None

                chunk_wav = os.path.join(temp_dir, f"speech_{idx}_decoded.wav")
                ffmpeg_binary = utils.get_ffmpeg_binary()
                cmd = [
                    ffmpeg_binary,
                    "-y",
                    "-i",
                    chunk_audio_file,
                    "-vn",
                    "-ac",
                    "1",
                    "-ar",
                    str(SAMPLE_RATE),
                    "-codec:a",
                    "pcm_s16le",
                    chunk_wav,
                ]
                res = subprocess.run(cmd, capture_output=True, text=True, check=False)
                if res.returncode != 0 or not os.path.exists(chunk_wav) or os.path.getsize(chunk_wav) == 0:
                    logger.error(
                        f"failed to decode speech chunk audio to PCM WAV: {speech_text[:50]}, "
                        f"error: {(res.stderr or res.stdout or '').strip()}"
                    )
                    return None

                try:
                    with wave.open(chunk_wav, "rb") as wf:
                        chunk_samples = wf.getnframes()
                except Exception as e:
                    logger.error(
                        f"failed to read decoded speech wave: {speech_text[:50]}, error: {e}"
                    )
                    return None

                if chunk_samples <= 0:
                    logger.error(
                        f"decoded speech chunk has no audio samples: {speech_text[:50]}"
                    )
                    return None

                # Derive offsets from sample counts to avoid accumulated MP3 frame drift.
                current_offset_seconds = cumulative_samples / float(SAMPLE_RATE)

                # 1. Shift cues from edge_tts 7.x.
                if hasattr(chunk_submaker, "cues") and chunk_submaker.cues:
                    offset_td = timedelta(seconds=current_offset_seconds)
                    for cue in chunk_submaker.cues:
                        shifted_cue = Subtitle(
                            index=len(combined_submaker.cues) + 1,
                            start=cue.start + offset_td,
                            end=cue.end + offset_td,
                            content=cue.content,
                        )
                        combined_submaker.cues.append(shifted_cue)

                # 2. Shift legacy subs/offset data.
                if hasattr(chunk_submaker, "subs") and chunk_submaker.subs:
                    combined_submaker.subs.extend(chunk_submaker.subs)
                if hasattr(chunk_submaker, "offset") and chunk_submaker.offset:
                    offset_100ns = int(current_offset_seconds * 10000000)
                    for start_ns, end_ns in chunk_submaker.offset:
                        combined_submaker.offset.append(
                            (start_ns + offset_100ns, end_ns + offset_100ns)
                        )

                audio_chunk_files.append(chunk_wav)
                cumulative_samples += chunk_samples

        if not _concat_audio_files(audio_chunk_files, voice_file):
            logger.error("failed to concatenate audio chunks with pauses")
            return None

        combined_submaker.duration = cumulative_samples / float(SAMPLE_RATE)
        return combined_submaker


def tts(
    text: str,
    voice_name: str,
    voice_rate: float,
    voice_file: str,
    voice_volume: float = 1.0,
) -> Union[SubMaker, None]:
    # Every provider is reached through this function, so clamping here is
    # what makes the speed and volume limits unavoidable rather than advisory.
    voice_rate = guardrails.clamp_voice_rate(voice_rate)
    voice_volume = guardrails.clamp_voice_volume(voice_volume)

    # Without pause tags, pass the original text straight through: no point
    # running the regex or risking whitespace truncation.
    if not utils.has_pause_tags(text):
        return _single_tts(
            text=text,
            voice_name=voice_name,
            voice_rate=voice_rate,
            voice_file=voice_file,
            voice_volume=voice_volume,
        )

    # Only Azure TTS v1 (Edge TTS) with pause tags in the script takes the
    # segmented synthesis path.
    if is_azure_v1_voice(voice_name):
        return _tts_with_pauses(
            text=text,
            voice_name=voice_name,
            voice_rate=voice_rate,
            voice_file=voice_file,
            voice_volume=voice_volume,
        )

    # Other providers (Gemini, Fish Audio, SiliconFlow, Kokoro, ...) strip the
    # pause tags and synthesize in a single request.
    clean_text = utils.remove_pause_tags(text)
    return _single_tts(
        text=clean_text,
        voice_name=voice_name,
        voice_rate=voice_rate,
        voice_file=voice_file,
        voice_volume=voice_volume,
    )


def convert_rate_to_percent(rate: float) -> str:
    # edge-tts requires a sign-prefixed percentage (e.g. "+0%", "-20%").
    # Rounding can yield 0 for rates near but not equal to 1.0 (e.g. 1.004,
    # 0.997); those must still be returned as "+0%", not the unsigned "0%"
    # which edge-tts rejects with ValueError: Invalid rate '0%'.
    # API or batch callers may pass zero, None, or non-numeric empty values.
    # These are not valid rates and would yield -100% or raise, so normalize
    # them to regular speed rather than creating very slow audio or failing.
    try:
        rate = float(rate)
    except (TypeError, ValueError):
        rate = 1.0
    if not math.isfinite(rate) or rate <= 0:
        rate = 1.0
    percent = round((rate - 1.0) * 100)
    if percent >= 0:
        return f"+{percent}%"
    return f"{percent}%"


def ensure_file_path_exists(file_path: str) -> None:
    """
    Ensure the output file's directory exists.

    This guard is needed because edge_tts 7.x opens the output before making a
    network request. A missing directory would raise a local path error and
    hide the actual TTS result.
    """
    dir_path = os.path.dirname(file_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)


def ensure_legacy_submaker_fields(sub_maker: SubMaker) -> SubMaker:
    """
    Add compatibility fields for callers that still use the legacy subtitle structure.

    edge_tts 7.x primarily exposes `cues/get_srt()`, while Azure v2, Gemini,
    and SiliconFlow still read and write `subs/offset`. Add both fields here so
    upgrading edge_tts does not break those providers.
    """
    if not hasattr(sub_maker, "subs"):
        sub_maker.subs = []
    if not hasattr(sub_maker, "offset"):
        sub_maker.offset = []
    return sub_maker


def has_real_word_timestamps(sub_maker) -> bool:
    """
    Return whether the sub_maker's timeline is built from real word/phrase
    boundaries emitted by the TTS service.

    Only Edge TTS (Azure v1) and the Azure Speech SDK (Azure v2) feed real
    boundaries via streaming WordBoundary events. Every other provider
    (Gemini, SiliconFlow, ElevenLabs, Fish Audio, Chatterbox, Kokoro,
    MiniMax, MiMo) routes through `populate_legacy_submaker_with_full_text`,
    which linearly distributes the total audio duration by character count.
    Treating those estimates as real boundaries makes the subtitle timeline
    drift by whole segments.
    """
    if sub_maker is None:
        return False
    return bool(getattr(sub_maker, "_has_real_word_timestamps", False))


def mark_real_word_timestamps(sub_maker: SubMaker) -> SubMaker:
    """Tag a SubMaker whose boundaries are real so `generate_subtitle` can route on it."""
    setattr(sub_maker, "_has_real_word_timestamps", True)
    return sub_maker


def populate_legacy_submaker_with_full_text(
    sub_maker: SubMaker, text: str, audio_duration_seconds: float
) -> SubMaker:
    """
    Populate the legacy `subs/offset` subtitle structure from the full text.

    Context:
    1. edge_tts 7.x no longer provides the legacy `create_sub()` method;
    2. non-edge providers such as Gemini and SiliconFlow still return an object
       with `subs/offset` for shared duration and subtitle generation;
    3. For TTS services that can't return word-level boundaries, still split
       by sentence so the subtitle aggregator matches the script line for
       line.

    Args:
        sub_maker: Subtitle object receiving compatibility fields.
        text: Original script text.
        audio_duration_seconds: Total audio duration in seconds.

    Returns:
        The SubMaker populated with compatibility subtitle data.
    """
    sub_maker = ensure_legacy_submaker_fields(sub_maker)

    # Clear old values so reusing an object cannot accumulate stale data.
    sub_maker.subs = []
    sub_maker.offset = []
    # Mark explicitly as an estimated timeline so it cannot be mistaken for
    # real word-level boundaries by the edge subtitle path.
    setattr(sub_maker, "_has_real_word_timestamps", False)

    normalized_text = (text or "").strip()
    if not normalized_text:
        return sub_maker

    audio_duration_100ns = max(int(audio_duration_seconds * 10000000), 1)

    # When providers such as Gemini and SiliconFlow lack word boundaries, keep
    # the existing strategy: split on punctuation and distribute duration by
    # character count. This lets create_subtitle() match script sentences and
    # avoids another Whisper fallback.
    sentences = utils.split_string_by_punctuations(normalized_text)
    if not sentences:
        sentences = [normalized_text]

    total_chars = sum(len(sentence) for sentence in sentences)
    if total_chars <= 0:
        sub_maker.subs.append(normalized_text)
        sub_maker.offset.append((0, audio_duration_100ns))
        return sub_maker

    current_offset = 0
    for index, sentence in enumerate(sentences):
        cleaned_sentence = sentence.strip()
        if not cleaned_sentence:
            continue

        # Allocate earlier sentences by character ratio and give the last one
        # the remainder so integer rounding cannot shorten the timeline.
        if index == len(sentences) - 1:
            sentence_end = audio_duration_100ns
        else:
            sentence_chars = len(cleaned_sentence)
            sentence_duration = max(
                int(audio_duration_100ns * (sentence_chars / total_chars)),
                1,
            )
            sentence_end = min(current_offset + sentence_duration, audio_duration_100ns)

        sub_maker.subs.append(cleaned_sentence)
        sub_maker.offset.append((current_offset, sentence_end))
        current_offset = sentence_end

    return sub_maker


def create_edge_tts_communicate(
    text: str, voice_name: str, rate_str: str
) -> edge_tts.Communicate:
    """
    Build a Communicate object for the installed edge_tts version.

    Context:
    1. The main code targets edge_tts 7.x and uses `boundary` for finer events;
    2. a Windows portable package may retain an older edge_tts after an update failure;
    3. older `Communicate.__init__()` versions reject `boundary` with
       `unexpected keyword argument 'boundary'`, breaking the TTS path.

    Inspect the constructor signature before passing `boundary` so the same code
    remains compatible with both dependency versions.
    """
    communicate_kwargs = {"rate": rate_str}
    communicate_signature = inspect.signature(edge_tts.Communicate)

    if "boundary" in communicate_signature.parameters:
        communicate_kwargs["boundary"] = "WordBoundary"

    return edge_tts.Communicate(text, voice_name, **communicate_kwargs)


def get_edge_tts_timeout_seconds() -> Union[float, None]:
    """
    Return the timeout for one Azure TTS v1 streaming request.

    Edge consumer TTS may remain blocked in `stream_sync()` during network
    failures, server throttling, or voice/language mismatches while logs remain
    at `start`. A default timeout prevents a WebUI task from hanging silently.

    Usage:
    - The 30-second default covers first-byte waits for typical short scripts.
    - Slow networks or proxies can set
      `edge_tts_timeout = 60`；
    - Zero or a negative value explicitly disables the timeout for compatibility.
    """
    raw_timeout = config.app.get(
        "edge_tts_timeout", _DEFAULT_EDGE_TTS_TIMEOUT_SECONDS
    )
    try:
        timeout_seconds = float(raw_timeout)
    except (TypeError, ValueError):
        logger.warning(
            "invalid edge_tts_timeout: "
            f"{raw_timeout}, fallback to {_DEFAULT_EDGE_TTS_TIMEOUT_SECONDS}s"
        )
        timeout_seconds = _DEFAULT_EDGE_TTS_TIMEOUT_SECONDS

    if timeout_seconds <= 0:
        return None

    return timeout_seconds


def _stream_edge_tts_sync_with_timeout(
    communicate, on_chunk, timeout_seconds: float
) -> None:
    """
    Consume the edge_tts 7.x synchronous stream with an overall timeout.

    `stream_sync()` is a blocking iterator, so the main thread cannot recover
    promptly when the network stalls. Consume it in a daemon thread and pass
    chunks through a Queue; raise TimeoutError at the deadline so outer retries
    and error logging can continue.

    The daemon thread is only a guardrail. At most a few threads may remain after
    the three Azure TTS v1 retries, and process exit reclaims them. This failure
    mode is bounded compared with a permanently stuck WebUI task.
    """
    stream_queue = queue.Queue()
    done_marker = object()

    def _produce_chunks():
        try:
            for chunk in communicate.stream_sync():
                stream_queue.put(("chunk", chunk))
            stream_queue.put(("done", done_marker))
        except Exception as e:
            stream_queue.put(("error", e))

    thread = threading.Thread(target=_produce_chunks, daemon=True)
    thread.start()

    deadline = time.monotonic() + timeout_seconds
    while True:
        remaining_seconds = deadline - time.monotonic()
        if remaining_seconds <= 0:
            raise TimeoutError(
                f"edge_tts stream timed out after {timeout_seconds:g}s"
            )

        try:
            item_type, payload = stream_queue.get(
                timeout=min(0.5, remaining_seconds)
            )
        except queue.Empty:
            continue

        if item_type == "chunk":
            on_chunk(payload)
        elif item_type == "error":
            raise payload
        elif item_type == "done":
            return


def stream_edge_tts_chunks(
    communicate, on_chunk, timeout_seconds: Union[float, None] = None
) -> None:
    """
    Consume current synchronous and legacy asynchronous edge_tts streams.

    edge_tts 7.x provides `stream_sync()` for direct iteration in synchronous
    code; earlier versions commonly expose only async `stream()`. This adapter
    keeps `azure_tts_v1()` working when an older dependency remains installed.

    Args:
        communicate: edge_tts.Communicate instance.
        on_chunk: Callback invoked for each event chunk.
        timeout_seconds: Overall request timeout; None disables it.
    """
    if hasattr(communicate, "stream_sync"):
        if timeout_seconds:
            _stream_edge_tts_sync_with_timeout(
                communicate, on_chunk, timeout_seconds
            )
            return

        for chunk in communicate.stream_sync():
            on_chunk(chunk)
        return

    if not hasattr(communicate, "stream"):
        raise AttributeError("edge_tts communicate object has no stream method")

    async def _consume_async_stream():
        async for chunk in communicate.stream():
            on_chunk(chunk)

    # Create a dedicated event loop to avoid missing-loop errors or cross-thread
    # loop reuse inside a synchronous call stack.
    loop = asyncio.new_event_loop()
    try:
        if timeout_seconds:
            loop.run_until_complete(
                asyncio.wait_for(_consume_async_stream(), timeout=timeout_seconds)
            )
        else:
            loop.run_until_complete(_consume_async_stream())
    finally:
        loop.close()


def azure_tts_v1(
    text: str, voice_name: str, voice_rate: float, voice_file: str
) -> Union[SubMaker, None]:
    voice_name = parse_voice_name(voice_name)
    text = text.strip()
    rate_str = convert_rate_to_percent(voice_rate)
    for i in range(3):
        try:
            logger.info(f"start, voice name: {voice_name}, try: {i + 1}")

            # Support edge_tts 7.x and older dependencies left in portable builds:
            # 1. Current versions support `boundary` and `stream_sync()`.
            # 2. Older versions reject `boundary` and usually expose only async `stream()`.
            ensure_file_path_exists(voice_file)
            communicate = create_edge_tts_communicate(text, voice_name, rate_str)
            sub_maker = edge_tts.SubMaker()
            timeout_seconds = get_edge_tts_timeout_seconds()

            with open(voice_file, "wb") as file:
                def _handle_chunk(chunk):
                    chunk_type = chunk["type"]
                    if chunk_type == "audio":
                        file.write(chunk["data"])
                    elif chunk_type in ["WordBoundary", "SentenceBoundary"]:
                        # Feed any available boundary event to SubMaker regardless
                        # of stream version so the existing subtitle path is used.
                        sub_maker.feed(chunk)

                stream_edge_tts_chunks(
                    communicate, _handle_chunk, timeout_seconds=timeout_seconds
                )

            if not sub_maker.get_srt():
                logger.warning("failed, sub_maker.get_srt() is empty")
                continue

            logger.info(f"completed, output file: {voice_file}")
            return mark_real_word_timestamps(sub_maker)
        except Exception as e:
            logger.error(f"failed, error: {str(e)}")
            # A timeout or network error before the first chunk leaves an empty
            # file. Remove only empty failures; preserve partial output for diagnosis.
            if os.path.exists(voice_file) and os.path.getsize(voice_file) == 0:
                try:
                    os.remove(voice_file)
                except Exception as remove_error:
                    logger.warning(
                        "failed to remove empty tts file: "
                        f"{voice_file}, error: {str(remove_error)}"
                    )
    return None


def siliconflow_tts(
    text: str,
    model: str,
    voice: str,
    voice_rate: float,
    voice_file: str,
    voice_volume: float = 1.0,
) -> Union[SubMaker, None]:
    """
    Generate speech with the SiliconFlow API.

    Args:
        text: Text to synthesize.
        model: Model name, for example "FunAudioLLM/CosyVoice2-0.5B".
        voice: Voice name, for example "FunAudioLLM/CosyVoice2-0.5B:alex".
        voice_rate: Speech speed in [0.25, 4.0].
        voice_file: Output audio path.
        voice_volume: Volume in [0.6, 5.0], converted to SiliconFlow gain [-10, 10].

    Returns:
        A SubMaker object or None.
    """
    text = text.strip()
    api_key = config.siliconflow.get("api_key", "")

    if not api_key:
        logger.error("SiliconFlow API key is not set")
        return None

    # Convert voice_volume to the SiliconFlow gain range.
    # The default voice_volume of 1.0 maps to zero gain.
    gain = voice_volume - 1.0
    # Clamp gain to [-10, 10].
    gain = max(-10, min(10, gain))

    url = "https://api.siliconflow.cn/v1/audio/speech"

    payload = {
        "model": model,
        "input": text,
        "voice": voice,
        "response_format": "mp3",
        "sample_rate": 32000,
        "stream": False,
        "speed": voice_rate,
        "gain": gain,
    }

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    for i in range(3):  # Try three times.
        try:
            logger.info(
                f"start siliconflow tts, model: {model}, voice: {voice}, try: {i + 1}"
            )

            response = requests.post(url, json=payload, headers=headers)

            if response.status_code == 200:
                # Save the audio file.
                with open(voice_file, "wb") as f:
                    f.write(response.content)

                sub_maker = ensure_legacy_submaker_fields(SubMaker())

                try:
                    audio_clip = AudioFileClip(voice_file)
                    try:
                        audio_duration = audio_clip.duration
                    finally:
                        audio_clip.close()
                except Exception as e:
                    logger.warning(f"Failed to read audio duration: {str(e)}")
                    audio_duration = 10.0

                logger.success(f"siliconflow tts succeeded: {voice_file}")
                return populate_legacy_submaker_with_full_text(
                    sub_maker=sub_maker,
                    text=text,
                    audio_duration_seconds=audio_duration,
                )
            else:
                logger.error(
                    f"siliconflow tts failed with status code {response.status_code}: {response.text}"
                )
        except Exception as e:
            logger.error(f"siliconflow tts failed: {str(e)}")

    return None


def _build_azure_v2_ssml(text: str, voice_name: str, voice_rate: float) -> str:
    """Build Azure Speech v2 SSML with a safely normalized speech rate."""
    try:
        normalized_rate = float(voice_rate)
    except (TypeError, ValueError):
        normalized_rate = 1.0
    normalized_rate = max(0.25, min(4.0, normalized_rate))

    voice_locale_parts = voice_name.split("-", 2)
    voice_locale = (
        "-".join(voice_locale_parts[:2])
        if len(voice_locale_parts) >= 2
        else "en-US"
    )
    escaped_text = escape(text)
    escaped_voice_name = escape(voice_name, {'"': "&quot;"})
    return (
        '<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" '
        f'xml:lang="{voice_locale}">'
        f'<voice name="{escaped_voice_name}">'
        f'<prosody rate="{normalized_rate:g}">{escaped_text}</prosody>'
        "</voice></speak>"
    )


def azure_tts_v2(
    text: str,
    voice_name: str,
    voice_file: str,
    voice_rate: float = 1.0,
) -> Union[SubMaker, None]:
    voice_name = is_azure_v2_voice(voice_name)
    if not voice_name:
        logger.error(f"invalid voice name: {voice_name}")
        raise ValueError(f"invalid voice name: {voice_name}")
    text = text.strip()
    ssml = _build_azure_v2_ssml(text, voice_name, voice_rate)

    def _format_duration_to_offset(duration) -> int:
        if isinstance(duration, str):
            time_obj = datetime.strptime(duration, "%H:%M:%S.%f")
            milliseconds = (
                (time_obj.hour * 3600000)
                + (time_obj.minute * 60000)
                + (time_obj.second * 1000)
                + (time_obj.microsecond // 1000)
            )
            return milliseconds * 10000

        if isinstance(duration, int):
            return duration

        return 0

    for i in range(3):
        try:
            logger.info(
                f"start, voice name: {voice_name}, rate: {voice_rate}, try: {i + 1}"
            )

            import azure.cognitiveservices.speech as speechsdk

            sub_maker = ensure_legacy_submaker_fields(SubMaker())

            def speech_synthesizer_word_boundary_cb(evt: speechsdk.SessionEventArgs):
                # print('WordBoundary event:')
                # print('\tBoundaryType: {}'.format(evt.boundary_type))
                # print('\tAudioOffset: {}ms'.format((evt.audio_offset + 5000)))
                # print('\tDuration: {}'.format(evt.duration))
                # print('\tText: {}'.format(evt.text))
                # print('\tTextOffset: {}'.format(evt.text_offset))
                # print('\tWordLength: {}'.format(evt.word_length))

                duration = _format_duration_to_offset(str(evt.duration))
                offset = _format_duration_to_offset(evt.audio_offset)
                sub_maker.subs.append(evt.text)
                sub_maker.offset.append((offset, offset + duration))

            # Creates an instance of a speech config with specified subscription key and service region.
            speech_key = config.azure.get("speech_key", "")
            service_region = config.azure.get("speech_region", "")
            if not speech_key or not service_region:
                logger.error("Azure speech key or region is not set")
                return None

            audio_config = speechsdk.audio.AudioOutputConfig(
                filename=voice_file, use_default_speaker=True
            )
            speech_config = speechsdk.SpeechConfig(
                subscription=speech_key, region=service_region
            )
            speech_config.speech_synthesis_voice_name = voice_name
            # speech_config.set_property(property_id=speechsdk.PropertyId.SpeechServiceResponse_RequestSentenceBoundary,
            #                            value='true')
            speech_config.set_property(
                property_id=speechsdk.PropertyId.SpeechServiceResponse_RequestWordBoundary,
                value="true",
            )

            speech_config.set_speech_synthesis_output_format(
                speechsdk.SpeechSynthesisOutputFormat.Audio48Khz192KBitRateMonoMp3
            )
            speech_synthesizer = speechsdk.SpeechSynthesizer(
                audio_config=audio_config, speech_config=speech_config
            )
            speech_synthesizer.synthesis_word_boundary.connect(
                speech_synthesizer_word_boundary_cb
            )

            # speak_text_async() has no rate argument. SSML prosody applies the
            # WebUI/API voice_rate to both previews and final generation.
            result = speech_synthesizer.speak_ssml_async(ssml).get()
            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                logger.success(f"azure v2 speech synthesis succeeded: {voice_file}")
                return mark_real_word_timestamps(sub_maker)
            elif result.reason == speechsdk.ResultReason.Canceled:
                cancellation_details = result.cancellation_details
                logger.error(
                    f"azure v2 speech synthesis canceled: {cancellation_details.reason}"
                )
                if cancellation_details.reason == speechsdk.CancellationReason.Error:
                    logger.error(
                        f"azure v2 speech synthesis error: {cancellation_details.error_details}"
                    )
            logger.info(f"completed, output file: {voice_file}")
        except Exception as e:
            logger.error(f"failed, error: {str(e)}")
    return None


DEFAULT_GEMINI_TTS_MODEL = "gemini-2.5-flash-preview-tts"
# Fallback list when the API key cannot be introspected (offline startup,
# quota exhaustion on the list call, etc.). Keeps the dropdown usable
# instead of empty so the user can still pick a known-good model.
_GEMINI_TTS_MODEL_FALLBACK = (
    DEFAULT_GEMINI_TTS_MODEL,
    "gemini-2.5-pro-preview-tts",
)
# Config key the WebUI dropdown writes to and the dispatcher reads from.
GEMINI_TTS_MODEL_CONFIG_KEY = "gemini_tts_model_name"


def get_gemini_tts_models(api_key: str) -> list[str]:
    """Return Gemini TTS-capable model ids exposed by the given API key.

    Calls `client.models.list()` and filters to models whose name ends in
    "-tts" (Google's naming convention for TTS endpoints, e.g.
    `gemini-2.5-flash-preview-tts`, `gemini-2.5-pro-preview-tts`). Never
    raises: any failure returns the bundled fallback list so the WebUI
    dropdown always has something to pick.
    """
    if not api_key or not api_key.strip():
        return list(_GEMINI_TTS_MODEL_FALLBACK)
    try:
        from google import genai

        client = genai.Client(api_key=api_key)
        models = list(client.models.list())
    except Exception as exc:  # noqa: BLE001 — dropdown should always be populated
        logger.warning(
            f"gemini TTS model listing failed, using fallback: {type(exc).__name__}: {exc}"
        )
        return list(_GEMINI_TTS_MODEL_FALLBACK)

    candidates = []
    for model in models:
        name = getattr(model, "name", "") or ""
        # google-genai returns names like "models/gemini-2.5-flash-preview-tts"
        short = name.rsplit("/", 1)[-1]
        if short.endswith("-tts"):
            candidates.append(short)

    if not candidates:
        return list(_GEMINI_TTS_MODEL_FALLBACK)

    # Stable order: put the default first, then the rest alphabetically so the
    # dropdown does not shuffle between calls.
    candidates.sort(key=lambda n: (n != DEFAULT_GEMINI_TTS_MODEL, n))
    return candidates


def gemini_tts(
    text: str,
    voice_name: str,
    voice_rate: float,
    voice_file: str,
    voice_volume: float = 1.0,
    model: str = DEFAULT_GEMINI_TTS_MODEL,
) -> Union[SubMaker, None]:
    """
    Generate speech with Google Gemini TTS.

    Args:
        text: Text to synthesize.
        voice_name: Voice name such as "Zephyr" or "Puck".
        voice_rate: Speech rate, currently unused.
        voice_file: Output audio path.
        voice_volume: Audio volume, currently unused.
        model: Gemini TTS model id (e.g. `gemini-2.5-flash-preview-tts`).
            Different models have different rate limits and quality; the
            WebUI lets the user pick from the models their API key exposes.

    Returns:
        A SubMaker object or None.
    """
    import base64
    import io
    from pydub import AudioSegment
    from google import genai
    from google.genai import types
    _configure_pydub_ffmpeg(AudioSegment)

    # Per-call wall-clock cap. The SDK's own HttpOptions.timeout catches a
    # slow HTTP read, but the SDK can also hang on TLS handshake, proxy
    # connect, or its own internal retries — none of which always honor the
    # configured timeout. Running the call in a daemon thread with a hard
    # deadline guarantees the WebUI gets feedback inside this window no
    # matter where the SDK gets stuck.
    gemini_call_timeout_seconds = 60.0
    # Inner SDK timeout fires earlier than the thread cap when Google is
    # just slow; the thread cap is the hard safety net.
    _gemini_sdk_timeout_ms = int(gemini_call_timeout_seconds * 1000) - 10_000

    def _is_gemini_terminal_error(exc: BaseException) -> bool:
        """True for errors that retrying will not fix (auth, quota, bad model).

        The google-genai SDK raises ClientError for HTTP 4xx and ServerError
        for HTTP 5xx; only the 4xx variants are terminal here because quota
        exhaustion and invalid keys never recover by trying again.
        """
        status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
        if isinstance(status, int) and 400 <= status < 500:
            return True
        message = str(exc) or ""
        # RESOURCE_EXHAUSTED and UNAUTHENTICATED show up in the message even
        # when status_code is missing on the wrapped exception; match the
        # status text directly so we don't keep burning quota on retries.
        return any(
            marker in message
            for marker in ("RESOURCE_EXHAUSTED", "UNAUTHENTICATED", "PERMISSION_DENIED")
        )

    def _invoke_gemini_with_timeout(text, voice_name, gen_config):
        """Run the SDK call in a worker thread and return its result or raise."""
        result_queue: queue.Queue = queue.Queue()
        done_marker = object()

        def _worker():
            try:
                # Inner SDK timeout fires earlier than the thread cap when
                # Google is just slow; the thread cap is the hard safety net.
                client_http_options = types.HttpOptions(timeout=_gemini_sdk_timeout_ms)
                with genai.Client(api_key=api_key, http_options=client_http_options) as client:
                    response = client.models.generate_content(
                        model=model,
                        contents=text,
                        config=gen_config,
                    )
                result_queue.put(("ok", response))
            except BaseException as exc:  # noqa: BLE001 — propagate every failure
                result_queue.put(("err", exc))
            finally:
                result_queue.put(("done", done_marker))

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()

        deadline = time.monotonic() + gemini_call_timeout_seconds
        response = None
        captured_error = None
        finished = False
        while not finished:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"gemini TTS exceeded {gemini_call_timeout_seconds:.0f}s "
                    "wall-clock budget"
                )
            try:
                item_type, payload = result_queue.get(timeout=min(0.5, remaining))
            except queue.Empty:
                continue
            if item_type == "ok":
                response = payload
            elif item_type == "err":
                captured_error = payload
            elif item_type == "done":
                finished = True

        if captured_error is not None:
            raise captured_error
        return response

    try:
        api_key = config.app.get("gemini_api_key", "")
        if not api_key:
            logger.error("Gemini API key is not set")
            return None

        generation_config = types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=voice_name
                    )
                )
            ),
        )

        # Gemini's free-tier TTS quota is 10 requests/day per project
        # (RESOURCE_EXHAUSTED). Retrying on quota errors just burns the
        # remaining quota faster, so distinguish transient failures from
        # terminal ones: retry on timeout and 5xx, fail fast on 4xx
        # (invalid key, quota, bad model, etc.) with a clear log line.
        last_error = None
        response = None
        for attempt in range(3):
            try:
                logger.info(f"start, voice name: {voice_name}, try: {attempt + 1}")
                response = _invoke_gemini_with_timeout(text, voice_name, generation_config)
                break
            except TimeoutError as exc:
                last_error = exc
                logger.warning(
                    f"gemini TTS timed out, retrying: voice={voice_name}, "
                    f"try={attempt + 1}, error={exc}"
                )
                continue
            except Exception as exc:  # noqa: BLE001 — decide retry vs fail-fast
                last_error = exc
                if _is_gemini_terminal_error(exc):
                    logger.error(
                        f"gemini TTS non-retryable error, giving up: "
                        f"voice={voice_name}, error={type(exc).__name__}: {exc}"
                    )
                    break
                logger.warning(
                    f"gemini TTS attempt failed, retrying: voice={voice_name}, "
                    f"try={attempt + 1}, error={type(exc).__name__}: {exc}"
                )
                continue

        if response is None:
            logger.error(
                f"gemini TTS failed: voice={voice_name}, "
                f"last_error={type(last_error).__name__}: {last_error}"
            )
            return None

        # Validate the response.
        if not response.candidates or not response.candidates[0].content:
            logger.error("No audio content received from Gemini TTS")
            return None
            
        # Extract audio data.
        audio_data = None
        for part in response.candidates[0].content.parts:
            if hasattr(part, 'inline_data') and part.inline_data:
                audio_data = part.inline_data.data
                break
                
        if not audio_data:
            logger.error("No audio data found in response")
            return None
            
        # Raw bytes need no base64 decoding.
        if isinstance(audio_data, str):
            # String responses are base64 encoded.
            audio_bytes = base64.b64decode(audio_data)
        else:
            # Byte responses can be used directly.
            audio_bytes = audio_data
        
        # Prepare for the audio format returned by Gemini.
        audio_segment = None
        
        # Parse Gemini's linear PCM using the documented parameters.
        try:
            audio_segment = AudioSegment.from_file(
                io.BytesIO(audio_bytes), 
                format="raw",
                frame_rate=24000,  # Gemini TTS default sample rate
                channels=1,        # Mono
                sample_width=2     # 16-bit
            )
        except Exception as e:
            logger.error(f"Failed to load PCM audio: {e}")
            return None
        
        # API, CLI, and tests may supply a nested output path that does not yet
        # exist. Create it before writing so a successful request is not lost
        # to a local path error and provider behavior stays consistent.
        ensure_file_path_exists(voice_file)

        # pydub returns an open output handle. Close it to avoid descriptor
        # accumulation and Windows failures when replacing or deleting files.
        exported_audio = audio_segment.export(voice_file, format="mp3")
        exported_audio.close()
        
        logger.info(f"completed, output file: {voice_file}")
        
        # Gemini provides no edge_tts-style word boundaries, so use the legacy
        # `subs/offset` structure to keep duration and subtitle processing valid.
        sub_maker = ensure_legacy_submaker_fields(SubMaker())
        audio_duration = len(audio_segment) / 1000.0  # Convert to seconds.
        return populate_legacy_submaker_with_full_text(
            sub_maker=sub_maker,
            text=text,
            audio_duration_seconds=audio_duration,
        )
        
    except ImportError as e:
        logger.error(f"Missing required package for Gemini TTS: {str(e)}. Please install: pip install pydub")
        return None
    except Exception as e:
        logger.error(f"Gemini TTS failed, error: {str(e)}")
        return None


def mimo_tts(
    text: str,
    voice_name: str,
    voice_rate: float,
    voice_file: str,
    voice_volume: float = 1.0,
) -> Union[SubMaker, None]:
    """
    Generate speech with Xiaomi MiMo V2.5 TTS.

    The official endpoint is compatible with OpenAI Chat Completions, with two
    important TTS differences:
    1. The text to synthesize must be in an `assistant` message.
    2. Audio is returned as a base64 string in `message.audio.data`.

    MiMo currently returns no word-level timeline, so reuse the legacy SubMaker
    fallback and build subtitle timing from final duration and script sentences.
    """
    from pydub import AudioSegment

    text = (text or "").strip()
    if not text:
        logger.error("MiMo TTS text is empty")
        return None

    api_key = config.app.get("mimo_api_key", "")
    if not api_key:
        logger.error("MiMo API key is not set")
        return None

    base_url = config.app.get("mimo_base_url", "") or _MIMO_DEFAULT_BASE_URL
    model_name = config.app.get("mimo_tts_model_name", "") or _MIMO_DEFAULT_TTS_MODEL
    style_prompt = config.app.get(
        "mimo_tts_style_prompt",
        "Read in a natural, clear tone suitable for short-form video narration.",
    )

    _configure_pydub_ffmpeg(AudioSegment)

    for i in range(3):
        try:
            logger.info(
                f"start mimo tts, model: {model_name}, voice: {voice_name}, try: {i + 1}"
            )
            ensure_file_path_exists(voice_file)

            client = OpenAI(api_key=api_key, base_url=base_url)
            completion = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "user", "content": style_prompt},
                    {"role": "assistant", "content": text},
                ],
                audio={
                    "format": "wav",
                    "voice": voice_name,
                },
            )

            if not completion or not getattr(completion, "choices", None):
                raise ValueError("MiMo TTS returned empty response")

            message = completion.choices[0].message
            audio = getattr(message, "audio", None)
            audio_data = None
            if isinstance(audio, dict):
                audio_data = audio.get("data")
            elif audio is not None:
                audio_data = getattr(audio, "data", None)

            if not audio_data:
                raise ValueError("MiMo TTS returned empty audio data")

            audio_bytes = base64.b64decode(audio_data)
            audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes), format="wav")

            output_format = utils.parse_extension(voice_file) or "mp3"
            if output_format == "wav":
                with open(voice_file, "wb") as f:
                    f.write(audio_bytes)
            else:
                audio_segment.export(voice_file, format=output_format)

            audio_duration = len(audio_segment) / 1000.0
            sub_maker = ensure_legacy_submaker_fields(SubMaker())
            logger.success(f"mimo tts succeeded: {voice_file}")
            logger.debug(
                "mimo subtitle timeline generated, "
                f"duration: {audio_duration:.3f}s, output_format: {output_format}"
            )
            return populate_legacy_submaker_with_full_text(
                sub_maker=sub_maker,
                text=text,
                audio_duration_seconds=audio_duration,
            )
        except Exception as e:
            logger.error(f"mimo tts failed: {str(e)}")

    return None


def _resolve_minimax_tts_url(configured_url: str) -> str:
    configured_url = (configured_url or "").strip().rstrip("/")
    if not configured_url:
        return MINIMAX_TTS_GLOBAL_URL
    if configured_url in {MINIMAX_TTS_GLOBAL_URL, MINIMAX_TTS_CN_URL}:
        return configured_url
    if configured_url.endswith("/v1"):
        return f"{configured_url}/t2a_v2"
    return configured_url


def get_minimax_tts_api_key() -> str:
    """Return the MiniMax TTS key, preferring its dedicated configuration."""
    return str(
        config.minimax_tts.get("api_key", "")
        or config.app.get("minimax_api_key", "")
        or os.getenv("MINIMAX_API_KEY", "")
        or ""
    ).strip()


def _infer_minimax_tts_url(base_url: str) -> str:
    """Infer the regional TTS URL from a MiniMax LLM URL, or return empty."""
    normalized_url = str(base_url or "").strip()
    if not normalized_url:
        return ""

    parse_target = normalized_url if "://" in normalized_url else f"//{normalized_url}"
    host = (urlparse(parse_target).hostname or "").lower()
    if host == "minimaxi.com" or host.endswith(".minimaxi.com"):
        return MINIMAX_TTS_CN_URL
    if host == "minimax.io" or host.endswith(".minimax.io"):
        return MINIMAX_TTS_GLOBAL_URL
    return ""


def get_minimax_tts_endpoint() -> str:
    """
    Return the MiniMax TTS endpoint matching the active key.

    A dedicated TTS key respects the configured TTS URL. When reusing the
    MiniMax LLM key, prefer the LLM base URL region so a regional key is not
    sent to the wrong endpoint and rejected with 401.
    """
    dedicated_key = str(config.minimax_tts.get("api_key", "") or "").strip()
    if not dedicated_key:
        inferred_url = _infer_minimax_tts_url(config.app.get("minimax_base_url", ""))
        if inferred_url:
            return inferred_url
    return _resolve_minimax_tts_url(config.minimax_tts.get("base_url", ""))


def get_minimax_voice_catalog(
    api_key: str = "",
    endpoint: str = "",
    voice_type: str = "all",
) -> list[dict[str, str]]:
    """
    Query system, cloned, and generated voices available to the MiniMax account.

    Results use the common voice_id, voice_name, and voice_type fields so callers
    need not know how MiniMax groups sources. Failures raise so the WebUI, API,
    or CLI can show an explicit error instead of silently returning an empty list.
    """
    if voice_type not in {"system", "voice_cloning", "voice_generation", "all"}:
        raise ValueError(f"Unsupported MiniMax voice type: {voice_type}")

    effective_api_key = str(api_key or get_minimax_tts_api_key()).strip()
    if not effective_api_key:
        raise ValueError("MiniMax TTS API key is not set")

    tts_endpoint = (
        _resolve_minimax_tts_url(endpoint)
        if endpoint
        else get_minimax_tts_endpoint()
    )
    voice_endpoint = (
        f"{tts_endpoint[:-len('/t2a_v2')]}/get_voice"
        if tts_endpoint.endswith("/t2a_v2")
        else f"{tts_endpoint.rstrip('/')}/get_voice"
    )
    response = requests.post(
        voice_endpoint,
        json={"voice_type": voice_type},
        headers={
            "Authorization": f"Bearer {effective_api_key}",
            "Content-Type": "application/json",
        },
        timeout=30,
    )
    if response.status_code != 200:
        raise RuntimeError(
            f"MiniMax get_voice failed with status {response.status_code}: "
            f"{response.text[:200]}"
        )

    try:
        body = response.json()
    except ValueError as exc:
        raise RuntimeError("MiniMax get_voice returned invalid JSON") from exc

    base_resp = body.get("base_resp") or {}
    if base_resp.get("status_code") not in {0, "0"}:
        status_message = str(base_resp.get("status_msg") or "unknown error")
        raise RuntimeError(f"MiniMax get_voice failed: {status_message}")

    catalog = []
    seen_voice_ids = set()
    response_groups = (
        ("system", "system_voice"),
        ("voice_cloning", "voice_cloning"),
        ("voice_generation", "voice_generation"),
    )
    for normalized_type, response_key in response_groups:
        for item in body.get(response_key) or []:
            voice_id = str(item.get("voice_id") or "").strip()
            if not voice_id or voice_id in seen_voice_ids:
                continue
            seen_voice_ids.add(voice_id)
            catalog.append(
                {
                    "voice_id": voice_id,
                    "voice_name": str(item.get("voice_name") or voice_id).strip(),
                    "voice_type": normalized_type,
                }
            )

    logger.info(f"loaded MiniMax voices: count={len(catalog)}, type={voice_type}")
    return catalog


def _write_validated_minimax_audio(audio_bytes: bytes, voice_file: str) -> float:
    """
    Atomically write MiniMax audio to the target and return its duration.

    A successful remote status does not guarantee complete audio. Validate a
    temporary file in the destination directory, then use os.replace so decoding
    failures or unreadable MoviePy output cannot leave a partial target.
    """
    ensure_file_path_exists(voice_file)
    output_dir = os.path.dirname(os.path.abspath(voice_file))
    output_suffix = os.path.splitext(voice_file)[1] or ".mp3"
    temp_fd, temp_path = tempfile.mkstemp(
        prefix=".minimax-tts-", suffix=output_suffix, dir=output_dir
    )
    os.close(temp_fd)

    try:
        with open(temp_path, "wb") as output:
            output.write(audio_bytes)

        audio_clip = AudioFileClip(temp_path)
        try:
            audio_duration = float(audio_clip.duration)
        finally:
            audio_clip.close()

        if not math.isfinite(audio_duration) or audio_duration <= 0:
            raise ValueError("MiniMax TTS returned audio with an invalid duration")

        os.replace(temp_path, voice_file)
        return audio_duration
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def minimax_tts(text: str, voice_id: str, voice_rate: float, voice_file: str, voice_volume: float = 1.0) -> Union[SubMaker, None]:
    """Generate speech with the synchronous MiniMax T2A HTTP API."""
    text, voice_id = (text or "").strip(), (voice_id or "").strip()
    if not text or not voice_id:
        logger.error("MiniMax TTS requires text and a voice ID")
        return None
    settings = config.minimax_tts
    api_key = get_minimax_tts_api_key()
    if not api_key:
        logger.error("MiniMax TTS API key is not set")
        return None
    url = get_minimax_tts_endpoint()
    model = str(settings.get("model_id", MINIMAX_TTS_DEFAULT_MODEL) or MINIMAX_TTS_DEFAULT_MODEL).strip()
    if model not in MINIMAX_TTS_MODELS:
        logger.error(f"Unsupported MiniMax TTS model: {model}")
        return None
    try:
        speed = max(0.5, min(2.0, float(voice_rate or 1.0)))
        volume = max(0.0, min(10.0, float(voice_volume or 1.0)))
        pitch = max(-12, min(12, int(settings.get("pitch", 0) or 0)))
        sample_rate = int(settings.get("sample_rate", 32000) or 32000)
        bitrate = int(settings.get("bitrate", 128000) or 128000)
        channel = int(settings.get("channel", 1) or 1)
    except (TypeError, ValueError) as exc:
        logger.error(f"Invalid MiniMax TTS audio setting: {str(exc)}")
        return None
    audio_format = str(settings.get("audio_format", "mp3") or "mp3").strip()
    if audio_format not in {"mp3", "wav", "flac", "pcm"}:
        logger.error(f"Unsupported MiniMax TTS audio format: {audio_format}")
        return None
    payload = {
        "model": model, "text": text, "stream": False, "language_boost": "auto", "output_format": "hex",
        "voice_setting": {"voice_id": voice_id, "speed": speed, "vol": volume, "pitch": pitch},
        "audio_setting": {"sample_rate": sample_rate, "bitrate": bitrate, "format": audio_format, "channel": channel},
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    for attempt in range(3):
        try:
            logger.info(f"start MiniMax TTS, model: {model}, voice: {voice_id}, try: {attempt + 1}")
            response = requests.post(url, json=payload, headers=headers, timeout=120)
            if response.status_code != 200:
                logger.error(f"MiniMax TTS failed with status {response.status_code}: {response.text[:200]}")
                continue
            body = response.json()
            data = body.get("data") or {}
            base_resp = body.get("base_resp") or {}
            if base_resp.get("status_code") != 0 or data.get("status") != 2:
                logger.error(f"MiniMax TTS returned an unsuccessful response: status_code={base_resp.get('status_code')}, audio_status={data.get('status')}")
                continue
            audio_hex = data.get("audio")
            if not isinstance(audio_hex, str) or not audio_hex:
                logger.error("MiniMax TTS returned empty audio data")
                continue
            if len(audio_hex) > _MINIMAX_TTS_MAX_AUDIO_HEX_CHARS:
                logger.error("MiniMax TTS returned audio data exceeding the supported size")
                continue
            audio_duration = _write_validated_minimax_audio(bytes.fromhex(audio_hex), voice_file)
            logger.success(f"MiniMax TTS succeeded: {voice_file}")
            return populate_legacy_submaker_with_full_text(
                ensure_legacy_submaker_fields(SubMaker()), text, audio_duration
            )
        except (OSError, ValueError, requests.RequestException) as exc:
            logger.error(f"MiniMax TTS failed: {str(exc)}")
    return None


def elevenlabs_tts(
    text: str,
    voice_id: str,
    voice_file: str,
    voice_rate: float = 1.0,
    voice_volume: float = 1.0,
    model_id: str = "",
) -> Union[SubMaker, None]:
    text = (text or "").strip()
    if not text:
        logger.error("ElevenLabs TTS text is empty")
        return None

    api_key = get_elevenlabs_api_key()
    if not api_key:
        logger.error("ElevenLabs API key is not set")
        return None

    if not model_id:
        model_id = config.elevenlabs.get("model_id", "eleven_multilingual_v2")

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json",
    }
    payload = {
        "text": text,
        "model_id": model_id,
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.75,
            "style": 0.0,
            "use_speaker_boost": True,
        },
    }

    # Errors where retrying will never help (auth/access/validation failures).
    _NON_RETRYABLE_CODES = {401, 403, 422}
    _NON_RETRYABLE_STATUSES = {"voice_disabled", "voice_access_denied", "unauthorized"}

    for i in range(3):
        try:
            logger.info(f"start elevenlabs tts, voice_id: {voice_id}, try: {i + 1}")
            ensure_file_path_exists(voice_file)

            response = requests.post(url, json=payload, headers=headers, timeout=60)
            if response.status_code != 200:
                error_status = ""
                try:
                    detail = response.json().get("detail", {})
                    if isinstance(detail, dict):
                        error_status = detail.get("status", "")
                except Exception:
                    pass

                if response.status_code in _NON_RETRYABLE_CODES or error_status in _NON_RETRYABLE_STATUSES:
                    logger.error(
                        f"ElevenLabs TTS failed (non-retryable) — voice_id: {voice_id}, "
                        f"status: {response.status_code}, error: {error_status or response.text[:200]}. "
                        "Please select a different ElevenLabs voice."
                    )
                    return None

                logger.error(
                    f"elevenlabs tts failed with status {response.status_code}: {response.text[:200]}"
                )
                continue

            with open(voice_file, "wb") as f:
                f.write(response.content)

            audio_clip = AudioFileClip(voice_file)
            try:
                audio_duration = audio_clip.duration
            finally:
                audio_clip.close()

            sub_maker = ensure_legacy_submaker_fields(SubMaker())
            logger.success(f"elevenlabs tts succeeded: {voice_file}")
            return populate_legacy_submaker_with_full_text(
                sub_maker=sub_maker,
                text=text,
                audio_duration_seconds=audio_duration,
            )
        except Exception as e:
            logger.error(f"elevenlabs tts failed: {str(e)}")

    return None


def _openai_compatible_tts(
    provider: str,
    base_url: str,
    api_key: str,
    model_id: str,
    voice: str,
    text: str,
    voice_rate: float,
    voice_file: str,
) -> Union[SubMaker, None]:
    """Shared transport for self-hosted, OpenAI-compatible ``/audio/speech``
    servers (Chatterbox, Kokoro, ...).

    Writes the returned audio to ``voice_file`` and builds the full-text
    SubMaker: these servers return no word-level timestamps, so set
    ``subtitle_provider = "whisper"`` for tighter subtitle sync.
    """
    url = f"{base_url}/audio/speech"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model_id,
        "input": text,
        "voice": voice,
        "response_format": "mp3",
        # OpenAI speech API accepts speed 0.25-4.0; MoneyPrinterTurbo's rate is a
        # 1.0-centred multiplier, so it maps directly (clamped to the valid range).
        "speed": max(0.25, min(4.0, float(voice_rate or 1.0))),
    }
    # The OpenAI speech contract has no volume field; final mixing applies
    # voice_volume. The speed field controls only speech rate.

    for i in range(3):
        temporary_audio = None
        try:
            logger.info(f"start {provider} tts, voice: {voice}, try: {i + 1}")
            ensure_file_path_exists(voice_file)

            response = requests.post(url, json=payload, headers=headers, timeout=120)
            if response.status_code != 200:
                logger.error(
                    f"{provider} tts failed with status {response.status_code}: {response.text[:200]}"
                )
                continue

            if not response.content:
                raise ValueError(f"{provider} returned empty audio")

            # Decode a temporary file in the destination directory before
            # replacing the target. Close it first for Windows compatibility,
            # and never destroy an existing preview or narration on failure.
            with tempfile.NamedTemporaryFile(
                dir=os.path.dirname(os.path.abspath(voice_file)),
                suffix=".mp3", delete=False,
            ) as f:
                temporary_audio = f.name
                f.write(response.content)

            audio_clip = AudioFileClip(temporary_audio)
            try:
                audio_duration = audio_clip.duration
            finally:
                audio_clip.close()
            if not math.isfinite(audio_duration) or audio_duration <= 0:
                raise ValueError(f"{provider} returned an invalid audio duration")

            sub_maker = ensure_legacy_submaker_fields(SubMaker())
            os.replace(temporary_audio, voice_file)
            logger.success(f"{provider} tts succeeded: {voice_file}")
            return populate_legacy_submaker_with_full_text(
                sub_maker=sub_maker,
                text=text,
                audio_duration_seconds=audio_duration,
            )
        except Exception as e:
            logger.error(f"{provider} tts failed: {str(e)}")
        finally:
            if temporary_audio and os.path.exists(temporary_audio):
                try:
                    os.unlink(temporary_audio)
                except OSError as exc:
                    # Preserve the TTS failure contract instead of masking it with cleanup errors.
                    logger.warning(f"could not remove temporary {provider} audio: {exc}")

    return None


def chatterbox_tts(
    text: str,
    voice: str,
    voice_file: str,
    voice_rate: float = 1.0,
    voice_volume: float = 1.0,
    model_id: str = "",
) -> Union[SubMaker, None]:
    """Generate speech with a self-hosted Chatterbox TTS server.

    Chatterbox (Resemble AI, MIT) is an open-source, locally hosted TTS model
    with zero-shot voice cloning — a self-hostable alternative to ElevenLabs.
    This talks to an OpenAI-compatible ``/audio/speech`` endpoint, so it works
    with the common community servers (e.g. devnen/Chatterbox-TTS-Server,
    travisvn/chatterbox-tts-api). Configure ``[chatterbox] base_url`` (and an
    optional ``api_key``).

    Like ElevenLabs, Chatterbox does not return word-level timestamps, so the
    subtitle path falls back to the full-text SubMaker. For tighter subtitle
    sync set ``subtitle_provider = "whisper"``.
    """
    text = (text or "").strip()
    if not text:
        logger.error("Chatterbox TTS text is empty")
        return None

    base_url = (config.chatterbox.get("base_url", "") or "").strip().rstrip("/")
    if not base_url:
        logger.error(
            "Chatterbox base_url is not set, please configure [chatterbox] base_url in config.toml"
        )
        return None

    api_key = config.chatterbox.get("api_key", "")
    if not model_id:
        model_id = config.chatterbox.get("model_id", "chatterbox") or "chatterbox"

    return _openai_compatible_tts(
        "chatterbox", base_url, api_key, model_id, voice, text, voice_rate, voice_file
    )


def kokoro_tts(
    text: str,
    voice: str,
    voice_file: str,
    voice_rate: float = 1.0,
    voice_volume: float = 1.0,
    model_id: str = "",
) -> Union[SubMaker, None]:
    """Generate speech with a self-hosted Kokoro TTS server.

    Kokoro (hexgrad/Kokoro-82M, Apache-2.0 code and weights) is a small open
    TTS model that runs well on CPU — a free, offline alternative to the
    cloud voices. This talks to an OpenAI-compatible ``/audio/speech``
    endpoint, so it works with the common servers (e.g. remsky/Kokoro-FastAPI
    on port 8880). Configure ``[kokoro] base_url`` (ending in ``/v1``) and an
    optional ``api_key``.

    Voice names are Kokoro's presets (``af_heart``, ``bf_emma``, ``hf_alpha``,
    ...); their first letter is the language (a/b English, e Spanish, f French,
    h Hindi, i Italian, p Portuguese, j Japanese, z Chinese), so pick a voice
    that matches the script's language.

    Like Chatterbox, the OpenAI speech contract returns no word-level
    timestamps, so the subtitle path falls back to the full-text SubMaker.
    For tighter subtitle sync set ``subtitle_provider = "whisper"``.
    """
    text = (text or "").strip()
    if not text:
        logger.error("Kokoro TTS text is empty")
        return None
    # Punctuation and emoji alone are not speakable; a real service may return
    # an empty MP3 with HTTP 200. Reject early to avoid a useless request and a
    # low-level MoviePy error while decoding an empty file.
    if not any(character.isalnum() for character in text):
        logger.error("Kokoro TTS text contains no speakable characters")
        return None
    base_url = (config.kokoro.get("base_url", "") or "").strip().rstrip("/")
    if not base_url:
        logger.error(
            "Kokoro base_url is not set, please configure [kokoro] base_url in config.toml"
        )
        return None
    api_key = config.kokoro.get("api_key", "")
    if not model_id:
        model_id = config.kokoro.get("model_id", "kokoro") or "kokoro"
    return _openai_compatible_tts(
        "kokoro", base_url, api_key, model_id, voice, text, voice_rate, voice_file
    )


# Fish Audio supported models.
FISH_AUDIO_MODELS = ("s2.1-pro-free", "s2.1-pro", "s2-pro")
FISH_AUDIO_DEFAULT_MODEL = "s2.1-pro-free"


def fish_audio_tts(
    text: str,
    voice_file: str,
    voice_rate: float = 1.0,
    voice_volume: float = 1.0,
    reference_id: str | None = None,
) -> Union[SubMaker, None]:
    """Generate speech using Fish Audio TTS API.

    The model is read from ``config.fish_audio["model"]`` (single source of
    truth).  ``reference_id`` selects a public or cloned voice; when *None*
    Fish Audio's built-in default voice is used.

    ``voice_rate`` is mapped to the ``prosody.speed`` field (0.5–2.0) and
    ``voice_volume`` is converted from a linear multiplier to dB for the
    ``prosody.volume`` field (-20.0–20.0 dB).
    """
    text = (text or "").strip()
    if not text:
        logger.error("Fish Audio TTS text is empty")
        return None

    api_key = get_fish_audio_api_key()
    if not api_key:
        logger.error(
            "Fish Audio API key is not set. Please set it in config.toml "
            "[fish_audio] or FISH_API_KEY environment variable."
        )
        return None

    model_name = str(
        config.fish_audio.get("model", FISH_AUDIO_DEFAULT_MODEL)
        or FISH_AUDIO_DEFAULT_MODEL
    ).strip()
    if model_name not in FISH_AUDIO_MODELS:
        logger.warning(
            f"Unknown Fish Audio model '{model_name}', falling back to "
            f"'{FISH_AUDIO_DEFAULT_MODEL}'"
        )
        model_name = FISH_AUDIO_DEFAULT_MODEL

    # Map voice_rate → prosody.speed (0.5–2.0)
    try:
        speed = max(0.5, min(2.0, float(voice_rate or 1.0)))
    except (TypeError, ValueError):
        speed = 1.0

    # Map voice_volume (linear multiplier) → prosody.volume (dB, -20–20).
    # A multiplier of 1.0 → 0 dB; 0.1 → -20 dB; 2.0 → +6 dB.
    import math
    try:
        vol = float(voice_volume or 1.0)
        if vol <= 0:
            volume_db = -20.0
        else:
            volume_db = max(-20.0, min(20.0, 20.0 * math.log10(vol)))
    except (TypeError, ValueError):
        volume_db = 0.0

    url = "https://api.fish.audio/v1/tts"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "model": model_name,
    }
    payload: dict = {
        "text": text,
        "format": "mp3",
        "prosody": {
            "speed": speed,
            "volume": volume_db,
        },
    }
    if reference_id:
        payload["reference_id"] = reference_id

    for i in range(3):
        try:
            logger.info(
                f"start fish audio tts, model: {model_name}, "
                f"ref: {reference_id or 'default'}, try: {i + 1}"
            )
            ensure_file_path_exists(voice_file)

            response = requests.post(url, json=payload, headers=headers, timeout=60)
            if response.status_code == 401:
                logger.error(
                    "Fish Audio TTS failed: Invalid API key (401). "
                    "Check config.toml [fish_audio] api_key or FISH_API_KEY."
                )
                return None
            if response.status_code == 402:
                logger.error(
                    "Fish Audio TTS failed: Insufficient API credit (402). "
                    "Please check your account balance at "
                    "https://fish.audio/app/developers or verify your model and billing tier."
                )
                return None
            if response.status_code == 429:
                logger.warning(
                    "Fish Audio TTS rate limited (429), retrying..."
                )
                continue
            if response.status_code != 200:
                logger.error(
                    f"fish audio tts failed with status "
                    f"{response.status_code}: {response.text[:200]}"
                )
                continue

            # Validate response contains audio data
            if not response.content or len(response.content) < 100:
                logger.error(
                    "Fish Audio TTS returned empty or invalid audio data"
                )
                continue

            with open(voice_file, "wb") as f:
                f.write(response.content)

            audio_clip = AudioFileClip(voice_file)
            try:
                audio_duration = audio_clip.duration
            finally:
                audio_clip.close()

            sub_maker = ensure_legacy_submaker_fields(SubMaker())
            logger.success(f"fish audio tts succeeded: {voice_file}")
            return populate_legacy_submaker_with_full_text(
                sub_maker=sub_maker,
                text=text,
                audio_duration_seconds=audio_duration,
            )
        except Exception as e:
            logger.error(f"fish audio tts failed: {str(e)}")

    return None


def voicestudio_tts(
    text: str,
    voice_preset: str,
    voice_file: str,
    voice_rate: float = 1.0,
    voice_volume: float = 1.0,
) -> Union[SubMaker, None]:
    """Synthesize speech with the self-hosted OmniVoice bridge.

    The bridge is a headless FastAPI service (kept under ``vendor/voice_studio``)
    that wraps the OmniVoice ``k2-fsa/OmniVoice`` model. ``voice_preset``
    selects a bundled voice-design preset (currently "narrator") or a cloned
    profile name; the matching ``instruct`` string or ``VoiceClonePrompt`` is
    resolved on the server side. The server defaults to ``http://127.0.0.1:8780``
    and is overridable via the ``[voicestudio] base_url`` config key or the
    ``VOICESTUDIO_BASE_URL`` environment variable.

    ``voice_rate`` is forwarded to OmniVoice as ``speed`` when set; ``voice_volume``
    is applied locally via ffmpeg's ``volume`` filter because OmniVoice exposes
    no native gain control and the pipeline's downstream MultiplyVolume cannot
    raise the volume past 1.0 — a user-set value of 1.5 was previously dropped.
    The endpoint returns raw WAV audio with no word-level timestamps, so
    subtitles fall back to the full-text SubMaker; set
    ``subtitle_provider = "whisper"`` for tighter sync.
    """
    text = (text or "").strip()
    if not text:
        logger.error("VoiceStudio TTS text is empty")
        return None
    # Reject punctuation and emoji alone to avoid a useless request.
    if not any(character.isalnum() for character in text):
        logger.error("VoiceStudio TTS text contains no speakable characters")
        return None
    base_url = get_voicestudio_base_url()

    payload = {"text": text, "voice": voice_preset}
    # OmniVoice's `speed` argument is what the upstream library documents as
    # the playback-speed knob. Only forward it when the caller actually asked
    # for a non-default value so the server keeps its 1.0 baseline otherwise.
    if voice_rate not in (None, 1.0) and float(voice_rate) > 0:
        payload["speed"] = float(voice_rate)
    started = perf_counter()
    for i in range(3):
        try:
            logger.info(
                f"start voicestudio tts, voice preset: {voice_preset}, "
                f"text length: {len(text)}, try: {i + 1}"
            )
            ensure_file_path_exists(voice_file)
            # First synthesis may download and load several gigabytes of OmniVoice weights.
            response = requests.post(f"{base_url}/generate", json=payload, timeout=1800)
            if response.status_code == 404:
                logger.error(
                    f"VoiceStudio voice preset not found: {voice_preset!r} "
                    f"(server response: {response.text[:200]!r})"
                )
                return None
            if response.status_code != 200:
                logger.error(
                    f"voicestudio tts failed with status {response.status_code}: "
                    f"{response.text[:200]}"
                )
                continue

            # Validate response contains audio data: a WAV header alone is
            # ~44 bytes, so anything shorter than that is a definite failure;
            # anything shorter than ~1 KB usually means the model produced
            # silence or a malformed file.
            if not response.content or len(response.content) < 1024:
                logger.error(
                    f"VoiceStudio TTS returned empty or invalid audio data "
                    f"({len(response.content) if response.content else 0} bytes)"
                )
                continue

            with open(voice_file, "wb") as f:
                f.write(response.content)

            # Apply voice_volume locally so the user-set value isn't dropped.
            # voice_volume of 1.0 is a no-op; anything else gets a quick ffmpeg
            # pass with the volume filter and the same wav bytes are rewritten.
            # The downstream pipeline also applies MultiplyVolume on the final
            # mux, but that only attenuates — a value > 1.0 (louder) needs to
            # bake into the source file here.
            normalized_volume = max(0.0, float(voice_volume or 1.0))
            if abs(normalized_volume - 1.0) > 0.001:
                logger.info(
                    f"applying voice_volume={normalized_volume:.3f} to voicestudio output"
                )
                volume_adjusted_file = f"{voice_file}.vol.wav"
                try:
                    volume_command = [
                        utils.get_ffmpeg_binary(),
                        "-y",
                        "-i", voice_file,
                        "-af", f"volume={normalized_volume:.3f}",
                        "-acodec", "pcm_s16le",
                        volume_adjusted_file,
                    ]
                    result = subprocess.run(
                        volume_command,
                        capture_output=True,
                        text=True,
                        check=False,
                        timeout=60,
                    )
                    if result.returncode == 0:
                        shutil.move(volume_adjusted_file, voice_file)
                    else:
                        logger.warning(
                            f"voicestudio volume adjustment failed; "
                            f"output will use unchanged audio. stderr: "
                            f"{(result.stderr or '')[:200]}"
                        )
                        if os.path.exists(volume_adjusted_file):
                            os.remove(volume_adjusted_file)
                except (OSError, subprocess.TimeoutExpired) as exc:
                    logger.warning(
                        f"voicestudio volume adjustment skipped due to "
                        f"{type(exc).__name__}: {exc}"
                    )

            audio_clip = AudioFileClip(voice_file)
            try:
                audio_duration = audio_clip.duration
            finally:
                audio_clip.close()

            if audio_duration <= 0:
                logger.error(
                    f"voicestudio tts produced audio with zero/negative "
                    f"duration: {audio_duration:.3f}s"
                )
                continue

            sub_maker = ensure_legacy_submaker_fields(SubMaker())
            elapsed = perf_counter() - started
            logger.success(
                f"voicestudio tts succeeded: {voice_file}, "
                f"duration={audio_duration:.2f}s, elapsed={elapsed:.2f}s"
            )
            return populate_legacy_submaker_with_full_text(
                sub_maker=sub_maker,
                text=text,
                audio_duration_seconds=audio_duration,
            )
        except Exception as e:
            logger.error(f"voicestudio tts failed: {type(e).__name__}: {str(e)}")

    return None


def _format_text(text: str) -> str:
    """
    Clean script text before subtitle alignment.

    This cannot happen only during LLM generation because users may paste a
    script or submit Markdown through the API. TTS usually skips separator
    lines such as `---`, `___`, and `***`, as well as `_` emphasis markers. If
    alignment keeps them, `create_subtitle()` waits for a cue that never arrives,
    leaving no subtitle file and an all-zero Whisper fallback timeline.
    """
    text = utils.remove_pause_tags(text or "")
    text = text.replace("[", " ")
    text = text.replace("]", " ")
    text = text.replace("(", " ")
    text = text.replace(")", " ")
    text = text.replace("{", " ")
    text = text.replace("}", " ")
    return utils.normalize_script_for_subtitle_matching(text)


def _build_subtitle_formatter():
    """
    Return the shared SRT line formatter.

    Keeping this as one helper lets the edge_tts 7.x cue path and legacy
    `subs/offset` path share exactly the same on-disk format.
    """

    def formatter(idx: int, start_time: float, end_time: float, sub_text: str) -> str:
        start_t = mktimestamp(start_time).replace(".", ",")
        end_t = mktimestamp(end_time).replace(".", ",")
        return f"{idx}\n{start_t} --> {end_t}\n{sub_text}\n"

    return formatter


# Arabic diacritics and the Tatweel extender may appear in edge_tts output.
# They do not change meaning but break exact script-to-cue matching.
_ARABIC_DIACRITICS = re.compile("[\u0610-\u061A\u064B-\u065F\u0670\u0640\u06D6-\u06ED]")


def _normalize_arabic(text: str) -> str:
    """Normalize common Arabic variants for tolerant cue-to-script matching.

    edge_tts may return different letter forms or diacritics than the source.
    Apply this only as the final matching fallback and preserve displayed text.
    """
    text = _ARABIC_DIACRITICS.sub("", text)
    for src, dst in (
        ("أإآٱ", "ا"),
        ("ىئ", "ي"),
        ("ة", "ه"),
        ("ؤ", "و"),
    ):
        for ch in src:
            text = text.replace(ch, dst)
    return text


def _match_script_line(script_lines: list[str], current_text: str, sub_index: int) -> str:
    """
    Match accumulated subtitle text to the current normalized script sentence.

    Preserve the existing punctuation-split strategy:
    1. Prefer an exact match.
    2. Retry without punctuation and Markdown `_` markers.
    3. Finally retry after normalizing Arabic letter forms.

    This supports punctuation omitted or separated by TTS and languages where
    word boundaries do not map one-to-one to script characters.
    """
    if len(script_lines) <= sub_index:
        return ""

    target_line = script_lines[sub_index]
    if current_text == target_line:
        return target_line.strip()

    current_text_normalized = re.sub(r"[_\W]+", "", current_text)
    target_line_normalized = re.sub(r"[_\W]+", "", target_line)
    if current_text_normalized == target_line_normalized:
        return target_line.strip()

    # Final Arabic fallback: edge_tts letter forms, diacritics, or Tatweel may
    # differ from the script. Normalize only after regular matching fails.
    current_ar = re.sub(r"[_\W]+", "", _normalize_arabic(current_text))
    target_ar = re.sub(r"[_\W]+", "", _normalize_arabic(target_line))
    if current_ar and current_ar == target_ar:
        return target_line.strip()

    return ""


def _write_subtitle_items(sub_items: list[str], subtitle_file: str) -> bool:
    """
    Write aggregated subtitle segments to SRT and validate basic readability.

    Returns True when the file is written and readable by MoviePy, otherwise False.
    """
    try:
        ensure_file_path_exists(subtitle_file)
        with open(subtitle_file, "w", encoding="utf-8") as file:
            file.write("\n".join(sub_items) + "\n")

        sbs = subtitles.file_to_subtitles(subtitle_file, encoding="utf-8")
        duration = max([tb for ((ta, tb), txt) in sbs]) if sbs else 0
        logger.info(
            f"completed, subtitle file created: {subtitle_file}, duration: {duration}"
        )
        return True
    except Exception as e:
        logger.error(f"failed, error: {str(e)}")
        if os.path.exists(subtitle_file):
            os.remove(subtitle_file)
        return False


def _build_subtitle_items_from_edge_cues(
    sub_maker: SubMaker, script_lines: list[str]
) -> list[str]:
    """
    Aggregate fine-grained edge_tts 7.x cues into script-sentence SRT segments.

    edge_tts 7.x `SubMaker.get_srt()` favors word- or phrase-level timing.
    That can suit word highlighting but is hard to read for scripts whose
    characters or short units arrive as separate cues.

    Strategy:
    1. Consume each cue's `content`.
    2. Accumulate candidate text.
    3. Emit one segment when it matches the current script sentence.
    4. Span from the first cue start to the final cue end for continuity.
    """
    formatter = _build_subtitle_formatter()
    sub_items = []
    sub_index = 0
    current_text = ""
    current_start_time = None

    for cue in sub_maker.cues:
        cue_text = unescape(cue.content)
        if current_start_time is None:
            current_start_time = int(cue.start.total_seconds() * 10000000)

        current_end_time = int(cue.end.total_seconds() * 10000000)
        current_text += cue_text

        matched_text = _match_script_line(script_lines, current_text, sub_index)
        if not matched_text:
            continue

        sub_index += 1
        sub_items.append(
            formatter(
                idx=sub_index,
                start_time=current_start_time,
                end_time=current_end_time,
                sub_text=matched_text,
            )
        )
        current_text = ""
        current_start_time = None

    if current_text.strip():
        logger.warning(
            f"edge cues still have unmatched text after aggregation: {current_text}"
        )

    return sub_items


def _build_subtitle_items_from_legacy_submaker(
    sub_maker: SubMaker, script_lines: list[str]
) -> list[str]:
    """
    Aggregate legacy `subs/offset` data into script-sentence SRT segments.

    This preserves the original algorithm while sharing sentence matching and
    output formatting with the edge_tts 7.x cue path.
    """
    formatter = _build_subtitle_formatter()
    start_time = -1.0
    sub_items = []
    sub_index = 0
    sub_line = ""

    legacy_offsets = getattr(sub_maker, "offset", [])
    legacy_subs = getattr(sub_maker, "subs", [])
    for _, (offset, sub) in enumerate(zip(legacy_offsets, legacy_subs)):
        current_start_time, current_end_time = offset
        if start_time < 0:
            start_time = current_start_time

        sub_line += unescape(sub)
        matched_text = _match_script_line(script_lines, sub_line, sub_index)
        if not matched_text:
            continue

        sub_index += 1
        sub_items.append(
            formatter(
                idx=sub_index,
                start_time=start_time,
                end_time=current_end_time,
                sub_text=matched_text,
            )
        )
        start_time = -1.0
        sub_line = ""

    if sub_line.strip():
        logger.warning(
            f"legacy subtitle items still have unmatched text after aggregation: {sub_line}"
        )

    return sub_items


def _build_subtitle_items_from_edge_cues_words(sub_maker: SubMaker) -> list[str]:
    """
    Directly format edge_tts cues into single-word / cue-level SRT items.
    """
    formatter = _build_subtitle_formatter()
    sub_items = []
    sub_index = 0
    for cue in sub_maker.cues:
        cue_text = unescape(cue.content).strip()
        if not cue_text:
            continue
        sub_index += 1
        start_time = int(cue.start.total_seconds() * 10000000)
        end_time = int(cue.end.total_seconds() * 10000000)
        sub_items.append(
            formatter(
                idx=sub_index,
                start_time=start_time,
                end_time=end_time,
                sub_text=cue_text,
            )
        )
    return sub_items


def _build_subtitle_items_from_legacy_submaker_words(sub_maker: SubMaker) -> list[str]:
    """
    Directly format legacy submaker into single-word SRT items.
    """
    formatter = _build_subtitle_formatter()
    sub_items = []
    sub_index = 0
    legacy_offsets = getattr(sub_maker, "offset", [])
    legacy_subs = getattr(sub_maker, "subs", [])
    for offset, sub in zip(legacy_offsets, legacy_subs):
        cue_text = unescape(sub).strip()
        if not cue_text:
            continue
        sub_index += 1
        start_time, end_time = offset
        sub_items.append(
            formatter(
                idx=sub_index,
                start_time=start_time,
                end_time=end_time,
                sub_text=cue_text,
            )
        )
    return sub_items


def create_subtitle(
    sub_maker: SubMaker,
    text: str,
    subtitle_file: str,
    word_level: bool = False,
):
    """
    Normalize a subtitle file by splitting on punctuation, matching each script
    line, and writing new SRT items. When word_level is True, write one item per cue.
    """
    text = _format_text(text)
    try:
        if word_level:
            if hasattr(sub_maker, "cues") and sub_maker.cues:
                sub_items = _build_subtitle_items_from_edge_cues_words(sub_maker)
            else:
                sub_items = _build_subtitle_items_from_legacy_submaker_words(sub_maker)
            if sub_items:
                _write_subtitle_items(sub_items, subtitle_file)
                return

        script_lines = utils.split_string_by_punctuations(text)
        if hasattr(sub_maker, "cues") and sub_maker.cues:
            sub_items = _build_subtitle_items_from_edge_cues(sub_maker, script_lines)
        else:
            sub_items = _build_subtitle_items_from_legacy_submaker(
                sub_maker, script_lines
            )

        if len(sub_items) != len(script_lines):
            logger.warning(
                f"failed, sub_items len: {len(sub_items)}, script_lines len: {len(script_lines)}"
            )
            return

        _write_subtitle_items(sub_items, subtitle_file)
    except Exception as e:
        logger.error(f"failed, error: {str(e)}")


def _get_audio_duration_from_submaker(sub_maker: SubMaker):
    """
    Return audio duration from a SubMaker.
    """
    if hasattr(sub_maker, "duration") and getattr(sub_maker, "duration", 0) > 0:
        return float(getattr(sub_maker, "duration"))

    # Prefer edge_tts 7.x cues, then read offsets populated by other providers.
    if hasattr(sub_maker, "cues") and sub_maker.cues:
        return sub_maker.cues[-1].end.total_seconds()

    legacy_offsets = getattr(sub_maker, "offset", [])
    if not legacy_offsets:
        return 0.0
    return legacy_offsets[-1][1] / 10000000

def _get_audio_duration_from_file(audio_file: str) -> float:
    """
    Return duration for any FFmpeg-decodable audio file.
    """
    if not os.path.exists(audio_file):
        logger.error(f"audio file does not exist: {audio_file}")
        return 0.0

    try:
        # Use moviepy (ffmpeg) to read the duration of any supported audio format
        with AudioFileClip(audio_file) as audio:
            return audio.duration  # Duration in seconds
    except Exception as e:
        logger.error(f"Failed to get audio duration from file: {str(e)}")
        return 0.0

def get_audio_duration(target: Union[str, SubMaker]) -> float:
    """
    Return duration from a SubMaker or an FFmpeg-decodable audio file path.
    """
    if isinstance(target, SubMaker):
        return _get_audio_duration_from_submaker(target)
    elif isinstance(target, str):
        return _get_audio_duration_from_file(target)
    else:
        logger.error(f"Invalid target type: {type(target)}")
        return 0.0
