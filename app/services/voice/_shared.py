"""
Voice routing predicates, voice-name parsing, voice catalogs, audio
utilities, SubMaker helpers, and the constants every provider needs.

Stage 1 of the voice.py split. Tests reach these helpers via
``voice.X`` after the package's ``__init__`` re-exports them, so
``patch.object(vs, "is_edge_tts_voice", ...)`` keeps intercepting the
dispatcher.
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
import time
import unicodedata
import wave
from urllib.parse import quote

import requests
from edge_tts import SubMaker
from loguru import logger

from app.config import config
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
    subtitle path still needs this formatter for provider-built timelines, so
    keep an equivalent implementation.
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


OMNIVOICE_DEFAULT_BASE_URL = "http://127.0.0.1:8780"


def get_omnivoice_base_url() -> str:
    """Return the base URL of the self-hosted OmniVoice server.

    An explicit ``OMNIVOICE_BASE_URL`` environment variable wins (used by
    the Docker deployment to reach a host-side OmniVoice service via
    ``host.docker.internal``), then the configured value, then the default.
    """
    env = os.environ.get("OMNIVOICE_BASE_URL", "")
    if env.strip():
        return env.strip().rstrip("/")
    configured = config.omnivoice.get("base_url", "") if hasattr(config, "omnivoice") else ""
    return str(configured or OMNIVOICE_DEFAULT_BASE_URL).strip().rstrip("/")


def _is_running_in_docker() -> bool:
    """True when this process is inside a container.

    The vendored OmniVoice service cannot run here on macOS (no Metal in the
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


def ensure_omnivoice_server_running(timeout: float = 2.0) -> bool:
    """
    Spawn the bundled OmniVoice server when the configured port is dead.

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
    base_url = get_omnivoice_base_url()
    try:
        response = requests.get(f"{base_url}/voices", timeout=timeout)
        if response.status_code == 200:
            return True
    except Exception:
        pass

    if _is_running_in_docker():
        # Compose owns OmniVoice on Linux and the host Metal service on macOS.
        logger.warning(
            "omnivoice not reachable at {} and we are inside a container, "
            "so the bundled server cannot be launched here. Start the stack "
            "with the platform's Docker Compose files.",
            base_url,
        )
        return False

    server_script = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "vendor",
        "omnivoice",
        "server.py",
    )
    if not os.path.isfile(server_script):
        logger.warning(
            f"omnivoice server script missing at {server_script}; "
            "start it manually before generating previews"
        )
        return False

    logger.info(f"omnivoice not reachable at {base_url}; launching bundled server")
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
        logger.warning(f"failed to spawn omnivoice server: {exc}")
        return False

    # give the server a moment to bind; do not block startup forever.
    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        try:
            response = requests.get(f"{base_url}/voices", timeout=1.0)
            if response.status_code == 200:
                logger.success(f"omnivoice server is up at {base_url}")
                return True
        except Exception:
            time.sleep(0.5)
    logger.warning(
        f"omnivoice server did not respond within 15s at {base_url}; "
        "check the server logs and try again"
    )
    return False


def get_omnivoice_voices() -> list[str]:
    """Read bundled voice-design presets from the local OmniVoice server.

    Each preset is returned as ``omnivoice:<preset_name>`` so it can be
    selected in the WebUI and dispatched by :func:`_single_tts`. The
    matching ``instruct`` string is resolved server-side; callers only
    see stable preset names.
    """
    base_url = get_omnivoice_base_url()
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
            return [f"omnivoice:{name}" for name in names]
        logger.warning(
            f"omnivoice voices request failed with status {response.status_code}"
        )
    except Exception as e:
        # Do not log URLs or exception bodies; query strings may contain credentials.
        logger.warning(f"omnivoice voice list unavailable ({type(e).__name__})")
    return []


def create_omnivoice_profile(
    profile_name: str,
    audio_bytes: bytes,
    original_filename: str,
    *,
    ref_text: str = "",
    instruct_override: str = "",
) -> tuple[bool, str]:
    """Clone a voice on the local OmniVoice bridge from an uploaded sample.

    The sample is POSTed as a multipart upload to ``{base_url}/profiles``.
    The server stores the reference audio, auto-transcribes it (or uses the
    optional transcript) and persists a reusable ``VoiceClonePrompt`` under
    its ``voice_profiles/`` directory. ``ref_text`` is the optional transcript;
    ``instruct_override`` is the per-profile emotion/delivery descriptor that
    every later generation will inherit. On success the profile shows up in
    the ``/voices`` catalog as ``omnivoice:<name>`` for the TTS drop-down.
    Returns ``(ok, message_or_name)`` mirroring the WebUI contract.
    """
    profile_name = (profile_name or "").strip()
    if not profile_name:
        return False, "profile name must be non-empty"
    if not audio_bytes:
        return False, "audio sample cannot be empty"
    base_url = get_omnivoice_base_url()
    data = {"name": profile_name}
    if ref_text.strip():
        data["ref_text"] = ref_text.strip()
    if instruct_override.strip():
        data["instruct_override"] = instruct_override.strip()[:240]
    try:
        response = requests.post(
            f"{base_url}/profiles",
            data=data,
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
        logger.warning(f"omnivoice profile creation failed: {exc}")
        return False, f"omnivoice server unreachable: {exc}"

    if response.status_code == 201 or response.status_code == 200:
        try:
            detail = response.json()
        except ValueError:
            detail = {}
        name = detail.get("name") or profile_name
        logger.success(f"omnivoice profile created: {name}")
        return True, name
    logger.warning(
        f"omnivoice profile creation failed with status "
        f"{response.status_code}: {response.text[:200]}"
    )
    return False, response.text[:200]


def get_omnivoice_profiles() -> list[str]:
    """List cloned profile names from the local OmniVoice bridge.

    Unlike :func:`get_omnivoice_voices` this returns only the cloned
    profiles (not the bundled presets), so the WebUI can offer them for
    deletion without having to know which names are presets.
    """
    base_url = get_omnivoice_base_url()
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
        logger.warning(f"omnivoice profile list unavailable: {exc}")
    return []


def get_omnivoice_profile_metadata(profile_name: str) -> dict:
    """Return the editable metadata for a cloned profile (empty dict on miss).

    The metadata carries ``instruct_override`` (the per-profile emotion
    descriptor), ``sample_duration_seconds``, and ``ref_text``. Used by the
    TTS client to merge the profile-level instruct onto the per-call style
    without exposing the rest of the metadata to other callers.
    """
    profile_name = (profile_name or "").strip()
    if not profile_name:
        return {}
    base_url = get_omnivoice_base_url()
    try:
        response = requests.get(f"{base_url}/profiles", timeout=5)
        if response.status_code == 200:
            data = response.json()
            for entry in data.get("profiles", []) or []:
                if isinstance(entry, dict) and entry.get("name") == profile_name:
                    return entry
    except Exception as exc:
        logger.warning(f"omnivoice profile metadata unavailable: {exc}")
    return {}


def delete_omnivoice_profile(profile_name: str) -> tuple[bool, str]:
    """Delete a cloned voice profile from the local OmniVoice bridge.

    Sends ``DELETE {base_url}/profiles/<name>``, which removes the saved
    ``VoiceClonePrompt`` (``.pt``), its metadata (``.json``), and the stored
    reference audio sample. Returns ``(ok, message_or_name)`` matching the
    contract used by :func:`create_omnivoice_profile`.
    """
    profile_name = (profile_name or "").strip()
    if not profile_name:
        return False, "profile name must be non-empty"
    base_url = get_omnivoice_base_url()
    try:
        response = requests.delete(
            f"{base_url}/profiles/{quote(profile_name, safe='')}", timeout=60
        )
    except Exception as exc:
        logger.warning(f"omnivoice profile deletion failed: {exc}")
        return False, f"omnivoice server unreachable: {exc}"

    if response.status_code == 200:
        try:
            detail = response.json()
        except ValueError:
            detail = {}
        name = detail.get("name") or profile_name
        logger.success(f"omnivoice profile removed: {name}")
        return True, name
    logger.warning(
        f"omnivoice profile removal failed with status "
        f"{response.status_code}: {response.text[:200]}"
    )
    return False, response.text[:200]


_EDGE_VOICES_DATA_FILE = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "edge_voices.json"
)
_edge_voices_cache = None


def _load_edge_voices() -> list[dict]:
    global _edge_voices_cache
    if _edge_voices_cache is None:
        with open(_EDGE_VOICES_DATA_FILE, "r", encoding="utf-8") as f:
            _edge_voices_cache = json.load(f)
    return _edge_voices_cache


def get_all_edge_voices(filter_locals=None) -> list[str]:
    voices = []
    for item in _load_edge_voices():
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


def is_omnivoice_voice(voice_name: str) -> bool:
    return (voice_name or "").startswith("omnivoice:")


def is_coqui_voice(voice_name: str | None) -> bool:
    """Return whether ``voice_name`` is routed through the Coqui XTTS provider.

    Accepted prefixes are ``coqui:`` (preset / builtin speaker) and any
    identifier with no prefix that the caller explicitly labelled as Coqui
    by passing the prefix. An empty / None value is treated as "not Coqui"
    so the dispatcher falls back to its default.
    """
    return (voice_name or "").startswith("coqui:")


def parse_coqui_voice_name(voice_name: str) -> tuple[str, str]:
    """Split a ``coqui:`` voice name into ``(kind, value)``.

    Supported shapes:

    - ``coqui:clone:<path>`` — zero-shot clone from a wav the user uploaded;
      ``<path>`` is absolute or ``~``-prefixed and is expanded before use.
    - ``coqui:speaker:<lang>`` — XTTS bundled speaker for the given language.
    - ``coqui:<speaker>`` — short form treated as a builtin speaker name.
    """
    raw = (voice_name or "").strip()
    body = raw[len("coqui:"):].strip() if raw.startswith("coqui:") else raw
    if not body:
        return "speaker", ""
    if body.startswith("clone:"):
        return "clone", os.path.expanduser(body[len("clone:"):].strip())
    if body.startswith("speaker:"):
        return "speaker", body[len("speaker:"):].strip()
    if os.path.isabs(body) or os.sep in body or body.startswith("~"):
        return "clone", os.path.expanduser(body)
    return "speaker", body


def coqui_clone_reference_path(voice_name: str) -> str:
    """Return the resolved clone reference path for ``coqui:clone:/path``.

    Centralised here so callers do not have to repeat the ``os.path.expanduser``
    step. Empty string when the voice name does not request a custom clone.
    """
    kind, value = parse_coqui_voice_name(voice_name)
    return value if kind == "clone" else ""


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


def is_edge_tts_voice(voice_name: str | None) -> bool:
    """
    Always returns False.

    Edge TTS was removed from this project; the function is preserved only
    so existing test fixtures and monkey-patches keep resolving. The voice
    dispatcher must route the request through omnivoice or coqui instead;
    any voice that arrives here without one of those prefixes will fall
    through to the bundled default voice.
    """
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
    if is_omnivoice_voice(name):
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

    edge_tts 7.x primarily exposes `cues/get_srt()`, while Gemini and
    SiliconFlow still read and write `subs/offset`. Add both fields here so
    upgrading edge_tts does not break those providers.
    """
    if not hasattr(sub_maker, "subs"):
        sub_maker.subs = []
    if not hasattr(sub_maker, "offset"):
        sub_maker.offset = []
    return sub_maker


# ------------------- Gemini dispatcher constants -------------------

# Default model when the WebUI dropdown is unconfigured.
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

# ------------------- Fish Audio model catalog -------------------

FISH_AUDIO_MODELS = ("s2.1-pro-free", "s2.1-pro", "s2-pro")
FISH_AUDIO_DEFAULT_MODEL = "s2.1-pro-free"

# Public alias for ``_MINIMAX_TTS_MAX_AUDIO_HEX_CHARS`` so providers.py can
# import it without depending on the underscore-prefixed name.
MINIMAX_TTS_MAX_AUDIO_HEX_CHARS = _MINIMAX_TTS_MAX_AUDIO_HEX_CHARS
def has_real_word_timestamps(sub_maker) -> bool:
    """
    Return whether the sub_maker's timeline is built from real word/phrase
    boundaries emitted by the TTS service.

    Only Edge TTS feeds real boundaries via streaming WordBoundary events.
    Every other provider
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
