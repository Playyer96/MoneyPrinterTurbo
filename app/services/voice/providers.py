"""TTS provider synthesis functions.

Stage 2 of the voice.py split. One provider per top-level function
(``edge_tts_synthesize``, ``gemini_tts``, ``minimax_tts``, ``omnivoice_tts``,
etc.) with their per-provider helpers grouped inline. The dispatcher
in ``voice.__init__`` looks these up via bare-name lookups against
the package's namespace, so test monkey-patches like
``patch.object(vs, "edge_tts_synthesize", ...)`` keep intercepting the
dispatch without test changes.
"""

from __future__ import annotations

import base64
import io
import math
import os
import queue
import tempfile
import threading
import time
from typing import Union
from urllib.parse import urlparse

import requests
from edge_tts import SubMaker
from loguru import logger
from moviepy.audio.io.AudioFileClip import AudioFileClip
from openai import OpenAI

from app.config import config
from app.utils import utils

# Shared TTS constants and utilities.
from app.services.voice._shared import (
    DEFAULT_GEMINI_TTS_MODEL,
    FISH_AUDIO_DEFAULT_MODEL,
    FISH_AUDIO_MODELS,
    MINIMAX_TTS_DEFAULT_MODEL,
    MINIMAX_TTS_GLOBAL_URL,
    MINIMAX_TTS_CN_URL,
    MINIMAX_TTS_MODELS,
    _GEMINI_TTS_MODEL_FALLBACK,
    _DEFAULT_EDGE_TTS_TIMEOUT_SECONDS,
    _MINIMAX_TTS_MAX_AUDIO_HEX_CHARS,
    _MIMO_DEFAULT_BASE_URL,
    _MIMO_DEFAULT_TTS_MODEL,
    _configure_pydub_ffmpeg,
    convert_rate_to_percent,
    ensure_file_path_exists,
    ensure_legacy_submaker_fields,
    get_elevenlabs_api_key,
    get_fish_audio_api_key,
    mark_real_word_timestamps,
    parse_voice_name,
    populate_legacy_submaker_with_full_text,
)





def coqui_xtts_tts(
    text: str,
    voice_name: str,
    voice_rate: float,
    voice_file: str,
    voice_volume: float = 1.0,
    voice_style: str = "",
) -> Union[SubMaker, None]:
    """
    Synthesize speech with the local Coqui XTTS v2 model.

    The XTTS checkpoint is downloaded automatically on the first call to
    ``TTS(model_name=...)`` and cached under ``$XDG_CACHE_HOME/tts`` (which
    defaults to ``$HOME/.local/share/tts``); no manual download step is
    needed and no path needs to be baked into the repo.

    Voice presets::

    - ``coqui:speaker:es`` — XTTS's bundled Spanish male/female sample.
    - ``coqi:clone:/abs/path/to/your_sample.wav`` — zero-shot clone of
      the audio file at ``/abs/path/to/your_sample.wav`` (``~`` expansion
      is supported). The wav must be at least 6 s long and ideally mono
      16- or 24-kHz PCM for the best results.

    ``voice_rate`` and ``voice_style`` are accepted for interface parity
    with the other providers but XTTS ignores them at the moment.
    """
    text = (text or "").strip()
    if not text:
        return None

    model_name = str(
        config.app.get("coqui_model_name", _COQUI_DEFAULT_MODEL)
        or _COQUI_DEFAULT_MODEL
    ).strip() or _COQUI_DEFAULT_MODEL
    speaker_wav = _resolve_coqui_speaker_wav(voice_name)
    language = _coqui_language_for_voice(voice_name)

    ensure_file_path_exists(voice_file)
    try:
        tts = _get_coqui_tts(model_name, language, speaker_wav)
        if speaker_wav:
            tts.tts_to_file(
                text=text,
                file_path=voice_file,
                speaker_wav=speaker_wav,
                language=language,
                speed=float(voice_rate) if voice_rate and voice_rate > 0 else 1.0,
            )
        else:
            tts.tts_to_file(
                text=text,
                file_path=voice_file,
                language=language,
                speaker=None,
                speed=float(voice_rate) if voice_rate and voice_rate > 0 else 1.0,
            )
    except Exception as exc:
        logger.error(f"coqui xtts failed: {type(exc).__name__}: {exc}")
        return None

    sub_maker = ensure_legacy_submaker_fields(SubMaker())
    populate_legacy_submaker_with_full_text(
        sub_maker=sub_maker,
        text=text,
        audio_duration_seconds=estimate_no_voice_duration(text) or 1.0,
    )
    return sub_maker


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
