"""Voice services — split into provider modules.

Public surface re-exported here so every existing import path
(``from app.services import voice`` / ``voice.edge_tts_synthesize`` etc.)
keeps working. The provider modules live under
``app.services.voice.providers``; the shared helpers live in
``app.services.voice._shared``; the subtitle cluster lives in
``app.services.voice.subtitle``. The dispatcher (``_single_tts``,
``tts``, ``_tts_with_pauses``) lives here so that ``patch.object(vs,
'X', ...)`` on provider names keeps intercepting the dispatcher's
bare-name lookups.
"""

from __future__ import annotations

import os
import re
import requests
import shutil
import subprocess
import tempfile
import wave
from datetime import timedelta
from typing import Optional, Union

import numpy as np

from edge_tts import SubMaker
from edge_tts.srt_composer import Subtitle
from loguru import logger
from moviepy.audio.io.AudioFileClip import AudioFileClip
from openai import OpenAI

from app.config import config
from app.services import guardrails
from app.services.audio_postprocess import soften_audio_transitions
from app.utils import utils

# Provider synthesis functions live here so that ``from app.services
# import voice; voice.edge_tts_synthesize(...)`` keeps working. The dispatcher's
# bare-name lookups (below) resolve against THIS package's namespace,
# so ``patch.object(vs, "edge_tts_synthesize", sentinel)`` swaps the binding the
# dispatcher reads on its next call.
from app.services.voice.providers import (  # noqa: F401  - re-exported
    _openai_compatible_tts,
    edge_tts_synthesize,
    chatterbox_tts,
    create_edge_tts_communicate,
    elevenlabs_tts,
    fish_audio_tts,
    gemini_tts,
    get_gemini_tts_models,
    get_minimax_voice_catalog,
    get_minimax_tts_api_key,
    get_minimax_tts_endpoint,
    get_edge_tts_timeout_seconds,
    _stream_edge_tts_sync_with_timeout,
    stream_edge_tts_chunks,
    kokoro_tts,
    mimo_tts,
    minimax_tts,
    _write_validated_minimax_audio,
    siliconflow_tts,
)
from app.services.voice.omnivoice import omnivoice_tts  # noqa: F401 - public API

# Subtitle cluster re-exported for backward compatibility.
from app.services.voice.subtitle import (  # noqa: F401  - re-exported
    create_subtitle,
    _format_text,
    _build_subtitle_formatter,
    _match_script_line,
    _normalize_arabic,
    _write_subtitle_items,
    _build_subtitle_items_from_edge_cues,
    _build_subtitle_items_from_legacy_submaker,
    _build_subtitle_items_from_edge_cues_words,
    _build_subtitle_items_from_legacy_submaker_words,
    _ARABIC_DIACRITICS,
    get_audio_duration,
)

# Shared helpers re-exported for backward compatibility.
from app.services.voice._shared import (  # noqa: F401  - re-exported
    DEFAULT_GEMINI_TTS_MODEL,
    FISH_AUDIO_DEFAULT_MODEL,
    FISH_AUDIO_MODELS,
    GEMINI_TTS_MODEL_CONFIG_KEY,
    GEMINI_TTS_VOICES,
    KOKORO_DEFAULT_VOICE,
    MINIMAX_TTS_DEFAULT_MODEL,
    MINIMAX_TTS_DEFAULT_VOICE,
    MINIMAX_TTS_GLOBAL_URL,
    MINIMAX_TTS_CN_URL,
    MINIMAX_TTS_MAX_AUDIO_HEX_CHARS,
    MINIMAX_TTS_MODELS,
    NO_VOICE_NAME,
    OMNIVOICE_DEFAULT_BASE_URL,
    _EDGE_VOICES_DATA_FILE,
    _GEMINI_TTS_MODEL_FALLBACK,
    _MINIMAX_TTS_MAX_AUDIO_HEX_CHARS,
    _MIMO_DEFAULT_BASE_URL,
    _MIMO_DEFAULT_TTS_MODEL,
    _NO_VOICE_ALIASES,
    _configure_pydub_ffmpeg,
    _is_running_in_docker,
    _load_edge_voices,
    _normalize_kokoro_voices,
    convert_rate_to_percent,
    create_omnivoice_profile,
    delete_omnivoice_profile,
    ensure_file_path_exists,
    ensure_legacy_submaker_fields,
    ensure_omnivoice_server_running,
    estimate_no_voice_duration,
    generate_silent_audio,
    get_all_edge_voices,
    get_chatterbox_voices,
    get_elevenlabs_api_key,
    get_elevenlabs_voices,
    get_fish_audio_api_key,
    get_fish_audio_voices,
    get_gemini_voices,
    get_kokoro_voices,
    get_minimax_voices,
    get_mimo_voices,
    get_siliconflow_voices,
    get_omnivoice_base_url,
    get_omnivoice_profile_metadata,
    get_omnivoice_profiles,
    get_omnivoice_voices,
    has_real_word_timestamps,
    is_edge_tts_voice,
    is_chatterbox_voice,
    is_elevenlabs_voice,
    is_fish_audio_voice,
    is_gemini_voice,
    is_kokoro_voice,
    is_minimax_voice,
    is_mimo_voice,
    is_no_voice,
    is_siliconflow_voice,
    is_omnivoice_voice,
    mark_real_word_timestamps,
    mktimestamp,
    parse_gemini_voice_name,
    parse_voice_name,
    populate_legacy_submaker_with_full_text,
)


# Fade length applied at every known chunk-join in ``_concat_audio_files``
# (e.g. a speech segment butting against a ``[pause: Ns]`` silence chunk).
# Unlike ``soften_audio_transitions``'s RMS-based detection, these boundaries
# are exact -- we're the ones building the chunk list -- so every join gets
# faded regardless of how short the adjoining silence is. The fade only
# scales existing samples in place (no samples are dropped or inserted), so
# total duration is unchanged and subtitle timing stays in sync.
_CHUNK_BOUNDARY_FADE_MS = 15


def _concat_audio_files(audio_files: list[str], output_file: str) -> bool:
    """
    Merge audio segments through PCM decoding and one final encode.

    Convert every input to standard 24 kHz 16-bit mono PCM for seamless
    concatenation, then encode the target once. This prevents accumulated MP3
    encoder delay and padding from causing audio/video or subtitle drift.

    Every chunk join gets a short in-place crossfade (see
    ``_CHUNK_BOUNDARY_FADE_MS``) before the generic ``soften_audio_transitions``
    pass runs, so the splice between e.g. a speech chunk and a pause's silence
    chunk never sounds like a hard cut.
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
    # Sample offset of every internal chunk join (i.e. not the very start of
    # the file), recorded right before that chunk's frames are appended.
    chunk_boundaries: list[int] = []
    ffmpeg_binary = utils.get_ffmpeg_binary()

    def _append_chunk(frames: bytes) -> None:
        if combined_pcm:
            chunk_boundaries.append(len(combined_pcm) // 2)  # 2 bytes/sample
        combined_pcm.extend(frames)

    with tempfile.TemporaryDirectory() as concat_temp:
        for idx, f in enumerate(audio_files):
            if not os.path.exists(f) or os.path.getsize(f) == 0:
                continue

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
                            _append_chunk(wf.readframes(wf.getnframes()))
                except Exception:
                    is_valid_pcm_wav = False

            if not is_valid_pcm_wav:
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
                if res.returncode != 0:
                    logger.warning(
                        f"ffmpeg decode failed for chunk {idx}: "
                        f"{(res.stderr or '').strip()[-200:]}"
                    )
                    continue
                try:
                    with wave.open(pcm_wav, "rb") as wf:
                        _append_chunk(wf.readframes(wf.getnframes()))
                except Exception:
                    continue

        if not combined_pcm:
            logger.error("audio concat: no usable samples after decoding")
            return False

        if chunk_boundaries:
            # In-place crossfade at every known join: fade the previous
            # chunk's tail out and the next chunk's head in, scaled to
            # whatever's actually available (a chunk shorter than the fade
            # window just gets a proportionally shorter ramp). No samples
            # are added or removed, so this cannot shift subtitle timing.
            boundary_samples = np.frombuffer(
                bytes(combined_pcm), dtype=np.int16
            ).astype(np.float32)
            max_fade = max(
                int(target_sample_rate * _CHUNK_BOUNDARY_FADE_MS / 1000), 1
            )
            for boundary in chunk_boundaries:
                fade_out_len = min(max_fade, boundary)
                if fade_out_len > 0:
                    boundary_samples[boundary - fade_out_len : boundary] *= (
                        np.linspace(1.0, 0.0, fade_out_len, dtype=np.float32)
                    )
                fade_in_len = min(max_fade, len(boundary_samples) - boundary)
                if fade_in_len > 0:
                    boundary_samples[boundary : boundary + fade_in_len] *= (
                        np.linspace(0.0, 1.0, fade_in_len, dtype=np.float32)
                    )
            combined_pcm = bytearray(boundary_samples.astype(np.int16).tobytes())

        pcm_target = os.path.join(concat_temp, "combined.wav")
        with wave.open(pcm_target, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(target_sample_rate)
            wf.writeframes(bytes(combined_pcm))

        encode_cmd = [
            ffmpeg_binary,
            "-y",
            "-i",
            pcm_target,
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(target_sample_rate),
            output_file,
        ]
        # Voice-only pipeline output. Bump bitrate to 192k so the
        # concatenated audio doesn't lose detail at every sentence break;
        # ffmpeg's default (~128k) is fine for mixed audio but eats the
        # sibilants and breath cues that make narration sound natural.
        if output_file.lower().endswith(".mp3"):
            encode_cmd.extend(["-c:a", "libmp3lame", "-b:a", "192k"])
        # Soften the silence boundaries in the PCM intermediate so the
        # final encode doesn't add an extra decode/re-encode cycle and the
        # micro-fades are baked into the output exactly once.
        try:
            soften_audio_transitions(pcm_target)
        except Exception as exc:
            logger.warning(
                f"audio concat silence softening skipped: "
                f"{type(exc).__name__}: {exc}"
            )
        res = subprocess.run(encode_cmd, capture_output=True, text=True, check=False)
        if res.returncode != 0:
            logger.error(
                f"audio concat encode failed: rc={res.returncode}, "
                f"stderr={(res.stderr or '').strip()[-200:]}"
            )
            return False
        return True


def _soften_voice_file(voice_file: str) -> None:
    """Apply the sentence-boundary click fix to any provider's raw output.

    ``soften_audio_transitions`` only understands 16-bit mono PCM WAV, but
    every provider writes its own native container (mp3 for edge_tts,
    elevenlabs, minimax, ...). ``_concat_audio_files`` already decodes,
    softens, and re-encodes once for the ``[pause: Ns]`` path; this does the
    same for the far more common path where a script has no pause tags and
    ``_single_tts`` writes straight to ``voice_file`` -- otherwise every
    natural sentence gap in that (default) path never gets its click fixed,
    no matter how good ``soften_audio_transitions`` itself is.

    Best-effort and silent: softening is cosmetic, so any decode/encode
    failure here must never turn a successful TTS call into a failed one.
    """
    if not voice_file or not os.path.isfile(voice_file) or os.path.getsize(voice_file) == 0:
        return
    ffmpeg_binary = utils.get_ffmpeg_binary()
    output_ext = os.path.splitext(voice_file)[1].lower()
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pcm_wav = os.path.join(tmp_dir, "decoded.wav")
            decode_cmd = [
                ffmpeg_binary, "-y", "-i", voice_file,
                "-vn", "-ac", "1", "-ar", "24000",
                "-codec:a", "pcm_s16le", pcm_wav,
            ]
            res = subprocess.run(decode_cmd, capture_output=True, text=True, check=False)
            if res.returncode != 0 or not os.path.exists(pcm_wav):
                return

            if not soften_audio_transitions(pcm_wav):
                return  # no silences worth softening; leave voice_file as-is

            re_encode_cmd = [
                ffmpeg_binary, "-y", "-i", pcm_wav,
                "-vn", "-ac", "1", "-ar", "24000",
            ]
            if output_ext == ".mp3":
                re_encode_cmd.extend(["-c:a", "libmp3lame", "-b:a", "192k"])
            re_encode_cmd.append(voice_file)
            res = subprocess.run(re_encode_cmd, capture_output=True, text=True, check=False)
            if res.returncode != 0:
                logger.warning(
                    "voice softening re-encode failed, keeping unsoftened "
                    f"audio: {(res.stderr or '').strip()[-200:]}"
                )
    except Exception as exc:
        logger.warning(f"voice softening skipped: {type(exc).__name__}: {exc}")


def _single_tts(
    text: str,
    voice_name: str,
    voice_rate: float,
    voice_file: str,
    voice_volume: float = 1.0,
    voice_style: str = "",
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

    if is_siliconflow_voice(voice_name):
        parts = voice_name.split(":")
        if len(parts) >= 3:
            model = parts[1]
            voice_with_gender = parts[2]
            voice = voice_with_gender.split("-")[0]
            full_voice = f"{model}:{voice}"
            return siliconflow_tts(
                text, model, full_voice, voice_rate, voice_file, voice_volume
            )
        logger.error(f"Invalid siliconflow voice name format: {voice_name}")
        return None
    elif is_gemini_voice(voice_name):
        voice = parse_gemini_voice_name(voice_name)
        if voice:
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
        logger.error(f"Invalid gemini voice name format: {voice_name}")
        return None
    elif is_mimo_voice(voice_name):
        parts = voice_name.split(":")
        if len(parts) >= 2:
            voice_with_gender = parts[1]
            voice = voice_with_gender.split("-")[0]
            return mimo_tts(text, voice, voice_rate, voice_file, voice_volume)
        logger.error(f"Invalid mimo voice name format: {voice_name}")
        return None
    elif is_minimax_voice(voice_name):
        voice_id = voice_name.split(":", 1)[1].strip()
        if voice_id:
            return minimax_tts(text, voice_id, voice_rate, voice_file, voice_volume)
        logger.error(f"Invalid MiniMax voice name format: {voice_name}")
        return None
    elif is_elevenlabs_voice(voice_name):
        parts = voice_name.split(":")
        if len(parts) >= 2:
            voice_id = parts[1]
            return elevenlabs_tts(text, voice_id, voice_file, voice_rate, voice_volume)
        logger.error(f"Invalid elevenlabs voice name format: {voice_name}")
        return None
    elif is_chatterbox_voice(voice_name):
        parts = voice_name.split(":", 1)
        if len(parts) >= 2 and parts[1].strip():
            chatterbox_voice = parts[1].strip()
            if chatterbox_voice.endswith(("-Female", "-Male")):
                chatterbox_voice = chatterbox_voice.rsplit("-", 1)[0]
            return chatterbox_tts(
                text, chatterbox_voice, voice_file, voice_rate, voice_volume
            )
        logger.error(f"Invalid chatterbox voice name format: {voice_name}")
        return None
    elif is_kokoro_voice(voice_name):
        parts = voice_name.split(":", 1)
        if len(parts) >= 2 and parts[1].strip():
            kokoro_voice = parts[1].strip()
            if kokoro_voice.endswith(("-Female", "-Male")):
                kokoro_voice = kokoro_voice.rsplit("-", 1)[0]
            return kokoro_tts(
                text, kokoro_voice, voice_file, voice_rate, voice_volume
            )
        logger.error(f"Invalid kokoro voice name format: {voice_name}")
        return None
    elif is_fish_audio_voice(voice_name):
        parts = voice_name.split(":")
        reference_id = parts[1] if len(parts) >= 2 else "default"
        if reference_id == "default":
            reference_id = None
        return fish_audio_tts(
            text, voice_file, voice_rate, voice_volume, reference_id=reference_id
        )
    elif is_omnivoice_voice(voice_name):
        parts = voice_name.split(":", 1)
        if len(parts) >= 2 and parts[1].strip():
            return omnivoice_tts(
                text,
                parts[1].strip(),
                voice_file,
                voice_rate,
                voice_volume,
                voice_style=voice_style,
            )
        logger.error(f"Invalid omnivoice voice name format: {voice_name}")
        return None
    return edge_tts_synthesize(text, voice_name, voice_rate, voice_file)


def _tts_with_pauses(
    text: str,
    voice_name: str,
    voice_rate: float,
    voice_file: str,
    voice_volume: float = 1.0,
    voice_style: str = "",
) -> Union[SubMaker, None]:
    """Synthesize scripts containing pause tags such as ``[pause: 2s]``.

    Routes every speech segment through ``_single_tts`` and interleaves exact
    PCM silence, then concatenates everything via ``_concat_audio_files``.
    Only Edge TTS takes this segmented path; other providers fall back
    to a single stripped-text call (see ``tts``).
    """
    segments = utils.parse_script_with_pauses(text)
    if not segments:
        return None

    speech_segments = [s for s in segments if s[0] == "speech"]
    pause_segments = [s for s in segments if s[0] == "pause"]

    if not pause_segments:
        clean_text = utils.remove_pause_tags(text)
        return _single_tts(
            clean_text,
            voice_name,
            voice_rate,
            voice_file,
            voice_volume,
            voice_style=voice_style,
        )

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
                generate_silent_audio(pause_duration, silence_wav)
                actual_pause_duration = pause_duration
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
                    voice_style=voice_style,
                )
                if (
                    not chunk_submaker
                    or not os.path.exists(chunk_audio_file)
                    or os.path.getsize(chunk_audio_file) == 0
                ):
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

                current_offset_seconds = cumulative_samples / float(SAMPLE_RATE)

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


# Pacing words that map to a rate multiplier for providers (Edge TTS) that
# accept `voice_rate` per call but no emotion instruction. The list is
# deliberately small; ambiguous cues fall back to the base rate. Longer
# phrases come first so they match before their shorter substrings ("slow
# and reflective" beats "slow").
_CUE_PACING_TABLE: dict[str, float] = {
    "very fast": 1.5,
    "very slow": 0.7,
    "slow and reflective": 0.8,
    "rapid": 1.4,
    "brisk": 1.15,
    "slow": 0.85,
    "fast": 1.3,
    "measured": 0.95,
    "moderate": 1.0,
}


def _cue_to_voice_rate(cue: str, base_rate: float) -> float:
    """
    Derive a per-paragraph voice_rate multiplier from a delivery cue.

    Edge TTS (and a few other providers) only consume a numeric rate, so this
    translates the first pacing word it finds in the cue into a multiplier
    applied on top of ``base_rate``. Emotion words (euphoric, somber, ...)
    are intentionally not mapped — those need ``voice_style`` support that
    Edge does not have. Empty or unmatched cues return ``base_rate`` unchanged.
    """
    if not cue:
        return base_rate
    text = cue.lower().strip()
    for keyword, multiplier in _CUE_PACING_TABLE.items():
        if keyword in text:
            return guardrails.clamp_voice_rate(base_rate * multiplier)
    return base_rate


def tts(
    text: str,
    voice_name: str,
    voice_rate: float,
    voice_file: str,
    voice_volume: float = 1.0,
    voice_style: str = "",
) -> Union[SubMaker, None]:
    """Single public TTS entry point — clamps + pause-tag routing.

    Every provider is reached through this function, so clamping here is
    what makes the speed and volume limits unavoidable rather than advisory.
    ``voice_style`` is a free-text emotion/delivery instruction forwarded
    to providers that understand it (today: OmniVoice). Other providers
    silently ignore it so the same call site can drive multiple backends.
    """
    voice_rate = guardrails.clamp_voice_rate(voice_rate)
    voice_volume = guardrails.clamp_voice_volume(voice_volume)

    if not utils.has_pause_tags(text):
        result = _single_tts(
            text=text,
            voice_name=voice_name,
            voice_rate=voice_rate,
            voice_file=voice_file,
            voice_volume=voice_volume,
            voice_style=voice_style,
        )
        if result is not None and not is_no_voice(voice_name):
            _soften_voice_file(voice_file)
        return result

    if is_edge_tts_voice(voice_name):
        # _tts_with_pauses -> _concat_audio_files already softens the
        # combined PCM internally before its own final encode.
        return _tts_with_pauses(
            text=text,
            voice_name=voice_name,
            voice_rate=voice_rate,
            voice_file=voice_file,
            voice_volume=voice_volume,
            voice_style=voice_style,
        )

    clean_text = utils.remove_pause_tags(text)
    result = _single_tts(
        text=clean_text,
        voice_name=voice_name,
        voice_rate=voice_rate,
        voice_file=voice_file,
        voice_volume=voice_volume,
        voice_style=voice_style,
    )
    if result is not None and not is_no_voice(voice_name):
        _soften_voice_file(voice_file)
    return result


def tts_with_styles(
    text: str,
    voice_name: str,
    voice_rate: float,
    voice_file: str,
    voice_volume: float = 1.0,
    voice_styles: Optional[list[str]] = None,
    default_voice_style: str = "",
) -> Union[SubMaker, None]:
    """TTS that applies a per-paragraph voice_style hint to each paragraph.

    Splits the script on blank lines (matching how ``llm.parse_delivery_cues``
    carves the response into paragraphs) and runs the single-shot TTS once
    per paragraph, forwarding the matching style from ``voice_styles`` (or
    ``default_voice_style`` when the entry is empty). Pause tags inside a
    paragraph are honoured for Edge TTS (silence interleaved between speech
    segments); for other providers pauses inside the paragraph are stripped,
    since they cannot be interleaved with custom voice_style calls.

    Falls back to ``tts`` when ``voice_styles`` is empty or the script has
    no paragraph breaks.

    ponytail: per-paragraph TTS multiplies provider calls and audio decoding by
    the paragraph count; the cheaper single-shot path stays the default.
    """
    if not voice_styles:
        return tts(
            text=text,
            voice_name=voice_name,
            voice_rate=voice_rate,
            voice_file=voice_file,
            voice_volume=voice_volume,
            voice_style=default_voice_style,
        )

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text or "") if p.strip()]
    if len(paragraphs) <= 1:
        style = voice_styles[0] if voice_styles else default_voice_style
        # When the whole script is one paragraph and it has pause tags, route
        # through the standard tts() so the pause-aware stitching path runs
        # unchanged (single paragraph still honours pause_tag semantics).
        if utils.has_pause_tags(text or ""):
            return tts(
                text=text,
                voice_name=voice_name,
                voice_rate=_cue_to_voice_rate(style, voice_rate),
                voice_file=voice_file,
                voice_volume=voice_volume,
                voice_style=style,
            )
        return tts(
            text=text,
            voice_name=voice_name,
            voice_rate=_cue_to_voice_rate(style, voice_rate),
            voice_file=voice_file,
            voice_volume=voice_volume,
            voice_style=style,
        )

    voice_rate = guardrails.clamp_voice_rate(voice_rate)
    voice_volume = guardrails.clamp_voice_volume(voice_volume)

    SAMPLE_RATE = 24000
    ffmpeg_binary = utils.get_ffmpeg_binary()
    chunk_files: list[str] = []
    combined_submaker = ensure_legacy_submaker_fields(SubMaker())
    cumulative_samples = 0

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            for idx, paragraph in enumerate(paragraphs):
                style = voice_styles[idx] if idx < len(voice_styles) else ""
                effective_style = style or default_voice_style
                # Derive a per-paragraph rate from the cue so Edge TTS, which
                # doesn't accept voice_style, still picks up pacing words.
                effective_rate = _cue_to_voice_rate(style, voice_rate)
                paragraph_chunk_files, paragraph_submaker, paragraph_samples = (
                    _synthesize_paragraph(
                        paragraph=paragraph,
                        voice_name=voice_name,
                        voice_rate=effective_rate,
                        voice_volume=voice_volume,
                        voice_style=effective_style,
                        temp_dir=temp_dir,
                        paragraph_index=idx,
                        ffmpeg_binary=ffmpeg_binary,
                        sample_rate=SAMPLE_RATE,
                    )
                )
                if paragraph_chunk_files is None:
                    logger.error(
                        f"failed to synthesize cue paragraph {idx}: "
                        f"{paragraph[:50]!r}"
                    )
                    return None

                offset_seconds = cumulative_samples / float(SAMPLE_RATE)
                if hasattr(paragraph_submaker, "cues") and paragraph_submaker.cues:
                    offset_td = timedelta(seconds=offset_seconds)
                    for cue in paragraph_submaker.cues:
                        combined_submaker.cues.append(
                            Subtitle(
                                index=len(combined_submaker.cues) + 1,
                                start=cue.start + offset_td,
                                end=cue.end + offset_td,
                                content=cue.content,
                            )
                        )
                if hasattr(paragraph_submaker, "subs") and paragraph_submaker.subs:
                    combined_submaker.subs.extend(paragraph_submaker.subs)
                if hasattr(paragraph_submaker, "offset") and paragraph_submaker.offset:
                    offset_100ns = int(offset_seconds * 10000000)
                    for start_ns, end_ns in paragraph_submaker.offset:
                        combined_submaker.offset.append(
                            (start_ns + offset_100ns, end_ns + offset_100ns)
                        )

                chunk_files.extend(paragraph_chunk_files)
                cumulative_samples += paragraph_samples

            if not _concat_audio_files(chunk_files, voice_file):
                logger.error("failed to concatenate per-cue audio chunks")
                return None

        combined_submaker.duration = cumulative_samples / float(SAMPLE_RATE)
        if not is_no_voice(voice_name):
            _soften_voice_file(voice_file)
        return combined_submaker
    except Exception:
        logger.exception("tts_with_styles: unexpected failure")
        return None


def _synthesize_paragraph(
    *,
    paragraph: str,
    voice_name: str,
    voice_rate: float,
    voice_volume: float,
    voice_style: str,
    temp_dir: str,
    paragraph_index: int,
    ffmpeg_binary: str,
    sample_rate: int,
):
    """
    Render a single paragraph to a list of PCM WAV chunks + its SubMaker.

    Honours pause tags inside the paragraph for Edge TTS (silence interleaved
    between speech segments via ``_tts_with_pauses``); for other providers
    pauses are stripped and a single ``_single_tts`` call renders the whole
    paragraph with the given ``voice_style``. Either way the returned chunks
    are PCM WAVs at ``sample_rate`` Hz, ready to concatenate.

    Returns ``(None, None, 0)`` when synthesis fails so the caller can abort
    the whole ``tts_with_styles`` run cleanly.
    """
    chunk_files: list[str] = []
    combined_submaker: SubMaker = ensure_legacy_submaker_fields(SubMaker())
    cumulative_samples = 0
    paragraph_prefix = f"p{paragraph_index:03d}_"

    if utils.has_pause_tags(paragraph):
        # The pause-aware path internally produces a list of PCM chunks
        # (speech + silence) that we walk to merge into our running offset.
        segments = utils.parse_script_with_pauses(paragraph)
        speech_segments = [s for s in segments if s[0] == "speech"]
        pause_segments = [s for s in segments if s[0] == "pause"]

        if not pause_segments:
            # Pause tags were present but stripped; fall back to single-shot.
            chunk_submaker = _single_tts(
                text=utils.remove_pause_tags(paragraph),
                voice_name=voice_name,
                voice_rate=voice_rate,
                voice_file=os.path.join(temp_dir, f"{paragraph_prefix}single.mp3"),
                voice_volume=voice_volume,
                voice_style=voice_style,
            )
            if chunk_submaker is None:
                return None, None, 0
        elif not speech_segments:
            # No speech, only silence: emit silent chunks at the right
            # duration so the cue paragraph still occupies air time.
            chunk_files, cumulative_samples = _emit_silence_chunks(
                pause_segments=pause_segments,
                temp_dir=temp_dir,
                prefix=paragraph_prefix,
                sample_rate=sample_rate,
            )
            combined_submaker = ensure_legacy_submaker_fields(SubMaker())
            combined_submaker.duration = cumulative_samples / float(sample_rate)
            return chunk_files, combined_submaker, cumulative_samples
        else:
            chunk_files, cumulative_samples, combined_submaker = (
                _render_pause_aware_chunks(
                    segments=segments,
                    voice_name=voice_name,
                    voice_rate=voice_rate,
                    voice_volume=voice_volume,
                    voice_style=voice_style,
                    temp_dir=temp_dir,
                    prefix=paragraph_prefix,
                    ffmpeg_binary=ffmpeg_binary,
                    sample_rate=sample_rate,
                )
            )
            if not chunk_files:
                return None, None, 0
            return chunk_files, combined_submaker, cumulative_samples

    # Plain paragraph (no pause tags): one shot, one chunk.
    chunk_audio = os.path.join(temp_dir, f"{paragraph_prefix}single.mp3")
    chunk_submaker = _single_tts(
        text=paragraph,
        voice_name=voice_name,
        voice_rate=voice_rate,
        voice_file=chunk_audio,
        voice_volume=voice_volume,
        voice_style=voice_style,
    )
    if (
        not chunk_submaker
        or not os.path.exists(chunk_audio)
        or os.path.getsize(chunk_audio) == 0
    ):
        return None, None, 0

    chunk_wav = os.path.join(temp_dir, f"{paragraph_prefix}single_decoded.wav")
    if not _decode_to_pcm_wav(
        chunk_audio, chunk_wav, ffmpeg_binary, sample_rate
    ):
        return None, None, 0

    chunk_samples = _pcm_wav_sample_count(chunk_wav)
    if chunk_samples <= 0:
        return None, None, 0
    return [chunk_wav], chunk_submaker, chunk_samples


def _emit_silence_chunks(
    *,
    pause_segments,
    temp_dir: str,
    prefix: str,
    sample_rate: int,
):
    """
    Generate PCM WAV silence chunks matching the requested pause durations.

    Returns ``(chunk_files, total_samples)`` so callers can stitch the silence
    into their running concatenation just like speech chunks. Used when a
    paragraph contains only ``[pause: ...]`` tags with no speech.
    """
    files: list[str] = []
    total_samples = 0
    for idx, (seg_type, seg_val) in enumerate(pause_segments):
        pause_duration = float(seg_val)
        silence_wav = os.path.join(temp_dir, f"{prefix}silence_{idx}.wav")
        num_silent_samples = int(round(pause_duration * sample_rate))
        with wave.open(silence_wav, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(b"\x00\x00" * num_silent_samples)
        generate_silent_audio(pause_duration, silence_wav)
        actual_pause_duration = pause_duration
        mock_check_duration = get_audio_duration(silence_wav)
        if mock_check_duration > 0 and abs(mock_check_duration - pause_duration) > 0.05:
            actual_pause_duration = mock_check_duration
            num_silent_samples = int(round(actual_pause_duration * sample_rate))
        files.append(silence_wav)
        total_samples += num_silent_samples
    return files, total_samples


def _render_pause_aware_chunks(
    *,
    segments,
    voice_name: str,
    voice_rate: float,
    voice_volume: float,
    voice_style: str,
    temp_dir: str,
    prefix: str,
    ffmpeg_binary: str,
    sample_rate: int,
):
    """
    Render a paragraph containing pause tags into a sequence of PCM WAV chunks
    plus a SubMaker with timestamps measured from the start of the paragraph.

    Only Edge TTS supports interleaving silence between speech segments with
    per-call voice_style; for other providers we fall back to single-shot
    synthesis with pauses stripped (matches the legacy ``_tts_with_pauses``
    contract). Returns ``(chunk_files, total_samples, submaker)``; an empty
    file list means the provider could not synthesise this paragraph.
    """
    chunk_files: list[str] = []
    cumulative_samples = 0
    combined_submaker = ensure_legacy_submaker_fields(SubMaker())

    if not is_edge_tts_voice(voice_name):
        clean_text = utils.remove_pause_tags(" ".join(str(v) for t, v in segments if t == "speech"))
        chunk_audio = os.path.join(temp_dir, f"{prefix}fallback.mp3")
        chunk_submaker = _single_tts(
            text=clean_text,
            voice_name=voice_name,
            voice_rate=voice_rate,
            voice_file=chunk_audio,
            voice_volume=voice_volume,
            voice_style=voice_style,
        )
        if (
            not chunk_submaker
            or not os.path.exists(chunk_audio)
            or os.path.getsize(chunk_audio) == 0
        ):
            return [], 0, combined_submaker
        chunk_wav = os.path.join(temp_dir, f"{prefix}fallback_decoded.wav")
        if not _decode_to_pcm_wav(chunk_audio, chunk_wav, ffmpeg_binary, sample_rate):
            return [], 0, combined_submaker
        chunk_samples = _pcm_wav_sample_count(chunk_wav)
        if chunk_samples <= 0:
            return [], 0, combined_submaker
        offset_seconds = cumulative_samples / float(sample_rate)
        if hasattr(chunk_submaker, "cues") and chunk_submaker.cues:
            offset_td = timedelta(seconds=offset_seconds)
            for cue in chunk_submaker.cues:
                combined_submaker.cues.append(
                    Subtitle(
                        index=len(combined_submaker.cues) + 1,
                        start=cue.start + offset_td,
                        end=cue.end + offset_td,
                        content=cue.content,
                    )
                )
        if hasattr(chunk_submaker, "subs") and chunk_submaker.subs:
            combined_submaker.subs.extend(chunk_submaker.subs)
        if hasattr(chunk_submaker, "offset") and chunk_submaker.offset:
            offset_100ns = int(offset_seconds * 10000000)
            for start_ns, end_ns in chunk_submaker.offset:
                combined_submaker.offset.append(
                    (start_ns + offset_100ns, end_ns + offset_100ns)
                )
        chunk_files.append(chunk_wav)
        cumulative_samples += chunk_samples
        combined_submaker.duration = cumulative_samples / float(sample_rate)
        return chunk_files, cumulative_samples, combined_submaker

    for idx, (seg_type, seg_val) in enumerate(segments):
        if seg_type == "pause":
            pause_files, pause_samples = _emit_silence_chunks(
                pause_segments=[(seg_type, seg_val)],
                temp_dir=temp_dir,
                prefix=f"{prefix}p{idx:03d}_",
                sample_rate=sample_rate,
            )
            chunk_files.extend(pause_files)
            cumulative_samples += pause_samples
            continue
        speech_text = str(seg_val).strip()
        if not speech_text:
            continue
        chunk_audio = os.path.join(temp_dir, f"{prefix}speech_{idx}.mp3")
        chunk_submaker = _single_tts(
            text=speech_text,
            voice_name=voice_name,
            voice_rate=voice_rate,
            voice_file=chunk_audio,
            voice_volume=voice_volume,
            voice_style=voice_style,
        )
        if (
            not chunk_submaker
            or not os.path.exists(chunk_audio)
            or os.path.getsize(chunk_audio) == 0
        ):
            return [], 0, combined_submaker
        chunk_wav = os.path.join(temp_dir, f"{prefix}speech_{idx}_decoded.wav")
        if not _decode_to_pcm_wav(chunk_audio, chunk_wav, ffmpeg_binary, sample_rate):
            return [], 0, combined_submaker
        chunk_samples = _pcm_wav_sample_count(chunk_wav)
        if chunk_samples <= 0:
            return [], 0, combined_submaker

        offset_seconds = cumulative_samples / float(sample_rate)
        if hasattr(chunk_submaker, "cues") and chunk_submaker.cues:
            offset_td = timedelta(seconds=offset_seconds)
            for cue in chunk_submaker.cues:
                combined_submaker.cues.append(
                    Subtitle(
                        index=len(combined_submaker.cues) + 1,
                        start=cue.start + offset_td,
                        end=cue.end + offset_td,
                        content=cue.content,
                    )
                )
        if hasattr(chunk_submaker, "subs") and chunk_submaker.subs:
            combined_submaker.subs.extend(chunk_submaker.subs)
        if hasattr(chunk_submaker, "offset") and chunk_submaker.offset:
            offset_100ns = int(offset_seconds * 10000000)
            for start_ns, end_ns in chunk_submaker.offset:
                combined_submaker.offset.append(
                    (start_ns + offset_100ns, end_ns + offset_100ns)
                )
        chunk_files.append(chunk_wav)
        cumulative_samples += chunk_samples

    combined_submaker.duration = cumulative_samples / float(sample_rate)
    return chunk_files, cumulative_samples, combined_submaker


def _decode_to_pcm_wav(
    source_audio: str,
    output_wav: str,
    ffmpeg_binary: str,
    sample_rate: int,
) -> bool:
    """Decode any audio to a mono PCM s16le WAV at ``sample_rate``. False on failure."""
    cmd = [
        ffmpeg_binary,
        "-y",
        "-i",
        source_audio,
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-codec:a",
        "pcm_s16le",
        output_wav,
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return (
        res.returncode == 0
        and os.path.exists(output_wav)
        and os.path.getsize(output_wav) > 0
    )


def _pcm_wav_sample_count(wav_path: str) -> int:
    """Return the number of PCM samples in a WAV (0 on read error)."""
    try:
        with wave.open(wav_path, "rb") as wf:
            return wf.getnframes()
    except Exception as exc:
        logger.warning(f"failed to read decoded wav {wav_path}: {exc}")
        return 0
