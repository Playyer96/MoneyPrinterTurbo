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
import requests
import shutil
import subprocess
import tempfile
import wave
from datetime import timedelta
from typing import Union

from edge_tts import SubMaker
from edge_tts.srt_composer import Subtitle
from loguru import logger
from moviepy.audio.io.AudioFileClip import AudioFileClip
from openai import OpenAI

from app.config import config
from app.services import guardrails
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
                        combined_pcm.extend(wf.readframes(wf.getnframes()))
                except Exception:
                    continue

        if not combined_pcm:
            logger.error("audio concat: no usable samples after decoding")
            return False

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
        res = subprocess.run(encode_cmd, capture_output=True, text=True, check=False)
        if res.returncode != 0:
            logger.error(
                f"audio concat encode failed: rc={res.returncode}, "
                f"stderr={(res.stderr or '').strip()[-200:]}"
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
                text, parts[1].strip(), voice_file, voice_rate, voice_volume
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


def tts(
    text: str,
    voice_name: str,
    voice_rate: float,
    voice_file: str,
    voice_volume: float = 1.0,
) -> Union[SubMaker, None]:
    """Single public TTS entry point — clamps + pause-tag routing.

    Every provider is reached through this function, so clamping here is
    what makes the speed and volume limits unavoidable rather than advisory.
    """
    voice_rate = guardrails.clamp_voice_rate(voice_rate)
    voice_volume = guardrails.clamp_voice_volume(voice_volume)

    if not utils.has_pause_tags(text):
        return _single_tts(
            text=text,
            voice_name=voice_name,
            voice_rate=voice_rate,
            voice_file=voice_file,
            voice_volume=voice_volume,
        )

    if is_edge_tts_voice(voice_name):
        return _tts_with_pauses(
            text=text,
            voice_name=voice_name,
            voice_rate=voice_rate,
            voice_file=voice_file,
            voice_volume=voice_volume,
        )

    clean_text = utils.remove_pause_tags(text)
    return _single_tts(
        text=clean_text,
        voice_name=voice_name,
        voice_rate=voice_rate,
        voice_file=voice_file,
        voice_volume=voice_volume,
    )
