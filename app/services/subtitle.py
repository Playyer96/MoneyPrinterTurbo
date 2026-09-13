import json
import os.path
import re
import sys
from types import SimpleNamespace
from timeit import default_timer as timer
from typing import Dict, List

import requests

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None
# mlx-whisper dispatches Whisper through Apple's MLX framework (Metal on
# Apple Silicon, with some operators landing on the Neural Engine via
# Core ML). On Mac hosts this is the fastest local path -- 3-5x faster
# than faster-whisper's CPU-only CTranslate2 backend -- and the primary
# choice when the VoiceStudio LaunchAgent is not reachable.
try:
    import mlx_whisper
except ImportError:
    mlx_whisper = None
from loguru import logger

from app.config import config
from app.services import guardrails
from app.utils import utils

model_size = config.whisper.get("model_size", "large-v3")
compute_type = config.whisper.get("compute_type", "default")
initial_prompt = config.whisper.get("initial_prompt", "") or None
model = None


def _remote_transcribe(audio_file: str):
    """Transcribe on the host-side GPU server, or return None when unavailable.

    faster-whisper's CTranslate2 backend is cpu/cuda only, so inside a Linux
    container on a Mac whisper can never leave the CPU. The VoiceStudio server
    already runs natively on the host for the same reason (see `make mac-setup`);
    it exposes /transcribe backed by MLX, which does run on Metal.
    """
    from app.services import voice

    base_url = voice.get_voicestudio_base_url()
    try:
        with open(audio_file, "rb") as fh:
            response = requests.post(
                f"{base_url}/transcribe",
                data=fh.read(),
                headers={"Content-Type": "application/octet-stream"},
                timeout=(2, 60),
            )
    except Exception as e:
        logger.warning(
            f"remote GPU whisper unavailable at {base_url}: {type(e).__name__}"
        )
        return None

    if response.status_code != 200:
        logger.warning(
            f"remote GPU whisper returned status {response.status_code}: "
            f"{response.text[:300]}"
        )
        return None

    payload = response.json()
    # Rebuild the attribute-access shape faster-whisper returns so the
    # segment-walking code below stays identical for both backends.
    segments = [
        SimpleNamespace(
            text=seg["text"],
            start=seg["start"],
            end=seg["end"],
            words=[SimpleNamespace(**w) for w in seg["words"]],
        )
        for seg in payload["segments"]
    ]
    info = SimpleNamespace(
        language=payload["language"],
        language_probability=payload["language_probability"],
    )
    logger.info(f"transcribed on remote GPU whisper at {base_url}")
    return segments, info


def _has_cuda_whisper() -> bool:
    """Return whether CTranslate2 can see a CUDA device."""
    try:
        import ctranslate2

        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


def log_gpu_backend_status() -> None:
    """Log the accelerated Whisper backend available on this host."""
    if os.environ.get("VOICESTUDIO_BASE_URL"):
        logger.info("Whisper backend: remote accelerator")
    elif sys.platform == "darwin" and mlx_whisper is not None:
        logger.info("Whisper backend: mlx-whisper (Apple Silicon Metal)")
    elif _has_cuda_whisper():
        logger.info("Whisper backend: faster-whisper (CUDA)")
    else:
        logger.warning("no accelerated Whisper backend; CPU transcription is disabled")


def _mlx_transcribe(audio_file: str):
    """Transcribe on the local MLX runtime (Apple Silicon Metal + ANE).

    mlx-whisper is the preferred local path on Mac because it dispatches
    Whisper through Apple's MLX framework, which targets Metal on Apple
    Silicon GPUs and routes a growing subset of operators to the Neural
    Engine via Core ML. That makes it 3-5x faster than faster-whisper
    (CPU-only CTranslate2) and removes the dependency on the VoiceStudio
    LaunchAgent being up.

    Returns a (segments, info) tuple in the same shape as the other
    backends, or None when the runtime is unavailable / fails.
    """
    if mlx_whisper is None or sys.platform != "darwin":
        return None
    mlx_model = str(
        config.whisper.get("mlx_model", "mlx-community/whisper-large-v3-turbo")
    )
    try:
        logger.info(f"mlx-whisper local transcription: model={mlx_model}")
        result = mlx_whisper.transcribe(
            audio_file,
            path_or_hf_repo=mlx_model,
            word_timestamps=True,
            verbose=False,
            **({"initial_prompt": initial_prompt} if initial_prompt else {}),
        )
    except Exception as exc:
        logger.warning(f"mlx-whisper local transcription failed: {exc}")
        return None

    if not result or not isinstance(result, dict):
        return None

    raw_segments = result.get("segments") or []
    segments = []
    for raw in raw_segments:
        words_raw = raw.get("words") or []
        words = [
            SimpleNamespace(
                start=float(w.get("start", 0.0)),
                end=float(w.get("end", 0.0)),
                word=str(w.get("word", "")),
            )
            for w in words_raw
            if w.get("word") is not None
        ]
        segments.append(
            SimpleNamespace(
                text=str(raw.get("text", "") or ""),
                start=float(raw.get("start", 0.0)),
                end=float(raw.get("end", 0.0)),
                words=words,
            )
        )
    info = SimpleNamespace(
        language=str(result.get("language", "en") or "en"),
        language_probability=float(result.get("language_probability", 1.0) or 1.0),
    )
    logger.info("mlx-whisper local transcription succeeded")
    return segments, info


def create(audio_file, subtitle_file: str = "", word_level: bool = False):
    global model

    # Order on Mac hosts:
    #   1. local mlx-whisper (Metal + ANE, fast, no server required)
    #   2. VoiceStudio remote (MLX, only if local MLX is unavailable)
    #   3. faster-whisper on CUDA
    # CPU inference is deliberately disabled on every platform.
    if sys.platform == "darwin" and mlx_whisper is not None:
        mlx_result = _mlx_transcribe(audio_file)
        if mlx_result is not None:
            segments, info = mlx_result
            return _write_subtitle(
                segments, info, audio_file, subtitle_file, word_level
            )

    remote = _remote_transcribe(audio_file)
    if remote is not None:
        segments, info = remote
        return _write_subtitle(segments, info, audio_file, subtitle_file, word_level)

    if WhisperModel is None or not _has_cuda_whisper():
        logger.warning(
            "no accelerated Whisper backend is available; skipping transcription"
        )
        return ""
    if not model:
        model_path = f"{utils.root_dir()}/models/whisper-{model_size}"
        model_bin_file = f"{model_path}/model.bin"
        if not os.path.isdir(model_path) or not os.path.isfile(model_bin_file):
            model_path = model_size

        logger.info(
            f"loading model: {model_path}, device: cuda, compute_type: {compute_type}"
        )
        try:
            model = WhisperModel(
                model_size_or_path=model_path, device="cuda", compute_type=compute_type
            )
        except Exception as e:
            logger.error(
                f"failed to load model: {e} \n\n"
                f"********************************************\n"
                f"this may be caused by network issue. \n"
                f"please download the model manually and put it in the 'models' folder. \n"
                f"see [README.md FAQ](https://github.com/harry0703/MoneyPrinterTurbo) for more details.\n"
                f"********************************************\n\n"
            )
            return None

    segments, info = model.transcribe(
        audio_file,
        beam_size=1,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
        **({"initial_prompt": initial_prompt} if initial_prompt else {}),
    )
    return _write_subtitle(segments, info, audio_file, subtitle_file, word_level)


def _write_subtitle(segments, info, audio_file, subtitle_file, word_level):
    """Turn transcribed segments into an SRT file. Backend-agnostic."""
    logger.info(f"start, output file: {subtitle_file}")
    if not subtitle_file:
        subtitle_file = f"{audio_file}.srt"

    logger.info(
        f"detected language: '{info.language}', probability: {info.language_probability:.2f}"
    )

    start = timer()
    subtitles = []
    # word-level entries captured during the same segment walk. Kept as
    # compact {"w","s","e"} dicts so the JSON sidecar stays under a few
    # hundred KB even for a 10-minute video. A real-time player reads
    # this file to drive karaoke-style highlighting without re-encoding.
    word_entries: List[Dict[str, float]] = []

    def recognized(seg_text, seg_start, seg_end):
        seg_text = seg_text.strip()
        if not seg_text:
            return

        msg = "[%.2fs -> %.2fs] %s" % (seg_start, seg_end, seg_text)
        logger.debug(msg)

        subtitles.append(
            {"msg": seg_text, "start_time": seg_start, "end_time": seg_end}
        )

    for segment in segments:
        # Capture every word regardless of mode: the JSON sidecar is a
        # progressive enhancement that costs a tiny dict append per word
        # and unlocks the real-time karaoke player for every subtitle file,
        # not only the explicitly word-level ones.
        if segment.words:
            for word in segment.words:
                cleaned_word = word.word.strip()
                if cleaned_word:
                    word_entries.append(
                        {
                            "w": cleaned_word,
                            "s": float(word.start),
                            "e": float(word.end),
                        }
                    )

        if word_level and segment.words:
            for word in segment.words:
                cleaned_word = word.word.strip()
                if cleaned_word:
                    recognized(cleaned_word, word.start, word.end)
            continue

        words_idx = 0
        words_len = len(segment.words)

        seg_start = 0
        seg_end = 0
        seg_text = ""

        if segment.words:
            is_segmented = False
            for word in segment.words:
                if not is_segmented:
                    seg_start = word.start
                    is_segmented = True

                seg_end = word.end
                # If it contains punctuation, then break the sentence.
                seg_text += word.word

                if utils.str_contains_punctuation(word.word):
                    # remove last char
                    seg_text = seg_text[:-1]
                    if not seg_text:
                        continue

                    recognized(seg_text, seg_start, seg_end)

                    is_segmented = False
                    seg_text = ""

                if words_idx == 0 and segment.start < word.start:
                    seg_start = word.start
                if words_idx == (words_len - 1) and segment.end > word.end:
                    seg_end = word.end
                words_idx += 1

        if not seg_text:
            continue

        recognized(seg_text, seg_start, seg_end)

    end = timer()

    diff = end - start
    logger.info(f"complete, elapsed: {diff:.2f} s")

    # Timing comes from the transcriber, which happily emits 0.2s flashes and
    # overlapping segments. Fix them here, once, for every backend.
    subtitles = guardrails.enforce_subtitle_cues(subtitles, word_level=word_level)

    idx = 1
    lines = []
    for subtitle in subtitles:
        text = subtitle.get("msg")
        if text:
            lines.append(
                utils.text_to_srt(
                    idx, text, subtitle.get("start_time"), subtitle.get("end_time")
                )
            )
            idx += 1

    sub = "\n".join(lines) + "\n"
    with open(subtitle_file, "w", encoding="utf-8") as f:
        f.write(sub)
    logger.info(f"subtitle file created: {subtitle_file}")

    # Word-level JSON sidecar. The SRT is the source of truth for legacy
    # burn-in paths; the JSON drives the real-time player that reads word
    # timestamps and styles them at playback. The two share the same guardrail
    # pass above, so timing stays consistent.
    if word_entries:
        try:
            from app.config import config as _cfg

            style_preset = str(
                _cfg.ui.get("subtitle_style_preset", "custom") or "custom"
            )
            font_size = int(_cfg.ui.get("font_size", 60) or 60)
        except Exception:
            style_preset = "custom"
            font_size = 60

        json_path = os.path.splitext(subtitle_file)[0] + ".words.json"
        cue_payload = [
            {
                "i": i + 1,
                "s": s.get("start_time"),
                "e": s.get("end_time"),
                "t": s.get("msg"),
            }
            for i, s in enumerate(subtitles)
            if s.get("msg")
        ]
        payload = {
            "version": 1,
            "language": info.language,
            "style": {"preset": style_preset, "fontSize": font_size},
            "cues": cue_payload,
            "words": word_entries,
        }
        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
            logger.info(f"word-level subtitle json created: {json_path}")
        except Exception as exc:
            # The JSON is a progressive enhancement; failing to write it must
            # not break the SRT path the rest of the pipeline depends on.
            logger.warning(f"failed to write word-level subtitle json: {exc}")


def file_to_subtitles(filename):
    if not filename or not os.path.isfile(filename):
        return []

    times_texts = []
    current_times = None
    current_text = ""
    index = 0
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            times = re.findall("([0-9]*:[0-9]*:[0-9]*,[0-9]*)", line)
            if times:
                current_times = line
            elif line.strip() == "" and current_times:
                index += 1
                times_texts.append((index, current_times.strip(), current_text.strip()))
                current_times, current_text = None, ""
            elif current_times:
                current_text += line

    # Flush the final block. SRT files whose last subtitle is not followed by a
    # trailing blank line never hit the blank-line branch above, so without this
    # the last subtitle would be silently dropped.
    if current_times:
        index += 1
        times_texts.append((index, current_times.strip(), current_text.strip()))
    return times_texts


def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def similarity(a, b):
    distance = levenshtein_distance(a.lower(), b.lower())
    max_length = max(len(a), len(b))
    return 1 - (distance / max_length)


def correct(subtitle_file, video_script):
    subtitle_items = file_to_subtitles(subtitle_file)
    normalized_script = utils.normalize_script_for_subtitle_matching(video_script)
    script_lines = utils.split_string_by_punctuations(normalized_script)

    corrected = False
    new_subtitle_items = []
    script_index = 0
    subtitle_index = 0

    while script_index < len(script_lines) and subtitle_index < len(subtitle_items):
        script_line = script_lines[script_index].strip()
        subtitle_line = subtitle_items[subtitle_index][2].strip()

        if script_line == subtitle_line:
            new_subtitle_items.append(subtitle_items[subtitle_index])
            script_index += 1
            subtitle_index += 1
        else:
            combined_subtitle = subtitle_line
            start_time = subtitle_items[subtitle_index][1].split(" --> ")[0]
            end_time = subtitle_items[subtitle_index][1].split(" --> ")[1]
            next_subtitle_index = subtitle_index + 1

            while next_subtitle_index < len(subtitle_items):
                next_subtitle = subtitle_items[next_subtitle_index][2].strip()
                if similarity(
                    script_line, combined_subtitle + " " + next_subtitle
                ) > similarity(script_line, combined_subtitle):
                    combined_subtitle += " " + next_subtitle
                    end_time = subtitle_items[next_subtitle_index][1].split(" --> ")[1]
                    next_subtitle_index += 1
                else:
                    break

            if similarity(script_line, combined_subtitle) > 0.8:
                logger.warning(
                    f"Merged/Corrected - Script: {script_line}, Subtitle: {combined_subtitle}"
                )
                new_subtitle_items.append(
                    (
                        len(new_subtitle_items) + 1,
                        f"{start_time} --> {end_time}",
                        script_line,
                    )
                )
                corrected = True
            else:
                logger.warning(
                    f"Mismatch - Script: {script_line}, Subtitle: {combined_subtitle}"
                )
                new_subtitle_items.append(
                    (
                        len(new_subtitle_items) + 1,
                        f"{start_time} --> {end_time}",
                        script_line,
                    )
                )
                corrected = True

            script_index += 1
            subtitle_index = next_subtitle_index

    # Process the remaining lines of the script.
    while script_index < len(script_lines):
        logger.warning(f"Extra script line: {script_lines[script_index]}")
        if subtitle_index < len(subtitle_items):
            new_subtitle_items.append(
                (
                    len(new_subtitle_items) + 1,
                    subtitle_items[subtitle_index][1],
                    script_lines[script_index],
                )
            )
            subtitle_index += 1
        else:
            new_subtitle_items.append(
                (
                    len(new_subtitle_items) + 1,
                    "00:00:00,000 --> 00:00:00,000",
                    script_lines[script_index],
                )
            )
        script_index += 1
        corrected = True

    if corrected:
        with open(subtitle_file, "w", encoding="utf-8") as fd:
            for i, item in enumerate(new_subtitle_items):
                fd.write(f"{i + 1}\n{item[1]}\n{item[2]}\n\n")
        logger.info("Subtitle corrected")
    else:
        logger.success("Subtitle is correct")
