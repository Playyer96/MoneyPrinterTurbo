import itertools
import io
import math
import os
import random
import re
import gc
import glob
import shutil
import subprocess
import sys
import tempfile
import unicodedata
from contextlib import ExitStack, redirect_stdout
from functools import lru_cache
from time import perf_counter
from typing import List, Optional, Set, Tuple
from loguru import logger
import numpy as np
from moviepy import (
    AudioFileClip,
    ColorClip,
    CompositeAudioClip,
    CompositeVideoClip,
    ImageClip,
    TextClip,
    VideoFileClip,
    afx,
)
from moviepy.video.io import ffmpeg_writer as moviepy_ffmpeg_writer
from moviepy.video.tools.subtitles import SubtitlesClip
from PIL import Image, ImageDraw, ImageFont

from app.config import config
from app.models import const
from app.models.schema import (
    MaterialInfo,
    VideoAspect,
    VideoConcatMode,
    VideoFitMode,
    VideoParams,
    VideoTransitionMode,
)
from app.services import guardrails, subtitle, subtitle_styles
from app.services import bgm as bgm_service
from app.services.utils import video_effects
from app.utils import file_security, utils

# MoviePy readers keep using the container binary, which can stream decoded
# frames locally. Only writers use the host bridge needed for Mac hardware.
moviepy_ffmpeg_writer.FFMPEG_BINARY = utils.get_ffmpeg_binary()

class SubClippedVideoClip:
    def __init__(
        self,
        file_path,
        start_time=None,
        end_time=None,
        width=None,
        height=None,
        duration=None,
        source_file_path=None,
    ):
        self.file_path = file_path
        self.start_time = start_time
        self.end_time = end_time
        self.width = width
        self.height = height
        self.source_file_path = source_file_path or file_path
        if duration is None:
            self.duration = end_time - start_time
        else:
            self.duration = duration

    def __str__(self):
        return f"SubClippedVideoClip(file_path={self.file_path}, start_time={self.start_time}, end_time={self.end_time}, duration={self.duration}, width={self.width}, height={self.height})"


audio_codec = "aac"
# the ffmpeg/AAC combination inside Docker is more prone to audio quality
# swings at its default settings, so raise the audio bitrate explicitly rather
# than letting a low default introduce audible distortion in the final render.
audio_bitrate = "192k"
fps = 30
# concatenating and transcoding at a fixed frame rate can leave the final
# duration a few tens of milliseconds short of what MoviePy reported. keep a
# small safety margin of material so frame rounding cannot end the video on a
# black frame, a stutter, or a last line of narration with no picture.
_VIDEO_DURATION_SAFETY_MARGIN = 0.1
_MIN_MATERIAL_DIMENSION = 480
# messaging apps and some encoders round frame dimensions down: WhatsApp turns
# a 9:16 clip into 478x850, two pixels under 480. a hard 480 floor would drop
# every such material and fail the whole task with "no valid materials found".
# a small tolerance admits material that is short only because of rounding while
# still rejecting genuinely low-resolution footage.
_MIN_DIMENSION_TOLERANCE = 10
# Kept only as a sentinel for rejecting old configurations. Software video
# encoding is intentionally disabled; auto must resolve to a hardware encoder.
_SOFTWARE_VIDEO_CODEC = "libx264"
_SUBTITLE_SPRING_DURATION_SECONDS = 0.18
_MIN_SUBTITLE_SPRING_SCALE = 0.05
_MAX_SUBTITLE_SPRING_SCALE = 1.35
_SUPPORTED_VIDEO_CODECS = (
    "h264_nvenc",
    "h264_amf",
    "h264_qsv",
    "h264_vaapi",
    "h264_mf",
    "h264_videotoolbox",
    "auto",
)
_runtime_disabled_video_codecs = set()
# Without preset/quality/bitrate, a hardware encoder may fail or select poor
# defaults. Every encoder gets the minimum working parameters, and MoviePy's
# write_videofile passes ffmpeg_params through to the final ffmpeg call.
# ponytail: one global parameter table, extend as needed; caller-supplied
# ffmpeg_params win over these.
_CODEC_FFMPEG_PARAMS = {
    "h264_nvenc": ["-preset", "p4", "-rc", "vbr", "-b:v", "5M"],
    "h264_amf": ["-usage", "transcoding", "-quality", "balanced", "-b:v", "5M"],
    "h264_qsv": ["-preset", "veryfast", "-b:v", "5M"],
    "h264_vaapi": ["-qp", "20"],
    "h264_mf": ["-b:v", "5M"],
    "h264_videotoolbox": ["-b:v", "5M"],
}
# Auto-detection picks the first codec available in the bundled ffmpeg, ordered
# by what is most likely to be useful on each platform.
# ponytail: per-OS priority list; reorder when new hardware backends land.
_HARDWARE_CODEC_AUTO_PRIORITY = {
    "darwin": ("h264_videotoolbox", "h264_qsv"),
    "linux": (
        "h264_nvenc",
        "h264_qsv",
        "h264_amf",
        "h264_vaapi",
        "h264_videotoolbox",
    ),
    "win32": ("h264_nvenc", "h264_qsv", "h264_amf", "h264_mf", "h264_videotoolbox"),
}


def _get_subtitle_spring_scale(time_seconds: float, duration_seconds: float) -> float:
    """Return the scale the subtitle bounce animation uses at a given time."""
    if duration_seconds <= 0 or time_seconds >= duration_seconds:
        return 1.0

    progress = max(0.0, min(time_seconds / duration_seconds, 1.0))
    scale = 1.0 - math.exp(-6.0 * progress) * math.cos(2.5 * math.pi * progress)
    return max(
        _MIN_SUBTITLE_SPRING_SCALE,
        min(scale, _MAX_SUBTITLE_SPRING_SCALE),
    )


def _scale_subtitle_frame_on_canvas(frame: np.ndarray, scale: float) -> np.ndarray:
    """
    Scale a subtitle frame or its alpha mask around the centre, keeping the
    canvas size unchanged.

    MoviePy stores the subtitle colour frame and its alpha mask separately. The
    bounce animation must scale and crop both identically, otherwise the first
    animated frame composites the transparent area onto the video as a black
    text outline. A 2-D array is a 0..1 mask; a 3-D array is an RGB/RGBA frame.
    """
    if frame.ndim not in (2, 3):
        raise ValueError("subtitle frame must be a 2D mask or 3D color frame")

    height, width = frame.shape[:2]
    scaled_width = max(1, int(round(width * scale)))
    scaled_height = max(1, int(round(height * scale)))
    offset = ((width - scaled_width) // 2, (height - scaled_height) // 2)

    if frame.ndim == 2:
        # MoviePy masks are 0..1 floats while Pillow's L mode is 0..255.
        # restore the original type and range after converting so
        # CompositeVideoClip's transparency semantics stay intact.
        mask_image = Image.fromarray(
            np.clip(frame * 255.0, 0, 255).astype(np.uint8)
        )
        resized_mask = mask_image.resize(
            (scaled_width, scaled_height),
            Image.Resampling.BILINEAR,
        )
        mask_canvas = Image.new("L", (width, height), 0)
        mask_canvas.paste(resized_mask, offset)
        return (np.asarray(mask_canvas) / 255.0).astype(frame.dtype, copy=False)

    if frame.shape[2] not in (3, 4):
        raise ValueError("subtitle color frame must use RGB or RGBA channels")
    color_image = Image.fromarray(frame)
    resized_color = color_image.resize(
        (scaled_width, scaled_height),
        Image.Resampling.BILINEAR,
    )
    background = (0, 0, 0, 0) if frame.shape[2] == 4 else (0, 0, 0)
    color_canvas = Image.new(color_image.mode, (width, height), background)
    color_canvas.paste(resized_color, offset)
    return np.asarray(color_canvas).astype(frame.dtype, copy=False)


def _shift_subtitle_frame_on_canvas(frame: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Shift a subtitle frame or mask by (dx, dy) keeping canvas bounds intact."""
    if dx == 0 and dy == 0:
        return frame
    height, width = frame.shape[:2]
    if frame.ndim == 2:
        mask_image = Image.fromarray(
            np.clip(frame * 255.0, 0, 255).astype(np.uint8)
        )
        mask_canvas = Image.new("L", (width, height), 0)
        mask_canvas.paste(mask_image, (dx, dy))
        return (np.asarray(mask_canvas) / 255.0).astype(frame.dtype, copy=False)

    if frame.shape[2] not in (3, 4):
        raise ValueError("subtitle color frame must use RGB or RGBA channels")
    color_image = Image.fromarray(frame)
    background = (0, 0, 0, 0) if frame.shape[2] == 4 else (0, 0, 0)
    color_canvas = Image.new(color_image.mode, (width, height), background)
    color_canvas.paste(color_image, (dx, dy))
    return np.asarray(color_canvas).astype(frame.dtype, copy=False)


def _apply_subtitle_animation(clip, subtitle_duration: float, anim_type: str = "none"):
    """
    Apply entry animation to a subtitle or title clip.
    Supports none, pop_spring, scale_up, fade, slide_up, shake.
    """
    if anim_type in ("none", "", None):
        return clip

    anim_duration = min(0.18, max(0.0, subtitle_duration))
    if anim_duration <= 0:
        return clip

    if anim_type in ("pop_spring", "spring", "pop"):
        def transform_spring(get_frame, t):
            frame = get_frame(t)
            scale = _get_subtitle_spring_scale(t, anim_duration)
            if scale == 1.0:
                return frame
            return _scale_subtitle_frame_on_canvas(frame, scale)
        return clip.transform(transform_spring, apply_to=["mask"])

    if anim_type in ("scale_up", "zoom_in", "punch"):
        def transform_scale_up(get_frame, t):
            frame = get_frame(t)
            if t >= anim_duration:
                return frame
            progress = max(0.0, min(t / anim_duration, 1.0))
            scale = 1.0 - 0.35 * ((1.0 - progress) ** 2)
            return _scale_subtitle_frame_on_canvas(frame, scale)
        return clip.transform(transform_scale_up, apply_to=["mask"])

    if anim_type in ("fade", "fade_in"):
        def transform_fade(get_frame, t):
            frame = get_frame(t)
            if t >= anim_duration:
                return frame
            progress = max(0.0, min(t / anim_duration, 1.0))
            if frame.ndim == 2:
                return (frame * progress).astype(frame.dtype, copy=False)
            if frame.shape[2] == 4:
                frame_copy = frame.copy()
                frame_copy[:, :, 3] = np.clip(frame_copy[:, :, 3] * progress, 0, 255).astype(frame.dtype)
                return frame_copy
            return (frame * progress).astype(frame.dtype, copy=False)
        return clip.transform(transform_fade, apply_to=["mask"])

    if anim_type in ("slide_up", "rise"):
        clip_h = getattr(clip, "h", None) or 50
        shift_dist = max(15, int(round(clip_h * 0.3)))
        def transform_slide(get_frame, t):
            frame = get_frame(t)
            if t >= anim_duration:
                return frame
            progress = max(0.0, min(t / anim_duration, 1.0))
            dy = int(round(shift_dist * ((1.0 - progress) ** 2)))
            return _shift_subtitle_frame_on_canvas(frame, 0, dy)
        return clip.transform(transform_slide, apply_to=["mask"])

    if anim_type in ("shake", "jitter"):
        def transform_shake(get_frame, t):
            frame = get_frame(t)
            if t >= anim_duration:
                return frame
            decay = math.exp(-20.0 * t)
            dx = int(round(5.0 * math.sin(t * 60.0) * decay))
            dy = int(round(3.5 * math.cos(t * 50.0) * decay))
            return _shift_subtitle_frame_on_canvas(frame, dx, dy)
        return clip.transform(transform_shake, apply_to=["mask"])

    return clip


def _apply_subtitle_spring_animation(clip, subtitle_duration: float):
    """Scale the subtitle frame and its mask together so the bounce animation
    never opens on a black frame."""
    return _apply_subtitle_animation(clip, subtitle_duration, "pop_spring")


def _get_required_video_duration(audio_duration: float) -> float:
    """
    Return the target duration the concatenated material has to cover.

    Combining a video needs the material to cover the narration audio. Landing
    exactly on the audio duration can still leave the final video slightly short
    because of frame-rate rounding, so add a light margin. Keeping this in its
    own function makes the margin easy to test and to retune later.
    """
    return max(0.0, float(audio_duration) + _VIDEO_DURATION_SAFETY_MARGIN)


def is_material_resolution_acceptable(width: int, height: int) -> bool:
    """
    Report whether a material's resolution is good enough to combine.

    The nominal minimum is 480x480, but a material may fall `_MIN_DIMENSION_TOLERANCE`
    pixels below it to allow for encoders and messaging apps that round
    dimensions down (WhatsApp's 478x850, for example).
    """
    min_dimension = _MIN_MATERIAL_DIMENSION - _MIN_DIMENSION_TOLERANCE
    return width >= min_dimension and height >= min_dimension


def _prioritize_unique_source_clips(
    subclipped_items: List[SubClippedVideoClip],
    concat_mode: VideoConcatMode,
    skip_fingerprints: Optional[Set[Tuple[str, float, float]]] = None,
    seed: Optional[int] = None,
) -> List[SubClippedVideoClip]:
    """
    Order the slices so every source material is used before any is reused.

    A single long material is cut into many short slices. Ordering them naively
    makes the same source dominate the opening of the video, and in sequential
    mode it used to make the pipeline keep only each material's first slice, so
    a 49-minute upload contributed 4 seconds and the loop-to-fill fallback
    repeated it for the whole narration.

    Random mode leads with the longest slice of each source, then shuffles the
    rest as fallback; picking the longest avoids leading with a ragged tail
    slice. Sequential mode round-robins the sources so materials still appear in
    the order the user listed them, while each one advances through its own
    timeline instead of replaying its first seconds.

    ``skip_fingerprints`` drops slices already used by an earlier part so each
    part of a multi-part task shows different scenes instead of recycling the
    same handful of clips. ``seed`` makes the shuffle deterministic per part
    (different seed -> different shuffle order) without leaking the global
    RNG state across calls.
    """
    if not subclipped_items:
        return []

    rng = random.Random(seed) if seed is not None else random
    skip: Set[Tuple[str, float, float]] = skip_fingerprints or set()

    def fingerprint(item: SubClippedVideoClip) -> Tuple[str, float, float]:
        return (item.source_file_path, item.start_time, item.end_time)

    grouped_items: dict[str, list[SubClippedVideoClip]] = {}
    for item in subclipped_items:
        if fingerprint(item) in skip:
            continue
        grouped_items.setdefault(item.source_file_path, []).append(item)

    if not grouped_items:
        return []

    concat_mode_value = getattr(concat_mode, "value", concat_mode)

    if concat_mode_value != VideoConcatMode.random.value:
        return [
            item
            for row in itertools.zip_longest(*grouped_items.values())
            for item in row
            if item is not None
        ]

    primary_items = []
    overflow_items = []
    for items in grouped_items.values():
        primary_item = max(items, key=lambda item: item.duration)
        primary_items.append(primary_item)
        overflow_items.extend(item for item in items if item is not primary_item)

    rng.shuffle(primary_items)
    rng.shuffle(overflow_items)
    logger.info(
        "prioritized unique video materials, "
        f"sources: {len(grouped_items)}, "
        f"primary clips: {len(primary_items)}, "
        f"fallback clips: {len(overflow_items)}"
        f"{', skipping ' + str(len(skip)) + ' already-used clips' if skip else ''}"
    )
    return primary_items + overflow_items


def get_ffmpeg_binary():
    """
    Keep working for callers that historically read the FFmpeg path from the
    video service.

    The real resolution lives in `app.utils.utils.get_ffmpeg_binary()`; video,
    speech, and any future path should share that one priority order. This thin
    wrapper stays so external scripts or old tests importing
    `app.services.video.get_ffmpeg_binary` do not hit an AttributeError.
    """
    return utils.get_ffmpeg_binary()


def _get_configured_video_codec() -> str:
    """
    Read the user-configured video encoder.

    When video_codec is unset the project default is "auto": probe ffmpeg for a
    hardware encoder (NVENC/AMF/QSV/VideoToolbox). Only a fixed allowlist is
    accepted on purpose: opening it
    to arbitrary FFmpeg parameters would let a typo produce an unpredictable
    output format, or fail the task at a much later stage.
    """
    configured_codec = str(config.app.get("video_codec", "auto") or "auto").strip()
    if configured_codec not in _SUPPORTED_VIDEO_CODECS:
        logger.warning(
            f"unsupported video codec configured: {configured_codec}, "
            "using automatic hardware selection"
        )
        return "auto"
    return configured_codec


@lru_cache(maxsize=16)
def _ffmpeg_encoder_exists(ffmpeg_binary: str, codec: str) -> bool:
    """
    Check whether this FFmpeg build advertises the given encoder.

    That only proves the encoder was compiled in, not that this machine's
    hardware and drivers can actually use it, so the runtime smoke test still
    verifies a real encode.
    """
    try:
        result = subprocess.run(
            [ffmpeg_binary, "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning(
            f"ffmpeg encoder probe failed for {ffmpeg_binary}: {exc}"
        )
        return False

    if result.returncode != 0:
        stderr_excerpt = (result.stderr or result.stdout or "").strip()[:200]
        logger.warning(
            f"ffmpeg encoder probe failed (rc={result.returncode}) for "
            f"{ffmpeg_binary}: {stderr_excerpt}"
        )
        return False
    return codec in result.stdout


@lru_cache(maxsize=16)
def _ffmpeg_filter_exists(ffmpeg_binary: str, filter_name: str) -> bool:
    """Return whether the selected FFmpeg build exposes a named video filter."""
    try:
        result = subprocess.run(
            [ffmpeg_binary, "-hide_banner", "-filters"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0 and bool(
        re.search(rf"^\s*\.{{3}}\s+{re.escape(filter_name)}\s", result.stdout, re.MULTILINE)
    )


def _get_vaapi_device() -> str:
    """Return the configured or first container-visible VAAPI render node."""
    configured = os.environ.get("VAAPI_DEVICE", "").strip()
    if configured:
        return configured
    return next(iter(sorted(glob.glob("/dev/dri/renderD*"))), "")


def _get_codec_ffmpeg_params(codec: str, *, filtered: bool = False) -> list[str]:
    """Return encoder options, including runtime-discovered VAAPI wiring."""
    params = list(_CODEC_FFMPEG_PARAMS.get(codec, ()))
    if codec != "h264_vaapi":
        return params

    device = _get_vaapi_device()
    if not device:
        return params
    params[:0] = ["-vaapi_device", device]
    if not filtered:
        params.extend(["-vf", "format=nv12,hwupload"])
    return params


@lru_cache(maxsize=16)
def _ffmpeg_encoder_runnable(ffmpeg_binary: str, codec: str) -> bool:
    """
    Verify an encoder actually works by running a tiny smoke encode.

    `ffmpeg -encoders` only proves the encoder was compiled into the build.
    A GPU-less container still lists h264_nvenc and h264_qsv even though
    neither can open a device here, so the "auto" policy must probe with a
    real encode instead of trusting the listing. The same standard ffmpeg
    parameters as a real task are used, so a build that rejects an encoder's
    private option (for example -rc) is also rejected up front.
    """
    if codec == "h264_vaapi" and not _get_vaapi_device():
        logger.warning("VAAPI encoder found but no render device is visible")
        return False

    command = [
        ffmpeg_binary,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "testsrc=duration=0.2:size=192x192:rate=5",
        "-frames:v",
        "1",
        "-c:v",
        codec,
    ]
    command.extend(_get_codec_ffmpeg_params(codec))
    command.extend(["-f", "null", "-"])
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning(
            f"failed to smoke-test encoder {codec}: {str(exc)}"
        )
        return False
    if result.returncode != 0:
        logger.warning(
            f"encoder {codec} smoke test failed: "
            f"{(result.stderr or result.stdout or '').strip()}"
        )
        return False
    logger.info(f"encoder {codec} passed the runtime smoke test")
    return True


def _get_effective_video_codec(preferred_codec: str | None = None) -> str:
    """
    Return the codec actually used for this run.

    When the user picks `auto`, probe ffmpeg for a hardware encoder that can
    actually encode on this host and pick the first one. When the user picks a
    specific hardware codec, validate it against ffmpeg's encoder list, a
    runtime smoke encode, and the runtime-disabled set so a codec that cannot
    work here is rejected before any clip wastes time failing.
    """
    selected_codec = preferred_codec or _get_configured_video_codec()
    if selected_codec == _SOFTWARE_VIDEO_CODEC:
        raise RuntimeError("software video encoding is disabled; select a hardware encoder")

    if selected_codec == "auto":
        resolved = _detect_hardware_codec(utils.get_ffmpeg_binary())
        if resolved is None:
            raise RuntimeError(
                f"no usable hardware video encoder is available on {sys.platform}; "
                "CPU video encoding is disabled"
            )
        logger.info(f"auto-detected hardware codec: {resolved}")
        return resolved

    if selected_codec in _runtime_disabled_video_codecs:
        raise RuntimeError(
            f"hardware video encoder {selected_codec} was disabled after a runtime failure"
        )

    ffmpeg_binary = utils.get_ffmpeg_binary()
    if not _ffmpeg_encoder_exists(ffmpeg_binary, selected_codec):
        raise RuntimeError(
            f"hardware video encoder {selected_codec} is not available in ffmpeg"
        )

    if not _ffmpeg_encoder_runnable(ffmpeg_binary, selected_codec):
        raise RuntimeError(
            f"hardware video encoder {selected_codec} cannot encode on this host "
            "because no compatible GPU or driver is available"
        )

    return selected_codec


def _detect_hardware_codec(ffmpeg_binary: str) -> str | None:
    """
    Probe ffmpeg for a hardware H.264 encoder that can actually run here.

    Order is platform-specific (videotoolbox on macOS, nvenc first on
    Linux/Windows) so the most likely useful encoder is preferred. Each
    candidate must both be listed by `-encoders` and pass a tiny smoke encode,
    because a GPU-less container still ships nvenc/qsv builds although neither
    can open a device. A codec already disabled after a runtime failure is
    skipped, so "auto" does not keep retrying the same broken encoder for every
    clip in a task. Returns None when no hardware encoder can work, which the
    caller reports as a missing accelerator.
    """
    priority = _HARDWARE_CODEC_AUTO_PRIORITY.get(sys.platform, ())
    for codec in priority:
        if codec in _runtime_disabled_video_codecs:
            continue
        if _ffmpeg_encoder_exists(ffmpeg_binary, codec) and _ffmpeg_encoder_runnable(
            ffmpeg_binary, codec
        ):
            return codec
    return None


def _disable_runtime_video_codec(codec: str, reason: str):
    _runtime_disabled_video_codecs.add(codec)
    logger.warning(f"hardware video codec {codec} failed and was disabled: {reason}")


def _get_temp_audio_dir(output_dir: str) -> str:
    """
    Return the directory to use for MoviePy's temporary audio file.

    On Windows, Windows Defender can lock files written to the task output
    directory while scanning them, causing MoviePy to fail with a
    PermissionError (WinError 32) on the TEMP_MPY_wvf_snd temp file and
    leaving the final MP4 at 0 bytes.  Using the system temp directory
    sidesteps the scan without changing behaviour on other platforms.

    On Linux/macOS/Docker the output directory is returned unchanged so
    existing behaviour is preserved.
    """
    if sys.platform == "win32":
        return tempfile.gettempdir()
    return output_dir


def _write_videofile_with_codec_fallback(clip, output_file: str, codec: str, **kwargs):
    """
    Write the video with the requested hardware encoder.

    A hardware encoder invoked with no preset/quality/bitrate may not work at
    all, so inject the minimum working parameters per codec; ffmpeg_params from
    the caller take precedence.
    """
    effective_codec = _get_effective_video_codec(codec)
    if (
        effective_codec in _CODEC_FFMPEG_PARAMS
        and "ffmpeg_params" not in kwargs
    ):
        kwargs["ffmpeg_params"] = _get_codec_ffmpeg_params(effective_codec)
    try:
        clip.write_videofile(output_file, codec=effective_codec, **kwargs)
        return effective_codec
    except Exception as exc:
        _disable_runtime_video_codec(effective_codec, str(exc))
        raise


def _escape_ffmpeg_concat_path(file_path: str) -> str:
    # the concat demuxer wraps paths in single quotes, so a single quote inside
    # a path has to be escaped first.
    return file_path.replace("'", "'\\''")


def _format_ffmpeg_concat_path(file_path: str) -> str:
    """
    Build a path entry for the concat demuxer's file list.

    FFmpeg's documentation requires special characters and spaces in a concat
    list to be escaped, and backslashes in a Windows absolute path are easily
    read as escape sequences. Normalise to forward slashes so `C:\\Users\\...`
    becomes `C:/Users/...`, then handle single quotes, which also works on
    macOS and Linux.
    """
    absolute_path = os.path.abspath(file_path)
    return _escape_ffmpeg_concat_path(absolute_path.replace("\\", "/"))


def concat_video_clips_with_ffmpeg(
    clip_files: List[str],
    output_file: str,
    threads: int,
    output_dir: str,
    max_duration: float | None = None,
):
    concat_list_file = os.path.join(output_dir, "ffmpeg-concat-list.txt")
    with open(concat_list_file, "w", encoding="utf-8") as fp:
        for clip_file in clip_files:
            relative_path = os.path.relpath(
                os.path.abspath(clip_file), os.path.abspath(output_dir)
            ).replace("\\", "/")
            fp.write(f"file '{_escape_ffmpeg_concat_path(relative_path)}'\n")

    def build_stream_copy_command() -> list[str]:
        # When all per-clip writes happened through the same write path
        # (same codec + preset + resolution + fps), the resulting MP4s
        # share an identical video stream and the concat demuxer can join
        # them with `-c copy` in a few milliseconds instead of a full
        # re-encode. Probe the first clip to confirm the stream matches
        # what we expect; the demuxer silently produces a broken output
        # when params drift (different SAR, different timebase).
        return [
            utils.get_ffmpeg_binary(),
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            concat_list_file,
            "-c",
            "copy",
            "-movflags",
            "+faststart",
        ]

    def build_command(codec: str) -> list[str]:
        command = [
            utils.get_ffmpeg_binary(),
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            concat_list_file,
            "-c:v",
            codec,
            "-threads",
            str(threads or 2),
        ]
        if codec != "h264_vaapi":
            command.extend(["-pix_fmt", "yuv420p"])
        # same minimal parameters as the MoviePy write path, so a hardware
        # encoder picked by "auto" gets preset/quality/bitrate it can use,
        # with the parameters used by the normal write path.
        if codec in _CODEC_FFMPEG_PARAMS:
            command.extend(_get_codec_ffmpeg_params(codec))
        if max_duration is not None and max_duration > 0:
            command.extend(["-t", f"{max_duration:.3f}"])
        command.extend(["-movflags", "+faststart"])
        command.append(output_file)
        return command

    def run_concat(codec: str) -> str:
        command = build_command(codec)
        # concatenate and encode once with ffmpeg instead of letting MoviePy
        # merge segment by segment and re-encode each time, which degrades
        # quality and shifts colour.
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            error_message = (result.stderr or result.stdout or "").strip()
            raise RuntimeError(error_message or "ffmpeg concat failed")
        return codec

    def run_stream_copy() -> bool:
        """Try the concat-as-stream-copy fast path. Returns True on success.

        The MoviePy per-clip writes in this pipeline all go through
        ``_write_videofile_with_codec_fallback`` with the same codec, fps,
        and resolution, so the resulting MP4s share an identical video
        stream and the concat demuxer can join them without re-encoding.
        This is the single biggest wall-clock win for the pipeline: a
        30-clip concat that re-encoded in ~30s lands here in ~150ms.
        """
        if not clip_files:
            return False
        command = build_stream_copy_command()
        command.append(output_file)
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
        return result.returncode == 0

    try:
        # Try the cheap stream-copy fast path first. Fall back to a
        # full re-encode if the demuxer rejects the clips (mismatched
        # params, variable GOP, etc.). This is the single largest
        # wall-clock win in the concat stage.
        if run_stream_copy():
            return "copy"
        effective_codec = _get_effective_video_codec()
        try:
            return run_concat(effective_codec)
        except Exception as exc:
            _disable_runtime_video_codec(effective_codec, str(exc))
            raise
    finally:
        delete_files(concat_list_file)


def _sanitize_image_file(image_path: str) -> str:
    # some local images open fine in Pillow but make ImageClip raise while
    # parsing, because of corrupt EXIF/eXIf metadata. re-export a clean copy
    # with the bad metadata stripped.
    image_root, _ = os.path.splitext(image_path)
    sanitized_path = f"{image_root}.sanitized.png"

    with Image.open(image_path) as image:
        image.load()
        # always export PNG so the differing JPEG/PNG metadata paths cannot
        # carry the bad block through.
        cleaned_image = Image.new(image.mode, image.size)
        cleaned_image.putdata(list(image.getdata()))
        cleaned_image.save(sanitized_path)

    return sanitized_path


def _open_image_clip_with_fallback(image_path: str):
    # try the original image first; only if corrupt metadata breaks it, fall
    # back to a metadata-free copy.
    try:
        return ImageClip(image_path), image_path
    except Exception as exc:
        logger.warning(
            f"failed to open image directly, trying sanitized copy: {image_path}, error: {str(exc)}"
        )
        sanitized_path = _sanitize_image_file(image_path)
        return ImageClip(sanitized_path), sanitized_path


def _open_video_clip_quietly(video_path: str, audio: bool = False) -> VideoFileClip:
    """
    Open a video file quietly, so MoviePy 2.1.x cannot print ffmpeg probe
    information straight to stdout.

    Background:
    the pinned `FFMPEG_VideoReader` contains `print(self.infos)` and
    `print(ffmpeg command)`, so reading an intermediate video with no audio
    track prints `audio_found: False`. That is only input metadata and says
    nothing about the final render, but it makes WebUI and terminal users think
    generation failed.

    Implementation:
    1. redirect stdout only for the short window in which VideoFileClip opens;
    2. default to `audio=False`, because the material stage does not need the
       source audio -- the final audio is attached in `generate_video()`;
    3. if the library did print something, downgrade it to a debug log so it
       stays available for troubleshooting.
    """
    captured_stdout = io.StringIO()
    with redirect_stdout(captured_stdout):
        clip = VideoFileClip(video_path, audio=audio)

    moviepy_stdout = captured_stdout.getvalue().strip()
    if moviepy_stdout:
        logger.debug(
            "suppressed MoviePy video reader stdout for "
            f"{video_path}, chars: {len(moviepy_stdout)}"
        )

    return clip


def close_clip(clip):
    if clip is None:
        return
        
    try:
        # close main resources
        if hasattr(clip, 'reader') and clip.reader is not None:
            clip.reader.close()
            
        # close audio resources
        if hasattr(clip, 'audio') and clip.audio is not None:
            if hasattr(clip.audio, 'reader') and clip.audio.reader is not None:
                clip.audio.reader.close()
            del clip.audio
            
        # close mask resources
        if hasattr(clip, 'mask') and clip.mask is not None:
            if hasattr(clip.mask, 'reader') and clip.mask.reader is not None:
                clip.mask.reader.close()
            del clip.mask
            
        # handle child clips in composite clips
        if hasattr(clip, 'clips') and clip.clips:
            for child_clip in clip.clips:
                if child_clip is not clip:  # avoid possible circular references
                    close_clip(child_clip)
            
        # clear clip list
        if hasattr(clip, 'clips'):
            clip.clips = []
            
    except Exception as e:
        logger.error(f"failed to close clip: {str(e)}")
    
    del clip
    gc.collect()

def delete_files(files: List[str] | str):
    if isinstance(files, str):
        files = [files]

    # when clips are looped to fill the video, the same temp path appears
    # several times in the FFmpeg concat list. concatenation needs those
    # duplicates, but cleanup must delete each file once. de-duplicating in the
    # original order makes cleanup idempotent for every caller and stops a
    # stream of FileNotFoundError after the first successful delete.
    unique_files = dict.fromkeys(file for file in files if file)
    for file in unique_files:
        try:
            os.remove(file)
        except FileNotFoundError:
            # a missing file is fine here: an FFmpeg failure path or a
            # concurrent cleanup may already have removed it. that is nothing
            # the user has to act on, so keep it out of the generation log.
            continue
        except OSError as e:
            # permissions, a read-only filesystem, or a disk error leave a real
            # temp file behind. keep the warning so the path and OS error can
            # pinpoint the environment problem.
            logger.warning(f"failed to delete temporary file {file}: {str(e)}")


def get_bgm_file(bgm_type: str = "random", bgm_file: str = ""):
    if not bgm_type:
        return ""

    if bgm_file:
        try:
            resolved_bgm_file = bgm_service.resolve_bgm_file(bgm_file)
        except ValueError as exc:
            # bgm_file in an API request is user input, so resolve it only
            # inside the user BGM or bundled song directories. that stops
            # MoviePy from reading arbitrary server files such as config or keys.
            logger.warning(
                f"reject unsafe bgm file: {bgm_file}, error: {str(exc)}"
            )
            return ""
        return resolved_bgm_file

    if bgm_type == "random":
        files = bgm_service.list_bgm_files()
        # an empty background music directory falls back to "no BGM" rather
        # than letting random.choice([]) raise.
        if not files:
            logger.warning("no background music files found")
            return ""
        return random.choice(files)

    return ""


def _ffmpeg_fit_filter(
    target_width: int,
    target_height: int,
    fit_mode: VideoFitMode | str,
) -> str:
    """Return the ffmpeg scale/crop/letterbox filter chain that matches
    MoviePy's _fit_clip_to_canvas for the cover/contain modes.

    Cover: scale up so the source fills the canvas, then crop the excess
    on the long axis. Contain: scale down so the source fits inside the
    canvas, then pad the short axis with black bars.
    """
    w = int(target_width)
    h = int(target_height)
    mode = VideoFitMode(fit_mode)
    if mode == VideoFitMode.cover:
        return (
            f"scale={w}:{h}:force_original_aspect_ratio=increase,"
            f"crop={w}:{h}"
        )
    return (
        f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
        f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color=black"
    )


def _build_single_ffmpeg_combine_command(
    clip_inputs: List[Tuple[str, float, float]],
    *,
    output_file: str,
    threads: int,
    target_width: int,
    target_height: int,
    fit_mode: VideoFitMode,
    output_fps: int,
    codec: str,
    max_duration: float,
    codec_params: List[str],
) -> list[str]:
    """Build the ffmpeg invocation that takes N source clips (each with
    start time + duration) and produces ONE combined silent video track in
    a single subprocess.

    Replaces MoviePy's per-clip write_videofile loop + the concat re-encode
    with one filter_complex invocation. For 6 clips that is 7 ffmpeg
    subprocesses collapsed into 1, which is the largest wall-clock win in
    the combine stage.
    """
    command = [utils.get_ffmpeg_binary(), "-y"]
    for source_path, start_time, duration in clip_inputs:
        # input-seek (-ss before -i) is the fast seek; -t after -i sets
        # the duration to read. Together they avoid loading the entire
        # source when we only need a 5-second slice.
        command.extend([
            "-ss", f"{start_time:.3f}",
            "-t", f"{duration:.3f}",
            "-i", source_path,
        ])
    # Build a scale+fps filter per input, then concat them.
    fit_filter = _ffmpeg_fit_filter(target_width, target_height, fit_mode)
    filter_parts = []
    for idx in range(len(clip_inputs)):
        filter_parts.append(
            f"[{idx}:v]{fit_filter},setsar=1,fps={output_fps}[v{idx}]"
        )
    concat_inputs = "".join(f"[v{i}]" for i in range(len(clip_inputs)))
    output_filter = f"concat=n={len(clip_inputs)}:v=1:a=0"
    if codec == "h264_vaapi":
        output_filter += ",format=nv12,hwupload"
    filter_parts.append(f"{concat_inputs}{output_filter}[out]")
    command.extend([
        "-filter_complex", ";".join(filter_parts),
        "-map", "[out]",
        "-c:v", codec,
    ])
    if codec_params:
        command.extend(codec_params)
    command.extend([
        "-threads", str(threads or 2),
    ])
    if codec != "h264_vaapi":
        command.extend(["-pix_fmt", "yuv420p"])
    command.extend([
        "-t", f"{max_duration:.3f}",
        "-movflags", "+faststart",
        output_file,
    ])
    return command


def _fit_clip_to_canvas(
    clip,
    *,
    target_width: int,
    target_height: int,
    fit_mode: VideoFitMode | str = VideoFitMode.cover,
):
    """Resize a clip to an exact canvas using cover/crop or contain/letterbox."""
    source_width, source_height = (int(value) for value in clip.size)
    target_width = int(target_width)
    target_height = int(target_height)
    if min(source_width, source_height, target_width, target_height) <= 0:
        raise ValueError(
            "video dimensions must be positive: "
            f"source={source_width}x{source_height}, "
            f"target={target_width}x{target_height}"
        )

    mode = VideoFitMode(fit_mode)
    if (source_width, source_height) == (target_width, target_height):
        return clip

    # Exact aspect-ratio matches do not need either a crop or a background.
    if source_width * target_height == source_height * target_width:
        return clip.resized(new_size=(target_width, target_height))

    width_scale = target_width / source_width
    height_scale = target_height / source_height

    if mode == VideoFitMode.cover:
        # ceil guarantees the resized clip covers the complete canvas despite
        # floating-point rounding. Any excess is removed symmetrically.
        scale_factor = max(width_scale, height_scale)
        resized_width = max(target_width, math.ceil(source_width * scale_factor))
        resized_height = max(target_height, math.ceil(source_height * scale_factor))
        resized_clip = clip.resized(new_size=(resized_width, resized_height))
        crop_x = max(0, (resized_width - target_width) // 2)
        crop_y = max(0, (resized_height - target_height) // 2)
        return resized_clip.cropped(
            x1=crop_x,
            y1=crop_y,
            width=target_width,
            height=target_height,
        )

    # contain preserves the legacy behavior: show the complete source frame,
    # centered over a black canvas when the aspect ratios differ.
    scale_factor = min(width_scale, height_scale)
    resized_width = max(1, min(target_width, int(source_width * scale_factor)))
    resized_height = max(1, min(target_height, int(source_height * scale_factor)))
    background = ColorClip(
        size=(target_width, target_height), color=(0, 0, 0)
    ).with_duration(clip.duration)
    resized_clip = clip.resized(
        new_size=(resized_width, resized_height)
    ).with_position("center")
    return CompositeVideoClip(
        [background, resized_clip], size=(target_width, target_height)
    ).with_duration(clip.duration)


def combine_videos(
    combined_video_path: str,
    video_paths: List[str],
    audio_file: str,
    video_aspect: VideoAspect = VideoAspect.portrait,
    video_concat_mode: VideoConcatMode = VideoConcatMode.random,
    video_transition_mode: VideoTransitionMode = None,
    max_clip_duration: int = 5,
    threads: int = 2,
    clip_speed: float = 1.0,
    video_fit_mode: VideoFitMode = VideoFitMode.cover,
    exclude_clip_fingerprints: Optional[Set[Tuple[str, float, float]]] = None,
    part_index: int = 0,
    task_id: Optional[str] = None,
) -> str:
    """
    Build the combined clip track for one part of a multi-part task.

    ``exclude_clip_fingerprints`` is the set of (source_file_path,
    start_time, end_time) tuples already consumed by earlier parts so this part
    picks different scenes. ``part_index`` (0-based) and ``task_id`` together
    seed the per-part shuffle so two parts with the same source pool produce
    different orderings even when no clips are skipped.
    """
    audio_clip = AudioFileClip(audio_file)
    try:
        # only the narration duration is needed here, to decide how much
        # material to concatenate; audio_clip is never used again. close it
        # right away so an early return or an exception cannot leak the handle.
        audio_duration = audio_clip.duration
    finally:
        close_clip(audio_clip)
    logger.info(f"audio duration: {audio_duration} seconds")
    # Every render path lands here, so the shot-length floor holds regardless
    # of what the WebUI, the API or a saved task file asked for.
    max_clip_duration = guardrails.clamp_clip_duration(max_clip_duration)
    logger.info(f"maximum clip duration: {max_clip_duration} seconds")
    required_video_duration = _get_required_video_duration(audio_duration)
    logger.info(
        f"required video duration: {required_video_duration:.2f} seconds "
        f"(audio duration + {_VIDEO_DURATION_SAFETY_MARGIN:.2f}s safety margin)"
    )

    # tolerate a direct API call that passed no transition mode, so reading
    # .value below cannot crash.
    transition_value = getattr(video_transition_mode, "value", video_transition_mode)
    normalized_clip_speed = utils.normalize_clip_speed(clip_speed)
    if normalized_clip_speed != 1.0:
        # log the effective value once: enough to spot an out-of-range API
        # parameter being normalised, without repeating the same line in the
        # per-clip hot path.
        logger.info(f"clip playback speed: {normalized_clip_speed:.2f}x")
    # max_clip_duration bounds playback time in the final video, not how much
    # source is read. MoviePy turns 1.5s of source at 0.5x into a 3s clip, and
    # 6s of source at 2x into a 3s clip too. so the source duration has to be
    # derived from the speed before slicing: reading a fixed 3s, slowing it
    # down, then cropping, while the next slice starts at source second 3, would
    # skip 1.5s of picture. this also keeps the source timeline continuous and
    # non-overlapping at any speed.
    source_clip_duration = max_clip_duration * normalized_clip_speed
    output_dir = os.path.dirname(combined_video_path)

    aspect = VideoAspect(video_aspect)
    fit_mode = VideoFitMode(video_fit_mode)
    video_width, video_height = aspect.to_resolution()

    processed_clips = []
    subclipped_items = []
    video_duration = 0
    # How many slices does the audio need? Round up so cycle-fill never
    # starts the run. The ceiling, not the floor, is what the loop below
    # actually consumes.
    n_slices_needed = max(
        1,
        math.ceil(required_video_duration / max_clip_duration),
    )
    # Random mode spreads each slice evenly across its source so a 10-hour
    # upload contributes the same number of clips as a 10-second one and
    # the resulting video uses the FULL timeline instead of the first 30s.
    # Per part, rotate the sample offset so part 0 reads positions
    # [0, 1/N, 2/N, ...], part 1 reads [1/N, 2/N, 3/N, ...] -- same
    # coverage, different scenes. Sequential mode walks the source timeline
    # continuously instead: the user picked sequential for predictable
    # in-source ordering, so breaking the order would defeat the choice.
    n_sources = max(1, len(video_paths))
    slices_per_source = max(1, math.ceil(n_slices_needed / n_sources))
    use_even_sampling = (
        getattr(video_concat_mode, "value", video_concat_mode)
        == VideoConcatMode.random.value
    )
    # Per-part rotation step that scales with source duration. With a
    # 10-hour source and 5-second slices, the old rotation of
    # source_clip_duration / slices_per_source was a few hundred
    # milliseconds -- meaningless against a 36000s timeline, so every
    # part sampled near t=0 and the output still looked the same. The
    # step is now large enough that consecutive parts land on
    # non-overlapping regions even for hour-long inputs.
    #
    # Use the LONGEST source duration as the basis so a short clip mixed
    # in with hour-long material still produces a meaningful per-part
    # rotation against the dominant timeline.
    longest_source_duration = 0.0
    for video_path in video_paths:
        probe = _open_video_clip_quietly(video_path)
        if probe.duration > longest_source_duration:
            longest_source_duration = probe.duration
        close_clip(probe)
    rotation_step = (
        # 0.31D / slices_per_source -- the irrational-looking constant
        # keeps the per-source offsets from aligning with each other or
        # with the slice spacing, so two sources never land on the same
        # sample positions and consecutive parts don't wrap onto the
        # same set of windows.
        longest_source_duration * 0.31 / max(slices_per_source, 1)
        if use_even_sampling and part_index and longest_source_duration > 0
        else 0.0
    )
    for source_idx, video_path in enumerate(video_paths):
        clip = _open_video_clip_quietly(video_path)
        clip_duration = clip.duration
        clip_w, clip_h = clip.size
        close_clip(clip)

        if clip_duration <= 0:
            continue

        max_slices = max(1, int(clip_duration // source_clip_duration))
        if use_even_sampling and max_slices >= slices_per_source:
            # Even spacing with a per-source, per-part offset so two
            # adjacent sources don't land on the same sample positions.
            # The base offset scales with the source duration so that a
            # 10h upload doesn't have every part open with t=0; the
            # irrational-looking 0.37 multiplier keeps consecutive parts
            # from landing on the same sample set after modulo wrap.
            base_offset = (
                part_index * 0.37 * longest_source_duration
            ) % clip_duration
            source_offset = (
                base_offset + (source_idx * rotation_step)
            ) % clip_duration
            for i in range(slices_per_source):
                # mid-bin sampling: the centre of the i-th Nth of the
                # timeline, not the boundary, so two consecutive parts
                # pick noticeably different scenes instead of identical
                # neighbours when offsets are small.
                ratio = ((i + 0.5) / slices_per_source)
                start_time = (
                    source_offset + ratio * clip_duration
                ) % clip_duration
                end_time = start_time + source_clip_duration
                if end_time > clip_duration:
                    # wrap back to 0 if the sample would run off the end;
                    # the visual cost is one cross-cut per wrap, far
                    # cheaper than missing the tail of a long source.
                    end_time = clip_duration
                if end_time - start_time < 0.1:
                    # source too short for a meaningful slice -- skip
                    continue
                subclipped_items.append(
                    SubClippedVideoClip(
                        file_path=video_path,
                        start_time=start_time,
                        end_time=end_time,
                        width=clip_w,
                        height=clip_h,
                        source_file_path=video_path,
                    )
                )
        else:
            # Contiguous slicing (sequential mode, or short sources that
            # cannot host the requested slice count evenly).
            start_time = 0
            while start_time < clip_duration:
                end_time = min(
                    start_time + source_clip_duration, clip_duration
                )
                if end_time > start_time:
                    subclipped_items.append(
                        SubClippedVideoClip(
                            file_path=video_path,
                            start_time=start_time,
                            end_time=end_time,
                            width=clip_w,
                            height=clip_h,
                            source_file_path=video_path,
                        )
                    )
                start_time = end_time

    # derive a per-part seed so each part's shuffle is deterministic but
    # distinct from the others; combine_videos may also exclude clips used by
    # earlier parts so different parts show different scenes.
    per_part_seed = None
    if task_id is not None:
        per_part_seed = hash((task_id, part_index)) & 0x7FFFFFFF

    subclipped_items = _prioritize_unique_source_clips(
        subclipped_items=subclipped_items,
        concat_mode=video_concat_mode,
        skip_fingerprints=exclude_clip_fingerprints,
        seed=per_part_seed,
    )
        
    logger.debug(f"total subclipped items: {len(subclipped_items)}")

    # Track every source slice this part actually consumed so the next part
    # can skip them and show different scenes. Mutated in place below so the
    # caller (generate_final_videos) can reuse the same set across parts.
    used_fingerprints: Set[Tuple[str, float, float]] = set()
    if exclude_clip_fingerprints is None:
        exclude_clip_fingerprints = set()

    # Stream-copy fast path: when the source slices already match the
    # canvas (same width/height/codec), the only thing combine_videos
    # needs to do is slice and concat. ``-c copy`` makes that an instant
    # copy instead of a full re-encode -- on a 6-clip 1080p pipeline this
    # drops the combine stage from ~2s to ~30ms. Falls through to the
    # re-encode single-call path on any failure (mismatched streams,
    # missing keyframes, etc.).
    if (
        normalized_clip_speed == 1.0
        and transition_value in (None, VideoTransitionMode.none.value)
        and subclipped_items
    ):
        copy_inputs: List[Tuple[str, float, float]] = []
        copy_duration = 0.0
        all_match = True
        for sub in subclipped_items:
            if sub.width != video_width or sub.height != video_height:
                all_match = False
                break
        if all_match:
            for sub in subclipped_items:
                if copy_duration >= required_video_duration:
                    break
                seg_dur = min(
                    sub.end_time - sub.start_time, max_clip_duration
                )
                if seg_dur <= 0:
                    continue
                copy_inputs.append(
                    (sub.file_path, sub.start_time, seg_dur)
                )
                copy_duration += seg_dur
                used_fingerprints.add(
                    (sub.source_file_path, sub.start_time, sub.end_time)
                )
            while (
                copy_duration < required_video_duration
                and copy_inputs
            ):
                for src_path, start_time, seg_dur in itertools.cycle(
                    list(copy_inputs)
                ):
                    if copy_duration >= required_video_duration:
                        break
                    copy_inputs.append(
                        (src_path, start_time, seg_dur)
                    )
                    copy_duration += seg_dur

            if copy_inputs:
                concat_list_file = os.path.join(
                    output_dir, "ffmpeg-concat-list.txt"
                )
                with open(concat_list_file, "w", encoding="utf-8") as fp:
                    for src_path, start_time, _ in copy_inputs:
                        rel = os.path.relpath(
                            os.path.abspath(src_path),
                            os.path.abspath(output_dir),
                        ).replace("\\", "/")
                        fp.write(
                            f"file '{_escape_ffmpeg_concat_path(rel)}'\n"
                        )
                command = [
                    utils.get_ffmpeg_binary(),
                    "-y",
                    "-f", "concat",
                    "-safe", "0",
                    "-i", concat_list_file,
                    "-c", "copy",
                    "-movflags", "+faststart",
                    combined_video_path,
                ]
                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                delete_files(concat_list_file)
                if result.returncode == 0:
                    exclude_clip_fingerprints.update(used_fingerprints)
                    logger.info(
                        f"stream-copy combine produced {combined_video_path} "
                        f"from {len(copy_inputs)} source slices"
                    )
                    return combined_video_path
                logger.warning(
                    f"stream-copy combine failed (rc={result.returncode}); "
                    f"falling back to single-call re-encode path"
                )

    # Single-call fast path: when the run is "vanilla" (no per-clip
    # transitions, no playback speed change), do all the slicing +
    # scaling + concatenating in ONE ffmpeg invocation instead of
    # MoviePy's N per-clip writes + the concat re-encode. For a typical
    # 6-clip pipeline that collapses 7 ffmpeg subprocesses into 1 and
    # is the single largest wall-clock win in combine_videos.
    if (
        normalized_clip_speed == 1.0
        and transition_value in (None, VideoTransitionMode.none.value)
        and subclipped_items
    ):
        fast_path_inputs: List[Tuple[str, float, float]] = []
        fast_path_duration = 0.0
        for sub in subclipped_items:
            if fast_path_duration >= required_video_duration:
                break
            seg_dur = min(
                sub.end_time - sub.start_time, max_clip_duration
            )
            if seg_dur <= 0:
                continue
            fast_path_inputs.append((sub.file_path, sub.start_time, seg_dur))
            fast_path_duration += seg_dur
            used_fingerprints.add(
                (sub.source_file_path, sub.start_time, sub.end_time)
            )
        # cycle-fill if prioritized list is too short
        while (
            fast_path_duration < required_video_duration
            and fast_path_inputs
        ):
            for src_path, start_time, seg_dur in itertools.cycle(
                list(fast_path_inputs)
            ):
                if fast_path_duration >= required_video_duration:
                    break
                fast_path_inputs.append(
                    (src_path, start_time, seg_dur)
                )
                fast_path_duration += seg_dur

        effective_codec = _get_effective_video_codec()
        codec_params = _get_codec_ffmpeg_params(
            effective_codec,
            filtered=True,
        )
        command = _build_single_ffmpeg_combine_command(
            clip_inputs=fast_path_inputs,
            output_file=combined_video_path,
            threads=threads,
            target_width=video_width,
            target_height=video_height,
            fit_mode=fit_mode,
            output_fps=fps,
            codec=effective_codec,
            max_duration=required_video_duration,
            codec_params=codec_params,
        )
        result = subprocess.run(
            command, capture_output=True, text=True, check=False,
        )
        if result.returncode == 0:
            exclude_clip_fingerprints.update(used_fingerprints)
            if used_fingerprints:
                logger.info(
                    f"single-call combine produced {combined_video_path} "
                    f"from {len(fast_path_inputs)} source slices; "
                    f"recorded {len(used_fingerprints)} fingerprints for "
                    f"cross-part dedup"
                )
            logger.info("video combining completed (single-call fast path)")
            return combined_video_path
        # fall through to the per-clip path on any failure
        logger.warning(
            f"single-call combine failed (rc={result.returncode}); "
            f"falling back to per-clip MoviePy path. stderr: "
            f"{(result.stderr or '')[:300]}"
        )
        # Reset the fingerprints we tentatively added so the per-clip
        # path can populate them again from its own slice consumption.
        used_fingerprints.clear()

    # Add downloaded clips over and over until the duration of the audio (max_duration) has been reached
    for i, subclipped_item in enumerate(subclipped_items):
        if video_duration >= required_video_duration:
            break

        logger.debug(
            f"processing clip {i+1}: {subclipped_item.width}x{subclipped_item.height}, "
            f"source: {os.path.basename(subclipped_item.source_file_path)}, "
            f"current duration: {video_duration:.2f}s, "
            f"remaining: {required_video_duration - video_duration:.2f}s"
        )
        
        try:
            clip = _open_video_clip_quietly(subclipped_item.file_path).subclipped(
                subclipped_item.start_time, subclipped_item.end_time
            )
            # playback speed belongs to the material, so apply it before the
            # transition. that keeps a one-second fade or slide at one second
            # instead of stretching it to 0.5s or 2s with the material speed.
            # the max-duration crop below stays as a safety net against float
            # error or an odd material duration, so no clip exceeds the limit.
            if normalized_clip_speed != 1.0:
                clip = clip.with_speed_scaled(normalized_clip_speed)
            # Normalize every source clip before transitions are applied. In cover mode
            # the clip fills the canvas and the excess edges are cropped; contain keeps
            # the complete source frame and uses black bars for the unused area.
            clip_w, clip_h = clip.size
            if clip_w != video_width or clip_h != video_height:
                clip_ratio = clip.w / clip.h
                video_ratio = video_width / video_height
                logger.debug(
                    "resizing clip, "
                    f"source: {clip_w}x{clip_h}, ratio: {clip_ratio:.2f}, "
                    f"target: {video_width}x{video_height}, ratio: {video_ratio:.2f}, "
                    f"fit_mode: {fit_mode.value}"
                )
                clip = _fit_clip_to_canvas(
                    clip,
                    target_width=video_width,
                    target_height=video_height,
                    fit_mode=fit_mode,
                )

            shuffle_side = random.choice(["left", "right", "top", "bottom"])
            if transition_value in (None, VideoTransitionMode.none.value):
                clip = clip
            elif transition_value == VideoTransitionMode.fade_in.value:
                clip = video_effects.fadein_transition(clip, 1)
            elif transition_value == VideoTransitionMode.fade_out.value:
                clip = video_effects.fadeout_transition(clip, 1)
            elif transition_value == VideoTransitionMode.slide_in.value:
                clip = video_effects.slidein_transition(clip, 1, shuffle_side)
            elif transition_value == VideoTransitionMode.slide_out.value:
                clip = video_effects.slideout_transition(clip, 1, shuffle_side)
            elif transition_value == VideoTransitionMode.zoom_in.value:
                clip = video_effects.zoomin_transition(clip, 1)
            elif transition_value == VideoTransitionMode.zoom_out.value:
                clip = video_effects.zoomout_transition(clip, 1)
            elif transition_value == VideoTransitionMode.shuffle.value:
                transition_funcs = [
                    lambda c: video_effects.fadein_transition(c, 1),
                    lambda c: video_effects.fadeout_transition(c, 1),
                    lambda c: video_effects.slidein_transition(c, 1, shuffle_side),
                    lambda c: video_effects.slideout_transition(c, 1, shuffle_side),
                    lambda c: video_effects.zoomin_transition(c, 1),
                    lambda c: video_effects.zoomout_transition(c, 1),
                ]
                shuffle_transition = random.choice(transition_funcs)
                clip = shuffle_transition(clip)

            if clip.duration > max_clip_duration:
                clip = clip.subclipped(0, max_clip_duration)
                
            # wirte clip to temp file
            clip_file = f"{output_dir}/temp-clip-{i+1}.mp4"
            _write_videofile_with_codec_fallback(
                clip,
                clip_file,
                codec=_get_configured_video_codec(),
                logger=None,
                fps=fps,
            )

            # Store clip duration before closing
            clip_duration_saved = clip.duration
            close_clip(clip)

            processed_clips.append(
                SubClippedVideoClip(
                    file_path=clip_file,
                    duration=clip_duration_saved,
                    width=clip_w,
                    height=clip_h,
                    source_file_path=subclipped_item.source_file_path,
                )
            )
            video_duration += clip_duration_saved
            # record the source slice fingerprint so cross-part dedup can
            # skip it in later parts. The base slice range is what matters
            # for scene selection; transitions, speed changes, and the
            # max_clip_duration crop below are render-time effects only.
            used_fingerprints.add(
                (
                    subclipped_item.source_file_path,
                    subclipped_item.start_time,
                    subclipped_item.end_time,
                )
            )

        except Exception as e:
            logger.error(f"failed to process clip: {str(e)}")
    
    # loop processed clips until the video duration covers the audio duration and the small safety margin.
    if video_duration < required_video_duration:
        logger.warning(
            f"video duration ({video_duration:.2f}s) is shorter than required duration "
            f"({required_video_duration:.2f}s), looping clips to match audio length."
        )
        base_clips = processed_clips.copy()
        for clip in itertools.cycle(base_clips):
            if video_duration >= required_video_duration:
                break
            processed_clips.append(clip)
            video_duration += clip.duration
            # cycle-fill repeats already-consumed slices, so its source
            # fingerprints were already recorded by the main loop above.
            # No new entries to add; dedup set is unchanged.
        logger.info(
            f"video duration: {video_duration:.2f}s, audio duration: {audio_duration:.2f}s, "
            f"required duration: {required_video_duration:.2f}s, "
            f"looped {len(processed_clips)-len(base_clips)} clips"
        )
     
    # merge video clips progressively, avoid loading all videos at once to avoid memory overflow
    logger.info("starting clip merging process")
    if not processed_clips:
        logger.warning("no clips available for merging")
        # still mutate the dedup set so callers see "0 fingerprints consumed"
        # consistently across the no-clips edge case.
        exclude_clip_fingerprints.update(used_fingerprints)
        return combined_video_path

    clip_files = [clip.file_path for clip in processed_clips]
    logger.info(f"concatenating {len(clip_files)} clips with ffmpeg")
    concat_video_clips_with_ffmpeg(
        clip_files=clip_files,
        output_file=combined_video_path,
        threads=threads,
        output_dir=output_dir,
        max_duration=audio_duration,
    )

    # clean temp files
    delete_files(clip_files)

    # hand the consumed fingerprints back to the caller (generate_final_videos)
    # so the next part can skip them. in-place mutation keeps the public
    # return type stable for existing tests and CLI scripts.
    exclude_clip_fingerprints.update(used_fingerprints)
    if used_fingerprints:
        logger.info(
            f"recorded {len(used_fingerprints)} source slice fingerprints for "
            f"cross-part dedup"
        )

    logger.info("video combining completed")
    return combined_video_path


def wrap_text(text, max_width, font="Arial", fontsize=60):
    # subtitle wrapping has to happen before the TextClip exists, otherwise
    # MoviePy sizes the render area from the unwrapped text. measure width with
    # PIL at the current font and size so every line stays within the usable
    # video width, and a large font or a long CJK sentence cannot overflow.
    font = ImageFont.truetype(font, fontsize)
    max_width = int(max_width)

    # getbbox() returns the visible ink height of these glyphs, not the font's
    # line height. text made only of A, m, n and other characters without a
    # descender loses the descent, and across several lines that error
    # accumulates until the canvas crops the TextClip's last line. ascent +
    # descent comes from the font itself, is independent of language and
    # character mix, and matches MoviePy's baseline drawing model.
    ascent, descent = font.getmetrics()
    line_height = int(ascent + descent)
    if line_height <= 0:
        # a normal TrueType/OpenType font never reaches this branch. keep the
        # diagnostic log and the font-size fallback so a corrupt or unusual font
        # returning bad metrics cannot produce zero-height subtitles.
        logger.warning(
            "invalid subtitle font metrics, fallback to font size: "
            f"ascent={ascent}, descent={descent}, fontsize={fontsize}"
        )
        line_height = max(1, int(fontsize))

    def get_text_size(inner_text):
        inner_text = inner_text.strip()
        if not inner_text:
            return 0, line_height
        left, top, right, bottom = font.getbbox(inner_text)
        # bbox still measures the real width wrapping needs; height must always
        # come from the stable font line height.
        return right - left, line_height

    width, height = get_text_size(text)
    if width <= max_width:
        # an SRT entry may carry the author's own line breaks. even when the
        # text needs no further wrapping by width, canvas height must follow the
        # existing line count, or the second and later lines get cropped.
        return text, (text.count("\n") + 1) * line_height

    def split_long_token(token):
        # when a single token is already too wide (common for a long CJK
        # sentence with no spaces, or a very long English word), fall back to
        # splitting per character. the key point: once the candidate is too
        # wide, commit the still-valid current line first and start the current
        # character on the next line -- never push the overflowing character
        # back onto the previous line.
        lines = []
        current = ""
        for char in token:
            candidate = f"{current}{char}"
            candidate_width, _ = get_text_size(candidate)
            if candidate_width <= max_width or not current:
                current = candidate
                continue
            lines.append(current)
            current = char
        if current:
            lines.append(current)
        return lines

    lines = []
    current = ""
    words = text.split(" ")
    for word in words:
        candidate = f"{current} {word}".strip() if current else word
        candidate_width, _ = get_text_size(candidate)
        if candidate_width <= max_width:
            current = candidate
            continue

        if current:
            lines.append(current)

        word_width, _ = get_text_size(word)
        if word_width <= max_width:
            current = word
        else:
            lines.extend(split_long_token(word))
            current = ""

    if current:
        lines.append(current)

    line_start_punctuation = "，。！？；：、,.!?;:)]}）】》」』”’"
    for index in range(1, len(lines)):
        # splitting a long CJK sentence per character can leave the closing
        # period or comma alone on the next line, which inflates the subtitle
        # background and reads as a stray dot below the text. without redesigning
        # the wrapping algorithm, move the previous line's last character in
        # front of the punctuation so it follows the text, which works for the
        # common closing punctuation in both CJK and Latin scripts.
        if not lines[index] or lines[index][0] not in line_start_punctuation:
            continue
        if len(lines[index - 1]) <= 1:
            continue

        candidate = f"{lines[index - 1][-1]}{lines[index]}"
        candidate_width, _ = get_text_size(candidate)
        if candidate_width <= max_width:
            lines[index] = candidate
            lines[index - 1] = lines[index - 1][:-1]

    result = "\n".join(line.strip() for line in lines if line.strip()).strip()
    # take the height from the final result. an explicit line break in the
    # source text can survive inside a token, in which case the temporary lines
    # list is not the number of lines MoviePy actually renders.
    height = (result.count("\n") + 1) * line_height
    return result, height


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    # the subtitle background colour comes from an API or WebUI parameter and
    # may be empty or malformed. accept only #RRGGBB and fall back to black, so
    # an invalid value cannot raise inside PIL and abort the task.
    if isinstance(color, str) and color.startswith("#") and len(color) == 7:
        try:
            return (int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16))
        except ValueError:
            pass
    return (0, 0, 0)


def _rounded_subtitle_background_clip(
    width: int,
    height: int,
    color: str,
    alpha: int = 140,
    radius: int = 16,
) -> ImageClip:
    # the new subtitle background is used only when the user opts in: draw a
    # rounded translucent plate as an RGBA image and hand it to MoviePy as a
    # transparent ImageClip. the default path is untouched, which makes a
    # softer subtitle look cheap to experiment with.
    rgb = _hex_to_rgb(color)
    safe_alpha = max(0, min(255, int(alpha)))
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle(
        [0, 0, max(0, width - 1), max(0, height - 1)],
        radius=max(0, int(radius)),
        fill=(rgb[0], rgb[1], rgb[2], safe_alpha),
    )
    return ImageClip(np.array(img), transparent=True)


def _highlight_tokens(text: str, active: bool) -> list[tuple[str, bool]]:
    """Split text into drawable tokens while keeping CJK glyphs wrappable."""
    tokens: list[tuple[str, bool]] = []
    buffer = ""
    for char in text:
        if char.isspace() or unicodedata.east_asian_width(char) in {"W", "F"}:
            if buffer:
                tokens.append((buffer, active))
                buffer = ""
            tokens.append((char, active))
        else:
            buffer += char
    if buffer:
        tokens.append((buffer, active))
    return tokens


def _render_highlighted_subtitle_clip(
    phrase: str,
    *,
    font_path: str,
    font_size: int,
    max_width: int,
    text_color: str,
    highlight_color: str,
    stroke_color: str,
    stroke_width: int,
    background_color: str | None,
    rounded_background: bool,
) -> ImageClip:
    """Render one karaoke cue with a differently colored active word."""
    active_start = phrase.find(subtitle_styles.HIGHLIGHT_OPEN)
    active_end = phrase.find(subtitle_styles.HIGHLIGHT_CLOSE)
    if active_start < 0 or active_end < active_start:
        raise ValueError("highlighted subtitle cue is missing its active-word marker")

    before = phrase[:active_start]
    active = phrase[active_start + len(subtitle_styles.HIGHLIGHT_OPEN) : active_end]
    after = phrase[active_end + len(subtitle_styles.HIGHLIGHT_CLOSE) :]
    tokens = [
        *_highlight_tokens(before, False),
        *_highlight_tokens(active, True),
        *_highlight_tokens(after, False),
    ]

    font = ImageFont.truetype(font_path, font_size)
    probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))

    def measure(value: str) -> float:
        box = probe.textbbox((0, 0), value, font=font, stroke_width=stroke_width)
        return float(box[2] - box[0])

    padding_x = int(font_size * (0.42 if background_color else 0.12))
    padding_y = int(font_size * (0.30 if background_color else 0.18))
    text_width = max(1, max_width - 2 * padding_x)
    lines: list[list[tuple[str, bool, float]]] = [[]]
    line_widths = [0.0]
    for token, is_active in tokens:
        if token.isspace() and not lines[-1]:
            continue
        token_width = measure(token)
        if lines[-1] and not token.isspace() and line_widths[-1] + token_width > text_width:
            lines.append([])
            line_widths.append(0.0)
        lines[-1].append((token, is_active, token_width))
        line_widths[-1] += token_width

    font_box = font.getbbox("Ag", stroke_width=stroke_width)
    line_height = max(1, font_box[3] - font_box[1])
    interline = int(font_size * 0.22)
    content_width = max(1, math.ceil(max(line_widths, default=1)))
    canvas_width = min(max_width, content_width + 2 * padding_x)
    canvas_height = (
        len(lines) * line_height + max(0, len(lines) - 1) * interline + 2 * padding_y
    )
    image = Image.new("RGBA", (canvas_width, canvas_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    if background_color:
        rgb = _hex_to_rgb(background_color)
        alpha = 140 if rounded_background else 255
        radius = int(font_size * 0.4) if rounded_background else 0
        draw.rounded_rectangle(
            (0, 0, canvas_width - 1, canvas_height - 1),
            radius=radius,
            fill=(*rgb, alpha),
        )

    normal_rgb = _hex_to_rgb(text_color)
    active_rgb = _hex_to_rgb(highlight_color)
    stroke_rgb = _hex_to_rgb(stroke_color)
    y = padding_y - font_box[1]
    for line, line_width in zip(lines, line_widths):
        x = (canvas_width - line_width) / 2
        for token, is_active, token_width in line:
            draw.text(
                (x, y),
                token,
                font=font,
                fill=active_rgb if is_active else normal_rgb,
                stroke_width=stroke_width,
                stroke_fill=stroke_rgb,
            )
            x += token_width
        y += line_height + interline
    return ImageClip(np.array(image), transparent=True)


def _get_visible_center_position(
    text_clip: TextClip,
    container_width: int,
    container_height: int,
) -> tuple[int, int]:
    """
    Centre a TextClip in its background container by the text's visible pixels.

    MoviePy's TextClip builds a transparent canvas from the font's line height
    and baseline. For many fonts the visible glyphs are not at that canvas's
    geometric centre, so a plain `with_position("center")` centres the whole
    transparent canvas and the subtitle looks too high or too low. Read the
    TextClip's alpha mask and derive the offset from the bbox of the pixels that
    are actually drawn, so the text the user sees is optically centred.
    """
    x = int(round((container_width - text_clip.w) / 2))
    y = int(round((container_height - text_clip.h) / 2))

    try:
        if text_clip.mask is None:
            return x, y

        mask_frame = text_clip.mask.get_frame(0)
        ys, _ = np.where(mask_frame > 0.01)
        if len(ys) == 0:
            return x, y

        visible_top = int(ys.min())
        visible_bottom = int(ys.max())
        visible_height = visible_bottom - visible_top + 1
        y = int(round((container_height - visible_height) / 2 - visible_top))
    except Exception as exc:
        logger.debug(f"failed to center subtitle text by visible mask: {str(exc)}")

    return x, y


def subtitle_colors_are_indistinguishable(params: VideoParams) -> bool:
    """Report whether subtitle text and background share a colour, so the user
    can be warned the subtitles may be unreadable."""
    if not params.subtitle_enabled or not params.text_background_color:
        return False

    def normalize_color(value):
        if isinstance(value, bool):
            return "#000000" if value else ""
        return str(value or "").strip().lower()

    text_color = normalize_color(params.text_fore_color)
    background_color = normalize_color(params.text_background_color)
    return bool(text_color and text_color == background_color)


@lru_cache(maxsize=64)
def _subtitle_font_supports_sample(font_path: str, sample: str) -> bool:
    """Check the font has the glyphs the sample text needs, caching repeats."""
    try:
        font = ImageFont.truetype(font_path, 30)
        missing_mask = font.getmask("\U0010ffff")
        missing_signature = (
            missing_mask.size,
            missing_mask.getbbox(),
            bytes(missing_mask),
        )
        for char in sample:
            char_mask = font.getmask(char)
            char_signature = (
                char_mask.size,
                char_mask.getbbox(),
                bytes(char_mask),
            )
            if char_mask.getbbox() is None or char_signature == missing_signature:
                return False
        return True
    except Exception as e:
        # a failed font probe must not block generation; keep the log for
        # troubleshooting environment compatibility.
        logger.warning(f"failed to inspect subtitle font glyphs: {font_path}, {e}")
        return True


def subtitle_font_supports_text(font_path: str, text: str) -> bool:
    """Check the font can draw the letters and digits in the text, ignoring
    whitespace and punctuation."""
    sample = "".join(
        dict.fromkeys(
            char
            for char in str(text or "")
            if unicodedata.category(char)[0] in {"L", "N"}
        )
    )[:64]
    if not sample:
        return True
    return _subtitle_font_supports_sample(font_path, sample)


def _resolve_subtitle_background_color_locally(value):
    """Module-level version of the legacy ``text_background_color`` resolver
    so the drawtext fast path can use it without instantiating the
    ``generate_video`` closure first."""
    if isinstance(value, bool):
        return "#000000" if value else None
    return value


def _escape_ffmpeg_drawtext_text(text: str) -> str:
    r"""Escape a subtitle string for safe inclusion in an ffmpeg drawtext
    filter argument.

    drawtext treats `:`, `\`, and `%` as filter-argument syntax characters,
    so each one has to be doubled before reaching ffmpeg's text parser.
    The single-quote wrapper used in the caller handles quotes. Newlines
    inside a cue become an explicit ``\\n`` drawtext escape so multi-line
    SRT cues stay readable.
    """
    if not text:
        return ""
    return (
        text.replace("\\", "\\\\")
        .replace(":", "\\:")
        .replace("%", "\\%")
        .replace(";", "\\;")
        .replace("'", "’")
        .replace("\r\n", "\n")
        .replace("\n", "\\n")
    )


_SUBTITLE_TIMING_RE = re.compile(
    r"(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)"
)


def _parse_subtitle_timing(line: str) -> tuple[float, float] | None:
    match = _SUBTITLE_TIMING_RE.search(line)
    if not match:
        return None
    h1, m1, s1, ms1, h2, m2, s2, ms2 = (int(value) for value in match.groups())
    return (
        h1 * 3600 + m1 * 60 + s1 + ms1 / 1000.0,
        h2 * 3600 + m2 * 60 + s2 + ms2 / 1000.0,
    )


def _ass_color(color: str, fallback: str) -> str:
    """Convert a web ``#RRGGBB`` colour to ASS ``&H00BBGGRR``."""
    value = (
        color
        if isinstance(color, str) and re.fullmatch(r"#[0-9A-Fa-f]{6}", color)
        else fallback
    )
    return f"&H00{value[5:7]}{value[3:5]}{value[1:3]}".upper()


def _ass_time(seconds: float) -> str:
    centiseconds = max(0, int(round(float(seconds) * 100)))
    hours, remainder = divmod(centiseconds, 360000)
    minutes, remainder = divmod(remainder, 6000)
    secs, fraction = divmod(remainder, 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{fraction:02d}"


def _escape_ass_text(text: str) -> str:
    """Escape user text so it cannot be interpreted as ASS override tags."""
    return (
        str(text or "")
        .replace("\\", "\uff3c")
        .replace("{", "\uff5b")
        .replace("}", "\uff5d")
        .replace("\r\n", "\n")
        .replace("\n", r"\N")
    )


def _ass_dialogue_text(phrase: str, normal_color: str, highlight_color: str) -> str:
    """Turn the internal active-word markers into ASS colour overrides."""
    start = phrase.find(subtitle_styles.HIGHLIGHT_OPEN)
    end = phrase.find(subtitle_styles.HIGHLIGHT_CLOSE)
    if start < 0 or end < start:
        return _escape_ass_text(phrase)
    before = _escape_ass_text(phrase[:start])
    active = _escape_ass_text(
        phrase[start + len(subtitle_styles.HIGHLIGHT_OPEN) : end]
    )
    after = _escape_ass_text(phrase[end + len(subtitle_styles.HIGHLIGHT_CLOSE) :])
    return f"{before}{{\\c{highlight_color}&}}{active}{{\\c{normal_color}&}}{after}"


def _build_ass_subtitles(
    *,
    srt_path: str,
    params: VideoParams,
    font_path: str,
    video_width: int,
    video_height: int,
    max_duration: float | None = None,
) -> str:
    """Build styled ASS events so FFmpeg can render animated karaoke natively."""
    timed_cues: list[tuple[tuple[float, float], str]] = []
    for _index, timing_line, text in subtitle.file_to_subtitles(srt_path):
        timing = _parse_subtitle_timing(timing_line)
        if timing is None or timing[1] <= timing[0] or not text:
            continue
        if max_duration is not None and timing[0] >= max_duration:
            continue
        timed_cues.append((timing, text))
    if not timed_cues:
        return ""

    normal_color = _ass_color(getattr(params, "text_fore_color", ""), "#FFFFFF")
    stroke_color = _ass_color(getattr(params, "stroke_color", ""), "#000000")
    preset = subtitle_styles.get_subtitle_preset(
        getattr(params, "subtitle_style_preset", "custom")
    ) or {}
    highlight_color = _ass_color(preset.get("highlight_color", ""), "#FFE600")
    font_name = os.path.splitext(os.path.basename(font_path))[0]
    font_size = int(getattr(params, "font_size", 60) or 60)
    ass_font_size = max(1, int(round(font_size * 1.15)))
    stroke_width = max(0, int(round(float(getattr(params, "stroke_width", 0) or 0))))
    margin_x = max(10, int(video_width * 0.05))

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {video_width}
PlayResY: {video_height}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font_name},{ass_font_size},{normal_color},{normal_color},{stroke_color},&H00000000,-1,0,0,0,100,100,0,0,1,{stroke_width},0,5,{margin_x},{margin_x},0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    position = getattr(params, "subtitle_position", "bottom")
    if position == "bottom":
        alignment, x, y = 2, video_width / 2, video_height * 0.95
    elif position == "top":
        alignment, x, y = 8, video_width / 2, video_height * 0.05
    elif position in ("two_thirds_bottom", "two_thirds", "2/3_bottom"):
        alignment, x, y = 8, video_width / 2, video_height * 0.30
    elif position == "custom":
        percent = max(0.0, min(100.0, float(getattr(params, "custom_position", 50))))
        alignment, x, y = 8, video_width / 2, video_height * percent / 100
    else:
        alignment, x, y = 5, video_width / 2, video_height / 2

    animation = getattr(params, "subtitle_animation", "none")
    if animation in ("scale_up", "zoom_in", "punch"):
        animation_tag = r"\fscx65\fscy65\t(0,180,0.5,\fscx100\fscy100)"
    elif animation in ("pop_spring", "spring", "pop"):
        animation_tag = r"\fscx5\fscy5\t(0,100,0.5,\fscx135\fscy135)\t(100,180,0.5,\fscx100\fscy100)"
    elif animation in ("fade", "fade_in"):
        animation_tag = r"\fad(180,0)"
    else:
        animation_tag = ""

    display_cues = subtitle_styles.build_display_cues(
        timed_cues,
        getattr(params, "subtitle_display_mode", "sentence"),
    )
    events: list[str] = []
    for timing, raw_phrase in display_cues:
        start, end = timing
        if max_duration is not None:
            end = min(end, max_duration)
        if end <= start:
            continue
        phrase = subtitle_styles.apply_text_casing(
            str(raw_phrase), getattr(params, "subtitle_casing", "as_is")
        )
        if animation in ("slide_up", "rise"):
            shift = max(15, int(round(font_size * 0.3)))
            position_tag = (
                f"\\an{alignment}\\move({x:.0f},{y + shift:.0f},{x:.0f},{y:.0f},0,180)"
            )
        else:
            position_tag = f"\\an{alignment}\\pos({x:.0f},{y:.0f})"
        text = _ass_dialogue_text(phrase, normal_color, highlight_color)
        events.append(
            f"Dialogue: 0,{_ass_time(start)},{_ass_time(end)},Default,,0,0,0,,"
            f"{{{position_tag}{animation_tag}}}{text}"
        )
    return header + "\n".join(events) + "\n" if events else ""


def _escape_ffmpeg_filter_path(path: str) -> str:
    """Escape a portable filesystem path for one FFmpeg filter argument."""
    return (
        path.replace("\\", "/")
        .replace(":", r"\:")
        .replace("'", r"\'")
        .replace(",", r"\,")
        .replace("[", r"\[")
        .replace("]", r"\]")
        .replace(";", r"\;")
    )


def _render_subtitle_pngs(
    cues_with_timing,
    *,
    font_path: str,
    font_size: int,
    text_color: str,
    stroke_color: Optional[str],
    stroke_width: float,
    canvas_width: int,
    canvas_height: int,
) -> List[str]:
    """Pre-render every subtitle cue to a PNG with Pillow, one file per
    cue. The caller then passes these PNGs to ffmpeg with an ``overlay``
    filter that activates each PNG only inside its own time window, so
    the entire subtitle burn-in is a single ffmpeg invocation.

    This is the right fast path on hosts whose ffmpeg was built without
    libfreetype/libass and therefore lacks the ``drawtext`` filter. It
    also stays dramatically faster than MoviePy's TextClip path because
    Pillow renders the text ONCE per cue (vs. once per frame in
    MoviePy), and the ffmpeg overlay+encode runs in a single pass.
    """
    from PIL import Image, ImageDraw, ImageFont

    if not os.path.exists(font_path):
        return []

    try:
        font_obj = ImageFont.truetype(font_path, int(font_size))
    except Exception:
        return []

    safe_color = text_color if (text_color or "").startswith("#") else "#FFFFFF"
    safe_stroke = stroke_color if (stroke_color or "").startswith("#") else None

    # Render the widest cue first so every PNG uses the same canvas
    # dimensions -- ffmpeg's overlay filter expects a single size and
    # composes each layer at its native resolution otherwise.
    rendered: list[tuple[str, float, float]] = []
    widest = max(
        (max((len(line) for line in text.split("\n")), default=0))
        for _t, text in cues_with_timing
    )
    char_width_estimate = max(int(font_size * 0.6), 1)
    est_w = max(1, widest * char_width_estimate)
    pad_x = max(8, int(font_size * 0.4))
    pad_y = max(6, int(font_size * 0.25))
    box_w = min(int(canvas_width * 0.95), est_w + 2 * pad_x)
    line_h = int(font_size * 1.3)
    max_lines = 2
    box_h = line_h * max_lines + 2 * pad_y

    png_paths: list[str] = []
    for index, (timing, text) in enumerate(cues_with_timing):
        start, end = timing
        if end <= start or not text:
            continue

        img = Image.new("RGBA", (box_w, box_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        lines = text.split("\n")[:max_lines]
        y = pad_y
        for line in lines:
            draw.text(
                (pad_x, y),
                line,
                font=font_obj,
                fill=safe_color,
                stroke_width=int(stroke_width) if stroke_width and stroke_width > 0 else 0,
                stroke_fill=safe_stroke or "#000000",
            )
            y += line_h

        png_path = os.path.join(
            tempfile.gettempdir(),
            f"mpt-subtitle-{os.getpid()}-{index:04d}.png",
        )
        try:
            img.save(png_path)
            png_paths.append(png_path)
            rendered.append((png_path, start, end))
        except Exception:
            continue

    return rendered


def _final_mux_with_ffmpeg(
    *,
    video_path: str,
    audio_path: str,
    output_file: str,
    voice_volume: float,
    threads: int,
) -> bool:
    """Final mux of audio onto the combined video, with optional voice
    volume adjustment. Re-encodes the video unless ``use_copy`` is True
    (the user-facing fast path) — in that case the video stream is
    stream-copied from the input, dropping the encode step entirely.

    Returns True on success, False on any ffmpeg error so the caller can
    fall back to MoviePy.
    """
    command = [
        utils.get_ffmpeg_binary(),
        "-y",
        "-i", video_path,
        "-i", audio_path,
    ]
    af_args: list[str] = []
    if voice_volume != 1.0:
        af_args.append(f"volume={float(voice_volume):.3f}")
    if af_args:
        command.extend(["-af", ",".join(af_args)])
    command.extend([
        "-map", "0:v",
        "-map", "1:a",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-ar", "44100",
        "-shortest",
        "-movflags", "+faststart",
        "-threads", str(threads or 2),
        output_file,
    ])
    result = subprocess.run(
        command, capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        stderr = result.stderr or ""
        last_lines = stderr.strip().splitlines()[-3:]
        logger.warning(
            f"final stream-copy mux failed (rc={result.returncode}); "
            f"falling back to MoviePy. last stderr: {' | '.join(last_lines)}"
        )
        return False
    return True


def _run_ffmpeg_subtitle_filter(
    *,
    video_path: str,
    audio_path: str,
    output_file: str,
    video_filter: str,
    voice_volume: float,
    threads: int,
    max_duration: float | None,
    overlay_path: str | None = None,
) -> tuple[bool, str]:
    """Run the shared hardware-encoded subtitle and narration render."""
    effective_codec = _get_effective_video_codec()
    if effective_codec == "h264_vaapi":
        if overlay_path:
            video_filter = video_filter.replace(
                "[subtitled]", ",format=nv12,hwupload[subtitled]"
            )
        else:
            video_filter = f"{video_filter},format=nv12,hwupload"
    command = [
        utils.get_ffmpeg_binary(),
        "-y",
        "-i",
        video_path,
        "-i",
        audio_path,
    ]
    if overlay_path:
        command.extend(["-i", overlay_path])
    if voice_volume != 1.0:
        command.extend(["-af", f"volume={float(voice_volume):.3f}"])
    if overlay_path:
        command.extend(
            ["-filter_complex", video_filter, "-map", "[subtitled]", "-map", "1:a"]
        )
    else:
        command.extend(["-vf", video_filter, "-map", "0:v", "-map", "1:a"])
    command.extend(["-c:v", effective_codec])
    command.extend(_get_codec_ffmpeg_params(effective_codec, filtered=True))
    command.extend(["-threads", str(threads or 2)])
    if effective_codec != "h264_vaapi":
        command.extend(["-pix_fmt", "yuv420p"])
    command.extend(
        [
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-ar",
            "44100",
            "-shortest",
            "-movflags",
            "+faststart",
        ]
    )
    if max_duration is not None and max_duration > 0:
        command.extend(["-t", f"{float(max_duration):.3f}"])
    command.append(output_file)
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    stderr = " | ".join((result.stderr or "").strip().splitlines()[-5:])
    return result.returncode == 0, stderr


def _burn_subtitles_with_ffmpeg_ass(
    *,
    video_path: str,
    audio_path: str,
    srt_path: str,
    output_file: str,
    font_path: str,
    params: VideoParams,
    video_width: int,
    video_height: int,
    voice_volume: float,
    threads: int,
    max_duration: float | None = None,
) -> bool:
    """Burn styled or animated subtitles through native FFmpeg filters."""
    ass_document = _build_ass_subtitles(
        srt_path=srt_path,
        params=params,
        font_path=font_path,
        video_width=video_width,
        video_height=video_height,
        max_duration=max_duration,
    )
    if not ass_document:
        return False

    ass_path = f"{output_file}.subtitles.ass"
    overlay_path = f"{output_file}.subtitles.mov"
    try:
        with open(ass_path, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(ass_document)
        ffmpeg_binary = utils.get_ffmpeg_binary()
        ass_filter = (
            f"ass=filename='{_escape_ffmpeg_filter_path(os.path.abspath(ass_path))}'"
            f":fontsdir='{_escape_ffmpeg_filter_path(os.path.abspath(os.path.dirname(font_path)))}'"
        )

        if not _ffmpeg_filter_exists(ffmpeg_binary, "ass"):
            container_ffmpeg = shutil.which("ffmpeg")
            if not (
                os.environ.get("FFMPEG_MAC_PROXY_URL")
                and container_ffmpeg
                and _ffmpeg_filter_exists(container_ffmpeg, "ass")
            ):
                return False

            cue_ends = [
                timing[1]
                for _index, timing_line, _text in subtitle.file_to_subtitles(srt_path)
                if (timing := _parse_subtitle_timing(timing_line)) is not None
            ]
            duration = float(max_duration) if max_duration else max(cue_ends, default=0.0)
            if duration <= 0:
                return False
            overlay_command = [
                container_ffmpeg,
                "-y",
                "-f",
                "lavfi",
                "-i",
                f"color=c=black@0.0:s={video_width}x{video_height}:r={fps},format=rgba",
                "-vf",
                f"{ass_filter}:alpha=1",
                "-t",
                f"{duration:.3f}",
                "-an",
                "-c:v",
                "qtrle",
                "-pix_fmt",
                "argb",
                overlay_path,
            ]
            overlay_result = subprocess.run(
                overlay_command, capture_output=True, text=True, check=False
            )
            if overlay_result.returncode != 0:
                stderr = " | ".join(
                    (overlay_result.stderr or "").strip().splitlines()[-5:]
                )
                logger.warning(
                    "container ASS overlay render failed; falling back to MoviePy. "
                    f"last stderr: {stderr}"
                )
                return False
            ass_filter = "[0:v][2:v]overlay=eof_action=pass:format=auto[subtitled]"

        succeeded, stderr = _run_ffmpeg_subtitle_filter(
            video_path=video_path,
            audio_path=audio_path,
            output_file=output_file,
            video_filter=ass_filter,
            voice_volume=voice_volume,
            threads=threads,
            max_duration=max_duration,
            overlay_path=overlay_path if os.path.exists(overlay_path) else None,
        )
        if not succeeded:
            logger.warning(
                "ffmpeg ASS subtitle burn-in failed; falling back to MoviePy. "
                f"last stderr: {stderr}"
            )
        return succeeded
    finally:
        for temporary_path in (ass_path, overlay_path):
            try:
                os.remove(temporary_path)
            except FileNotFoundError:
                pass


def _burn_subtitles_with_ffmpeg_drawtext(
    *,
    video_path: str,
    audio_path: str,
    srt_path: str,
    output_file: str,
    font_path: str,
    font_size: int,
    text_color: str,
    stroke_color: Optional[str],
    stroke_width: float,
    voice_volume: float,
    threads: int,
    fps: int,
    max_duration: float | None = None,
) -> bool:
    """Burn subtitles into the video with ONE ffmpeg invocation using
    ``drawtext`` filters, replacing MoviePy's per-cue ``TextClip`` path.

    Returns True on success. The caller falls back to MoviePy when this
    path is not applicable (e.g., when the SRT can't be parsed).

    Why this is so much faster than MoviePy's path: MoviePy renders each
    TextClip frame-by-frame through Pillow and writes an intermediate PNG
    per cue; for a 60s video with 20 cues that is hundreds of PIL renders
    plus a final composite + write. drawtext is implemented in libharfbuzz
    inside ffmpeg, so all 20 cues are drawn in a single filter graph pass
    while the video streams through. Wall-clock goes from ~5-15s to ~1-2s
    on a typical 30s clip.
    """
    if not os.path.exists(srt_path):
        return False
    raw_cues = subtitle.file_to_subtitles(srt_path)
    if not raw_cues:
        return False

    # ffmpeg drawtext wants the font file path. Skip if not present so we
    # fall back to MoviePy rather than failing the whole generation.
    if not font_path or not os.path.exists(font_path):
        return False

    # Compose one drawtext filter per cue. ``enable='between(t,start,end)'``
    # keeps each cue visible only in its own time window. ``x=(w-tw)/2``
    # centres horizontally; ``y=h*0.85`` sits the cue near the bottom like
    # the default MoviePy position.
    filters: list[str] = []
    for _index, timing_line, text in raw_cues:
        parsed = _parse_subtitle_timing(timing_line)
        if parsed is None:
            continue
        start, end = parsed
        if max_duration is not None and start >= max_duration:
            continue
        if end <= start or not text:
            continue
        # ffmpeg's drawtext is built around a single-line argument string;
        # the comma between drawtext entries is the filter-chain separator
        # and must not appear inside an entry, so we replace any commas in
        # the cue with full-width equivalents.
        safe_text = _escape_ffmpeg_drawtext_text(text).replace(",", "，")
        # fontcolor accepts #RRGGBB and named colors; if the user gave
        # something else (rgba, etc.) drop to white to keep the call valid.
        safe_color = text_color if (text_color or "").startswith("#") else "white"
        safe_stroke = (
            stroke_color if (stroke_color or "").startswith("#") else None
        )
        stroke_part = (
            f":bordercolor={safe_stroke}:borderw={int(stroke_width)}"
            if safe_stroke and stroke_width and stroke_width > 0
            else ""
        )
        filter_str = (
            f"drawtext=text='{safe_text}'"
            f":enable='between(t,{float(start):.3f},{float(end):.3f})'"
            f":fontfile='{_escape_ffmpeg_drawtext_text(font_path)}'"
            f":fontsize={int(font_size)}"
            f":fontcolor={safe_color}"
            f"{stroke_part}"
            f":x=(w-tw)/2"
            f":y=h*0.85"
        )
        filters.append(filter_str)

    if not filters:
        return False

    # Chain every drawtext into one -vf argument. Newlines between filter
    # entries keep ffmpeg's parser happy on long chains.
    vf_arg = ",".join(filters)
    succeeded, stderr = _run_ffmpeg_subtitle_filter(
        video_path=video_path,
        audio_path=audio_path,
        output_file=output_file,
        video_filter=vf_arg,
        voice_volume=voice_volume,
        threads=threads,
        max_duration=max_duration,
    )
    if not succeeded:
        logger.warning(
            "ffmpeg drawtext subtitle burn-in failed; falling back to MoviePy. "
            f"last stderr: {stderr}"
        )
        return False
    return True


def _burn_subtitles_with_png_overlay(
    *,
    video_path: str,
    audio_path: str,
    srt_path: str,
    output_file: str,
    font_path: str,
    font_size: int,
    text_color: str,
    stroke_color: Optional[str],
    stroke_width: float,
    voice_volume: float,
    canvas_width: int,
    canvas_height: int,
    threads: int,
    fps: int,
    max_duration: float | None = None,
) -> bool:
    """Burn subtitles with one ffmpeg invocation using PNG overlays.

    Works on every ffmpeg build (no libfreetype / libass required) and
    replaces MoviePy's per-frame PIL rendering with a one-render-per-cue
    approach: Pillow draws the cue text into a PNG ONCE, then ffmpeg
    overlays each PNG on the video inside its time window while doing a
    single encode pass. MoviePy does TextClip×fps×duration PIL renders;
    this path does Pillow×cues renders, which is the single biggest
    speed win on hosts whose ffmpeg lacks the drawtext filter.
    """
    if not os.path.exists(srt_path):
        return False
    raw_cues = subtitle.file_to_subtitles(srt_path)
    if not raw_cues:
        return False
    if not font_path or not os.path.exists(font_path):
        return False

    cues_with_timing: list[tuple[tuple[float, float], str]] = []
    for _i, timing_line, text in raw_cues:
        parsed = _parse_subtitle_timing(timing_line)
        if parsed is None:
            continue
        start, end = parsed
        if max_duration is not None and start >= max_duration:
            continue
        if end <= start or not text:
            continue
        cues_with_timing.append((parsed, text))
    if not cues_with_timing:
        return False

    png_specs = _render_subtitle_pngs(
        cues_with_timing,
        font_path=font_path,
        font_size=int(font_size),
        text_color=text_color or "#FFFFFF",
        stroke_color=stroke_color,
        stroke_width=float(stroke_width or 0),
        canvas_width=int(canvas_width),
        canvas_height=int(canvas_height),
    )
    if not png_specs:
        return False

    # Build a filter_complex with one overlay per PNG, activated by the
    # cue's own [start, end] window. Inputs are 0:video, 1:audio, and
    # 2..N+1: the PNGs. Output is [out_v].
    parts: list[str] = []
    parts.append("[0:v]format=yuv420p[base]")
    overlay_inputs = "[base]"
    for idx, (_png_path, start, end) in enumerate(png_specs):
        input_idx = idx + 2
        parts.append(f"[{input_idx}:v]format=rgba[layer{idx}]")
        parts.append(
            f"{overlay_inputs}[layer{idx}]overlay="
            f"x=(W-w)/2:y=H*0.85-h:"
            f"enable='between(t,{start:.3f},{end:.3f})'[ov{idx}]"
        )
        overlay_inputs = f"[ov{idx}]"
    effective_codec = _get_effective_video_codec()
    output_format = (
        "format=nv12,hwupload"
        if effective_codec == "h264_vaapi"
        else "format=yuv420p"
    )
    parts.append(f"{overlay_inputs}{output_format}[out_v]")

    af_arg = f"volume={float(voice_volume):.3f}" if voice_volume != 1.0 else None

    command = [
        utils.get_ffmpeg_binary(),
        "-y",
        "-i", video_path,
        "-i", audio_path,
    ]
    for png_path, _s, _e in png_specs:
        command.extend(["-i", png_path])
    command.extend(["-filter_complex", ";".join(parts)])
    command.extend(["-map", "[out_v]", "-map", "1:a"])
    if af_arg:
        command.extend(["-af", af_arg])
    command.extend([
        "-c:v", effective_codec,
    ])
    command.extend(_get_codec_ffmpeg_params(effective_codec, filtered=True))
    command.extend([
        "-threads", str(threads or 2),
    ])
    if effective_codec != "h264_vaapi":
        command.extend(["-pix_fmt", "yuv420p"])
    command.extend([
        "-c:a", "aac",
        "-b:a", "192k",
        "-ar", "44100",
        "-shortest",
        "-movflags", "+faststart",
    ])
    if max_duration is not None and max_duration > 0:
        command.extend(["-t", f"{float(max_duration):.3f}"])
    command.append(output_file)

    result = subprocess.run(
        command, capture_output=True, text=True, check=False,
    )
    # Best-effort cleanup of PNGs we wrote into the system tempdir.
    for png_path, _s, _e in png_specs:
        try:
            os.remove(png_path)
        except OSError:
            pass

    if result.returncode != 0:
        stderr = result.stderr or ""
        last_lines = stderr.strip().splitlines()[-3:]
        logger.warning(
            f"ffmpeg png-overlay subtitle burn-in failed "
            f"(rc={result.returncode}); falling back to MoviePy. "
            f"last stderr: {' | '.join(last_lines)}"
        )
        return False
    return True


def _create_title_clip(
    params: VideoParams,
    video_width: int,
    video_height: int,
    video_duration: float,
):
    """
    Render an on-screen title or hook banner overlay clip for TikTok/Reels/Shorts.
    Returns None if title overlay is not enabled or title text is empty.
    """
    if not getattr(params, "title_enabled", False):
        return None

    title_text = (getattr(params, "title_text", "") or "").strip()
    if not title_text:
        title_text = (getattr(params, "video_subject", "") or "").strip()
    if not title_text:
        return None

    style_id = getattr(params, "title_style", "tiktok_yellow") or "tiktok_yellow"
    style_cfg = (
        subtitle_styles.get_title_style(style_id)
        or subtitle_styles.TITLE_STYLES["tiktok_yellow"]
    )

    # Honour a per-call casing override first so a user pasting raw text and
    # choosing "as_is" sees exactly what they typed, then fall back to the
    # style's preset casing (the previous behaviour).
    user_casing = getattr(params, "title_casing", None)
    casing = user_casing if user_casing else style_cfg.get("casing", "uppercase")
    title_text = subtitle_styles.apply_text_casing(title_text, casing)

    requested_font_name = getattr(params, "title_font_name", None)
    style_font_name = style_cfg.get("font_name")
    fallback_font_name = (
        getattr(params, "font_name", "Anton-Regular.ttf") or "Anton-Regular.ttf"
    )
    font_name = requested_font_name or style_font_name or fallback_font_name
    available_fonts = [
        f for f in os.listdir(utils.font_dir()) if f.endswith((".ttf", ".ttc"))
    ]
    if font_name not in available_fonts:
        # User-selected (or style-default) font is missing — log it so the
        # gap is visible instead of silently swapping to an unrelated font.
        logger.warning(
            "title font not found on disk: %s "
            "(available: %s); falling back to a system default",
            font_name,
            ", ".join(sorted(available_fonts)) or "<none>",
        )
        font_name = (
            "STHeitiMedium.ttc"
            if "STHeitiMedium.ttc" in available_fonts
            else (available_fonts[0] if available_fonts else "")
        )
    font_path = os.path.join(utils.font_dir(), font_name)
    if os.name == "nt":
        font_path = font_path.replace("\\", "/")

    font_size = getattr(params, "title_font_size", None)
    if not font_size:
        font_size = max(36, int(video_width * 0.055))
    font_size = int(font_size)

    max_title_width = video_width * 0.85
    wrapped_title, _ = wrap_text(
        title_text,
        max_width=max_title_width,
        font=font_path,
        fontsize=font_size,
    )

    text_color = style_cfg.get("text_color", "#FFFFFF")
    stroke_color = style_cfg.get("stroke_color")
    stroke_width = float(style_cfg.get("stroke_width", 0.0))
    bg_color = style_cfg.get("bg_color")
    rounded = style_cfg.get("rounded", True)

    pad_x = int(font_size * 0.5) if bg_color else 0
    pad_y = int(font_size * 0.3) if bg_color else 0

    try:
        font_obj = ImageFont.truetype(font_path, font_size)
        text_w = max(
            int(font_obj.getbbox(line)[2] - font_obj.getbbox(line)[0])
            for line in wrapped_title.split("\n")
        )
    except Exception:
        text_w = int(max_title_width)

    box_w = max(1, min(int(max_title_width), text_w + 2 * pad_x))
    interline = int(font_size * 0.2)
    line_count = wrapped_title.count("\n") + 1
    clip_h = int(font_size * line_count * 1.3 + 2 * pad_y)

    text_clip = TextClip(
        text=wrapped_title,
        font=font_path,
        font_size=font_size,
        color=text_color,
        stroke_color=stroke_color,
        stroke_width=int(stroke_width),
        interline=interline,
        size=(box_w, None),
        text_align="center",
    )
    clip_h = max(clip_h, text_clip.h + 2 * pad_y)

    if bg_color:
        radius = max(8, int(font_size * 0.4)) if rounded else 0
        bg_clip = _rounded_subtitle_background_clip(
            width=box_w,
            height=clip_h,
            color=bg_color,
            alpha=235,
            radius=radius,
        )
        text_pos = _get_visible_center_position(text_clip, box_w, clip_h)
        title_clip = CompositeVideoClip(
            [bg_clip, text_clip.with_position(text_pos)],
            size=(box_w, clip_h),
        )
    else:
        title_clip = text_clip

    duration_mode = getattr(params, "title_duration", "intro")
    if duration_mode == "full":
        title_duration = video_duration
    else:
        title_duration = min(4.5, video_duration)

    title_clip = (
        title_clip.with_start(0.0)
        .with_duration(title_duration)
        .with_end(title_duration)
    )

    anim = getattr(params, "title_animation", "pop_spring")
    title_clip = _apply_subtitle_animation(title_clip, title_duration, anim)

    if duration_mode != "full" and title_duration > 1.0:
        fade_out_dur = min(0.4, title_duration * 0.2)

        def fade_out_transform(get_frame, t):
            frame = get_frame(t)
            time_left = title_duration - t
            if time_left >= fade_out_dur:
                return frame
            p = max(0.0, min(time_left / fade_out_dur, 1.0))
            if frame.ndim == 2:
                return (frame * p).astype(frame.dtype, copy=False)
            if frame.shape[2] == 4:
                fc = frame.copy()
                fc[:, :, 3] = np.clip(fc[:, :, 3] * p, 0, 255).astype(frame.dtype)
                return fc
            return (frame * p).astype(frame.dtype, copy=False)

        title_clip = title_clip.transform(fade_out_transform, apply_to=["mask"])

    pos = getattr(params, "title_position", "top")
    if pos == "top":
        y_pos = max(20.0, video_height * 0.08)
    elif pos == "center":
        y_pos = "center"
    elif pos == "bottom":
        y_pos = max(20.0, video_height * 0.82 - title_clip.h)
    elif pos == "custom":
        # Mirror the subtitle custom-position path: percent maps to a y
        # coordinate from the top of the canvas. Without this branch the
        # previous default fell through to "top" and silently ignored the
        # user's chosen position.
        percent = max(0.0, min(100.0, float(getattr(params, "custom_position", 50))))
        y_pos = max(20.0, video_height * percent / 100 - title_clip.h / 2)
    else:
        y_pos = max(20.0, video_height * 0.08)

    title_clip = title_clip.with_position(("center", y_pos))
    return title_clip


def _try_fast_subtitle_render(
    *,
    video_path: str,
    audio_path: str,
    subtitle_path: str,
    output_file: str,
    font_path: str,
    params: VideoParams,
    video_width: int,
    video_height: int,
    bgm_file_override: str | None,
) -> bool:
    """Render subtitles without entering the frame-by-frame MoviePy pipeline."""
    animation = getattr(params, "subtitle_animation", "none")
    supported_animations = {
        "none",
        "",
        None,
        "scale_up",
        "zoom_in",
        "punch",
        "pop_spring",
        "spring",
        "pop",
        "fade",
        "fade_in",
        "slide_up",
        "rise",
    }
    if not (
        params.subtitle_enabled
        and subtitle_path
        and os.path.exists(subtitle_path)
        and not params.title_enabled
        and not bgm_file_override
        and not _resolve_subtitle_background_color_locally(
            getattr(params, "text_background_color", False)
        )
        and animation in supported_animations
        and font_path
        and os.path.exists(font_path)
    ):
        return False

    common = {
        "video_path": video_path,
        "audio_path": audio_path,
        "srt_path": subtitle_path,
        "output_file": output_file,
        "font_path": font_path,
        "font_size": int(getattr(params, "font_size", 60)),
        "text_color": getattr(params, "text_fore_color", "#FFFFFF") or "#FFFFFF",
        "stroke_color": getattr(params, "stroke_color", None),
        "stroke_width": float(getattr(params, "stroke_width", 0) or 0),
        "voice_volume": float(getattr(params, "voice_volume", 1.0) or 1.0),
        "threads": int(getattr(params, "n_threads", 2) or 2),
        "fps": int(fps),
    }
    try:
        started = perf_counter()
        if _burn_subtitles_with_ffmpeg_ass(
            video_path=video_path,
            audio_path=audio_path,
            srt_path=subtitle_path,
            output_file=output_file,
            font_path=font_path,
            params=params,
            video_width=video_width,
            video_height=video_height,
            voice_volume=common["voice_volume"],
            threads=common["threads"],
        ):
            logger.info(
                f"ASS subtitle burn-in succeeded in "
                f"{perf_counter() - started:.2f}s"
            )
            return True
    except Exception as exc:
        logger.warning(f"ASS subtitle fast path raised: {exc}; falling back")

    display_mode = getattr(params, "subtitle_display_mode", "sentence")
    if animation not in ("none", "", None) or display_mode not in {
        "sentence",
        "word_by_word",
    }:
        return False

    try:
        started = perf_counter()
        if _burn_subtitles_with_ffmpeg_drawtext(**common):
            logger.info(
                f"drawtext subtitle burn-in succeeded in "
                f"{perf_counter() - started:.2f}s"
            )
            return True
    except Exception as exc:
        logger.warning(f"drawtext subtitle fast path raised: {exc}; falling back")

    try:
        started = perf_counter()
        if _burn_subtitles_with_png_overlay(
            **common,
            canvas_width=int(video_width),
            canvas_height=int(video_height),
        ):
            logger.info(
                f"png-overlay subtitle burn-in succeeded in "
                f"{perf_counter() - started:.2f}s"
            )
            return True
    except Exception as exc:
        logger.warning(
            f"png-overlay subtitle fast path raised: {exc}; falling back to MoviePy"
        )
    return False


def _try_fast_final_mux(
    *,
    video_path: str,
    audio_path: str,
    output_file: str,
    params: VideoParams,
    bgm_file_override: str | None,
) -> bool:
    """Stream-copy the video when the final stage only adds narration."""
    if params.subtitle_enabled or params.title_enabled or bgm_file_override:
        return False
    try:
        started = perf_counter()
        if not _final_mux_with_ffmpeg(
            video_path=video_path,
            audio_path=audio_path,
            output_file=output_file,
            voice_volume=float(getattr(params, "voice_volume", 1.0) or 1.0),
            threads=int(getattr(params, "n_threads", 2) or 2),
        ):
            return False
        logger.info(f"final stream-copy mux succeeded in {perf_counter() - started:.2f}s")
        return True
    except Exception as exc:
        logger.warning(f"final stream-copy fast path raised: {exc}; falling back to MoviePy")
        return False


def generate_video(
    video_path: str,
    audio_path: str,
    subtitle_path: str,
    output_file: str,
    params: VideoParams,
    bgm_file_override: str | None = None,
) -> bool:
    """
    Render the final video and report whether background music succeeded.

    The return value describes the BGM stage only: True when no BGM was
    requested or the mix succeeded, False when BGM was requested but loading,
    the effect, or the mix failed. A BGM failure still produces a
    narration-only video, leaving it to the task layer to decide whether to
    show the user a degraded-output warning.
    """
    aspect = VideoAspect(params.video_aspect)
    video_width, video_height = aspect.to_resolution()

    logger.info(f"generating video: {video_width} x {video_height}")
    logger.info(f"  ① video: {video_path}")
    logger.info(f"  ② audio: {audio_path}")
    logger.info(f"  ③ subtitle: {subtitle_path}")
    logger.info(f"  ④ output: {output_file}")

    # https://github.com/harry0703/MoneyPrinterTurbo/issues/217
    # PermissionError: [WinError 32] The process cannot access the file because it is being used by another process: 'final-1.mp4.tempTEMP_MPY_wvf_snd.mp3'
    # write into the same directory as the output file
    output_dir = os.path.dirname(output_file)

    font_path = ""
    if params.subtitle_enabled:
        if not params.font_name:
            params.font_name = "STHeitiMedium.ttc"
        font_path = os.path.join(utils.font_dir(), params.font_name)
        if os.name == "nt":
            font_path = font_path.replace("\\", "/")

        logger.info(f"  ⑤ font: {font_path}")

    if _try_fast_subtitle_render(
        video_path=video_path,
        audio_path=audio_path,
        subtitle_path=subtitle_path,
        output_file=output_file,
        font_path=font_path,
        params=params,
        video_width=video_width,
        video_height=video_height,
        bgm_file_override=bgm_file_override,
    ):
        return True

    if _try_fast_final_mux(
        video_path=video_path,
        audio_path=audio_path,
        output_file=output_file,
        params=params,
        bgm_file_override=bgm_file_override,
    ):
        return True

    def resolve_subtitle_background_color():
        # legacy parameter: the API's `text_background_color` may be a boolean
        # or an actual colour string. normalise it here so True/False never
        # reaches TextClip and renders unpredictably.
        if isinstance(params.text_background_color, bool):
            return "#000000" if params.text_background_color else None
        return params.text_background_color

    def finish_text_clip(clip, subtitle_item):
        duration = subtitle_item[0][1] - subtitle_item[0][0]
        clip = clip.with_start(subtitle_item[0][0])
        clip = clip.with_end(subtitle_item[0][1])
        clip = clip.with_duration(duration)

        anim_type = getattr(params, "subtitle_animation", "none")
        clip = _apply_subtitle_animation(clip, duration, anim_type)

        if params.subtitle_position == "bottom":
            clip = clip.with_position(("center", video_height * 0.95 - clip.h))
        elif params.subtitle_position == "top":
            clip = clip.with_position(("center", video_height * 0.05))
        elif params.subtitle_position in ("two_thirds_bottom", "two_thirds", "2/3_bottom"):
            clip = clip.with_position(("center", (video_height - clip.h) / 3.0))
        elif params.subtitle_position == "custom":
            margin = 10
            max_y = video_height - clip.h - margin
            custom_y = (video_height - clip.h) * (params.custom_position / 100)
            clip = clip.with_position(("center", max(margin, min(custom_y, max_y))))
        else:
            clip = clip.with_position(("center", "center"))
        return clip

    def create_text_clip(subtitle_item):
        params.font_size = int(params.font_size)
        params.stroke_width = int(params.stroke_width)
        phrase = subtitle_item[1]
        casing = getattr(params, "subtitle_casing", "as_is")
        phrase = subtitle_styles.apply_text_casing(phrase, casing)
        max_width = video_width * 0.9
        bg_color = resolve_subtitle_background_color()
        rounded_bg_enabled = bool(
            getattr(params, "rounded_subtitle_background", False) and bg_color
        )
        if subtitle_styles.HIGHLIGHT_OPEN in phrase:
            preset = subtitle_styles.get_subtitle_preset(
                getattr(params, "subtitle_style_preset", "custom")
            ) or {}
            highlighted_clip = _render_highlighted_subtitle_clip(
                phrase,
                font_path=font_path,
                font_size=params.font_size,
                max_width=int(max_width),
                text_color=params.text_fore_color,
                highlight_color=preset.get("highlight_color", "#FFE600"),
                stroke_color=params.stroke_color,
                stroke_width=params.stroke_width,
                background_color=bg_color,
                rounded_background=rounded_bg_enabled,
            )
            return finish_text_clip(highlighted_clip, subtitle_item)

        has_subtitle_background = bool(bg_color)
        # the rounded background is sized to the real text width, so it needs
        # less horizontal padding. the old rectangular background keeps the
        # larger safety margin so long subtitles from existing configs are not
        # flush to the edge or cropped.
        padding_ratio = 0.4 if rounded_bg_enabled else 0.6
        pad_x = int(params.font_size * padding_ratio) if has_subtitle_background else 0
        # a subtitle background needs explicit horizontal padding around the
        # text. subtract the padding from the usable width before wrapping, so
        # long English text or a large font that exactly fills 90% of the video
        # width does not sit flush against the plate and look cropped. both the
        # rectangular and rounded backgrounds use this; subtitles without a
        # background keep the original maximum width.
        text_max_width = max(1, int(max_width) - 2 * pad_x)
        wrapped_txt, txt_height = wrap_text(
            phrase,
            max_width=text_max_width,
            font=font_path,
            fontsize=params.font_size,
        )
        interline = int(params.font_size * 0.25)
        line_count = wrapped_txt.count("\n") + 1
        vertical_padding = int(params.font_size * 0.35)
        # Pillow and MoviePy expand a stroke above and below the glyphs and
        # count that in each line's advance height. adding stroke padding once
        # around the whole subtitle block still accumulates error line by line
        # with a thick stroke. account for both sides per line instead: a thin
        # default stroke adds only a little height, while "small font + thick
        # stroke + several lines" still renders in full.
        stroke_padding = int(params.stroke_width * 2 * line_count)
        text_clip_margin_y = max(
            int(params.font_size * 0.3), int(params.stroke_width * 2)
        )
        # with `method=label` MoviePy shrinks the text box height on its own
        # and readily crops the bottom half of the last line once subtitles are
        # multi-line, stroked, or have a background colour. pass a more
        # conservative height that includes line spacing and extra vertical
        # padding, so both the background plate and the text render in full.
        clip_h = int(
            txt_height
            + vertical_padding
            + (interline * line_count)
            + stroke_padding
        )

        if rounded_bg_enabled:
            # the rounded background hugs the text width rather than reusing
            # 90% of the video width. measure the longest line with PIL and add
            # horizontal padding, so a short subtitle does not get an
            # over-wide plate.
            try:
                font = ImageFont.truetype(font_path, params.font_size)
                text_w = max(
                    int(font.getbbox(line)[2] - font.getbbox(line)[0])
                    for line in wrapped_txt.split("\n")
                )
            except Exception as exc:
                logger.warning(
                    f"failed to measure subtitle text width, fallback to max width: {str(exc)}"
                )
                text_w = int(max_width)

            box_w = max(1, min(int(max_width), text_w + 2 * pad_x))
            radius = max(8, int(params.font_size * 0.4))
            text_clip = TextClip(
                text=wrapped_txt,
                font=font_path,
                font_size=params.font_size,
                color=params.text_fore_color,
                bg_color=None,
                stroke_color=params.stroke_color,
                stroke_width=params.stroke_width,
                interline=interline,
                size=(box_w, None),
                text_align="center",
                margin=(0, text_clip_margin_y),
            )
            clip_h = max(clip_h, text_clip.h)
            bg_clip = _rounded_subtitle_background_clip(
                width=box_w,
                height=clip_h,
                color=bg_color,
                alpha=140,
                radius=radius,
            )
            text_position = _get_visible_center_position(text_clip, box_w, clip_h)
            _clip = CompositeVideoClip(
                [bg_clip, text_clip.with_position(text_position)],
                size=(box_w, clip_h),
            )
        elif bg_color:
            size = (
                int(max_width),
                clip_h,
            )
            text_clip = TextClip(
                text=wrapped_txt,
                font=font_path,
                font_size=params.font_size,
                color=params.text_fore_color,
                bg_color=None,
                stroke_color=params.stroke_color,
                stroke_width=params.stroke_width,
                interline=interline,
                size=(int(max_width), None),
                text_align="center",
                margin=(0, text_clip_margin_y),
            )
            size = (size[0], max(size[1], text_clip.h))
            bg_clip = _rounded_subtitle_background_clip(
                width=size[0],
                height=size[1],
                color=bg_color,
                alpha=255,
                radius=0,
            )
            text_position = _get_visible_center_position(text_clip, size[0], size[1])
            _clip = CompositeVideoClip(
                [bg_clip, text_clip.with_position(text_position)],
                size=size,
            )
        else:
            size = (
                int(max_width),
                clip_h,
            )
            _clip = TextClip(
                text=wrapped_txt,
                font=font_path,
                font_size=params.font_size,
                color=params.text_fore_color,
                bg_color=None,
                stroke_color=params.stroke_color,
                stroke_width=params.stroke_width,
                interline=interline,
                size=size,
                text_align="center",
            )
        return finish_text_clip(_clip, subtitle_item)

    # MoviePy's CompositeAudioClip.close() does not close the child
    # AudioFileClips. hold every source reader in an ExitStack so the FFmpeg
    # subprocesses are released on success, on a subtitle error, on a mix
    # failure, and on a write failure -- which above all keeps files from
    # staying locked on Windows.
    with ExitStack() as clip_stack:
        source_video_clip = clip_stack.enter_context(
            _open_video_clip_quietly(video_path)
        )
        voice_source_clip = clip_stack.enter_context(AudioFileClip(audio_path))
        video_clip = source_video_clip
        audio_clip = voice_source_clip.with_effects(
            [afx.MultiplyVolume(params.voice_volume)]
        )

        def make_textclip(text):
            return TextClip(
                text=text,
                font=font_path,
                font_size=params.font_size,
            )

        clips_to_composite = [video_clip]
        if subtitle_path and os.path.exists(subtitle_path):
            sub = clip_stack.enter_context(
                SubtitlesClip(
                    subtitles=subtitle_path,
                    encoding="utf-8",
                    make_textclip=make_textclip,
                )
            )
            text_clips = []
            display_cues = subtitle_styles.build_display_cues(
                sub.subtitles,
                getattr(params, "subtitle_display_mode", "sentence"),
            )
            for item in display_cues:
                clip = create_text_clip(subtitle_item=item)
                text_clips.append(clip)
            clips_to_composite.extend(text_clips)

        title_clip = _create_title_clip(
            params=params,
            video_width=video_width,
            video_height=video_height,
            video_duration=video_clip.duration,
        )
        if title_clip is not None:
            clips_to_composite.append(title_clip)

        if len(clips_to_composite) > 1:
            video_clip = CompositeVideoClip(clips_to_composite)
            clip_stack.callback(video_clip.close)

        bgm_enabled = bgm_service.should_use_bgm(
            params.bgm_type, params.bgm_volume
        )
        if not bgm_enabled and params.bgm_type:
            # every BGM source shares this short circuit. at a volume of zero
            # or less, neither a random nor a custom file is resolved and a
            # provider-supplied file is not loaded, avoiding pointless IO and
            # mixing.
            logger.info(
                f"skipping background music because volume is not positive: "
                f"type={params.bgm_type}, volume={params.bgm_volume}"
            )

        # the task layer may pass a provider's soundtrack file directly. None
        # keeps the random/custom BGM resolution, an empty string disables BGM
        # explicitly -- but every source still has to pass the volume rule above.
        bgm_file = ""
        if bgm_enabled:
            bgm_file = (
                bgm_file_override
                if bgm_file_override is not None
                else get_bgm_file(
                    bgm_type=params.bgm_type,
                    bgm_file=params.bgm_file,
                )
            )
        bgm_mix_succeeded = True
        if bgm_file:
            try:
                bgm_effects = [
                    afx.MultiplyVolume(params.bgm_volume),
                    afx.AudioFadeOut(3),
                ]
                # random or custom music resolved inside this service may be
                # shorter than the video and has to be looped. a file passed in
                # by the task layer means the provider already matched the
                # duration. decide by origin rather than by a name allowlist
                # that would need editing for every new provider.
                if bgm_file_override is None:
                    bgm_effects.append(afx.AudioLoop(duration=video_clip.duration))
                bgm_source_clip = clip_stack.enter_context(AudioFileClip(bgm_file))
                bgm_clip = bgm_source_clip.with_effects(bgm_effects)
                audio_clip = CompositeAudioClip([audio_clip, bgm_clip])
            except Exception:
                bgm_mix_succeeded = False
                # log the full stack and stable context so a file-decoding
                # failure, a MoviePy effect failure, and a CompositeAudioClip
                # failure stay distinguishable. no file content or API key
                # reaches the log.
                logger.exception(
                    f"failed to mix background music: type={params.bgm_type}, "
                    f"file={bgm_file}"
                )

        final_video_clip = video_clip.with_audio(audio_clip)
        clip_stack.callback(final_video_clip.close)
        # reuse the input audio's sample rate explicitly, falling back to
        # MoviePy's 44100 Hz default when it cannot be read. that avoids another
        # resample and the quality swings it causes across environments,
        # Docker in particular.
        output_audio_fps = int(getattr(audio_clip, "fps", 0) or 44100)
        _write_videofile_with_codec_fallback(
            final_video_clip,
            output_file=output_file,
            codec=_get_configured_video_codec(),
            audio_codec=audio_codec,
            audio_fps=output_audio_fps,
            audio_bitrate=audio_bitrate,
            temp_audiofile_path=_get_temp_audio_dir(output_dir),
            threads=params.n_threads or 2,
            logger=None,
            fps=fps,
        )
        return bgm_mix_succeeded


def render_image_zoom_video(image_path: str, clip_duration: int = 5) -> str:
    """
    Render one local image into an mp4 clip with a slow zoom, returning the
    output path.

    Local material preprocessing and OpenAI-compatible text-to-image material
    share this "image -> clip" rendering: an ImageClip plays for a fixed
    clip_duration with roughly 3% zoom per second, so a still frame does not
    look inert in the final video. Rendering errors are left to the caller,
    which handles them per its own material source's failure contract.
    """
    clip = ImageClip(image_path).with_duration(clip_duration).with_position("center")
    try:
        # Apply a zoom effect using the resize method.
        # A lambda function is used to make the zoom effect dynamic over time.
        # The zoom effect starts from the original size and gradually scales up to 120%.
        # t represents the current time, and clip.duration is the total duration of the clip.
        # Note: 1 represents 100% size, so 1.2 represents 120%.
        zoom_clip = clip.resized(
            lambda t: 1 + (clip_duration * 0.03) * (t / clip.duration)
        )

        # Optionally, create a composite video clip containing the zoomed clip.
        # This is useful if you want to add other elements to the video.
        final_clip = CompositeVideoClip([zoom_clip])
        try:
            # Output the video to a file.
            video_file = f"{image_path}.mp4"
            _write_videofile_with_codec_fallback(
                final_clip,
                video_file,
                _get_configured_video_codec(),
                fps=30,
                logger=None,
            )
            return video_file
        finally:
            close_clip(final_clip)
    finally:
        close_clip(clip)


def preprocess_video(materials: List[MaterialInfo], clip_duration=4):
    # in some re-generation flows the WebUI passes an empty material list;
    # return an empty result rather than raising a NoneType error.
    if not materials:
        return []

    # return only material that passed preprocessing, so a low-resolution image
    # never reaches the video combination stage.
    valid_materials = []
    local_videos_dir = utils.storage_dir("local_videos", create=True)

    for material in materials:
        if not material.url:
            continue

        try:
            material_source_path = file_security.resolve_path_within_directory(
                local_videos_dir, material.url
            )
        except ValueError as exc:
            # a local video_source path comes from an API parameter and must
            # stay inside the dedicated material directory. a bare filename is
            # allowed, as is a legacy absolute path, but nothing may escape
            # elsewhere on the system -- that would be arbitrary file read, or
            # probing sensitive local files through MoviePy.
            logger.warning(
                f"skip unsafe local material: {material.url}, "
                f"local_videos_dir: {local_videos_dir}, error: {str(exc)}"
            )
            continue

        ext = utils.parse_extension(material_source_path)
        try:
            # read image material as an image directly, instead of letting
            # VideoFileClip misjudge it and trigger the flaky fallback branch.
            if ext in const.FILE_TYPE_IMAGES:
                clip, material_source_path = _open_image_clip_with_fallback(
                    material_source_path
                )
            else:
                clip = _open_video_clip_quietly(material_source_path)
        except Exception:
            # on an unusual extension or a failed probe, fall back to image
            # mode, which keeps working for callers that historically passed a
            # local image path directly.
            try:
                clip, material_source_path = _open_image_clip_with_fallback(
                    material_source_path
                )
            except Exception as exc:
                logger.warning(
                    f"skip unreadable local material: {material.url}, error: {str(exc)}"
                )
                continue
        try:
            width = clip.size[0]
            height = clip.size[1]
            if not is_material_resolution_acceptable(width, height):
                logger.warning(
                    f"low resolution material: {width}x{height}, minimum "
                    f"{_MIN_MATERIAL_DIMENSION}x{_MIN_MATERIAL_DIMENSION} required "
                    f"(tolerance {_MIN_DIMENSION_TOLERANCE}px)"
                )
                # close the handle as soon as low-resolution material is
                # detected, and do not return that material downstream.
                close_clip(clip)
                continue

            if ext in const.FILE_TYPE_IMAGES:
                logger.info(f"processing image: {material_source_path}")
                # the material was already opened once to read its size;
                # release that handle before rendering the clip to export.
                close_clip(clip)
                video_file = render_image_zoom_video(
                    material_source_path, clip_duration
                )
                material.url = video_file
                logger.success(f"image processed: {video_file}")
            else:
                # ordinary video material is only opened to validate its size,
                # so release the handle as soon as that is done.
                close_clip(clip)
                # Update url to the resolved absolute path so that downstream
                # stages (combine_videos) can open the file without re-resolving.
                material.url = material_source_path
        except Exception:
            close_clip(clip)
            raise

        valid_materials.append(material)

    return valid_materials
