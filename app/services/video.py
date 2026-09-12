import itertools
import io
import math
import os
import random
import gc
import subprocess
import sys
import tempfile
import unicodedata
from contextlib import ExitStack, redirect_stdout
from functools import lru_cache
from typing import List
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
from app.services import guardrails, subtitle_styles
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
# libx264 is the software fallback used whenever a hardware encoder is absent
# or failed at runtime. The default policy for an unset video_codec is "auto":
# probe for a platform hardware encoder and only then fall back to libx264, so
# a GPU passes through by default and a CPU-only host keeps working unchanged.
_DEFAULT_VIDEO_CODEC = "libx264"
_SUBTITLE_SPRING_DURATION_SECONDS = 0.18
_MIN_SUBTITLE_SPRING_SCALE = 0.05
_MAX_SUBTITLE_SPRING_SCALE = 1.35
_SUPPORTED_VIDEO_CODECS = (
    "libx264",
    "h264_nvenc",
    "h264_amf",
    "h264_qsv",
    "h264_mf",
    "h264_videotoolbox",
    "auto",
)
_runtime_disabled_video_codecs = set()
# without preset/quality/bitrate, a hardware encoder makes ffmpeg either error
# out or quietly fall back to software encoding. give each hardware encoder the
# minimum ffmpeg parameters that work; MoviePy's write_videofile passes
# ffmpeg_params straight through to the final ffmpeg call.
# ponytail: one global parameter table, extend as needed; caller-supplied
# ffmpeg_params win over these.
_HARDWARE_CODEC_FFMPEG_PARAMS = {
    "h264_nvenc": ["-preset", "p4", "-rc", "vbr", "-b:v", "5M"],
    "h264_amf": ["-usage", "transcoding", "-quality", "balanced", "-b:v", "5M"],
    "h264_qsv": ["-preset", "veryfast", "-b:v", "5M"],
    "h264_mf": ["-b:v", "5M"],
    "h264_videotoolbox": ["-b:v", "5M"],
}
# Auto-detection picks the first codec available in the bundled ffmpeg, ordered
# by what is most likely to be useful on each platform.
# ponytail: per-OS priority list; reorder when new hardware backends land.
_HARDWARE_CODEC_AUTO_PRIORITY = {
    "darwin": ("h264_videotoolbox", "h264_qsv"),
    "linux": ("h264_nvenc", "h264_qsv", "h264_amf", "h264_videotoolbox"),
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
    """
    if not subclipped_items:
        return []

    concat_mode_value = getattr(concat_mode, "value", concat_mode)

    grouped_items: dict[str, list[SubClippedVideoClip]] = {}
    for item in subclipped_items:
        grouped_items.setdefault(item.source_file_path, []).append(item)

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

    random.shuffle(primary_items)
    random.shuffle(overflow_items)
    logger.info(
        "prioritized unique video materials, "
        f"sources: {len(grouped_items)}, "
        f"primary clips: {len(primary_items)}, "
        f"fallback clips: {len(overflow_items)}"
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
    hardware encoder (NVENC/AMF/QSV/VideoToolbox) and fall back to libx264 when
    none can be used. Only a fixed allowlist is accepted on purpose: opening it
    to arbitrary FFmpeg parameters would let a typo produce an unpredictable
    output format, or fail the task at a much later stage.
    """
    configured_codec = str(config.app.get("video_codec", "auto") or "auto").strip()
    if configured_codec not in _SUPPORTED_VIDEO_CODECS:
        logger.warning(
            f"unsupported video codec configured: {configured_codec}, "
            f"fallback to {_DEFAULT_VIDEO_CODEC}"
        )
        return _DEFAULT_VIDEO_CODEC
    return configured_codec


@lru_cache(maxsize=16)
def _ffmpeg_encoder_exists(ffmpeg_binary: str, codec: str) -> bool:
    """
    Check whether this FFmpeg build advertises the given encoder.

    That only proves the encoder was compiled in, not that this machine's
    hardware and drivers can actually use it, so the runtime smoke test and
    real encoding path still fall back to libx264.
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
            f"ffmpeg encoder probe failed for {ffmpeg_binary}, "
            f"fallback to {_DEFAULT_VIDEO_CODEC}: {exc}"
        )
        return False

    if result.returncode != 0:
        stderr_excerpt = (result.stderr or result.stdout or "").strip()[:200]
        logger.warning(
            f"ffmpeg encoder probe failed (rc={result.returncode}) for "
            f"{ffmpeg_binary}, fallback to {_DEFAULT_VIDEO_CODEC}: {stderr_excerpt}"
        )
        return False
    return codec in result.stdout


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
    command.extend(_HARDWARE_CODEC_FFMPEG_PARAMS.get(codec, []))
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
            f"failed to smoke-test encoder {codec}, "
            f"fallback to {_DEFAULT_VIDEO_CODEC}: {str(exc)}"
        )
        return False
    if result.returncode != 0:
        logger.warning(
            f"encoder {codec} smoke test failed, "
            f"fallback to {_DEFAULT_VIDEO_CODEC}: "
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
    if selected_codec == _DEFAULT_VIDEO_CODEC:
        return _DEFAULT_VIDEO_CODEC

    if selected_codec == "auto":
        resolved = _detect_hardware_codec(utils.get_ffmpeg_binary())
        if resolved is None:
            logger.info(
                f"no hardware encoder available on {sys.platform}, "
                f"fallback to {_DEFAULT_VIDEO_CODEC}"
            )
            return _DEFAULT_VIDEO_CODEC
        logger.info(f"auto-detected hardware codec: {resolved}")
        return resolved

    if selected_codec in _runtime_disabled_video_codecs:
        logger.warning(
            f"video codec {selected_codec} was disabled after a runtime failure, "
            f"fallback to {_DEFAULT_VIDEO_CODEC}"
        )
        return _DEFAULT_VIDEO_CODEC

    ffmpeg_binary = utils.get_ffmpeg_binary()
    if not _ffmpeg_encoder_exists(ffmpeg_binary, selected_codec):
        logger.warning(
            f"ffmpeg encoder {selected_codec} is not available, "
            f"fallback to {_DEFAULT_VIDEO_CODEC}"
        )
        return _DEFAULT_VIDEO_CODEC

    if not _ffmpeg_encoder_runnable(ffmpeg_binary, selected_codec):
        logger.warning(
            f"ffmpeg encoder {selected_codec} cannot encode on this host "
            f"(no compatible GPU or driver), fallback to {_DEFAULT_VIDEO_CODEC}"
        )
        return _DEFAULT_VIDEO_CODEC

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
    caller maps to the software fallback.
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
    if codec == _DEFAULT_VIDEO_CODEC:
        return
    _runtime_disabled_video_codecs.add(codec)
    logger.warning(
        f"video codec {codec} failed, fallback to {_DEFAULT_VIDEO_CODEC}. "
        f"reason: {reason}"
    )


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


def _fallback_write_videofile(clip, output_file: str, failed_codec: str, reason: str, **kwargs):
    """
    Retry with libx264 after a hardware encode fails, and disable the hardware
    encoder only if that retry succeeds.

    On Windows an FFmpeg failure has many possible causes: an unsupported GPU or
    driver, but equally a locked output file, directory permissions, or
    antivirus interference. Only when libx264 writes successfully is the
    original failure likely the hardware encoder itself, so later tasks are not
    penalised for a generic IO problem.

    The retry drops the hardware-specific ffmpeg_params so options like -rc and
    preset, which only mean something to a hardware encoder, are never fed to
    libx264.
    """
    kwargs.pop("ffmpeg_params", None)
    clip.write_videofile(output_file, codec=_DEFAULT_VIDEO_CODEC, **kwargs)
    _disable_runtime_video_codec(failed_codec, reason)
    return _DEFAULT_VIDEO_CODEC


def _write_videofile_with_codec_fallback(clip, output_file: str, codec: str, **kwargs):
    """
    Write the video with the requested encoder, retrying once with libx264.

    Whether a hardware encoder works depends not only on FFmpeg but on the GPU,
    the driver, and the runtime environment. A generation task must not fail
    outright because an advanced encoder is unavailable, so the fallback is
    handled in one place.

    A hardware encoder invoked with no preset/quality/bitrate may not work at
    all, so inject the minimum working parameters per codec; ffmpeg_params from
    the caller take precedence.
    """
    effective_codec = _get_effective_video_codec(codec)
    if (
        effective_codec in _HARDWARE_CODEC_FFMPEG_PARAMS
        and "ffmpeg_params" not in kwargs
    ):
        kwargs["ffmpeg_params"] = _HARDWARE_CODEC_FFMPEG_PARAMS[effective_codec]
    try:
        clip.write_videofile(output_file, codec=effective_codec, **kwargs)
        return effective_codec
    except Exception as exc:
        if effective_codec == _DEFAULT_VIDEO_CODEC:
            raise
        return _fallback_write_videofile(
            clip,
            output_file,
            failed_codec=effective_codec,
            reason=str(exc),
            **kwargs,
        )


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
            "-pix_fmt",
            "yuv420p",
        ]
        # same minimal parameters as the MoviePy write path, so a hardware
        # encoder picked by "auto" gets preset/quality/bitrate it can use.
        if codec in _HARDWARE_CODEC_FFMPEG_PARAMS:
            command.extend(_HARDWARE_CODEC_FFMPEG_PARAMS[codec])
        if max_duration is not None and max_duration > 0:
            command.extend(["-t", f"{max_duration:.3f}"])
        command.append(output_file)
        return command

    def run_concat(codec: str):
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

    try:
        effective_codec = _get_effective_video_codec()
        try:
            return run_concat(effective_codec)
        except Exception as exc:
            if effective_codec == _DEFAULT_VIDEO_CODEC:
                raise
            result_codec = run_concat(_DEFAULT_VIDEO_CODEC)
            _disable_runtime_video_codec(effective_codec, str(exc))
            return result_codec
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
) -> str:
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
    for video_path in video_paths:
        clip = _open_video_clip_quietly(video_path)
        clip_duration = clip.duration
        clip_w, clip_h = clip.size
        close_clip(clip)
        
        start_time = 0

        while start_time < clip_duration:
            end_time = min(start_time + source_clip_duration, clip_duration)

            # keep every valid slice. that neither drops a material shorter
            # than max_clip_duration in its entirety, nor swallows the short
            # tail left over at the end of a long video.
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

    subclipped_items = _prioritize_unique_source_clips(
        subclipped_items=subclipped_items,
        concat_mode=video_concat_mode,
    )
        
    logger.debug(f"total subclipped items: {len(subclipped_items)}")
    
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
        logger.info(
            f"video duration: {video_duration:.2f}s, audio duration: {audio_duration:.2f}s, "
            f"required duration: {required_video_duration:.2f}s, "
            f"looped {len(processed_clips)-len(base_clips)} clips"
        )
     
    # merge video clips progressively, avoid loading all videos at once to avoid memory overflow
    logger.info("starting clip merging process")
    if not processed_clips:
        logger.warning("no clips available for merging")
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

    casing = style_cfg.get("casing", "uppercase")
    title_text = subtitle_styles.apply_text_casing(title_text, casing)

    font_name = (
        getattr(params, "title_font_name", None)
        or style_cfg.get("font_name")
        or getattr(params, "font_name", "Anton-Regular.ttf")
    )
    available_fonts = [
        f for f in os.listdir(utils.font_dir()) if f.endswith((".ttf", ".ttc"))
    ]
    if font_name not in available_fonts:
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
    else:
        y_pos = max(20.0, video_height * 0.08)

    title_clip = title_clip.with_position(("center", y_pos))
    return title_clip


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
