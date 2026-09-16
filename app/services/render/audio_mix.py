"""Single ffmpeg audio chain: narration + ducked BGM, loudness-normalized once.

Produces one mixed, loudness-normalized WAV that the render stage's fast
paths (and the MoviePy fallback) use as their sole audio input, so every
final video lands at the same target loudness regardless of whether BGM is
present. This replaces per-path ad-hoc volume multiplication with the two-
pass ``loudnorm`` ffmpeg already ships, run on the actual mixed signal.

ponytail: the sidechain ducking constants (threshold/ratio/attack/release)
are fixed, not tuned per BGM track; revisit if a loud BGM still masks
narration.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from typing import Optional

from loguru import logger

from app.utils import utils

# First pass stabilizes the narration level so the sidechain compressor
# ducks against a predictable signal; the final pass hits the platform target.
_PRENORM_I, _PRENORM_TP = -16.0, -1.5
_TARGET_I, _TARGET_TP, _TARGET_LRA = -14.0, -1.0, 11.0
_DUCK_THRESHOLD, _DUCK_RATIO = 0.05, 8
_DUCK_ATTACK_MS, _DUCK_RELEASE_MS = 200, 600

_LOUDNORM_JSON_RE = re.compile(r"\{[^{}]*\"input_i\"[^{}]*\}", re.DOTALL)


def _ffprobe_binary() -> str:
    """Resolve ffprobe -- a read-only prober, so unlike ffmpeg it never needs
    to go through a host-encoder proxy/wrapper (e.g. the Mac VideoToolbox
    bridge script). It ships next to ffmpeg on every platform (Docker,
    Linux, Windows, native Mac), so a plain PATH lookup is enough.
    """
    configured = os.environ.get("FFPROBE_BINARY", "").strip()
    if configured:
        return configured
    found = shutil.which("ffprobe")
    if found:
        return found
    return "ffprobe"


def probe_duration(path: str) -> float:
    result = subprocess.run(
        [_ffprobe_binary(), "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
        capture_output=True, text=True, check=False, timeout=30,
    )
    try:
        return float(result.stdout.strip() or 0.0)
    except ValueError:
        return 0.0


def _mix_filter(*, has_bgm: bool, voice_volume: float, bgm_volume: float, duration: float, final_pass: Optional[dict]) -> str:
    voice_chain = f"[0:a]volume={voice_volume:.3f}[voice]"
    if has_bgm:
        bgm_chain = (
            f"[1:a]aloop=loop=-1:size=2000000000,atrim=0:{duration:.3f},"
            f"asetpts=N/SR/TB,volume={bgm_volume:.3f}[bgm_raw];"
            f"[bgm_raw][voice]sidechaincompress=threshold={_DUCK_THRESHOLD}:ratio={_DUCK_RATIO}:"
            f"attack={_DUCK_ATTACK_MS}:release={_DUCK_RELEASE_MS}[bgm_ducked]"
        )
        mix_chain = f"{voice_chain};{bgm_chain};[voice][bgm_ducked]amix=inputs=2:duration=first:normalize=0[premix]"
    else:
        mix_chain = f"{voice_chain};[voice]anull[premix]"

    if final_pass is None:
        loudnorm = f"loudnorm=I={_TARGET_I}:TP={_TARGET_TP}:LRA={_TARGET_LRA}:print_format=json"
    else:
        loudnorm = (
            f"loudnorm=I={_TARGET_I}:TP={_TARGET_TP}:LRA={_TARGET_LRA}:"
            f"measured_I={final_pass['input_i']}:measured_TP={final_pass['input_tp']}:"
            f"measured_LRA={final_pass['input_lra']}:measured_thresh={final_pass['input_thresh']}:"
            f"offset={final_pass.get('target_offset', 0)}:linear=true"
        )
    return f"{mix_chain};[premix]{loudnorm}[aout]"


def _run_ffmpeg(voice_path: str, bgm_path: Optional[str], filter_complex: str, extra_output: list[str]) -> subprocess.CompletedProcess:
    command = [utils.get_ffmpeg_binary(), "-y", "-i", voice_path]
    if bgm_path:
        command += ["-i", bgm_path]
    command += ["-filter_complex", filter_complex, "-map", "[aout]"] + extra_output
    return subprocess.run(command, capture_output=True, text=True, check=False, timeout=300)


def render_mixed_audio(
    *,
    voice_path: str,
    bgm_path: Optional[str],
    voice_volume: float,
    bgm_volume: float,
    output_path: str,
) -> bool:
    """Write a loudness-normalized, ducked mix of narration (+ optional BGM).

    Two ffmpeg passes, both audio-only (no video decode/encode): measure the
    mixed signal's loudness, then apply the platform target with the
    measured values so ``loudnorm`` runs in its accurate linear mode.
    """
    if not voice_path or not os.path.isfile(voice_path):
        return False
    has_bgm = bool(bgm_path and os.path.isfile(bgm_path))
    duration = probe_duration(voice_path)
    if duration <= 0:
        logger.warning(f"audio_mix: could not probe duration of {voice_path}")
        return False

    measure = _run_ffmpeg(
        voice_path, bgm_path if has_bgm else None,
        _mix_filter(has_bgm=has_bgm, voice_volume=voice_volume, bgm_volume=bgm_volume, duration=duration, final_pass=None),
        ["-f", "null", "-"],
    )
    match = _LOUDNORM_JSON_RE.search(measure.stderr or "")
    if not match:
        logger.warning(
            f"audio_mix: loudnorm measurement pass produced no JSON; stderr tail: "
            f"{(measure.stderr or '')[-300:]}"
        )
        return False
    try:
        measured = json.loads(match.group(0))
    except ValueError:
        return False

    apply_result = _run_ffmpeg(
        voice_path, bgm_path if has_bgm else None,
        _mix_filter(has_bgm=has_bgm, voice_volume=voice_volume, bgm_volume=bgm_volume, duration=duration, final_pass=measured),
        ["-ar", "48000", "-ac", "2", "-c:a", "pcm_s16le", output_path],
    )
    if apply_result.returncode != 0:
        logger.warning(
            f"audio_mix: apply pass failed (rc={apply_result.returncode}); "
            f"stderr tail: {(apply_result.stderr or '')[-300:]}"
        )
        return False
    return True
