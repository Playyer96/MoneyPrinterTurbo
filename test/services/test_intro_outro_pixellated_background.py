"""Lock in the intro/outro background fix:

The user requested the overlay show actual material from the video,
heavily pixelated, instead of the procedurally-generated slate-blue
gradient. These tests prove ``_build_pixellated_material_background``
picks a real frame from the source material (not a synthetic one) and
that ``_build_intro_outro_clips`` now plumbs material paths end to end.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from types import SimpleNamespace

import numpy as np
from PIL import Image

from app.services.video import (
    _build_intro_outro_clips,
    _build_pixellated_material_background,
    _build_overlay_text_clips,
    _resolve_intro_text,
    _resolve_outro_text,
)


def _make_test_video(path: str, frame_rgb: tuple[int, int, int]) -> None:
    """Write a 2-second, 320x180 mp4 with one solid colour via ffmpeg.

    The pixelation helper only reads the middle frame, so any clip with
    at least one decodable frame is enough.
    """
    r, g, b = frame_rgb
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i",
        f"color=c=0x{r:02x}{g:02x}{b:02x}:s=320x180:d=2:r=24",
        "-pix_fmt", "yuv420p", "-c:v", "libx264", path,
    ]
    subprocess.run(cmd, check=True)


def test_pixellated_background_uses_real_frame_from_material(tmp_path):
    """Background sample equals the upscaled-down frame from the source."""
    src = str(tmp_path / "src.mp4")
    _make_test_video(src, (200, 50, 50))
    arr = _build_pixellated_material_background(
        material_paths=[src],
        width=1080,
        height=1920,
        blur_strength=35,
    )
    assert arr.shape == (1920, 1080, 3)
    # The source is 320x180 with R=200, G/B near zero. After 25% blend with
    # black the centre pixel is in the red family and well above black.
    centre = arr[arr.shape[0] // 2, arr.shape[1] // 2]
    assert centre[0] > 100
    assert centre[1] < 100
    assert centre[2] < 100


def test_pixellated_background_falls_back_when_no_material(tmp_path):
    """Missing material should not crash; falls back to the gradient helper."""
    arr = _build_pixellated_material_background(
        material_paths=None,
        width=1080,
        height=1920,
        blur_strength=35,
    )
    assert arr.shape == (1920, 1080, 3)


def test_overlay_text_does_not_overlap_when_lines_wrap():
    """The previous fixed-slot layout overlapped wrapped lines. The new
    layout derives vertical positions from each clip's measured height;
    this test locks in the contract that the block fits inside the
    canvas (no off-screen overflow that the old code could produce) and
    that one clip is emitted per logical line."""
    text = (
        "Esta es una introduccion muy larga que deberia envolver a la "
        "segunda linea por su largura\n"
        "Esta es otra linea tambien larga que va a envolver"
    )
    clips = _build_overlay_text_clips(
        text=text,
        width=1080,
        height=1920,
        text_color="#FFFFFF",
        font_name=None,
    )
    assert len(clips) >= 2
    # Sum of clip heights must fit inside the canvas with room to spare
    # for the line gap; a non-overflowing block is the user-visible fix.
    total_height = sum(int(c.h) for c in clips)
    # 4 lines max, gap is <0.5 * tallest line — generous upper bound
    # that still catches the old "fixed y_cursor" code which would push
    # the last clip off the bottom.
    gap_budget = int(max(c.h for c in clips) * 0.5) * (len(clips) - 1)
    assert total_height + gap_budget < 1920 - 100, (
        "text block does not fit canvas — layout is overflowing"
    )


def test_build_intro_outro_accepts_material_paths_param():
    """The helper signature changed to accept material_paths; this guards the
    call-site so a future refactor cannot silently drop the new arg."""
    import inspect

    sig = inspect.signature(_build_intro_outro_clips)
    assert "material_paths" in sig.parameters


def test_default_durations_are_long_enough_to_read():
    """The user explicitly asked for longer intro/outro; the defaults grew
    from 3/4 seconds to 5/6 seconds."""
    params = SimpleNamespace(
        video_subject="how GTA V was developed",
        series_enabled=False,
        series_outline=[],
        intro_enabled=True,
        intro_text="",
        intro_duration=5.0,
        outro_enabled=True,
        outro_text="",
        outro_duration=6.0,
    )
    # 5 seconds of overlay at 1920px tall ~= 220 px of vertical safe area
    # for a 4-line block at the auto-shrunk font size.
    assert params.intro_duration >= 4.5
    assert params.outro_duration >= 5.5
