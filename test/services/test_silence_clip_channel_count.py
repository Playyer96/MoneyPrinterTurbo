"""Pin MoviePy's mono-``AudioArrayClip`` write bug: a single-channel silence
array is written to disk at DOUBLE its claimed duration.

Discovered while debugging a real render whose intro/outro overlays each
came out exactly 2x their configured length (a 5s intro became 10s of
actual video, throwing the whole concatenated timeline 6+ seconds out of
sync). Every place in the render code that builds silence must use a
stereo (n, 2) array, never mono (n, 1) -- this test proves why, against
the actual installed MoviePy/ffmpeg writer, not just the in-memory
``.duration`` property (which reports correctly for both and would not
have caught this).
"""

from __future__ import annotations

import subprocess

import numpy as np
from moviepy import AudioArrayClip


def _probe_duration(path: str) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
        capture_output=True, text=True, check=True,
    )
    return float(result.stdout.strip())


def test_stereo_silence_writes_its_real_claimed_duration(tmp_path):
    sample_rate = 44100
    claimed_seconds = 3.0
    stereo_silence = AudioArrayClip(
        np.zeros((int(round(claimed_seconds * sample_rate)), 2), dtype=np.float32),
        fps=sample_rate,
    )
    assert stereo_silence.duration == claimed_seconds

    out_path = str(tmp_path / "stereo.wav")
    stereo_silence.write_audiofile(out_path, fps=sample_rate, logger=None)

    written = _probe_duration(out_path)
    assert abs(written - claimed_seconds) < 0.05, (
        f"stereo silence should write {claimed_seconds}s, got {written}s"
    )


def test_mono_silence_reproduces_the_double_duration_bug(tmp_path):
    """Documents the exact bug this fix works around, so an upgraded
    MoviePy/ffmpeg that no longer has it is visible here rather than
    silently -- if this starts failing, the stereo workaround in
    ``app/services/video.py`` can likely be simplified back to mono."""
    sample_rate = 44100
    claimed_seconds = 3.0
    mono_silence = AudioArrayClip(
        np.zeros((int(round(claimed_seconds * sample_rate)), 1), dtype=np.float32),
        fps=sample_rate,
    )
    assert mono_silence.duration == claimed_seconds

    out_path = str(tmp_path / "mono.wav")
    mono_silence.write_audiofile(out_path, fps=sample_rate, logger=None)

    written = _probe_duration(out_path)
    assert written > claimed_seconds * 1.5, (
        "expected the known mono-AudioArrayClip write bug (~2x duration); "
        f"got {written}s for a claimed {claimed_seconds}s -- if this now "
        "passes, the library bug may be fixed and the stereo workaround "
        "in app/services/video.py could be revisited"
    )
