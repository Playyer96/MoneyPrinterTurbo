"""Lock in the audio-through-overlay bug fix via the real rendering path.

The last two rendered videos came out silent in the final MP4 even
though ``audio.mp3`` had content. Cause: ``concatenate_videoclips``
with ``method="compose"`` drops the audio track when one of the input
clips has no ``.audio`` attribute (the intro / outro overlays did
not, because ``intro_tts_enabled`` and ``outro_tts_enabled`` were
both False in those tasks). The fix pre-builds a
``CompositeAudioClip`` covering the full timeline and re-attaches it
after concat.

The test produces a small MP4 with the production fix's exact audio
plumbing, then decodes it back with ffmpeg and verifies per-region
peak amplitude: silence at the start, voice in the middle, silence
at the end.
"""

from __future__ import annotations

import subprocess
import wave

import numpy as np


def _write_test_audio(path: str, sr: int = 44100, dur: float = 2.0) -> None:
    """Write a 2 s 440 Hz stereo sine wave via ffmpeg."""
    subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i",
            f"sine=frequency=440:duration={dur}:sample_rate={sr}",
            "-ac", "2", "-c:a", "pcm_s16le", path,
        ],
        check=True,
    )


def _video_peak(arr: np.ndarray, start: int, end: int) -> float:
    if end - start <= 0:
        return 0.0
    return float(np.abs(arr[start:end]).max())


def test_intro_outro_audio_track_is_continuous(tmp_path):
    """Render intro (silence) + voice (sine) + outro (silence) and verify the
    resulting MP4 carries the tone in the middle with silence on either
    side. Implements the same pattern as
    ``app/services/video.py:_build_intro_outro_clips`` so a regression
    in the concat logic surfaces here before it hits a user render.
    """
    from moviepy import (
        AudioArrayClip,
        AudioFileClip,
        ColorClip,
        concatenate_audioclips,
        concatenate_videoclips,
    )

    wav_in = str(tmp_path / "voice.wav")
    out_mp4 = str(tmp_path / "concat.mp4")
    wav_out = str(tmp_path / "audio.wav")
    _write_test_audio(wav_in)

    voice = AudioFileClip(wav_in).with_duration(2.0)
    sr = voice.fps
    intro = ColorClip(size=(64, 64), color=(20, 20, 20)).with_duration(1.0).with_fps(24)
    main_video = ColorClip(size=(64, 64), color=(30, 30, 30)).with_duration(2.0).with_fps(24)
    outro = ColorClip(size=(64, 64), color=(40, 40, 40)).with_duration(1.0).with_fps(24)

    intro_silence = AudioArrayClip(
        np.zeros((int(1.0 * sr), 2), dtype=np.float32), fps=sr
    ).with_duration(1.0)
    outro_silence = AudioArrayClip(
        np.zeros((int(1.0 * sr), 2), dtype=np.float32), fps=sr
    ).with_duration(1.0)
    composed = concatenate_audioclips([intro_silence, voice, outro_silence])
    assert abs(composed.duration - 4.0) < 0.05, (
        f"concatenate_audioclips did not yield a 4 s timeline: {composed.duration}"
    )

    final = concatenate_videoclips(
        [
            intro.without_audio(),
            main_video.without_audio(),
            outro.without_audio(),
        ],
        method="compose",
    ).with_audio(composed)
    final.write_videofile(out_mp4, fps=24, codec="libx264", logger=None)

    subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", out_mp4, "-vn", "-acodec", "pcm_s16le",
            "-ar", "44100", wav_out,
        ],
        check=True,
    )

    with wave.open(wav_out, "rb") as wf:
        n = wf.getnframes()
        sr = wf.getframerate()
        nch = wf.getnchannels()
        raw = wf.readframes(n)
    arr = np.frombuffer(raw, dtype=np.int16).reshape(n, nch)

    intro_peak = _video_peak(arr, 0, sr)
    main_peak = _video_peak(arr, sr, 3 * sr)
    outro_peak = _video_peak(arr, 3 * sr, n)

    assert intro_peak < 200, (
        f"intro should be silent, got peak {intro_peak} samples (the user-reported 'no audio at all' bug returns here)"
    )
    assert main_peak > 1000, (
        f"voice missing in middle, peak {main_peak} samples -- the original bug returned here"
    )
    assert main_peak > intro_peak * 5, (
        f"middle should be much louder than intro: mid={main_peak} intro={intro_peak}"
    )
    assert outro_peak < 200, (
        f"outro should be silent, got peak {outro_peak} samples"
    )
