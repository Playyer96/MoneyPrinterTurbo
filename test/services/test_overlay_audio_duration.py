"""``_pad_or_trim_audio_to_duration`` must never claim more audio than it can decode.

The bug this guards: ``AudioFileClip.with_duration(longer_value)`` only
rewrites the clip's metadata, not the underlying decodable audio. A
downstream compositing read near the fake extended boundary then crashes
ffmpeg with an OSError ("Accessing time t=3.16-3.20 seconds, with clip
duration=3.12 seconds"). The fix pads with real silence instead of lying
about the duration.
"""

from __future__ import annotations

from app.services.video import _pad_or_trim_audio_to_duration


def test_shorter_clip_is_padded_with_real_silence_not_a_fake_duration():
    from moviepy import AudioArrayClip
    import numpy as np

    sample_rate = 44100
    short = AudioArrayClip(
        np.ones((int(3.12 * sample_rate), 1), dtype=np.float32) * 0.5, fps=sample_rate
    )

    padded = _pad_or_trim_audio_to_duration(short, 3.20)

    assert abs(padded.duration - 3.20) < 1e-2
    # The clip must actually be readable at every point up to its claimed
    # duration -- this is what crashed before the fix.
    frame = padded.get_frame(3.18)
    assert frame is not None


def test_longer_clip_is_trimmed():
    from moviepy import AudioArrayClip
    import numpy as np

    sample_rate = 44100
    long_clip = AudioArrayClip(
        np.ones((int(5.0 * sample_rate), 1), dtype=np.float32) * 0.5, fps=sample_rate
    )

    trimmed = _pad_or_trim_audio_to_duration(long_clip, 2.0)

    assert abs(trimmed.duration - 2.0) < 1e-2


def test_none_clip_and_zero_duration_pass_through_unchanged():
    assert _pad_or_trim_audio_to_duration(None, 3.0) is None
    from moviepy import AudioArrayClip
    import numpy as np

    clip = AudioArrayClip(np.zeros((100, 1), dtype=np.float32), fps=44100)
    assert _pad_or_trim_audio_to_duration(clip, 0) is clip
    assert _pad_or_trim_audio_to_duration(clip, None) is clip
