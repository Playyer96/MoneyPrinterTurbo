"""``_build_segment_audio_sequence`` must never reorder or drop a segment.

The bug this guards: an earlier build prepended BOTH intro and outro
narration ahead of the main voice track (with BGM appended as its own
trailing block), so a video whose visual timeline was
[intro, main, outro] played audio in the order
[intro speech, outro speech, main speech, music] -- the outro narration
audibly played right after the intro, before the main content even
started, and grew more desynced as each segment's real duration diverged
from what the scrambled audio assumed.
"""

from __future__ import annotations

from types import SimpleNamespace

from app.services.video import _build_segment_audio_sequence


def _clip(duration):
    return SimpleNamespace(duration=duration)


def test_order_is_intro_then_main_then_outro_with_both_narrations():
    intro_audio = object()
    main_audio = object()
    outro_audio = object()

    sequence = _build_segment_audio_sequence(
        intro_clip=_clip(3.0),
        intro_audio_clip=intro_audio,
        main_audio_clip=main_audio,
        outro_clip=_clip(4.0),
        outro_audio_clip=outro_audio,
        sample_rate=44100,
    )

    # This is the exact ordering the bug violated: outro narration must
    # never appear before the main audio.
    assert sequence == [intro_audio, main_audio, outro_audio]


def test_missing_narration_is_filled_with_real_silence_of_the_right_length():
    main_audio = object()

    sequence = _build_segment_audio_sequence(
        intro_clip=_clip(2.0),
        intro_audio_clip=None,  # intro_tts_enabled was False
        main_audio_clip=main_audio,
        outro_clip=_clip(1.5),
        outro_audio_clip=None,
        sample_rate=44100,
    )

    assert len(sequence) == 3
    intro_silence, mid, outro_silence = sequence
    assert mid is main_audio
    assert intro_silence.duration == 2.0
    assert outro_silence.duration == 1.5
    # Real silence, not just a claimed duration: readable at any point up
    # to the claimed length (the sibling overlay-duration bug this test
    # complements was exactly a claimed-vs-actually-decodable gap).
    frame = intro_silence.get_frame(1.9)
    assert frame is not None and float(abs(frame).max()) == 0.0


def test_disabled_intro_or_outro_are_simply_absent_not_silent_placeholders():
    main_audio = object()

    only_outro = _build_segment_audio_sequence(
        intro_clip=None,
        intro_audio_clip=None,
        main_audio_clip=main_audio,
        outro_clip=_clip(2.0),
        outro_audio_clip=None,
        sample_rate=44100,
    )
    assert len(only_outro) == 2
    assert only_outro[0] is main_audio

    neither = _build_segment_audio_sequence(
        intro_clip=None,
        intro_audio_clip=None,
        main_audio_clip=main_audio,
        outro_clip=None,
        outro_audio_clip=None,
        sample_rate=44100,
    )
    assert neither == [main_audio]
