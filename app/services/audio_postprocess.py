"""Audio post-processing helpers.

Single responsibility: take a finished TTS audio file and make it sound
less "abrupt" at the silent boundaries. Natural pauses in TTS output are
hard zeros, which the human ear reads as digital clicks between
sentences. The ``soften_audio_transitions`` helper detects every silence
>= ~150 ms and applies a short linear fade to the first/last ~30 ms of
voiced samples on either side of the gap.

The fade is what hides the splice: each TTS chunk starts at full
amplitude the instant the silence ends, so without it every sentence
boundary sounds like a cut. Earlier revisions of this helper also
injected a deterministic noise floor into the silent samples ("room
tone") so the ear would read the gap as analogue instead of a digital
mute. That trick backfires on headphones and quiet rooms: a -50 dBFS
floor is loud enough to be perceptible, and the diffusion models the
pipeline pairs with (OmniVoice, ...) already carry some hiss in the
voiced signal — the floor only added more audible noise without fixing
the real cause. Drop the floor entirely; the affine edge fade hides the
cut, and ``afftdn`` in the broader pipeline removes the model hiss.

Standalone module so the dependency on numpy stays optional at import
time (the rest of the package keeps working when numpy is unavailable)
and the softer can be unit-tested without spinning up the whole TTS
pipeline.
"""

from __future__ import annotations

import os
import tempfile
import wave

import numpy as np

# Edge-fade length applied at every silence/voice boundary. Without
# this, concatenated TTS chunks jump from near-zero to full voice in
# roughly 30 ms — an obvious click/thud that any Teams caller hears. A
# 30 ms fade keeps the transition under the ~10 ms/side threshold the
# human ear reads as "abrupt".
_EDGE_FADE_MS = 30


def soften_audio_transitions(
    audio_path: str,
    *,
    min_silence_ms: int = 150,
    rms_threshold: float = 300.0,
) -> bool:
    """Apply short edge fades around every silence long enough to need them.

    Returns ``True`` when the file was rewritten (any silence was found
    and softened). Returns ``False`` when the file is missing, has no
    decodable PCM, or has no silences long enough to be worth softening;
    in those cases the input file is left untouched.

    Detection uses 20 ms RMS windows so the boundaries land near the
    actual transition from voiced to silence rather than at the next
    sample. The first/last ``_EDGE_FADE_MS`` of voiced samples adjacent
    to every qualifying silence get a linear 0 -> 1 ramp so the voice
    onset doesn't pop against the silence.

    Only the fade ramps are touched; the silent samples themselves are
    left at their natural amplitude so the voiced signal is preserved
    bit-for-bit.
    """
    if not audio_path or not os.path.isfile(audio_path):
        return False

    with open(audio_path, "rb") as handle:
        raw_header = handle.read(44)
    if raw_header[:4] != b"RIFF" or raw_header[8:12] != b"WAVE":
        return False

    with wave.open(audio_path, "rb") as wf:
        channels = wf.getnchannels()
        sample_rate = wf.getframerate()
        sample_width = wf.getsampwidth()
        n_frames = wf.getnframes()
        if channels != 1 or sample_width != 2 or n_frames == 0:
            return False
        raw = wf.readframes(n_frames)

    if sample_rate <= 0:
        return False

    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    win = max(int(sample_rate * 0.02), 1)  # 20 ms
    n_windows = len(samples) // win
    if n_windows < 2:
        return False

    # Per-window RMS so the silence boundaries land at a real
    # voiced-to-silent transition rather than the next arbitrary sample.
    trimmed = samples[: n_windows * win].reshape(n_windows, win)
    rms_per_window = np.sqrt(np.mean(trimmed ** 2, axis=1))

    # Edges of voiced regions adjacent to a qualifying silence need a
    # short fade so the voice onset/offset doesn't pop. Populated only
    # for silences long enough to matter.
    fade_targets: list[tuple[int, int]] = []  # (start_sample, end_sample)
    in_silence = False
    silence_start_win = 0
    for i in range(n_windows):
        window_silent = rms_per_window[i] < rms_threshold
        if window_silent and not in_silence:
            silence_start_win = i
            in_silence = True
        elif not window_silent and in_silence:
            sil_dur_ms = (i - silence_start_win) * 20
            if sil_dur_ms >= min_silence_ms:
                fade_targets.append((silence_start_win * win, i * win))
            in_silence = False
    if in_silence and (n_windows - silence_start_win) * 20 >= min_silence_ms:
        fade_targets.append((silence_start_win * win, len(samples)))

    if not fade_targets:
        return False

    # Edge fades: a linear ramp from 0 -> 1 over _EDGE_FADE_MS at each
    # end of every qualifying silence, applied to the voiced samples
    # just outside the silence. This is what hides the splice.
    fade_samples = min(int(sample_rate * _EDGE_FADE_MS / 1000), len(samples) // 2)
    fade_curve = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
    for sil_start, sil_end in fade_targets:
        # Fade-in just after silence ends
        fade_in_end = min(sil_end + fade_samples, len(samples))
        fade_in_len = fade_in_end - sil_end
        if fade_in_len > 0:
            samples[sil_end:fade_in_end] *= fade_curve[:fade_in_len]
        # Fade-out just before silence begins
        fade_out_start = max(sil_start - fade_samples, 0)
        fade_out_len = sil_start - fade_out_start
        if fade_out_len > 0:
            samples[fade_out_start:sil_start] *= fade_curve[:fade_out_len]

    # Write to a sibling temp file and atomically replace the original
    # so a mid-write failure never leaves the user with a half-softened
    # file the rest of the pipeline would otherwise consume.
    output_dir = os.path.dirname(os.path.abspath(audio_path)) or "."
    fd, temp_path = tempfile.mkstemp(
        prefix=".soften-",
        suffix=".wav",
        dir=output_dir,
    )
    os.close(fd)
    try:
        with wave.open(temp_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(samples.astype(np.int16).tobytes())
        os.replace(temp_path, audio_path)
        return True
    except Exception:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass
        raise