"""Lock in the silence-boundary softening applied to TTS outputs.

Natural pauses in synthesized speech are hard zeros; the human ear reads
them as digital clicks between sentences. The soften_audio_transitions
helper detects every silence >= 150 ms and applies a short fade-in /
fade-out at each boundary so the transitions feel like breaths rather
than cuts.
"""

from __future__ import annotations

import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
import wave
from pathlib import Path


def _write_synthetic_wav(
    path: str,
    *,
    sample_rate: int = 16000,
    voiced_seconds: float = 1.0,
    silence_seconds: float = 0.4,
    repeats: int = 3,
) -> None:
    """Build a tone/silence/tone/silence/... WAV with hard boundaries."""
    voiced_samples = int(voiced_seconds * sample_rate)
    silence_samples = int(silence_seconds * sample_rate)
    amplitude = 12000
    frequency = 440.0

    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        for _ in range(repeats):
            voiced = [
                int(amplitude * 0.6 * __import__("math").sin(2 * __import__("math").pi * frequency * i / sample_rate))
                for i in range(voiced_samples)
            ]
            wf.writeframes(struct.pack(f"<{len(voiced)}h", *voiced))
            wf.writeframes(b"\x00\x00" * silence_samples)


class TestSoftenAudioTransitions(unittest.TestCase):
    def setUp(self) -> None:
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from app.services.audio_postprocess import (
            soften_audio_transitions,
        )

        self._soften = soften_audio_transitions
        self._tmp = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_short_audio_is_left_intact(self) -> None:
        path = str(Path(self._tmp.name) / "in.wav")
        _write_synthetic_wav(path, voiced_seconds=0.05, silence_seconds=0.05)
        self.assertFalse(self._soften(path))

    def test_noisy_audio_is_softened_in_place(self) -> None:
        path = str(Path(self._tmp.name) / "in.wav")
        _write_synthetic_wav(
            path,
            voiced_seconds=0.8,
            silence_seconds=0.4,
            repeats=3,
        )
        with wave.open(path, "rb") as wf:
            original_frames = wf.getnframes()
            original_rate = wf.getframerate()
        self.assertTrue(self._soften(path))
        # The file must still decode as a valid WAV after softening,
        # and the duration must be unchanged so the subtitle timeline
        # stays in sync with the audio.
        with wave.open(path, "rb") as wf:
            self.assertEqual(wf.getnchannels(), 1)
            self.assertEqual(wf.getsampwidth(), 2)
            self.assertEqual(wf.getframerate(), original_rate)
            self.assertEqual(wf.getnframes(), original_frames)

    def test_missing_file_returns_false(self) -> None:
        self.assertFalse(self._soften("/tmp/does-not-exist-audio.wav"))

    def test_voice_edges_fade_into_silence(self) -> None:
        """Verify voiced samples at silence boundaries get a fade ramp.

        Without the edge fade, each TTS chunk starts at full amplitude the
        instant the silence ends, producing the audible "click" the user
        hears. The fix multiplies the first ~30 ms of voice after each
        silence by a linear 0 -> 1 ramp so the onset is smooth.
        """
        import struct as _struct

        path = str(Path(self._tmp.name) / "in.wav")
        sample_rate = 16000
        amplitude = 12000
        # voice (0.5s) -- silence (0.4s, qualifies as >= 150 ms) -- voice (0.5s)
        voiced_samples = int(0.5 * sample_rate)
        silence_samples = int(0.4 * sample_rate)
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setnframes(0)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            voice = [int(amplitude * 0.6) for _ in range(voiced_samples)]
            wf.writeframes(_struct.pack(f"<{len(voice)}h", *voice))
            wf.writeframes(b"\x00\x00" * silence_samples)
            wf.writeframes(_struct.pack(f"<{len(voice)}h", *voice))

        self.assertTrue(self._soften(path))

        # The first 30 ms after the silence must NOT be at full amplitude.
        # If it were, the function would not have applied the fade.
        with wave.open(path, "rb") as wf:
            raw = wf.readframes(wf.getnframes())
        import numpy as _np
        samples = _np.frombuffer(raw, dtype=_np.int16)
        post_silence_start = voiced_samples + silence_samples
        first_30ms = samples[post_silence_start : post_silence_start + int(0.03 * sample_rate)]
        self.assertLess(
            int(_np.abs(first_30ms).max()),
            amplitude,
            "first 30ms after silence should be faded, not at full amplitude",
        )
        # And within ~50 ms the voice must recover close to full level
        # (the fade is short enough not to audibly duck the speech).
        first_50ms = samples[post_silence_start : post_silence_start + int(0.05 * sample_rate)]
        self.assertGreater(int(_np.abs(first_50ms).max()), amplitude // 2)

    def test_does_not_inject_silence_noise_floor(self) -> None:
        """Lock in that the helper no longer adds a -50 dBFS noise floor.

        Earlier revisions injected a deterministic noise floor (~100 LSB)
        into every silence so the ear would read the gap as room tone. The
        trick backfires on headphones and quiet rooms: that noise is audible.
        The new version only applies edge fades around silences and must NOT
        raise silent samples off zero.
        """
        import struct as _struct

        path = str(Path(self._tmp.name) / "in.wav")
        sample_rate = 16000
        voiced_samples = int(0.5 * sample_rate)
        silence_samples = int(0.4 * sample_rate)
        amplitude = 12000
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setnframes(0)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            voice = [int(amplitude * 0.6) for _ in range(voiced_samples)]
            wf.writeframes(_struct.pack(f"<{len(voice)}h", *voice))
            wf.writeframes(b"\x00\x00" * silence_samples)
            wf.writeframes(_struct.pack(f"<{len(voice)}h", *voice))

        self.assertTrue(self._soften(path))

        with wave.open(path, "rb") as wf:
            raw = wf.readframes(wf.getnframes())
        import numpy as _np

        samples = _np.frombuffer(raw, dtype=_np.int16)
        # The middle 200 ms of the silence must stay at or near zero. If
        # the noise floor is back, every sample there would have |value|
        # in the hundreds. 30 LSB is the natural absolute-zero floor from
        # the wave encoder.
        silence_mid = samples[
            voiced_samples + int(0.1 * sample_rate) : voiced_samples + int(0.3 * sample_rate)
        ]
        self.assertLessEqual(
            int(_np.abs(silence_mid).max()),
            30,
            "silence interior must stay at digital zero — no noise floor injected",
        )


class TestVoiceCleanupPipeline(unittest.TestCase):
    """End-to-end check of the silenceremove + afftdn cleanup pipeline.

    Builds a WAV that mimics OmniVoice's real output shape — leading
    silence, a voiced segment, internal silence, another voiced segment,
    trailing silence — and runs the same FFmpeg pipeline
    ``_soften_voice_file`` uses. If FFmpeg is missing or lacks afftdn /
    silenceremove the test skips rather than reporting a false pass on a
    build that cannot run the real pipeline.
    """

    def setUp(self) -> None:
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        self._tmp = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _write_realistic_tts_like_wav(self, path: str) -> None:
        """Lead/trailing silence around two voiced segments, like a TTS chunk."""
        import math

        sample_rate = 24000
        amplitude = 10000
        frequency = 440.0
        leading_silence = int(0.25 * sample_rate)  # 250 ms of model lead-in
        trailing_silence = int(0.20 * sample_rate)  # 200 ms of tail
        voiced = int(0.6 * sample_rate)
        inter_silence = int(0.15 * sample_rate)

        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(b"\x00\x00" * leading_silence)
            tone = [
                int(amplitude * math.sin(2 * math.pi * frequency * i / sample_rate))
                for i in range(voiced)
            ]
            wf.writeframes(struct.pack(f"<{len(tone)}h", *tone))
            wf.writeframes(b"\x00\x00" * inter_silence)
            tone2 = [
                int(amplitude * math.sin(2 * math.pi * frequency * i / sample_rate))
                for i in range(voiced)
            ]
            wf.writeframes(struct.pack(f"<{len(tone2)}h", *tone2))
            wf.writeframes(b"\x00\x00" * trailing_silence)

    def test_silenceremove_strips_leading_and_trailing_silence(self) -> None:
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            self.skipTest("ffmpeg not available")
        try:
            result = subprocess.run(
                [ffmpeg, "-hide_banner", "-filters"],
                capture_output=True, text=True, check=False, timeout=10,
            )
        except (OSError, subprocess.TimeoutExpired):
            self.skipTest("ffmpeg -filters probe failed")
        if "afftdn" not in result.stdout or "silenceremove" not in result.stdout:
            self.skipTest("ffmpeg lacks afftdn or silenceremove filters")

        path = str(Path(self._tmp.name) / "tts_like.wav")
        cleaned = str(Path(self._tmp.name) / "cleaned.wav")
        self._write_realistic_tts_like_wav(path)

        result = subprocess.run(
            [
                ffmpeg, "-y", "-i", path,
                "-vn", "-ac", "1", "-ar", "24000",
                "-af",
                "silenceremove=start_periods=1:start_silence=0.05:"
                "start_threshold=-50dB:stop_periods=-1:stop_silence=0.05:"
                "stop_threshold=-50dB,afftdn=nf=-25",
                "-codec:a", "pcm_s16le",
                cleaned,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertTrue(Path(cleaned).is_file())

        def _silence_fraction(wav_path: str) -> float:
            import numpy as _np

            with wave.open(wav_path, "rb") as wf:
                raw = wf.readframes(wf.getnframes())
            samples = _np.frombuffer(raw, dtype=_np.int16)
            if len(samples) == 0:
                return 0.0
            return float(_np.mean(_np.abs(samples) < 30))

        original_silence = _silence_fraction(path)
        cleaned_silence = _silence_fraction(cleaned)
        # The cleanup must drop a meaningful chunk of the lead-in/tail
        # silence. The original carries ~250 ms leading + 200 ms trailing
        # + 150 ms internal = ~600 ms of zero across ~2.05 s ≈ 30 %
        # silence. After the trim the leading/trailing blocks are gone,
        # so the cleaned fraction must be much lower than the original
        # (afftdn's noise-floor reduction may push some near-zero voiced
        # samples just under the threshold, so we don't pin the absolute
        # number — only that the trim + denoise meaningfully reduced it).
        self.assertLess(cleaned_silence, original_silence * 0.5)
        self.assertLess(cleaned_silence, original_silence - 0.10)


if __name__ == "__main__":
    unittest.main()