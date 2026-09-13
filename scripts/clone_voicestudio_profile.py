"""Clone a WAV sample into a VoiceStudio voice profile.

The OmniVoice bridge under ``vendor/voice_studio`` exposes ``POST /profiles``
which encodes a reference sample as a reusable ``VoiceClonePrompt`` and
persists it under ``voice_profiles/``. This script is a thin wrapper so
operators can clone a new voice with one command without writing Python:

    .venv/bin/python scripts/clone_voicestudio_profile.py /path/to/sample.wav boy_voice

The profile name is normalised to ``[a-z0-9_-]+`` by the server; a name with
illegal characters is rejected with HTTP 400. After cloning, the voice shows
up as ``voicestudio:<name>`` in the WebUI TTS drop-down and the
``VideoParams.voice_name`` default is now ``voicestudio:boy_voice``.

Quality guidance:
    A short reference sample (under ~10 seconds) produces a recognisable
    but flat clone. Aim for 20-60 seconds of clean, single-speaker audio
    for the strongest voice match. Background music, reverb, or multiple
    speakers all degrade the cloned prompt.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services import voice


def _probe_duration_seconds(path: str) -> float | None:
    """Return the audio duration in seconds via ffprobe, or None on failure."""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None
    try:
        result = subprocess.run(
            [
                ffprobe, "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                path,
            ],
            capture_output=True, text=True, timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return None
    try:
        return float(result.stdout.strip())
    except ValueError:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Clone a WAV sample into a VoiceStudio voice profile.",
    )
    parser.add_argument(
        "sample_path",
        help="Path to a WAV (or other audio) sample the cloned voice will sound like.",
    )
    parser.add_argument(
        "profile_name",
        help="Short slug for the new profile. The server normalises to [a-z0-9_-]+.",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Override the OmniVoice base URL (default: $VOICESTUDIO_BASE_URL or [voicestudio] base_url).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip the minimum-sample-length warning.",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.sample_path):
        print(f"sample file not found: {args.sample_path}", file=sys.stderr)
        return 2
    with open(args.sample_path, "rb") as fh:
        audio_bytes = fh.read()
    if not audio_bytes:
        print(f"sample file is empty: {args.sample_path}", file=sys.stderr)
        return 2

    duration = _probe_duration_seconds(args.sample_path)
    if duration is not None and duration < 10.0 and not args.force:
        print(
            f"warning: sample is only {duration:.1f}s long. "
            "OmniVoice produces noticeably flat clones from short references; "
            "20-60 seconds of clean single-speaker audio is recommended for "
            "the strongest voice match. Re-run with --force to upload anyway.",
            file=sys.stderr,
        )
        return 3

    if args.base_url:
        os.environ["VOICESTUDIO_BASE_URL"] = args.base_url

    ok, message = voice.create_voicestudio_profile(
        profile_name=args.profile_name,
        audio_bytes=audio_bytes,
        original_filename=os.path.basename(args.sample_path),
    )
    if ok:
        print(f"profile created: voicestudio:{message}")
        return 0
    print(f"profile creation failed: {message}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())