"""One-shot post-processor: re-run an existing MP4's audio through the
softening + 192 kbps MP3 re-encode pipeline so the user can hear the
new audio quality on videos that were generated before the recent fixes.

The script is intentionally thin: extract the audio track to PCM WAV,
soften the silence boundaries (replaces hard zeros with 40 ms fades),
re-encode at 192 kbps, then re-mux the new audio with the original video
stream into a sibling output file. The original video is left intact
in case the user wants to compare or re-process again with different
parameters.

Usage:
    .venv/bin/python scripts/repair_video_audio.py input.mp4
    .venv/bin/python scripts/repair_video_audio.py input.mp4 -o repaired.mp4
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.audio_postprocess import soften_audio_transitions
from app.utils import utils


def _ffmpeg() -> str:
    binary = utils.get_ffmpeg_binary()
    if not binary:
        sys.exit("ffmpeg not found on PATH; install ffmpeg and retry")
    return binary


def _probe_duration(ffmpeg_binary: str, media_path: str) -> float:
    result = subprocess.run(
        [
            ffmpeg_binary,
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            media_path,
        ],
        capture_output=True, text=True, timeout=30, check=False,
    )
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0


def _has_audio_stream(ffmpeg_binary: str, media_path: str) -> bool:
    """Return whether the media file contains at least one audio stream.

    ``-select_streams`` is a ffprobe-only option; the ffmpeg binary rejects
    it with ``Unrecognized option``. Probe with ffprobe when available;
    if ffprobe is not on PATH, accept the file (ffmpeg will surface a
    concrete failure at extraction time).
    """
    import shutil

    ffprobe_binary = shutil.which("ffprobe")
    if not ffprobe_binary:
        return True
    try:
        result = subprocess.run(
            [
                ffprobe_binary, "-v", "error",
                "-select_streams", "a",
                "-show_entries", "stream=index",
                "-of", "default=noprint_wrappers=1:nokey=1",
                media_path,
            ],
            capture_output=True, text=True, timeout=30, check=False,
        )
        return bool(result.stdout.strip())
    except OSError:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Re-run an existing MP4's audio through the silence-softening "
            "and 192 kbps re-encode pipeline."
        )
    )
    parser.add_argument("input", help="Path to the input video file.")
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output video path (default: <input>-repaired.mp4 next to the input).",
    )
    parser.add_argument(
        "--bitrate",
        default="192k",
        help="Target MP3 bitrate for the re-encoded audio (default: 192k).",
    )
    args = parser.parse_args()

    input_path = os.path.abspath(args.input)
    if not os.path.isfile(input_path):
        sys.exit(f"input file not found: {input_path}")
    output_path = os.path.abspath(
        args.output or os.path.splitext(input_path)[0] + "-repaired.mp4"
    )

    ffmpeg_binary = _ffmpeg()
    if not _has_audio_stream(ffmpeg_binary, input_path):
        sys.exit(f"input has no audio stream: {input_path}")

    duration = _probe_duration(ffmpeg_binary, input_path)
    print(
        f"repairing {os.path.basename(input_path)} "
        f"({duration:.1f}s) -> {os.path.basename(output_path)}"
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        decoded_wav = os.path.join(temp_dir, "decoded.wav")
        softened_wav = os.path.join(temp_dir, "softened.wav")
        encoded_mp3 = os.path.join(temp_dir, "softened.mp3")

        # 1. Extract audio to 24 kHz mono 16-bit PCM. The soften helper
        #    reads this format directly so no decode/re-encode is needed
        #    between extraction and softening.
        print("[1/4] extracting audio to PCM")
        result = subprocess.run(
            [
                ffmpeg_binary, "-y", "-i", input_path,
                "-vn", "-ac", "1", "-ar", "24000",
                "-acodec", "pcm_s16le",
                decoded_wav,
            ],
            capture_output=True, text=True, timeout=600, check=False,
        )
        if result.returncode != 0:
            sys.exit(f"audio extraction failed:\n{result.stderr.strip()[-400:]}")

        # 2. Apply the silence softening (numpy-based, runs in-process).
        print("[2/4] softening silence boundaries (noise floor)")
        softened = False
        try:
            # The helper writes back to its input path; copy the decoded
            # WAV first so we don't mutate the temp directory in place.
            import shutil

            shutil.copyfile(decoded_wav, softened_wav)
            softened = soften_audio_transitions(softened_wav)
        except Exception as exc:  # noqa: BLE001 - surface any failure
            sys.exit(f"silence softening failed: {exc}")
        if not softened:
            print("    no silences detected; softening skipped")
            softened_wav = decoded_wav

        # 3. Re-encode to MP3 at the requested bitrate.
        print(f"[3/4] re-encoding audio at {args.bitrate}")
        result = subprocess.run(
            [
                ffmpeg_binary, "-y", "-i", softened_wav,
                "-vn", "-ac", "1", "-ar", "24000",
                "-c:a", "libmp3lame", "-b:a", args.bitrate,
                encoded_mp3,
            ],
            capture_output=True, text=True, timeout=600, check=False,
        )
        if result.returncode != 0:
            sys.exit(f"audio re-encode failed:\n{result.stderr.strip()[-400:]}")

        # 4. Mux the new audio with the original video stream.
        print("[4/4] muxing new audio into the original video")
        result = subprocess.run(
            [
                ffmpeg_binary, "-y",
                "-i", input_path,
                "-i", encoded_mp3,
                "-map", "0:v", "-map", "1:a",
                "-c:v", "copy",
                "-c:a", "libmp3lame", "-b:a", args.bitrate,
                "-shortest", "-movflags", "+faststart",
                output_path,
            ],
            capture_output=True, text=True, timeout=600, check=False,
        )
        if result.returncode != 0:
            sys.exit(f"final mux failed:\n{result.stderr.strip()[-400:]}")

    print(f"done: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())