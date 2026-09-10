"""
Transcribe one audio file with MLX whisper and print the result as JSON.

Runs as a short-lived subprocess rather than inside server.py on purpose.
OmniVoice keeps a torch-MPS context resident in the server process, and MLX
allocating Metal buffers alongside it collapses throughput -- measured at
72s in-process versus 8.7s in a clean one for the same audio and model. A
fresh process per transcription costs a model load and still wins by ~8x.

Usage: python whisper_worker.py <audio_path> [--no-word-timestamps]
"""

from __future__ import annotations

import json
import os
import sys

DEFAULT_REPO = "mlx-community/whisper-large-v3-turbo"


def transcribe(audio_path: str, word_timestamps: bool = True) -> dict:
    import mlx_whisper

    result = mlx_whisper.transcribe(
        audio_path,
        path_or_hf_repo=os.environ.get("MLX_WHISPER_REPO", DEFAULT_REPO),
        word_timestamps=word_timestamps,
    )
    # Mirror the faster-whisper shape the caller walks. Timestamps arrive as
    # numpy floats, which json cannot serialize.
    return {
        "language": result.get("language", ""),
        "language_probability": 1.0,
        "segments": [
            {
                "text": seg.get("text", ""),
                "start": float(seg.get("start", 0.0)),
                "end": float(seg.get("end", 0.0)),
                "words": [
                    {
                        "word": w.get("word", ""),
                        "start": float(w.get("start", 0.0)),
                        "end": float(w.get("end", 0.0)),
                    }
                    for w in (seg.get("words") or [])
                ],
            }
            for seg in result.get("segments", [])
        ],
    }


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: whisper_worker.py <audio_path>", file=sys.stderr)
        return 2
    payload = transcribe(sys.argv[1], "--no-word-timestamps" not in sys.argv)
    # stdout carries only the JSON; model-loading chatter goes to stderr.
    json.dump(payload, sys.stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
