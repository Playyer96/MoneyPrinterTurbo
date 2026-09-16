"""End-to-end audio regeneration: transcribe the existing audio back to
text, re-synthesize through OmniVoice with the upgraded defaults
(32 diffusion steps / 0.45 position temperature), and re-mux into the
original video.

This is the path that actually addresses the user's complaints about
voice timbre breaks and emotion inconsistency — both are generation-time
artifacts of OmniVoice's stochastic diffusion that cannot be repaired
with post-processing alone.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile

import torch

# OmniVoice on Apple Silicon (MPS) crashes with SIGSEGV when loaded in
# float16 because PyTorch's MPS memory allocator cannot reclaim
# intermediate buffers fast enough during the diffusion sampling loop.
# Pin float32 and disable the MPS high-watermark eviction policy so the
# process holds the model weights in RAM and never asks the allocator
# to shrink under pressure. CUDA hosts skip this entirely.
if torch.backends.mps.is_available():
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.utils import utils


def _ffmpeg() -> str:
    binary = utils.get_ffmpeg_binary()
    if not binary:
        sys.exit("ffmpeg not found on PATH; install ffmpeg and retry")
    return binary


def _device_and_dtype() -> tuple[str, str]:
    if torch.backends.mps.is_available():
        return "mps", "float32"  # float16 on MPS SIGSEGVs OmniVoice
    if torch.cuda.is_available():
        return "cuda", "float16"
    return "cpu", "float32"


def _transcribe(mlx_model: str, wav_path: str) -> tuple[str, str]:
    """Run mlx-whisper on a 16 kHz WAV file.

    mlx-whisper dispatches Whisper through Apple's MLX framework which
    targets Metal on Apple Silicon GPUs. Returns (language, transcript).
    """
    import mlx_whisper

    print(f"    transcribing with {mlx_model}")
    result = mlx_whisper.transcribe(
        wav_path,
        path_or_hf_repo=mlx_model,
        word_timestamps=False,
        verbose=False,
    )
    return (
        result.get("language", "en"),
        (result.get("text") or "").strip(),
    )


def _omnivoice_generate(
    text: str,
    voice: str,
    audio_path: str,
    *,
    steps: int = 32,
    temperature: float = 0.45,
    device: str,
    dtype: str,
) -> None:
    """Load OmniVoice once, run a single TTS call, write WAV to ``audio_path``.

    The numbers come from the upgraded server defaults so the regenerated
    audio benefits from the same voice-quality / consistency bumps the
    pipeline applies to new runs.

    ``audio_chunk_duration`` and ``audio_chunk_threshold`` are pinned to
    a large value so OmniVoice generates the full script in one shot
    rather than internally splitting it into ~15 s chunks with 100 ms
    crossfades. The internal chunking is the source of the "voice
    breaks / rumbo diferente" artifact the user reported.
    """
    from omnivoice import OmniVoice
    from omnivoice.models.omnivoice import OmniVoiceGenerationConfig

    print(f"    loading OmniVoice on {device} ({dtype})")
    model = OmniVoice.from_pretrained(
        "k2-fsa/OmniVoice", device_map=device, dtype=dtype
    )

    config = OmniVoiceGenerationConfig(
        num_step=steps,
        position_temperature=temperature,
        # Pin chunking off: the threshold / duration defaults (30 s / 15 s)
        # cause OmniVoice to internally split a long script and crossfade
        # 100 ms between chunks. The crossfade is short enough that the
        # ear hears it as a voice break. Setting both to a large value
        # forces one continuous generation for the whole script.
        audio_chunk_duration=9999.0,
        audio_chunk_threshold=9999.0,
    )

    print(f"    generating {len(text)} chars (steps={steps}, temp={temperature})")
    waveforms = model.generate(
        text=text,
        voice=voice,
        generation_config=config,
    )
    import numpy as np
    import soundfile as sf

    wav = waveforms[0]
    if hasattr(wav, "detach"):
        wav = wav.detach().cpu().float().numpy()
    elif not isinstance(wav, np.ndarray):
        wav = np.asarray(wav, dtype=np.float32)
    sf.write(audio_path, wav, 24000, format="WAV")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Transcribe an MP4's audio, re-synthesize through OmniVoice "
            "with the upgraded defaults, and re-mux the new audio."
        )
    )
    parser.add_argument("input", help="Path to the input video file.")
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output video path (default: <input>-regen.mp4 next to the input).",
    )
    parser.add_argument(
        "--voice",
        default="narrator",
        help="OmniVoice voice preset (default: narrator).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=32,
        help="OmniVoice diffusion steps (default: 32).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.45,
        help="OmniVoice position temperature (default: 0.45).",
    )
    parser.add_argument(
        "--mlx-model",
        default="mlx-community/whisper-large-v3-turbo",
        help="mlx-whisper model repo (default: large-v3-turbo).",
    )
    parser.add_argument(
        "--bitrate",
        default="192k",
        help="Target MP3 bitrate for the final audio (default: 192k).",
    )
    args = parser.parse_args()

    input_path = os.path.abspath(args.input)
    if not os.path.isfile(input_path):
        sys.exit(f"input file not found: {input_path}")
    output_path = os.path.abspath(
        args.output or os.path.splitext(input_path)[0] + "-regen.mp4"
    )

    ffmpeg_binary = _ffmpeg()
    device, dtype = _device_and_dtype()

    with tempfile.TemporaryDirectory() as temp_dir:
        whisper_wav = os.path.join(temp_dir, "whisper_input.wav")
        new_wav = os.path.join(temp_dir, "regenerated.wav")
        new_mp3 = os.path.join(temp_dir, "regenerated.mp3")

        # 1. Extract audio to 16 kHz mono PCM for mlx-whisper.
        print("[1/4] extracting audio for transcription")
        result = subprocess.run(
            [
                ffmpeg_binary, "-y", "-i", input_path,
                "-vn", "-ac", "1", "-ar", "16000",
                "-acodec", "pcm_s16le",
                whisper_wav,
            ],
            capture_output=True, text=True, timeout=600, check=False,
        )
        if result.returncode != 0:
            sys.exit(f"audio extraction failed:\n{result.stderr.strip()[-400:]}")

        # 2. Transcribe.
        print("[2/4] transcribing with mlx-whisper")
        language, transcript = _transcribe(args.mlx_model, whisper_wav)
        if not transcript:
            sys.exit("mlx-whisper returned an empty transcript; aborting")
        print(f"    detected language: {language} ({len(transcript)} chars)")

        # 3. Re-synthesize through OmniVoice with the new defaults.
        print("[3/4] re-synthesizing through OmniVoice")
        try:
            _omnivoice_generate(
                transcript,
                args.voice,
                new_wav,
                steps=args.steps,
                temperature=args.temperature,
                device=device,
                dtype=dtype,
            )
        except Exception as exc:  # noqa: BLE001 - surface failures clearly
            sys.exit(f"OmniVoice generation failed: {exc}")
        if not os.path.isfile(new_wav) or os.path.getsize(new_wav) < 1024:
            sys.exit(f"OmniVoice output missing or too small: {new_wav}")

        # 4. Re-encode to 192 kbps MP3 and mux with the original video.
        print(f"[4/4] re-encoding at {args.bitrate} and muxing")
        result = subprocess.run(
            [
                ffmpeg_binary, "-y", "-i", new_wav,
                "-vn", "-ac", "1", "-ar", "24000",
                "-c:a", "libmp3lame", "-b:a", args.bitrate,
                new_mp3,
            ],
            capture_output=True, text=True, timeout=600, check=False,
        )
        if result.returncode != 0:
            sys.exit(f"audio re-encode failed:\n{result.stderr.strip()[-400:]}")

        result = subprocess.run(
            [
                ffmpeg_binary, "-y",
                "-i", input_path,
                "-i", new_mp3,
                "-map", "0:v", "-map", "1:a",
                "-c:v", "copy",
                "-c:a", "copy",
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