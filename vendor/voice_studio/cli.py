"""
Command-line interface (CLI) for VoiceStudio.

Usage:
  python cli.py create --audio <path> --name <profile_name>
  python cli.py generate --text "<string>" --profile <profile_name> --out <filename>
"""

from __future__ import annotations

import argparse
import sys

import core


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="VoiceStudio — Minimal Headless Voice Cloning & Speech Synthesis CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="Subcommand to run")

    # create --audio <path> --name <profile_name>
    create_parser = subparsers.add_parser(
        "create", help="Extract and save target voice profile locally"
    )
    create_parser.add_argument(
        "--audio",
        required=True,
        type=str,
        help="Path to the reference audio file",
    )
    create_parser.add_argument(
        "--name",
        required=True,
        type=str,
        help="Name of the target voice profile to create",
    )

    # generate --text "<string>" --profile <profile_name> --out <filename>
    gen_parser = subparsers.add_parser(
        "generate", help="Synthesize speech audio from text using a saved profile"
    )
    gen_parser.add_argument(
        "--text",
        required=True,
        type=str,
        help="Text string to synthesize into speech",
    )
    gen_parser.add_argument(
        "--profile",
        required=True,
        type=str,
        help="Name of the saved voice profile to clone",
    )
    gen_parser.add_argument(
        "--out",
        required=True,
        type=str,
        help="Output audio filename (saved in ./output/)",
    )

    return parser


def main(args: list[str] | None = None) -> int:
    parser = build_parser()
    parsed = parser.parse_args(args)

    try:
        if parsed.command == "create":
            saved_path = core.create_profile(
                audio_path=parsed.audio,
                profile_name=parsed.name,
            )
            print(f"Voice profile successfully created: {saved_path}")
            return 0
        elif parsed.command == "generate":
            out_path = core.generate_audio(
                text=parsed.text,
                profile_name=parsed.profile,
                output_filename=parsed.out,
            )
            print(f"Audio successfully generated: {out_path}")
            return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
