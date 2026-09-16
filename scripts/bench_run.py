#!/usr/bin/env python3
"""Run one real end-to-end generation through the FastAPI service and bench it.

Submits ``POST /api/v1/videos`` with the loop's test prompt, polls the task
until it finishes, records per-stage wall time from the progress transitions,
then hands the output to ``bench_render.py``. Exit code is the bench's, so a
failing gate fails this script too.

    python scripts/bench_run.py --mode local
    python scripts/bench_run.py --mode long-video --long-video material-36c9...mp4
    python scripts/bench_run.py --mode links --links https://youtu.be/...

Standard library only so it runs the same on every host.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCAL_VIDEOS = os.path.join(ROOT, "storage", "local_videos")
BENCH_DIR = os.path.join(ROOT, "storage", "bench")

TEST_PROMPT = "Cómo Rockstar hizo caber el mapa de GTA V en 512 MB de RAM"
TEST_LANG = "es-ES"

# Progress checkpoints written by app/services/pipeline/single.py; a jump past
# one of them means the stage before it finished.
STAGE_CHECKPOINTS = [
    (10, "script"),
    (30, "tts"),
    (40, "subtitles_materials"),
    (50, "prepare"),
    (75, "combine"),
    (100, "render"),
]


def api(base: str, method: str, path: str, body: dict | None = None) -> dict:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    request = urllib.request.Request(base.rstrip("/") + path, data=data, method=method, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def pick_local_materials(count: int, max_seconds: float = 300.0) -> list[str]:
    """Short clips from storage/local_videos, by file size as a duration proxy."""
    files = sorted(glob.glob(os.path.join(LOCAL_VIDEOS, "*.mp4")), key=os.path.getsize)
    chosen = []
    for path in files:
        result = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path], capture_output=True, text=True, check=False)
        try:
            duration = float(result.stdout.strip() or 0)
        except ValueError:
            duration = 0.0
        if 0 < duration <= max_seconds:
            chosen.append(os.path.basename(path))
        if len(chosen) >= count:
            break
    return chosen


def build_params(args: argparse.Namespace) -> dict:
    params = {
        "video_subject": args.subject,
        "video_language": args.lang,
        "video_aspect": "9:16",
        "video_fit_mode": "cover",
        "video_concat_mode": "random",
        "video_clip_duration": args.clip_duration,
        "video_count": 1,
        "voice_name": args.voice,
        "voice_volume": 1.0,
        "voice_rate": 1.0,
        "bgm_type": args.bgm_type,
        "bgm_volume": args.bgm_volume,
        "subtitle_enabled": True,
        "subtitle_format": "both",
        "subtitle_position": "two_thirds_bottom",
        "subtitle_display_mode": "karaoke",
        "subtitle_animation": "pop_spring",
        "subtitle_style_preset": args.preset,
        "subtitle_casing": args.casing,
        "font_name": args.font,
        "text_fore_color": args.text_color,
        "stroke_color": "#000000",
        "stroke_width": 4.0,
        "font_size": args.font_size,
        "paragraph_number": 1,
        "delivery_cues_enabled": True,
        "title_enabled": False,
        "intro_enabled": not args.no_intro_outro,
        "intro_tts_enabled": False,
        "outro_enabled": not args.no_intro_outro,
        "outro_tts_enabled": False,
        "n_threads": 4,
    }
    if args.mode == "local":
        materials = args.materials or pick_local_materials(args.material_count)
        if not materials:
            raise SystemExit("no short clips in storage/local_videos; pass --materials")
        params["video_source"] = "local"
        params["video_materials"] = [{"provider": "local", "url": name, "duration": 0} for name in materials]
    elif args.mode == "long-video":
        if not args.long_video:
            raise SystemExit("--long-video <file in storage/local_videos> is required for long-video mode")
        params["video_source"] = "local"
        params["video_materials"] = [{"provider": "local", "url": os.path.basename(args.long_video), "duration": 0}]
    elif args.mode == "links":
        if not args.links:
            raise SystemExit("--links url[,url] is required for links mode")
        params["video_source"] = "links"
        params["video_materials"] = [{"provider": "links", "url": url.strip(), "duration": 0} for url in args.links.split(",") if url.strip()]
    else:
        raise SystemExit(f"unknown mode {args.mode}")
    return params


def poll(base: str, task_id: str, timeout: float) -> tuple[dict, dict]:
    started = time.time()
    last_progress = -1
    checkpoint_times: dict[str, float] = {}
    last_checkpoint_time = started
    task = {}
    while time.time() - started < timeout:
        try:
            task = api(base, "GET", f"/api/v1/tasks/{task_id}").get("data", {})
        except (urllib.error.URLError, ValueError) as exc:
            print(f"poll error: {exc}", file=sys.stderr)
            time.sleep(3)
            continue
        progress = int(task.get("progress", 0) or 0)
        if progress != last_progress:
            now = time.time()
            for threshold, stage in STAGE_CHECKPOINTS:
                if last_progress < threshold <= progress and stage not in checkpoint_times:
                    checkpoint_times[stage] = round(now - last_checkpoint_time, 2)
                    last_checkpoint_time = now
            print(f"  progress {progress:3d}%  state={task.get('state')}  +{now - started:.0f}s", flush=True)
            last_progress = progress
        state = int(task.get("state", 0) or 0)
        if state in (-1, 1):  # failed / complete
            break
        time.sleep(2)
    checkpoint_times["total"] = round(time.time() - started, 2)
    return task, checkpoint_times


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("local", "links", "long-video"), default="local")
    parser.add_argument("--api", default=os.environ.get("MPT_API", "http://127.0.0.1:8080"))
    parser.add_argument("--subject", default=TEST_PROMPT)
    parser.add_argument("--lang", default=TEST_LANG)
    parser.add_argument("--voice", default="omnivoice:boy_voice")
    parser.add_argument("--preset", default="tiktok_yellow")
    parser.add_argument("--font", default="Anton-Regular.ttf")
    parser.add_argument("--font-size", type=int, default=60)
    parser.add_argument("--text-color", default="#FFFFFF")
    parser.add_argument("--casing", default="as_is")
    parser.add_argument("--bgm-type", default="random")
    parser.add_argument("--bgm-volume", type=float, default=0.2)
    parser.add_argument("--clip-duration", type=int, default=4)
    parser.add_argument("--materials", nargs="*", help="file names inside storage/local_videos")
    parser.add_argument("--material-count", type=int, default=6)
    parser.add_argument("--long-video", help="file name inside storage/local_videos")
    parser.add_argument("--links", help="comma separated public video URLs")
    parser.add_argument("--no-intro-outro", action="store_true")
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--label", default="")
    args = parser.parse_args(argv)

    params = build_params(args)
    os.makedirs(BENCH_DIR, exist_ok=True)
    print(f"submitting {args.mode} task to {args.api}: {args.subject!r}")
    try:
        created = api(args.api, "POST", "/api/v1/videos", params)
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"task creation failed: {exc.code} {exc.read().decode(errors='replace')[:500]}")
    task_id = created["data"]["task_id"]
    print(f"task {task_id}")
    task, timings = poll(args.api, task_id, args.timeout)
    state = int(task.get("state", 0) or 0)
    task_dir = os.path.join(ROOT, "storage", "tasks", task_id)
    timings_file = os.path.join(BENCH_DIR, f"{task_id}_timings.json")
    with open(timings_file, "w", encoding="utf-8") as handle:
        json.dump(timings, handle, indent=2)
    if state != 1:
        print(f"task ended in state {state}: stage={task.get('failed_stage')} error={task.get('error')}", file=sys.stderr)
        return 2
    print(f"task complete in {timings['total']}s; videos={task.get('videos')}")
    encode_passes = 2 if glob.glob(os.path.join(task_dir, "combined-*.mp4")) else 1
    command = [
        sys.executable, os.path.join(ROOT, "scripts", "bench_render.py"), task_dir,
        "--timings", timings_file, "--encode-passes", str(encode_passes), "--mode", args.mode,
    ]
    if args.label:
        command += ["--label", args.label]
    return subprocess.call(command)


if __name__ == "__main__":
    sys.exit(main())
