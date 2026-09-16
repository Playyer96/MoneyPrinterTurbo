#!/usr/bin/env python3
"""Measure a rendered MoneyPrinterTurbo video against the quality gates.

Standard library + ffmpeg/ffprobe only, so it runs on any host or inside the
container. Given a task directory (or one final .mp4) it writes
``storage/bench/<task_id>.json``, appends one row to
``storage/bench/history.jsonl``, renders a contact sheet plus three
full-resolution frames to look at, and exits non-zero when any gate fails.

    python scripts/bench_render.py storage/tasks/<task_id>
    python scripts/bench_render.py storage/tasks/<task_id>/part-01/final-1.mp4 \
        --timings timings.json --encode-passes 1 --mode local
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time
import unicodedata
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH_DIR = os.path.join(ROOT, "storage", "bench")
CONTAINER_ROOT = "/MoneyPrinterTurbo"
CONTAINER_NAME = os.environ.get("MPT_BENCH_CONTAINER", "moneyprinterturbo-api")

PLATFORM_CAPS = {"youtube": 180.0, "tiktok": 600.0, "instagram": 180.0}
SAFE_ZONE = {"x0": 60, "x1": 1020, "y0": 250, "y1": 1500}  # for a 1080x1920 canvas

GATES = {
    "x_realtime": ("<=", 1.5),
    "integrated_lufs": ("range", (-15.0, -13.0)),
    "true_peak_dbtp": ("<=", -1.0),
    "silence_gaps": ("==", 0),
    "black_frames": ("==", 0),
    "freeze_frames": ("==", 0),
    "av_drift_ms": ("<=", 40.0),
    "sub_max_chars_per_line": ("<=", 32),
    "sub_max_lines_per_cue": ("<=", 2),
    "sub_broken_words": ("==", 0),
    "sub_short_cues": ("==", 0),
    "sub_long_cues": ("==", 0),
    "sub_overlapping_cues": ("==", 0),
    "safe_zone": ("==", True),
    "duration_within_cap": ("==", True),
    "encode_passes": ("<=", 1),
}


def ffmpeg_bin() -> str:
    return os.environ.get("FFMPEG_BINARY") or shutil.which("ffmpeg") or "ffmpeg"


def ffprobe_bin() -> str:
    return os.environ.get("FFPROBE_BINARY") or shutil.which("ffprobe") or "ffprobe"


def run(command: list[str], timeout: int = 600) -> subprocess.CompletedProcess:
    return subprocess.run(command, capture_output=True, text=True, check=False, timeout=timeout)


def probe(path: str) -> dict:
    result = run(
        [
            ffprobe_bin(), "-v", "error", "-print_format", "json",
            "-show_format", "-show_streams", path,
        ]
    )
    if result.returncode != 0:
        raise SystemExit(f"ffprobe failed on {path}: {result.stderr.strip()[-300:]}")
    return json.loads(result.stdout)


def stream_info(path: str) -> dict:
    data = probe(path)
    video = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), {})
    audio = next((s for s in data.get("streams", []) if s.get("codec_type") == "audio"), {})
    fmt = data.get("format", {})
    fps = 0.0
    rate = video.get("r_frame_rate", "0/1")
    try:
        num, den = rate.split("/")
        fps = float(num) / float(den) if float(den) else 0.0
    except ValueError:
        fps = float(rate or 0)
    video_duration = float(video.get("duration") or fmt.get("duration") or 0.0)
    audio_duration = float(audio.get("duration") or fmt.get("duration") or 0.0)
    return {
        "duration_s": round(float(fmt.get("duration") or 0.0), 3),
        "width": int(video.get("width") or 0),
        "height": int(video.get("height") or 0),
        "fps": round(fps, 3),
        "v_codec": video.get("codec_name", ""),
        "v_bitrate_kbps": int(int(video.get("bit_rate") or fmt.get("bit_rate") or 0) / 1000),
        "a_codec": audio.get("codec_name", ""),
        "a_sample_rate": int(audio.get("sample_rate") or 0),
        "a_channels": int(audio.get("channels") or 0),
        "av_drift_ms": round(abs(audio_duration - video_duration) * 1000.0, 1),
    }


def loudness(path: str) -> dict:
    result = run([ffmpeg_bin(), "-nostats", "-hide_banner", "-i", path, "-af", "ebur128=peak=true", "-f", "null", "-"])
    text = result.stderr
    summary = text[text.rfind("Summary:"):] if "Summary:" in text else text

    def grab(pattern: str) -> float | None:
        match = re.search(pattern, summary)
        return float(match.group(1)) if match else None

    return {
        "integrated_lufs": grab(r"I:\s*(-?[\d.]+)\s*LUFS"),
        "lra": grab(r"LRA:\s*(-?[\d.]+)\s*LU"),
        "true_peak_dbtp": grab(r"Peak:\s*(-?[\d.]+)\s*dBFS"),
    }


def silence_gaps(path: str, duration: float, min_len: float = 1.2, noise_db: int = -35) -> list[dict]:
    result = run([ffmpeg_bin(), "-nostats", "-hide_banner", "-i", path, "-af", f"silencedetect=n={noise_db}dB:d={min_len}", "-f", "null", "-"])
    gaps = []
    starts = [float(m) for m in re.findall(r"silence_start:\s*(-?[\d.]+)", result.stderr)]
    ends = re.findall(r"silence_end:\s*(-?[\d.]+)\s*\|\s*silence_duration:\s*([\d.]+)", result.stderr)
    for index, start in enumerate(starts):
        end = float(ends[index][0]) if index < len(ends) else duration
        # Leading and trailing silence is the intro/outro or the encoder's
        # padding, not a hole in the narration.
        if start <= 0.5 or end >= duration - 0.5:
            continue
        gaps.append({"start": round(start, 2), "end": round(end, 2)})
    return gaps


def black_frames(path: str) -> int:
    result = run([ffmpeg_bin(), "-nostats", "-hide_banner", "-i", path, "-vf", "blackdetect=d=0.2:pix_th=0.10", "-an", "-f", "null", "-"])
    return len(re.findall(r"black_start:", result.stderr))


def freeze_frames(path: str, min_len: float = 1.5) -> int:
    result = run([ffmpeg_bin(), "-nostats", "-hide_banner", "-i", path, "-vf", f"freezedetect=n=-60dB:d={min_len}", "-an", "-f", "null", "-"])
    return len(re.findall(r"freeze_start:", result.stderr))


_ASS_TAG_RE = re.compile(r"\{[^}]*\}")
_ASS_TIME_RE = re.compile(r"(\d+):(\d\d):(\d\d)\.(\d\d)")
_SRT_TIME_RE = re.compile(r"(\d+):(\d\d):(\d\d)[,.](\d{1,3})")


def _ass_seconds(value: str) -> float:
    h, m, s, cs = _ASS_TIME_RE.match(value.strip()).groups()
    return int(h) * 3600 + int(m) * 60 + int(s) + int(cs) / 100.0


def _srt_seconds(value: str) -> float:
    h, m, s, ms = _SRT_TIME_RE.match(value.strip()).groups()
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms.ljust(3, "0")) / 1000.0


def read_ass_cues(path: str) -> list[dict]:
    events = []
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.startswith("Dialogue:"):
                continue
            parts = line[len("Dialogue:"):].split(",", 9)
            if len(parts) < 10:
                continue
            text = _ASS_TAG_RE.sub("", parts[9].strip())
            lines = [segment.strip() for segment in text.split(r"\N") if segment.strip()]
            events.append({"start": _ass_seconds(parts[1]), "end": _ass_seconds(parts[2]), "lines": lines})
    # Karaoke renders one event per word with the same plain text; the
    # readable cue is the run of identical-text events.
    cues: list[dict] = []
    for event in events:
        if cues and cues[-1]["lines"] == event["lines"] and abs(cues[-1]["end"] - event["start"]) < 0.05:
            cues[-1]["end"] = event["end"]
        else:
            cues.append(dict(event))
    return cues


def read_srt_cues(path: str) -> list[dict]:
    cues = []
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        blocks = re.split(r"\n\s*\n", handle.read().strip())
    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        timing = next((line for line in lines if "-->" in line), None)
        if not timing:
            continue
        start, end = [_srt_seconds(part) for part in timing.split("-->")]
        text_lines = [line for line in lines if line != timing and not line.isdigit()]
        cues.append({"start": start, "end": end, "lines": text_lines})
    return cues


def _norm(token: str) -> str:
    decomposed = unicodedata.normalize("NFD", token.lower())
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return re.sub(r"[^0-9a-z]+", "", stripped)


def subtitle_audit(subtitle_path: str, script: str) -> dict:
    cues = read_ass_cues(subtitle_path) if subtitle_path.endswith(".ass") else read_srt_cues(subtitle_path)
    script_tokens = {_norm(token) for token in re.split(r"\s+", script) if _norm(token)}
    max_chars = max((len(line) for cue in cues for line in cue["lines"]), default=0)
    max_lines = max((len(cue["lines"]) for cue in cues), default=0)
    short = [cue for cue in cues if cue["end"] - cue["start"] < 0.7 - 1e-6]
    long_ = [cue for cue in cues if cue["end"] - cue["start"] > 6.0 + 1e-6]
    overlaps = sum(1 for a, b in zip(cues, cues[1:]) if a["end"] > b["start"] + 1e-3)
    broken = []
    if script_tokens:
        for cue in cues:
            for line in cue["lines"]:
                for token in line.split():
                    key = _norm(token)
                    if key and not key.isdigit() and key not in script_tokens:
                        broken.append(token)
    longest = max(cues, key=lambda cue: len(" ".join(cue["lines"])), default=None)
    return {
        "subtitle_file": os.path.relpath(subtitle_path, ROOT),
        "sub_cue_count": len(cues),
        "sub_max_chars_per_line": max_chars,
        "sub_max_lines_per_cue": max_lines,
        "sub_short_cues": len(short),
        "sub_long_cues": len(long_),
        "sub_overlapping_cues": overlaps,
        "sub_broken_words": len(broken),
        "sub_broken_examples": broken[:8],
        "sub_longest_cue": longest,
        "sub_first_cue": cues[0] if cues else None,
        "sub_mid_cue": cues[len(cues) // 2] if cues else None,
    }


def count_encode_passes(task_id: str) -> int | None:
    """Best-effort video encode count from the API container's own log lines.

    A ``combined-*.mp4`` file always exists (the combine stage always writes
    one), so its presence alone does not prove a second real encode -- the
    stream-copy combine path also writes it with zero encodes. Count actual
    encoder invocations instead: the combine stage logs whether it took the
    zero-encode stream-copy path or the one-encode single-call path, and the
    final stage logs whether it burned subtitles via the one-encode ffmpeg
    fast path or fell back to MoviePy's per-clip re-encode (which the log
    line count cannot fully capture, so that path returns None -- unknown,
    not a false "1"). Falls back to None when the log is unavailable, so a
    caller never reports a confident-looking wrong number.
    """
    log_text = ""
    try:
        result = subprocess.run(
            ["docker", "logs", CONTAINER_NAME, "--since", "2h"],
            capture_output=True, text=True, check=False, timeout=30,
        )
        log_text = (result.stdout or "") + (result.stderr or "")
    except (OSError, FileNotFoundError):
        pass
    if task_id not in log_text:
        return None
    task_lines = [line for line in log_text.splitlines() if task_id in line or "combine" in line or "burn-in" in line or "stream-copy" in line]
    window = "\n".join(task_lines[-200:])
    combine_encodes = 0
    if "single-call combine produced" in window:
        combine_encodes = 1
    elif "stream-copy combine produced" in window:
        combine_encodes = 0
    else:
        return None
    if "ASS subtitle burn-in succeeded" in window or "prebuilt ASS subtitle burn-in succeeded" in window:
        return combine_encodes + 1
    if "final stream-copy mux succeeded" in window:
        return combine_encodes  # audio-only mux, no extra video encode
    # MoviePy fallback re-encodes every clip individually; the exact count
    # is not visible from these log lines.
    return None


def contact_sheet(video: str, output: str) -> None:
    run([ffmpeg_bin(), "-y", "-hide_banner", "-loglevel", "error", "-i", video, "-vf", "fps=1/5,scale=270:-1,tile=6x4", "-frames:v", "1", output])


def extract_frame(video: str, at: float, output: str) -> None:
    run([ffmpeg_bin(), "-y", "-hide_banner", "-loglevel", "error", "-ss", f"{max(0.0, at):.3f}", "-i", video, "-frames:v", "1", output])


def _host_has_ass_filter() -> bool:
    result = run([ffmpeg_bin(), "-hide_banner", "-filters"], timeout=30)
    return bool(re.search(r"^\s*\S+\s+ass\s", result.stdout, re.MULTILINE))


def _to_container(path: str) -> str:
    return CONTAINER_ROOT + os.path.abspath(path)[len(ROOT):]


def render_ass_alpha(ass_path: str, width: int, height: int, at: float) -> bytes | None:
    """Rasterise the ASS at ``at`` seconds on a transparent canvas; return the alpha plane."""
    fonts_dir = os.path.join(ROOT, "resource", "fonts")
    use_host = _host_has_ass_filter()
    if not use_host and not shutil.which("docker"):
        return None
    ass_arg = _to_container(ass_path) if not use_host else os.path.abspath(ass_path)
    fonts_arg = _to_container(fonts_dir) if not use_host else fonts_dir
    # ``alpha=1`` makes libass write the alpha plane; without it the transparent
    # canvas stays fully transparent and nothing can be measured.
    filter_arg = f"ass=filename='{ass_arg}':fontsdir='{fonts_arg}':alpha=1,alphaextract"
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
        "-i", f"color=c=black@0.0:s={width}x{height}:r=25,format=rgba",
        "-ss", f"{at:.3f}", "-vf", filter_arg, "-frames:v", "1",
        "-f", "rawvideo", "-pix_fmt", "gray", "-",
    ]
    if not use_host:
        command = ["docker", "exec", CONTAINER_NAME] + command
    else:
        command[0] = ffmpeg_bin()
    result = subprocess.run(command, capture_output=True, check=False, timeout=120)
    if result.returncode != 0 or len(result.stdout) != width * height:
        sys.stderr.write(f"safe-zone render failed: {result.stderr.decode(errors='replace')[-300:]}\n")
        return None
    return result.stdout


def safe_zone_check(ass_path: str, width: int, height: int, at: float) -> dict:
    alpha = render_ass_alpha(ass_path, width, height, at)
    if alpha is None:
        return {"safe_zone": None, "safe_zone_bbox": None}
    xs, ys = [], []
    for y in range(height):
        row = alpha[y * width:(y + 1) * width]
        if not any(row):
            continue
        first = next(i for i, v in enumerate(row) if v)
        last = len(row) - 1 - next(i for i, v in enumerate(reversed(row)) if v)
        xs.extend((first, last))
        ys.append(y)
    if not ys:
        return {"safe_zone": False, "safe_zone_bbox": None, "safe_zone_note": "no subtitle pixels rendered"}
    scale_x, scale_y = width / 1080.0, height / 1920.0
    bbox = {"x0": min(xs), "x1": max(xs), "y0": min(ys), "y1": max(ys)}
    inside = (
        bbox["x0"] >= SAFE_ZONE["x0"] * scale_x and bbox["x1"] <= SAFE_ZONE["x1"] * scale_x
        and bbox["y0"] >= SAFE_ZONE["y0"] * scale_y and bbox["y1"] <= SAFE_ZONE["y1"] * scale_y
    )
    return {"safe_zone": bool(inside), "safe_zone_bbox": bbox}


def read_stage_timings(task_id: str, timings_file: str | None) -> dict:
    if timings_file and os.path.isfile(timings_file):
        with open(timings_file, "r", encoding="utf-8") as handle:
            return json.load(handle)
    log_path = os.path.join(ROOT, "storage", "logs", "pipeline.jsonl")
    if not os.path.isfile(log_path):
        return {}
    starts, timings = {}, {}
    with open(log_path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if str(record.get("task_id", "")).split("/")[0] != task_id:
                continue
            event, stage = record.get("event"), record.get("stage")
            if event == "stage.start":
                starts[stage] = float(record.get("ts", 0))
            elif event == "stage.end" and stage in starts:
                elapsed = record.get("elapsed_ms")
                timings[stage] = round((elapsed / 1000.0) if elapsed is not None else float(record.get("ts", 0)) - starts[stage], 2)
    return timings


def evaluate(metrics: dict) -> dict:
    results = {}
    for name, (op, threshold) in GATES.items():
        value = metrics.get(name)
        if value is None:
            results[name] = None
            continue
        if op == "<=":
            results[name] = value <= threshold
        elif op == "==":
            results[name] = value == threshold
        elif op == "range":
            results[name] = threshold[0] <= value <= threshold[1]
    return results


def find_final(target: str) -> tuple[str, str]:
    """Return (final_mp4, task_dir) for a task dir, part dir, or mp4 path."""
    if os.path.isfile(target):
        return os.path.abspath(target), os.path.dirname(os.path.abspath(target))
    candidates = sorted(glob.glob(os.path.join(target, "final-*.mp4"))) or sorted(glob.glob(os.path.join(target, "part-*", "final-*.mp4")))
    if not candidates:
        raise SystemExit(f"no final-*.mp4 under {target}")
    return os.path.abspath(candidates[0]), os.path.dirname(os.path.abspath(candidates[0]))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("target", help="task directory, part directory, or final mp4")
    parser.add_argument("--timings", help="JSON file with per-stage wall seconds (from bench_run.py)")
    parser.add_argument("--encode-passes", type=int, default=None, help="how many video encodes produced the file")
    parser.add_argument("--mode", default="", help="source mode label (local, links, long-video)")
    parser.add_argument("--platforms", default="youtube,tiktok,instagram")
    parser.add_argument("--label", default="", help="free-form label stored in the row (e.g. baseline)")
    parser.add_argument("--no-history", action="store_true")
    args = parser.parse_args(argv)

    final_mp4, part_dir = find_final(args.target)
    task_dir = part_dir
    if os.path.basename(part_dir).startswith("part-"):
        task_dir = os.path.dirname(part_dir)
    task_id = os.path.basename(task_dir)
    tag = task_id if part_dir == task_dir else f"{task_id}_{os.path.basename(part_dir)}"
    os.makedirs(BENCH_DIR, exist_ok=True)

    started = time.time()
    metrics: dict = {
        "task_id": task_id,
        "part": os.path.basename(part_dir) if part_dir != task_dir else "",
        "video": os.path.relpath(final_mp4, ROOT),
        "mode": args.mode,
        "label": args.label,
        "measured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    metrics.update(stream_info(final_mp4))
    duration = metrics["duration_s"]
    metrics.update(loudness(final_mp4))
    gaps = silence_gaps(final_mp4, duration)
    metrics["silence_gaps"] = len(gaps)
    metrics["silence_gap_list"] = gaps
    metrics["black_frames"] = black_frames(final_mp4)
    metrics["freeze_frames"] = freeze_frames(final_mp4)

    script = ""
    script_json = os.path.join(part_dir, "script.json")
    if os.path.isfile(script_json):
        try:
            with open(script_json, "r", encoding="utf-8") as handle:
                script = str(json.load(handle).get("script", ""))
        except ValueError:
            script = ""
    subtitle_file = next(
        (path for path in (os.path.join(part_dir, "subtitle.ass"), os.path.join(part_dir, "subtitle.srt")) if os.path.isfile(path)),
        None,
    )
    if subtitle_file:
        metrics.update(subtitle_audit(subtitle_file, script))

    timings = read_stage_timings(task_id, args.timings)
    metrics["stage_seconds"] = timings
    total = timings.get("total")
    if total is None and timings:
        total = round(sum(v for k, v in timings.items() if isinstance(v, (int, float))), 2)
    metrics["total_seconds"] = total
    metrics["x_realtime"] = round(total / duration, 3) if total and duration else None

    passes = args.encode_passes
    if passes is None:
        passes = count_encode_passes(task_id)
    metrics["encode_passes"] = passes

    caps = [PLATFORM_CAPS[p.strip()] for p in args.platforms.split(",") if p.strip() in PLATFORM_CAPS]
    metrics["duration_cap_s"] = min(caps) if caps else None
    metrics["duration_within_cap"] = (duration <= min(caps)) if caps else True

    sheet = os.path.join(BENCH_DIR, f"{tag}_sheet.png")
    contact_sheet(final_mp4, sheet)
    metrics["contact_sheet"] = os.path.relpath(sheet, ROOT)
    frame_times = []
    if metrics.get("sub_first_cue"):
        cue = metrics["sub_first_cue"]
        frame_times.append(("first_cue", (cue["start"] + cue["end"]) / 2))
    if metrics.get("sub_mid_cue"):
        cue = metrics["sub_mid_cue"]
        frame_times.append(("mid_cue", (cue["start"] + cue["end"]) / 2))
    frame_times.append(("outro", max(0.0, duration - 1.5)))
    metrics["frames"] = {}
    for name, at in frame_times:
        frame = os.path.join(BENCH_DIR, f"{tag}_frame_{name}.png")
        extract_frame(final_mp4, at, frame)
        metrics["frames"][name] = os.path.relpath(frame, ROOT)

    if subtitle_file and subtitle_file.endswith(".ass") and metrics.get("sub_longest_cue"):
        cue = metrics["sub_longest_cue"]
        metrics.update(safe_zone_check(subtitle_file, metrics["width"], metrics["height"], (cue["start"] + cue["end"]) / 2))
    else:
        metrics["safe_zone"] = None

    metrics["gates"] = evaluate(metrics)
    metrics["bench_seconds"] = round(time.time() - started, 1)

    out_json = os.path.join(BENCH_DIR, f"{tag}.json")
    with open(out_json, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
    if not args.no_history:
        with open(os.path.join(BENCH_DIR, "history.jsonl"), "a", encoding="utf-8") as handle:
            handle.write(json.dumps({k: v for k, v in metrics.items() if k not in ("sub_longest_cue", "sub_first_cue", "sub_mid_cue", "silence_gap_list")}, ensure_ascii=False) + "\n")

    print(f"bench: {metrics['video']}  ({duration:.1f}s, {metrics['width']}x{metrics['height']}@{metrics['fps']:.0f}, {metrics['v_bitrate_kbps']} kbps)")
    print(f"{'gate':<26}{'value':>14}  result")
    failed = []
    for name, ok in metrics["gates"].items():
        value = metrics.get(name)
        mark = "n/a" if ok is None else ("PASS" if ok else "FAIL")
        if ok is False:
            failed.append(name)
        print(f"{name:<26}{str(value):>14}  {mark}")
    print(f"stage seconds: {json.dumps(timings)}")
    print(f"contact sheet: {metrics['contact_sheet']}; frames: {list(metrics['frames'].values())}")
    print(f"json: {os.path.relpath(out_json, ROOT)}")
    if failed:
        print(f"FAILED gates: {', '.join(failed)}")
        return 1
    print("all gates passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
