#!/usr/bin/env python3
"""
Host-side ffmpeg proxy for Docker-on-Mac.

Why this exists
---------------
Docker Desktop for Mac runs containers in a Linux VM. That VM has no
VideoToolbox framework, so ffmpeg inside the container cannot advertise
`h264_videotoolbox` -- the macOS encoder the app picks by default. The
host's ffmpeg (e.g. Homebrew's `/opt/homebrew/bin/ffmpeg`) does have it,
but its Mach-O binary cannot execute inside the Linux VM.

This proxy runs on the Mac host as a LaunchAgent (port 8781, loopback
only). The container's `app/utils/utils.py:get_ffmpeg_binary()` returns a
Python wrapper that POSTs each ffmpeg invocation here; we translate the
container's `/MoneyPrinterTurbo/...` paths to the host's repo path, run
the host's ffmpeg, and return its exit code + stdout + stderr.

Path translation is required because the bind mount
(`./:/MoneyPrinterTurbo`) is the same directory on both sides but the
absolute path differs (host: whatever the user cloned to; container:
`/MoneyPrinterTurbo`). Other paths (`/tmp/...`) are passed through
verbatim and rely on Docker Desktop sharing `/tmp` with the host VM,
which the VirtioFS/gRPC-FUSE backends do by default.

Protocol
--------
  POST /run        body: {"args": [...], "cwd": "..."}     returns {"returncode", "stdout", "stderr"}
  GET  /health     returns 200 OK with the resolved host ffmpeg path
  GET  /encoders   returns the host ffmpeg `-encoders` output (for diagnostics)
"""
import base64
import json
import logging
import os
import shutil
import subprocess
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CONTAINER_REPO = "/MoneyPrinterTurbo"
HOST_FFMPEG = os.environ.get("MPT_FFMPEG_BIN") or shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
LOG_PATH = Path(os.environ.get("MPT_FFMPEG_PROXY_LOG", REPO / "storage" / "logs" / "ffmpeg-mac-proxy.log"))
LISTEN_HOST = os.environ.get("MPT_FFMPEG_PROXY_HOST", "127.0.0.1")
LISTEN_PORT = int(os.environ.get("MPT_FFMPEG_PROXY_PORT", "8781"))

LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename=str(LOG_PATH),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("ffmpeg-mac-proxy")


def translate_path(arg: str) -> str:
    """
    Rewrite `/MoneyPrinterTurbo/...` to the host's repo path so ffmpeg on
    the host can read the bind-mounted storage. Other paths pass through.
    """
    if arg.startswith(CONTAINER_REPO + "/") or arg == CONTAINER_REPO:
        rel = arg[len(CONTAINER_REPO):].lstrip("/")
        return str(REPO / rel) if rel else str(REPO)
    return arg


def run_ffmpeg(args, cwd):
    translated = [translate_path(a) for a in args]
    host_cwd = translate_path(cwd) if cwd else str(REPO)
    log.info("running ffmpeg: %s (cwd=%s)", translated, host_cwd)
    try:
        proc = subprocess.run(
            [HOST_FFMPEG, *translated],
            capture_output=True,
            text=True,
            check=False,
            cwd=host_cwd,
            timeout=3600,
        )
    except subprocess.TimeoutExpired:
        log.warning("ffmpeg timed out after 3600s")
        return {"returncode": 124, "stdout": "", "stderr": "ffmpeg timed out after 3600s\n"}
    except Exception as exc:
        log.exception("ffmpeg invocation failed")
        return {"returncode": 1, "stdout": "", "stderr": f"proxy error: {exc}\n"}
    log.info("ffmpeg rc=%d stdout=%d stderr=%d", proc.returncode, len(proc.stdout), len(proc.stderr))
    return {
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def _decode_stream_metadata(value: str):
    return json.loads(base64.urlsafe_b64decode(value.encode("ascii")))


def _iter_request_body(handler):
    if handler.headers.get("Transfer-Encoding", "").lower() == "chunked":
        while True:
            size = int(handler.rfile.readline().split(b";", 1)[0], 16)
            if not size:
                handler.rfile.readline()
                return
            yield handler.rfile.read(size)
            handler.rfile.read(2)
    else:
        remaining = int(handler.headers.get("Content-Length", "0"))
        while remaining:
            chunk = handler.rfile.read(min(65536, remaining))
            if not chunk:
                return
            remaining -= len(chunk)
            yield chunk


def run_ffmpeg_stream(args, cwd, chunks):
    translated = [translate_path(a) for a in args]
    host_cwd = translate_path(cwd) if cwd else str(REPO)
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        proc = subprocess.Popen(
            [HOST_FFMPEG, *translated],
            stdin=subprocess.PIPE,
            stdout=stdout,
            stderr=stderr,
            cwd=host_cwd,
        )
        writable = True
        for chunk in chunks:
            if writable:
                try:
                    proc.stdin.write(chunk)
                except BrokenPipeError:
                    writable = False
        if writable:
            proc.stdin.close()
        try:
            returncode = proc.wait(timeout=3600)
        except subprocess.TimeoutExpired:
            proc.kill()
            returncode = 124
        stdout.seek(0)
        stderr.seek(0)
        return {
            "returncode": returncode,
            "stdout": stdout.read().decode("utf-8", errors="replace"),
            "stderr": stderr.read().decode("utf-8", errors="replace"),
        }


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send_json(200, {"ok": True, "ffmpeg": HOST_FFMPEG, "repo": str(REPO)})
            return
        if self.path == "/encoders":
            try:
                proc = subprocess.run(
                    [HOST_FFMPEG, "-hide_banner", "-encoders"],
                    capture_output=True, text=True, check=False, timeout=10,
                )
                self._send_json(200, {"returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr})
            except Exception as exc:
                self._send_json(500, {"error": str(exc)})
            return
        self._send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path == "/run-stream":
            try:
                payload = _decode_stream_metadata(self.headers["X-MPT-FFmpeg"])
                args = payload["args"]
                cwd = payload.get("cwd", "")
                if not isinstance(args, list) or not all(isinstance(a, str) for a in args):
                    raise ValueError("args must be a list of strings")
                self._send_json(200, run_ffmpeg_stream(args, cwd, _iter_request_body(self)))
            except (KeyError, ValueError, TypeError, json.JSONDecodeError) as exc:
                self._send_json(400, {"error": f"invalid stream request: {exc}"})
            return
        if self.path != "/run":
            self._send_json(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length", "0"))
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError as exc:
            self._send_json(400, {"error": f"invalid JSON: {exc}"})
            return
        args = payload.get("args")
        if not isinstance(args, list) or not all(isinstance(a, str) for a in args):
            self._send_json(400, {"error": "args must be a list of strings"})
            return
        cwd = payload.get("cwd") or ""
        self._send_json(200, run_ffmpeg(args, cwd))

    def log_message(self, format, *args):
        # Quiet the default access log; the file logger already captures real work.
        pass


def main():
    if not Path(HOST_FFMPEG).exists():
        log.error("host ffmpeg not found at %s; set MPT_FFMPEG_BIN or install ffmpeg", HOST_FFMPEG)
    log.info("starting ffmpeg-mac-proxy on %s:%d (ffmpeg=%s repo=%s)", LISTEN_HOST, LISTEN_PORT, HOST_FFMPEG, REPO)
    server = ThreadingHTTPServer((LISTEN_HOST, LISTEN_PORT), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
