#!/usr/bin/env python3
"""
Client wrapper that replaces the ffmpeg binary inside a Mac-hosted Docker
container. Forwards every argument to the host-side proxy (see
`scripts/ffmpeg_mac_proxy.py`) and returns the host ffmpeg's exit code,
stdout, and stderr. The bind-mount path translation happens on the host.

Set `FFMPEG_MAC_PROXY_URL` to the proxy endpoint. `get_ffmpeg_binary()`
returns this wrapper when that env var is set and the process is running
inside a container.
"""
import json
import base64
import http.client
import os
import subprocess
import stat
import sys
import urllib.error
import urllib.request
import urllib.parse

DEFAULT_URL = "http://host.docker.internal:8781"
TIMEOUT_SECONDS = 3600
CONTAINER_REPO = "/MoneyPrinterTurbo"
CONTAINER_FFMPEG = os.environ.get("MPT_CONTAINER_FFMPEG_BIN", "/usr/bin/ffmpeg")


def _stdin_is_pipe() -> bool:
    return stat.S_ISFIFO(os.fstat(sys.stdin.fileno()).st_mode)


def _requires_container_ffmpeg(args: list[str]) -> bool:
    """Return True when an absolute argument is outside the shared repository."""
    return any(
        os.path.isabs(arg)
        and arg != CONTAINER_REPO
        and not arg.startswith(CONTAINER_REPO + "/")
        for arg in args
    )


def _run_stream(proxy_url: str, args: list[str]) -> bytes:
    parsed = urllib.parse.urlsplit(proxy_url)
    connection = http.client.HTTPConnection(
        parsed.hostname,
        parsed.port or 80,
        timeout=TIMEOUT_SECONDS,
    )
    metadata = base64.urlsafe_b64encode(
        json.dumps({"args": args, "cwd": os.getcwd()}).encode("utf-8")
    ).decode("ascii")
    connection.request(
        "POST",
        f"{parsed.path.rstrip('/')}/run-stream",
        body=sys.stdin.buffer,
        headers={"X-MPT-FFmpeg": metadata},
        encode_chunked=True,
    )
    response = connection.getresponse()
    body = response.read()
    connection.close()
    if response.status != 200:
        raise ConnectionError(f"proxy returned HTTP {response.status}: {body.decode(errors='replace')}")
    return body


def main() -> int:
    proxy_url = os.environ.get("FFMPEG_MAC_PROXY_URL", DEFAULT_URL).rstrip("/")
    args = sys.argv[1:]
    if _requires_container_ffmpeg(args):
        return subprocess.call([CONTAINER_FFMPEG, *args])
    try:
        if _stdin_is_pipe():
            body = _run_stream(proxy_url, args)
        else:
            payload = json.dumps({"args": args, "cwd": os.getcwd()}).encode("utf-8")
            request = urllib.request.Request(
                f"{proxy_url}/run",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as resp:
                body = resp.read()
    except (urllib.error.URLError, http.client.HTTPException, ConnectionError, OSError, TimeoutError) as exc:
        sys.stderr.write(
            f"ffmpeg-mac-wrapper: cannot reach host proxy at {proxy_url}: {exc}\n"
            "Hint: run `make mac-setup` on the Mac host to install the LaunchAgent,\n"
            "or unset FFMPEG_MAC_PROXY_URL to fall back to the in-container ffmpeg.\n"
        )
        return 1
    try:
        result = json.loads(body)
    except json.JSONDecodeError as exc:
        sys.stderr.write(f"ffmpeg-mac-wrapper: invalid response from proxy: {exc}\n")
        return 1
    if result.get("stdout"):
        sys.stdout.write(result["stdout"])
    if result.get("stderr"):
        sys.stderr.write(result["stderr"])
    return int(result.get("returncode", 1))


if __name__ == "__main__":
    sys.exit(main())
