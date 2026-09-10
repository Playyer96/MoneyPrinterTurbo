#!/usr/bin/env python3
"""Print "china" or "default": which package mirrors this build should prefer.

The project ships Aliyun/Tsinghua mirrors because it started as a China-hosted
project, but hardcoding them makes every build outside China crawl -- pip
pulls the whole dependency tree over a cross-border link at a few hundred KB/s,
and the official index is only reached after both mirrors time out.

Rather than ask the user where they are, measure it. TCP connect time to a
China-hosted host versus a globally-CDN'd one separates the two cases by an
order of magnitude (single-digit ms on the near side, hundreds on the far
one), needs no HTTP client, no API and no third-party geo service, and costs
a couple of seconds once per image build.

Override the guess with --build-arg DOCKER_BUILD_MIRROR=china|default.
"""

import socket
import sys
import time

# One host per side. Both are anycast/CDN-fronted for their own audience, so
# the comparison reflects network distance rather than one server's load.
CHINA_HOST = "mirrors.aliyun.com"
GLOBAL_HOST = "pypi.org"

TIMEOUT = 3.0
PROBES = 3
UNREACHABLE = float("inf")


def connect_rtt(host: str) -> float:
    """Best-of-N TCP handshake time to host:443, or inf if it never answers.

    Best-of rather than mean: a single slow handshake is usually a retransmit,
    and one bad sample should not decide the mirror for the whole build.
    """
    best = UNREACHABLE
    for _ in range(PROBES):
        started = time.monotonic()
        try:
            with socket.create_connection((host, 443), timeout=TIMEOUT):
                pass
        except OSError:
            continue
        best = min(best, time.monotonic() - started)
    return best


def main() -> int:
    china = connect_rtt(CHINA_HOST)
    world = connect_rtt(GLOBAL_HOST)

    # Ties and total failures both fall to "default": the global index is the
    # safe guess when there is nothing to go on, and unlike the china branch it
    # is what the release images are built with.
    choice = "china" if china < world else "default"

    def show(rtt: float) -> str:
        return "unreachable" if rtt == UNREACHABLE else f"{rtt * 1000:.0f}ms"

    print(
        f"mirror probe: {CHINA_HOST}={show(china)} {GLOBAL_HOST}={show(world)}"
        f" -> {choice}",
        file=sys.stderr,
    )
    print(choice)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
