from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
PROVIDER = ROOT / "scripts" / "mpt-mac-accelerators"


def test_mac_compose_wires_host_acceleration_before_app_services():
    if shutil.which("docker") is None:
        pytest.skip("Docker is unavailable")

    subprocess.run(["sh", "-n", str(PROVIDER)], check=True)
    environment = os.environ.copy()
    environment["PWD"] = str(ROOT)
    result = subprocess.run(
        [
            "docker",
            "compose",
            "-f",
            str(ROOT / "docker-compose.yml"),
            "-f",
            str(ROOT / "docker-compose.mac.yml"),
            "config",
            "--format",
            "json",
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=ROOT,
        env=environment,
    )
    services = json.loads(result.stdout)["services"]

    assert services["host-accelerators"]["provider"]["type"] == str(PROVIDER)
    assert "host-accelerators" in services["voicestudio-host"]["depends_on"]
    assert "host-accelerators" in services["ffmpeg-proxy-host"]["depends_on"]
    assert services["webui"]["depends_on"]["voicestudio-host"]["condition"] == "service_healthy"
    assert services["api"]["depends_on"]["ffmpeg-proxy-host"]["condition"] == "service_healthy"
