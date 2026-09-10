"""Checks for docker/detect-stack.sh, which decides the compose overrides.

Getting this wrong is not a soft failure: layering docker-compose.gpu.yml onto
a host with no NVIDIA runtime makes `docker compose up` refuse to start at all,
so the "no GPU" and "pinned by hand" paths are the ones worth pinning down.
"""

import platform
import subprocess
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "docker" / "detect-stack.sh"
BASE = "docker-compose.yml"


def run(env_overrides=None, path_override=None):
    env = {"PATH": path_override if path_override is not None else "/usr/bin:/bin"}
    env.update(env_overrides or {})
    result = subprocess.run(
        ["sh", str(SCRIPT)],
        capture_output=True,
        text=True,
        check=True,
        env=env,
        timeout=30,
    )
    return result.stdout.strip()


class TestStackDetection(unittest.TestCase):
    def test_explicit_compose_file_wins(self):
        """A pinned COMPOSE_FILE must survive untouched, or a user who
        deliberately forced the CPU stack silently gets the GPU one back."""
        pinned = f"{BASE}:docker-compose.rocm.yml"
        self.assertEqual(run({"COMPOSE_FILE": pinned}), pinned)

    def test_every_named_override_exists(self):
        """Each file the script can emit has to be a real file; a typo here
        surfaces as a confusing compose error far from the cause."""
        repo = SCRIPT.parents[1]
        for name in (
            BASE,
            "docker-compose.mac.yml",
            "docker-compose.gpu.yml",
            "docker-compose.rocm.yml",
        ):
            self.assertTrue((repo / name).is_file(), f"{name} is missing")

    def test_result_is_a_known_chain(self):
        """Whatever this host is, the answer must be the base file plus at
        most one override -- never an empty string or a stray message."""
        parts = run().split(":")
        self.assertEqual(parts[0], BASE)
        self.assertLessEqual(len(parts), 2)

    @unittest.skipUnless(platform.system() == "Darwin", "Darwin-only branch")
    def test_mac_takes_the_host_gpu_route(self):
        """macOS never has a container-visible GPU, so it must land on the mac
        override regardless of what the docker daemon reports."""
        self.assertEqual(run(), f"{BASE}:docker-compose.mac.yml")

    def test_no_docker_and_no_gpu_falls_back_to_the_base_file(self):
        """With docker unreachable the probe cannot answer. It must degrade to
        the plain CPU stack rather than guessing a GPU that is not there."""
        if platform.system() == "Darwin":
            self.skipTest("the Darwin branch returns before the docker probe")
        # An empty PATH removes `docker`, standing in for a daemon that cannot
        # be reached; /dev/kfd is absent on any non-ROCm host.
        self.assertEqual(run(path_override=""), BASE)


if __name__ == "__main__":
    unittest.main()
