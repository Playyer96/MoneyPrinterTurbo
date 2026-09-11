import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

# add project root to python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.utils import utils


class TestCheckFfmpegReady(unittest.TestCase):
    """Cover the four possible probe outcomes from utils.check_ffmpeg_ready()."""

    def test_returns_true_when_ffmpeg_probe_succeeds(self):
        completed = subprocess.CompletedProcess(args=["ffmpeg", "-version"], returncode=0)
        with (
            patch.object(utils, "get_ffmpeg_binary", return_value="/usr/bin/ffmpeg"),
            patch.object(utils.subprocess, "run", return_value=completed) as run,
        ):
            self.assertTrue(utils.check_ffmpeg_ready())

        run.assert_called_once()
        self.assertEqual(run.call_args.args[0], ["/usr/bin/ffmpeg", "-version"])

    def test_returns_false_when_executable_is_missing(self):
        with (
            patch.object(utils, "get_ffmpeg_binary", return_value="ffmpeg"),
            patch.object(utils.subprocess, "run", side_effect=FileNotFoundError()),
            patch.object(utils.logger, "warning") as warning,
        ):
            self.assertFalse(utils.check_ffmpeg_ready())

        warning.assert_called_once()
        self.assertIn("no usable ffmpeg executable found", warning.call_args.args[0])

    def test_returns_false_on_non_zero_exit_code(self):
        completed = subprocess.CompletedProcess(args=["ffmpeg", "-version"], returncode=1)
        with (
            patch.object(utils, "get_ffmpeg_binary", return_value="/usr/bin/ffmpeg"),
            patch.object(utils.subprocess, "run", return_value=completed),
            patch.object(utils.logger, "warning") as warning,
        ):
            self.assertFalse(utils.check_ffmpeg_ready())

        warning.assert_called_once()
        self.assertIn("exited with status 1", warning.call_args.args[0])

    def test_returns_false_on_timeout(self):
        with (
            patch.object(utils, "get_ffmpeg_binary", return_value="/usr/bin/ffmpeg"),
            patch.object(
                utils.subprocess,
                "run",
                side_effect=subprocess.TimeoutExpired(cmd="ffmpeg", timeout=10),
            ),
            patch.object(utils.logger, "warning") as warning,
        ):
            self.assertFalse(utils.check_ffmpeg_ready(timeout=10))

        warning.assert_called_once()
        self.assertIn("failed to probe ffmpeg", warning.call_args.args[0])


class TestMacDockerFfmpegWrapper(unittest.TestCase):
    """
    `get_ffmpeg_binary` returns the host-ffmpeg forwarding wrapper when the
    process is inside a container AND `FFMPEG_MAC_PROXY_URL` is set. Every
    other configuration falls through to the normal ffmpeg lookup so native
    macOS, Linux hosts, and non-Mac Docker keep working unchanged.
    """

    def test_wrapper_returned_when_in_docker_and_proxy_url_set(self):
        with (
            patch.object(utils, "_running_in_docker", return_value=True),
            patch.dict(
                os.environ,
                {"FFMPEG_MAC_PROXY_URL": "http://ffmpeg-mac-proxy:8781"},
                clear=True,
            ),
            patch.object(utils, "shutil") as shutil_mock,
        ):
            result = utils.get_ffmpeg_binary()
            self.assertTrue(
                result.endswith("scripts/ffmpeg_mac_wrapper.py"),
                f"expected wrapper path, got {result!r}",
            )
            # The shutil.which fallback must NOT be consulted when the
            # wrapper path is in use; that proves the early return works.
            shutil_mock.which.assert_not_called()

    def test_no_wrapper_when_proxy_url_unset(self):
        with (
            patch.object(utils, "_running_in_docker", return_value=True),
            patch.dict(os.environ, {}, clear=True),
            patch.object(utils.shutil, "which", return_value="/usr/bin/ffmpeg"),
        ):
            self.assertEqual(utils.get_ffmpeg_binary(), "/usr/bin/ffmpeg")

    def test_no_wrapper_on_native_macos(self):
        with (
            patch.object(utils, "_running_in_docker", return_value=False),
            patch.dict(
                os.environ,
                {"FFMPEG_MAC_PROXY_URL": "http://ffmpeg-mac-proxy:8781"},
                clear=True,
            ),
            patch.object(utils.shutil, "which", return_value="/opt/homebrew/bin/ffmpeg"),
        ):
            self.assertEqual(
                utils.get_ffmpeg_binary(), "/opt/homebrew/bin/ffmpeg"
            )

    def test_running_in_docker_via_dot_dockerenv(self):
        with patch.object(utils.Path, "exists", return_value=True):
            self.assertTrue(utils._running_in_docker())


if __name__ == "__main__":
    unittest.main()
