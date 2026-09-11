import subprocess
from unittest.mock import patch

from scripts import ffmpeg_mac_proxy, ffmpeg_mac_wrapper


def test_run_ffmpeg_translates_container_working_directory():
    completed = subprocess.CompletedProcess([], 0, "ok", "")
    with patch.object(ffmpeg_mac_proxy.subprocess, "run", return_value=completed) as run:
        result = ffmpeg_mac_proxy.run_ffmpeg([], "/MoneyPrinterTurbo")

    assert result["returncode"] == 0
    assert run.call_args.kwargs["cwd"] == str(ffmpeg_mac_proxy.REPO)


def test_wrapper_keeps_container_private_paths_in_container():
    assert ffmpeg_mac_wrapper._requires_container_ffmpeg(
        ["-i", "/tmp/input.mp4", "/MoneyPrinterTurbo/storage/output.mp4"]
    )
    assert not ffmpeg_mac_wrapper._requires_container_ffmpeg(
        ["-i", "/MoneyPrinterTurbo/storage/input.mp4", "-"]
    )
