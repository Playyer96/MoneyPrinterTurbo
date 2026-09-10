import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

# vendor/voice_studio is not a package on sys.path; load server.py by file.
_SERVER = Path(__file__).parent.parent.parent / "vendor" / "voice_studio" / "server.py"
_spec = importlib.util.spec_from_file_location("voicestudio_server", _SERVER)
server = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(server)


def _fake_torch(cuda: bool, mps: bool):
    return SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: cuda),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps)),
        float16="float16",
        float32="float32",
    )


class TestPickDevice(unittest.TestCase):
    def test_cuda_wins(self):
        self.assertEqual(
            server._pick_device(_fake_torch(cuda=True, mps=False))[0], "cuda"
        )

    def test_mps_used_on_apple_host(self):
        self.assertEqual(
            server._pick_device(_fake_torch(cuda=False, mps=True))[0], "mps"
        )

    def test_no_gpu_refuses_instead_of_silently_using_cpu(self):
        """The whole point: a containerised Mac must fail loudly, not crawl."""
        with patch.dict(server.os.environ, {}, clear=False):
            server.os.environ.pop("VOICESTUDIO_ALLOW_CPU", None)
            with self.assertRaises(RuntimeError) as ctx:
                server._pick_device(_fake_torch(cuda=False, mps=False))
        self.assertIn("mac-setup", str(ctx.exception))

    def test_cpu_allowed_when_explicitly_opted_in(self):
        with patch.dict(server.os.environ, {"VOICESTUDIO_ALLOW_CPU": "1"}):
            self.assertEqual(
                server._pick_device(_fake_torch(cuda=False, mps=False))[0], "cpu"
            )


if __name__ == "__main__":
    unittest.main()
