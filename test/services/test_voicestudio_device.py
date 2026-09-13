import importlib.util
from array import array
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
        with patch.dict(server.os.environ, {"VOICESTUDIO_ALLOW_CPU": "1"}):
            with self.assertRaises(RuntimeError) as ctx:
                server._pick_device(_fake_torch(cuda=False, mps=False))
        self.assertIn("docker-compose.mac.yml", str(ctx.exception))

    @patch.object(server, "_load_model")
    def test_warmup_loads_model_once_before_video_requests(self, load_model):
        self.assertEqual(server.warmup(), {"ok": True})
        load_model.assert_called_once_with()

    @patch.object(server, "_load_model")
    def test_generate_uses_better_default_quality_settings(self, load_model):
        load_model.return_value.generate.return_value = [array("f", [0.0] * 240)]

        response = server.generate_audio(
            SimpleNamespace(text="Hello world", voice="narrator", speed=None)
        )

        self.assertEqual(response.media_type, "audio/wav")
        self.assertTrue(response.body.startswith(b"RIFF"))
        # 16 steps is the new default: noticeably better output than OmniVoice's
        # 8-step greedy default at roughly 2x render time. Operators who need
        # the fast path can still set OMNIVOICE_NUM_STEPS=8 in the env.
        self.assertEqual(
            load_model.return_value.generate.call_args.kwargs["num_step"], 16
        )
        # Position temperature > 0 keeps cloned voices from sounding flat.
        self.assertEqual(
            load_model.return_value.generate.call_args.kwargs["position_temperature"],
            0.2,
        )


if __name__ == "__main__":
    unittest.main()
