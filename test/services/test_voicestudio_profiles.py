import importlib.util
import os
import tempfile
import unittest
from pathlib import Path

# vendor/voice_studio is not a package on sys.path; load server.py by file.
_SERVER = Path(__file__).parent.parent.parent / "vendor" / "voice_studio" / "server.py"
_spec = importlib.util.spec_from_file_location("voicestudio_server", _SERVER)
server = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(server)


class TestProfileNameNormalization(unittest.TestCase):
    def test_lowercases_and_slugs(self):
        self.assertEqual(
            server._normalize_profile_name("  Cristiano Ronaldo "),
            "cristiano_ronaldo",
        )
        self.assertEqual(server._normalize_profile_name("A.B/C!"), "a_b_c")

    def test_empty_name_stays_empty(self):
        self.assertEqual(server._normalize_profile_name(""), "")
        self.assertEqual(server._normalize_profile_name("   !!! "), "")

    def test_truncates_long_names(self):
        self.assertEqual(len(server._normalize_profile_name("x" * 100)), 48)


class TestProfileDiscovery(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._orig_dir = server.PROFILES_DIR
        server.PROFILES_DIR = self._tmp.name

    def tearDown(self):
        server.PROFILES_DIR = self._orig_dir
        self._tmp.cleanup()

    def test_lists_prompt_files_only(self):
        Path(self._tmp.name, "demo.pt").write_text("p")
        Path(self._tmp.name, "junk.json").write_text("{}")
        Path(self._tmp.name, "junk.mp3").write_text("m")
        self.assertEqual(server._list_profiles(), ["demo"])

    def test_prompt_path_resolution(self):
        Path(self._tmp.name, "demo.pt").write_text("p")
        expected = os.path.join(self._tmp.name, "demo.pt")
        self.assertEqual(server._profile_prompt_path("demo"), expected)
        self.assertIsNone(server._profile_prompt_path("missing"))

    def test_voices_include_cloned_profiles(self):
        Path(self._tmp.name, "myclone.pt").write_text("p")
        names = {entry["name"] for entry in server.list_voices()["voices"]}
        self.assertIn("myclone", names)
        self.assertIn("narrator", names)


if __name__ == "__main__":
    unittest.main()