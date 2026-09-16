import importlib.util
import os
import tempfile
import unittest
from pathlib import Path

# vendor/omnivoice is not a package on sys.path; load server.py by file.
_SERVER = Path(__file__).parent.parent.parent / "vendor" / "omnivoice" / "server.py"
_spec = importlib.util.spec_from_file_location("omnivoice_server", _SERVER)
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

    def test_presets_have_nonempty_instructions(self):
        self.assertTrue(all(value.strip() for value in server.VOICE_PRESETS.values()))


class TestProfileDeletion(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._orig_dir = server.PROFILES_DIR
        server.PROFILES_DIR = self._tmp.name

    def tearDown(self):
        server.PROFILES_DIR = self._orig_dir
        self._tmp.cleanup()

    def _make_files(self, name: str, exts: tuple[str, ...] = (".pt", ".json", ".mp3")):
        for ext in exts:
            Path(self._tmp.name, f"{name}{ext}").write_text("x")

    def test_delete_removes_all_profile_files(self):
        self._make_files("demo")
        result = server.delete_profile("demo")
        self.assertTrue(result["ok"])
        self.assertEqual(result["name"], "demo")
        self.assertEqual(result["removed_files"], 3)
        self.assertEqual(result["voices"], [])
        self.assertEqual(sorted(os.listdir(self._tmp.name)), [])

    def test_delete_keeps_other_profiles(self):
        self._make_files("keep")
        self._make_files("drop")
        result = server.delete_profile("drop")
        self.assertEqual(result["voices"], ["keep"])
        self.assertTrue(Path(self._tmp.name, "keep.pt").exists())
        self.assertFalse(Path(self._tmp.name, "drop.pt").exists())
        self.assertFalse(Path(self._tmp.name, "drop.json").exists())
        self.assertFalse(Path(self._tmp.name, "drop.mp3").exists())

    def test_delete_normalizes_name(self):
        self._make_files("myclone")
        result = server.delete_profile(" MyClone ")
        self.assertEqual(result["name"], "myclone")
        self.assertEqual(sorted(os.listdir(self._tmp.name)), [])

    def test_delete_missing_raises_404(self):
        with self.assertRaises(server.HTTPException) as ctx:
            server.delete_profile("ghost")
        self.assertEqual(ctx.exception.status_code, 404)

    def test_delete_empty_name_raises_400(self):
        with self.assertRaises(server.HTTPException) as ctx:
            server.delete_profile("   !!! ")
        self.assertEqual(ctx.exception.status_code, 400)

    def test_delete_preset_name_reports_missing(self):
        self.assertFalse(Path(self._tmp.name, "narrator.pt").exists())
        with self.assertRaises(server.HTTPException) as ctx:
            server.delete_profile("narrator")
        self.assertEqual(ctx.exception.status_code, 404)

    def test_voices_excludes_deleted_profile(self):
        self._make_files("a")
        self._make_files("b")
        self.assertEqual(sorted(server._list_profiles()), ["a", "b"])
        server.delete_profile("a")
        self.assertEqual(server._list_profiles(), ["b"])


class TestListProfilesEndpoint(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._orig_dir = server.PROFILES_DIR
        server.PROFILES_DIR = self._tmp.name

    def tearDown(self):
        server.PROFILES_DIR = self._orig_dir
        self._tmp.cleanup()

    def test_returns_only_clone_names(self):
        for name in ("demo", "alt"):
            Path(self._tmp.name, f"{name}.pt").write_text("p")
        result = server.list_profiles_endpoint()
        self.assertEqual(
            [entry["name"] for entry in result["profiles"]], ["alt", "demo"]
        )

    def test_empty_profiles_returns_empty_list(self):
        result = server.list_profiles_endpoint()
        self.assertEqual(result["profiles"], [])

    def test_non_pt_files_excluded(self):
        Path(self._tmp.name, "notaprofile.json").write_text("{}")
        result = server.list_profiles_endpoint()
        self.assertEqual(result["profiles"], [])


if __name__ == "__main__":
    unittest.main()
