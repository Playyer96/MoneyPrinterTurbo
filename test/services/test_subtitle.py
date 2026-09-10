import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

# Let the app package be imported from the repo root when this test file is run directly.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.services import subtitle


class TestSubtitleService(unittest.TestCase):
    def test_file_to_subtitles_returns_empty_for_missing_input(self):
        """An empty path and a missing file both return an empty list safely."""
        self.assertEqual(subtitle.file_to_subtitles(""), [])
        with tempfile.TemporaryDirectory() as tmp_dir:
            missing_file = Path(tmp_dir) / "missing.srt"
            self.assertEqual(subtitle.file_to_subtitles(str(missing_file)), [])

    def test_levenshtein_distance_and_similarity_cover_common_boundaries(self):
        """
        Subtitle correction uses edit distance to decide whether to keep merging
        adjacent subtitles, so cover four boundaries -- empty string, swapped
        arguments, case-insensitivity, and clearly dissimilar input -- to guard
        against wrong merges if the algorithm is ever tuned.
        """
        self.assertEqual(subtitle.levenshtein_distance("kitten", "sitting"), 3)
        self.assertEqual(subtitle.levenshtein_distance("a", "longer"), 6)
        self.assertEqual(subtitle.levenshtein_distance("hello", ""), 5)
        self.assertEqual(subtitle.similarity("Hello", "hello"), 1.0)
        self.assertLess(subtitle.similarity("hello", "world"), 0.5)

    def test_create_returns_empty_when_whisper_is_unavailable(self):
        """When the optional Whisper dependency is missing, skip rather than raising in the task thread."""
        with patch.object(subtitle, "WhisperModel", None):
            self.assertEqual(subtitle.create("audio.mp3"), "")

    def test_create_returns_none_when_whisper_model_cannot_load(self):
        """A failed model download or init must return a failure result so the task layer can update status."""
        with patch.object(subtitle, "model", None), patch.object(
            subtitle,
            "WhisperModel",
            side_effect=RuntimeError("model unavailable"),
        ):
            self.assertIsNone(subtitle.create("audio.mp3"))

    def test_create_writes_punctuated_and_trailing_segments(self):
        """
        A fake Whisper model exercises word-level timestamp handling without
        network access or loading the real model. One segment carries both a
        punctuation-terminated sentence and unpunctuated trailing text, covering
        both key write paths.
        """

        class _FakeWhisperModel:
            def __init__(self, **kwargs):
                self.init_kwargs = kwargs

            def transcribe(self, audio_file, **kwargs):
                words = [
                    SimpleNamespace(start=0.0, end=0.4, word="Hello"),
                    SimpleNamespace(start=0.4, end=0.9, word=" world."),
                    SimpleNamespace(start=1.0, end=1.5, word="Again"),
                ]
                segment = SimpleNamespace(
                    start=0.0,
                    end=1.8,
                    words=words,
                )
                info = SimpleNamespace(language="en", language_probability=0.99)
                return [segment], info

        with tempfile.TemporaryDirectory() as tmp_dir:
            subtitle_file = Path(tmp_dir) / "generated.srt"
            with patch.object(subtitle, "model", None), patch.object(
                subtitle,
                "WhisperModel",
                _FakeWhisperModel,
            ):
                subtitle.create("audio.mp3", str(subtitle_file))

            items = subtitle.file_to_subtitles(str(subtitle_file))

        self.assertEqual([item[2] for item in items], ["Hello world", "Again"])

    def test_create_word_level_writes_each_whisper_word_with_its_timing(self):
        """逐词模式应保留 Whisper 的每个词及其独立起止时间。"""
        transcribe_kwargs = {}

        class _FakeWhisperModel:
            def __init__(self, **_kwargs):
                pass

            def transcribe(self, _audio_file, **kwargs):
                transcribe_kwargs.update(kwargs)
                words = [
                    SimpleNamespace(start=0.1, end=0.4, word="Hello"),
                    SimpleNamespace(start=0.4, end=0.8, word=" world"),
                ]
                segment = SimpleNamespace(start=0.1, end=0.8, words=words)
                info = SimpleNamespace(language="en", language_probability=0.99)
                return [segment], info

        with tempfile.TemporaryDirectory() as tmp_dir:
            subtitle_file = Path(tmp_dir) / "word-level.srt"
            with patch.object(subtitle, "model", None), patch.object(
                subtitle,
                "WhisperModel",
                _FakeWhisperModel,
            ):
                subtitle.create(
                    "audio.mp3",
                    str(subtitle_file),
                    word_level=True,
                )

            items = subtitle.file_to_subtitles(str(subtitle_file))

        self.assertEqual([item[2] for item in items], ["Hello", "world"])
        self.assertIs(transcribe_kwargs["word_timestamps"], True)
        self.assertIs(transcribe_kwargs["vad_filter"], True)
        self.assertIn("00:00:00,100 --> 00:00:00,400", items[0][1])
        self.assertIn("00:00:00,400 --> 00:00:00,800", items[1][1])

    def test_correct_ignores_markdown_separator_lines(self):
        """
        The Whisper fallback correction stage must also ignore unspeakable script
        lines such as `---`.

        If the Markdown separator were kept here, `correct()` would see more script
        lines than subtitle lines and pad with `00:00:00,000 --> 00:00:00,000`,
        which editing software treats as an unimportable SRT.
        """
        original_srt = (
            "1\n"
            "00:00:00,100 --> 00:00:01,000\n"
            "第一段\n\n"
            "2\n"
            "00:00:01,100 --> 00:00:02,000\n"
            "第二段\n\n"
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            subtitle_file = Path(tmp_dir) / "subtitle.srt"
            subtitle_file.write_text(original_srt, encoding="utf-8")

            subtitle.correct(
                subtitle_file=str(subtitle_file),
                video_script="第一段\n---\n第二段",
            )

            corrected_srt = subtitle_file.read_text(encoding="utf-8")

        self.assertIn("第一段", corrected_srt)
        self.assertIn("第二段", corrected_srt)
        self.assertNotIn("---", corrected_srt)
        self.assertNotIn("00:00:00,000 --> 00:00:00,000", corrected_srt)

    def test_correct_merges_adjacent_subtitles_for_one_script_sentence(self):
        """
        Whisper may split one scripted sentence across several time blocks. The
        correction logic should merge the time range and restore the original
        script text so the final subtitles are not needlessly fragmented.
        """
        original_srt = (
            "1\n00:00:00,100 --> 00:00:01,000\nHello\n\n"
            "2\n00:00:01,000 --> 00:00:02,000\nworld\n\n"
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            subtitle_file = Path(tmp_dir) / "subtitle.srt"
            subtitle_file.write_text(original_srt, encoding="utf-8")

            subtitle.correct(str(subtitle_file), "Hello world")
            items = subtitle.file_to_subtitles(str(subtitle_file))

        self.assertEqual(len(items), 1)
        self.assertEqual(items[0][1], "00:00:00,100 --> 00:00:02,000")
        self.assertEqual(items[0][2], "Hello world")

    def test_correct_replaces_mismatch_and_appends_missing_script_line(self):
        """
        When the transcript disagrees with the script entirely, the script wins.
        Extra script sentences with no reusable timeline get an explicit zero-time
        placeholder, which keeps the text and preserves existing behavior.
        """
        original_srt = "1\n00:00:00,100 --> 00:00:01,000\nWrong text\n\n"

        with tempfile.TemporaryDirectory() as tmp_dir:
            subtitle_file = Path(tmp_dir) / "subtitle.srt"
            subtitle_file.write_text(original_srt, encoding="utf-8")

            subtitle.correct(str(subtitle_file), "Expected sentence. Extra sentence.")
            items = subtitle.file_to_subtitles(str(subtitle_file))

        self.assertEqual(
            [item[2] for item in items],
            ["Expected sentence", "Extra sentence"],
        )
        self.assertEqual(items[1][1], "00:00:00,000 --> 00:00:00,000")

    def test_file_to_subtitles_keeps_last_block_without_trailing_newline(self):
        """
        The final subtitle must be parsed even when the SRT file does not end
        with a trailing blank line. Many tools omit it, and previously the last
        block was silently dropped because only a blank line flushed a block.
        """
        srt_without_trailing_blank = (
            "1\n"
            "00:00:00,000 --> 00:00:01,000\n"
            "Hello\n\n"
            "2\n"
            "00:00:01,000 --> 00:00:02,000\n"
            "World"
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            subtitle_file = Path(tmp_dir) / "subtitle.srt"
            subtitle_file.write_text(srt_without_trailing_blank, encoding="utf-8")

            items = subtitle.file_to_subtitles(str(subtitle_file))

        self.assertEqual(len(items), 2)
        self.assertEqual(items[0][2], "Hello")
        self.assertEqual(items[1][2], "World")

    def test_file_to_subtitles_parses_blocks_with_trailing_newline(self):
        """A normal SRT ending in a blank line still parses all blocks."""
        srt_with_trailing_blank = (
            "1\n"
            "00:00:00,000 --> 00:00:01,000\n"
            "Hello\n\n"
            "2\n"
            "00:00:01,000 --> 00:00:02,000\n"
            "World\n\n"
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            subtitle_file = Path(tmp_dir) / "subtitle.srt"
            subtitle_file.write_text(srt_with_trailing_blank, encoding="utf-8")

            items = subtitle.file_to_subtitles(str(subtitle_file))

        self.assertEqual([item[2] for item in items], ["Hello", "World"])

    def test_create_passes_auto_device_through_to_whisper_model(self):
        """
        The device default must reach CTranslate2 as "auto".

        That is what lets one image use the GPU when the container has one
        attached and the CPU when it does not, so a hardcoded "cpu" here would
        silently pin every GPU host to the CPU.
        """
        self.assertEqual(subtitle.device, "auto")

        captured = {}

        class _FakeWhisperModel:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            def transcribe(self, *args, **kwargs):
                info = SimpleNamespace(language="en", language_probability=1.0)
                return [], info

        with tempfile.TemporaryDirectory() as tmp_dir:
            subtitle_file = Path(tmp_dir) / "subtitle.srt"
            with patch.object(subtitle, "model", None), patch.object(
                subtitle, "WhisperModel", _FakeWhisperModel
            ):
                subtitle.create("audio.mp3", str(subtitle_file))

        self.assertEqual(captured["device"], "auto")
        self.assertEqual(captured["compute_type"], "default")


if __name__ == "__main__":
    unittest.main()
