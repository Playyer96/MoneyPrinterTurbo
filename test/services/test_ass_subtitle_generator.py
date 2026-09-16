"""Lock in the standalone ASS subtitle generator.

The pipeline writes ``subtitle.ass`` next to ``subtitle.srt`` so the user can
hand-author every override tag, swap styles wholesale, and edit timings
without re-running the whole task. The generator has to keep working when:

- the SRT is well-formed (blank-line separated cues),
- the SRT is hand-edited and missing a blank line between cues,
- the user pastes a raw ``[V4+ Styles]`` block to override the default,
- per-event override tags / shadow / blur / rotation are non-zero.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


class TestAssSubtitleGenerator(unittest.TestCase):
    def _params(self, **overrides) -> SimpleNamespace:
        defaults = dict(
            subtitle_style_preset="custom",
            text_fore_color="#FFFFFF",
            stroke_color="#000000",
            font_size=60,
            stroke_width=4.0,
            subtitle_position="bottom",
            custom_position=70.0,
            subtitle_animation="none",
            subtitle_display_mode="sentence",
            subtitle_casing="as_is",
            font_name="Anton-Regular.ttf",
            subtitle_background_enabled=False,
            rounded_subtitle_background=False,
            subtitle_ass_background_color="",
            subtitle_ass_style_override="",
            subtitle_ass_event_overrides="",
            subtitle_ass_shadow=0.0,
            subtitle_ass_blur=0.0,
            subtitle_ass_rotation=0.0,
        )
        defaults.update(overrides)
        return SimpleNamespace(**defaults)

    def _write_srt(self, srt_path: str, content: str) -> None:
        Path(srt_path).write_text(content, encoding="utf-8")

    def test_well_formed_srt_produces_one_dialogue_per_cue(self):
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from app.services import subtitle

        with tempfile.TemporaryDirectory() as tmp_dir:
            srt_path = str(Path(tmp_dir) / "in.srt")
            ass_path = str(Path(tmp_dir) / "out.ass")
            self._write_srt(
                srt_path,
                "1\n00:00:00,000 --> 00:00:02,000\nFirst line\n\n"
                "2\n00:00:02,500 --> 00:00:04,500\nSecond line\n\n",
            )

            ok = subtitle.create_ass_subtitle(srt_path, ass_path, self._params())
            self.assertTrue(ok)
            content = Path(ass_path).read_text(encoding="utf-8")
            self.assertEqual(content.count("Dialogue: 0,"), 2)
            self.assertIn("First line", content)
            self.assertIn("Second line", content)

    def test_missing_blank_line_does_not_collapse_cues(self):
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from app.services import subtitle

        with tempfile.TemporaryDirectory() as tmp_dir:
            srt_path = str(Path(tmp_dir) / "in.srt")
            ass_path = str(Path(tmp_dir) / "out.ass")
            # No blank line between cue 1's text and cue 2's timing line.
            self._write_srt(
                srt_path,
                "1\n00:00:00,000 --> 00:00:02,000\nFirst line\n"
                "2\n00:00:02,500 --> 00:00:04,500\nSecond line\n",
            )

            ok = subtitle.create_ass_subtitle(srt_path, ass_path, self._params())
            self.assertTrue(ok)
            content = Path(ass_path).read_text(encoding="utf-8")
            self.assertEqual(
                content.count("Dialogue: 0,"),
                2,
                "missing blank lines in source SRT must not collapse cues",
            )

    def test_raw_style_override_replaces_generated_block(self):
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from app.services import subtitle

        raw_style = (
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
            "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
            "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
            "Alignment, MarginL, MarginR, MarginV, Encoding\n"
            "Style: PinkBlur,Arial,80,&H00FF80FF,&H00FFFFFF,&H00000000,"
            "&H00000000,-1,0,0,0,100,100,0,0,1,3,2,2,40,40,40,1"
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            srt_path = str(Path(tmp_dir) / "in.srt")
            ass_path = str(Path(tmp_dir) / "out.ass")
            self._write_srt(
                srt_path,
                "1\n00:00:00,000 --> 00:00:02,000\nHello\n\n",
            )

            ok = subtitle.create_ass_subtitle(
                srt_path,
                ass_path,
                self._params(subtitle_ass_style_override=raw_style),
            )
            self.assertTrue(ok)
            content = Path(ass_path).read_text(encoding="utf-8")
            self.assertIn("Style: PinkBlur,Arial,80", content)
            # The auto-generated Style line should NOT appear when an
            # explicit override block is supplied.
            self.assertNotIn("Style: Default,Anton-Regular", content)

    def test_per_event_overrides_and_transforms_appear_in_every_cue(self):
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from app.services import subtitle

        with tempfile.TemporaryDirectory() as tmp_dir:
            srt_path = str(Path(tmp_dir) / "in.srt")
            ass_path = str(Path(tmp_dir) / "out.ass")
            self._write_srt(
                srt_path,
                "1\n00:00:00,000 --> 00:00:02,000\nHello\n\n"
                "2\n00:00:02,500 --> 00:00:04,500\nWorld\n\n",
            )

            ok = subtitle.create_ass_subtitle(
                srt_path,
                ass_path,
                self._params(
                    subtitle_ass_event_overrides=r"\fad(200,0)",
                    subtitle_ass_shadow=3.0,
                    subtitle_ass_blur=1.0,
                    subtitle_ass_rotation=15.0,
                ),
            )
            self.assertTrue(ok)
            content = Path(ass_path).read_text(encoding="utf-8")
            self.assertEqual(content.count(r"\shad3"), 2)
            self.assertEqual(content.count(r"\blur1"), 2)
            self.assertEqual(content.count(r"\frz15"), 2)
            self.assertEqual(content.count(r"\fad(200,0)"), 2)


if __name__ == "__main__":
    unittest.main()