"""Unit tests for subtitle and title style presets and helper functions."""

from __future__ import annotations

import unittest

from app.services.subtitle_styles import (
    SUBTITLE_PRESETS,
    SUPPORTED_SUBTITLE_ANIMATIONS,
    SUPPORTED_SUBTITLE_DISPLAY_MODES,
    SUPPORTED_TITLE_DURATIONS,
    SUPPORTED_TITLE_POSITIONS,
    TITLE_STYLES,
    apply_text_casing,
    build_display_cues,
    get_subtitle_preset,
    get_title_style,
)


class TestSubtitlePresets(unittest.TestCase):
    """Verify every subtitle preset has the required fields and sane values."""

    REQUIRED_KEYS = frozenset(
        {
            "id",
            "name",
            "description",
            "font_name",
            "text_fore_color",
            "font_size",
            "stroke_color",
            "stroke_width",
            "subtitle_background_enabled",
            "subtitle_background_color",
            "rounded_subtitle_background",
            "subtitle_animation",
            "subtitle_casing",
            "subtitle_display_mode",
            "subtitle_position",
            "highlight_color",
        }
    )

    def test_at_least_ten_presets(self):
        """The picker offers a broad set of distinct short-form presets."""
        self.assertGreaterEqual(len(SUBTITLE_PRESETS), 15)

    def test_custom_preset_exists(self):
        self.assertIn("custom", SUBTITLE_PRESETS)

    def test_all_presets_contain_required_keys(self):
        for preset_id, preset in SUBTITLE_PRESETS.items():
            with self.subTest(preset_id=preset_id):
                self.assertEqual(
                    self.REQUIRED_KEYS - set(preset.keys()),
                    set(),
                    f"Missing keys in preset '{preset_id}'",
                )

    def test_id_field_matches_dict_key(self):
        for preset_id, preset in SUBTITLE_PRESETS.items():
            with self.subTest(preset_id=preset_id):
                self.assertEqual(preset["id"], preset_id)

    def test_font_size_is_positive_int(self):
        for preset_id, preset in SUBTITLE_PRESETS.items():
            with self.subTest(preset_id=preset_id):
                self.assertIsInstance(preset["font_size"], int)
                self.assertGreater(preset["font_size"], 0)

    def test_stroke_width_is_non_negative(self):
        for preset_id, preset in SUBTITLE_PRESETS.items():
            with self.subTest(preset_id=preset_id):
                self.assertGreaterEqual(preset["stroke_width"], 0.0)

    def test_animation_values_are_supported(self):
        for preset_id, preset in SUBTITLE_PRESETS.items():
            with self.subTest(preset_id=preset_id):
                self.assertIn(
                    preset["subtitle_animation"], SUPPORTED_SUBTITLE_ANIMATIONS
                )

    def test_display_modes_are_supported(self):
        for preset_id, preset in SUBTITLE_PRESETS.items():
            with self.subTest(preset_id=preset_id):
                self.assertIn(
                    preset["subtitle_display_mode"], SUPPORTED_SUBTITLE_DISPLAY_MODES
                )

    def test_casing_values_are_valid(self):
        valid_casing = {"as_is", "uppercase", "lowercase", "capitalize"}
        for preset_id, preset in SUBTITLE_PRESETS.items():
            with self.subTest(preset_id=preset_id):
                self.assertIn(preset["subtitle_casing"], valid_casing)

    def test_color_strings_are_hex(self):
        import re

        hex_color = re.compile(r"^#[0-9A-Fa-f]{6}$")
        for preset_id, preset in SUBTITLE_PRESETS.items():
            with self.subTest(preset_id=preset_id):
                self.assertRegex(preset["text_fore_color"], hex_color)
                self.assertRegex(preset["stroke_color"], hex_color)
                self.assertRegex(preset["highlight_color"], hex_color)

    def test_binance_preset_uses_gold_karaoke(self):
        preset = SUBTITLE_PRESETS["binance_karaoke"]

        self.assertEqual(preset["subtitle_display_mode"], "karaoke")
        self.assertEqual(preset["highlight_color"], "#F0B90B")


class TestGetSubtitlePreset(unittest.TestCase):
    def test_returns_known_preset(self):
        result = get_subtitle_preset("tiktok_yellow")
        self.assertIsNotNone(result)
        self.assertEqual(result["id"], "tiktok_yellow")

    def test_returns_none_for_unknown(self):
        self.assertIsNone(get_subtitle_preset("nonexistent_style"))

    def test_returns_none_for_empty_string(self):
        self.assertIsNone(get_subtitle_preset(""))


class TestTitleStyles(unittest.TestCase):
    """Verify every title style has the required fields."""

    REQUIRED_KEYS = frozenset(
        {
            "id",
            "name",
            "description",
            "font_name",
            "text_color",
            "rounded",
            "casing",
        }
    )

    def test_at_least_seven_title_styles(self):
        """We defined 7 viral title styles."""
        self.assertGreaterEqual(len(TITLE_STYLES), 7)

    def test_all_styles_contain_required_keys(self):
        for style_id, style in TITLE_STYLES.items():
            with self.subTest(style_id=style_id):
                self.assertEqual(
                    self.REQUIRED_KEYS - set(style.keys()),
                    set(),
                    f"Missing keys in title style '{style_id}'",
                )

    def test_id_field_matches_dict_key(self):
        for style_id, style in TITLE_STYLES.items():
            with self.subTest(style_id=style_id):
                self.assertEqual(style["id"], style_id)


class TestGetTitleStyle(unittest.TestCase):
    def test_returns_known_style(self):
        result = get_title_style("red_banner")
        self.assertIsNotNone(result)
        self.assertEqual(result["id"], "red_banner")

    def test_returns_none_for_unknown(self):
        self.assertIsNone(get_title_style("nonexistent"))


class TestApplyTextCasing(unittest.TestCase):
    def test_uppercase(self):
        self.assertEqual(apply_text_casing("hello world", "uppercase"), "HELLO WORLD")

    def test_lowercase(self):
        self.assertEqual(apply_text_casing("Hello World", "lowercase"), "hello world")

    def test_capitalize(self):
        self.assertEqual(
            apply_text_casing("hello world test", "capitalize"), "Hello World Test"
        )

    def test_as_is(self):
        self.assertEqual(apply_text_casing("MiXeD CaSe", "as_is"), "MiXeD CaSe")

    def test_none_casing(self):
        self.assertEqual(apply_text_casing("Hello", None), "Hello")

    def test_empty_string(self):
        self.assertEqual(apply_text_casing("", "uppercase"), "")

    def test_unknown_casing_returns_unchanged(self):
        self.assertEqual(apply_text_casing("Hello", "foobar"), "Hello")

    def test_casing_preserves_karaoke_markers(self):
        self.assertEqual(
            apply_text_casing("hello {{active}}world{{/active}}", "uppercase"),
            "HELLO {{active}}WORLD{{/active}}",
        )


class TestBuildDisplayCues(unittest.TestCase):
    CUES = [
        ((0.0, 0.2), "Make"),
        ((0.2, 0.4), "every"),
        ((0.4, 0.6), "second"),
        ((0.6, 0.8), "count."),
    ]

    def test_word_groups_keep_the_outer_timing(self):
        self.assertEqual(
            build_display_cues(self.CUES, "two_words"),
            [((0.0, 0.4), "Make every"), ((0.4, 0.8), "second count.")],
        )

    def test_progressive_reveal_builds_the_phrase(self):
        cues = build_display_cues(self.CUES, "progressive")

        self.assertEqual([cue[1] for cue in cues], [
            "Make",
            "Make every",
            "Make every second",
            "Make every second count.",
        ])

    def test_karaoke_marks_only_the_active_word(self):
        cues = build_display_cues(self.CUES, "karaoke")

        self.assertEqual(cues[1][1], "Make {{active}}every{{/active}} second count.")


class TestSupportedConstants(unittest.TestCase):
    def test_animation_tuple_contains_none(self):
        self.assertIn("none", SUPPORTED_SUBTITLE_ANIMATIONS)

    def test_title_positions(self):
        self.assertIn("top", SUPPORTED_TITLE_POSITIONS)
        self.assertIn("center", SUPPORTED_TITLE_POSITIONS)
        self.assertIn("bottom", SUPPORTED_TITLE_POSITIONS)

    def test_title_durations(self):
        self.assertIn("intro", SUPPORTED_TITLE_DURATIONS)
        self.assertIn("full", SUPPORTED_TITLE_DURATIONS)


if __name__ == "__main__":
    unittest.main()
