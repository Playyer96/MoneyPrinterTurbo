import sys
import unittest
from pathlib import Path

# add project root to python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.services import guardrails


class TestScriptGuardrails(unittest.TestCase):
    def test_meta_lines_and_stage_directions_are_removed(self):
        script = (
            "Sure, here's your script:\n"
            "Paragraph 1:\n"
            "[upbeat music]\n"
            "Narrator: The reef lost half its coral in a decade.\n\n"
            "Warming water bleaches the polyps. 🐠 #ocean"
        )
        cleaned = guardrails.scrub_script(script)

        self.assertNotIn("Sure, here", cleaned)
        self.assertNotIn("Paragraph 1", cleaned)
        self.assertNotIn("[upbeat music]", cleaned)
        self.assertNotIn("Narrator:", cleaned)
        self.assertNotIn("#ocean", cleaned)
        self.assertNotIn("🐠", cleaned)
        self.assertIn("The reef lost half its coral in a decade.", cleaned)
        self.assertIn("Warming water bleaches the polyps.", cleaned)

    def test_filler_sentences_are_dropped_but_content_survives(self):
        script = (
            "Let's dive right in. The volcano erupted for nine hours. "
            "Buckle up! Ash reached the stratosphere. "
            "Don't forget to like and subscribe."
        )
        cleaned = guardrails.scrub_script(script)

        self.assertIn("The volcano erupted for nine hours.", cleaned)
        self.assertIn("Ash reached the stratosphere.", cleaned)
        for filler in ("dive right in", "Buckle up", "subscribe"):
            self.assertNotIn(filler, cleaned)

    def test_find_slop_reports_generated_prose_vocabulary(self):
        found = guardrails.find_slop(
            "Let's delve into this rich tapestry of ever-evolving ideas."
        )
        self.assertIn("delve into", found)
        self.assertIn("tapestry", found)
        self.assertIn("ever-evolving", found)

    def test_clean_script_reports_no_slop(self):
        cleaned, slop = guardrails.enforce_script(
            "The dam holds back nine billion litres. It was built in 1974."
        )
        self.assertEqual(slop, [])
        self.assertIn("nine billion litres", cleaned)

    def test_scrubbing_never_empties_a_real_paragraph(self):
        cleaned = guardrails.scrub_script("Ash reached the stratosphere.")
        self.assertEqual(cleaned, "Ash reached the stratosphere.")


class TestResearchGuardrails(unittest.TestCase):
    def test_one_result_per_domain(self):
        results = [
            {"url": "https://a.test/1", "snippet": "first"},
            {"url": "https://www.a.test/2", "snippet": "second"},
            {"url": "https://b.test/1", "snippet": "third"},
        ]
        kept = guardrails.filter_research_results(results)
        self.assertEqual(
            [r["url"] for r in kept], ["https://a.test/1", "https://b.test/1"]
        )

    def test_duplicate_snippets_are_dropped(self):
        results = [
            {"url": "https://a.test/1", "snippet": "Same syndicated paragraph."},
            {"url": "https://b.test/1", "snippet": "Same syndicated paragraph."},
        ]
        self.assertEqual(len(guardrails.filter_research_results(results)), 1)

    def test_cookie_walls_are_not_usable_pages(self):
        self.assertFalse(guardrails.is_usable_page("We value your privacy"))
        self.assertFalse(guardrails.is_usable_page(""))
        self.assertTrue(
            guardrails.is_usable_page(
                "The eruption began at dawn. " * 10 + "It lasted nine hours."
            )
        )


class TestMediaGuardrails(unittest.TestCase):
    def test_voice_rate_and_volume_are_clamped(self):
        self.assertEqual(guardrails.clamp_voice_rate(9.0), guardrails.MAX_VOICE_RATE)
        self.assertEqual(guardrails.clamp_voice_rate(0.01), guardrails.MIN_VOICE_RATE)
        self.assertEqual(guardrails.clamp_voice_rate(1.2), 1.2)
        self.assertEqual(guardrails.clamp_voice_rate("nonsense"), 1.0)
        self.assertEqual(guardrails.clamp_voice_rate(float("nan")), 1.0)
        self.assertEqual(
            guardrails.clamp_voice_volume(99.0), guardrails.MAX_VOICE_VOLUME
        )

    def test_clip_duration_floor(self):
        self.assertEqual(guardrails.clamp_clip_duration(1), guardrails.MIN_CLIP_SECONDS)
        self.assertEqual(guardrails.clamp_clip_duration(8), 8)
        self.assertEqual(guardrails.clamp_clip_duration(None), 5)


class TestSubtitleGuardrails(unittest.TestCase):
    def test_short_cue_is_extended_into_free_time(self):
        cues = [
            {
                "msg": "A sentence that needs reading time.",
                "start_time": 0.0,
                "end_time": 0.3,
            },
            {"msg": "Next.", "start_time": 10.0, "end_time": 12.0},
        ]
        fixed = guardrails.enforce_subtitle_cues(cues)
        self.assertGreaterEqual(
            fixed[0]["end_time"] - fixed[0]["start_time"], guardrails.MIN_CUE_SECONDS
        )

    def test_overlapping_cues_never_share_the_screen(self):
        cues = [
            {"msg": "First cue text.", "start_time": 0.0, "end_time": 5.0},
            {"msg": "Second cue text.", "start_time": 2.0, "end_time": 6.0},
        ]
        fixed = guardrails.enforce_subtitle_cues(cues)
        self.assertLessEqual(fixed[0]["end_time"], fixed[1]["start_time"])

    def test_a_cue_is_never_extended_over_the_next_one(self):
        cues = [
            {
                "msg": "A long sentence that would need several seconds to read.",
                "start_time": 0.0,
                "end_time": 0.5,
            },
            {"msg": "Next.", "start_time": 0.8, "end_time": 2.0},
        ]
        fixed = guardrails.enforce_subtitle_cues(cues)
        self.assertLessEqual(fixed[0]["end_time"], fixed[1]["start_time"])

    def test_word_level_cues_keep_their_short_timings(self):
        cues = [
            {"msg": "one", "start_time": 0.0, "end_time": 0.2},
            {"msg": "two", "start_time": 0.25, "end_time": 0.4},
        ]
        fixed = guardrails.enforce_subtitle_cues(cues, word_level=True)
        self.assertEqual(len(fixed), 2)
        self.assertAlmostEqual(fixed[0]["end_time"], 0.2, places=3)

    def test_empty_and_reversed_cues_are_dropped(self):
        cues = [
            {"msg": "   ", "start_time": 0.0, "end_time": 1.0},
            {"msg": "backwards", "start_time": 5.0, "end_time": 4.0},
            {"msg": "kept", "start_time": 6.0, "end_time": 7.0},
        ]
        fixed = guardrails.enforce_subtitle_cues(cues)
        self.assertEqual([c["msg"] for c in fixed], ["kept"])


if __name__ == "__main__":
    unittest.main()
