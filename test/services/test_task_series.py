import sys
import unittest
from pathlib import Path
from unittest.mock import patch

# add project root to python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.models import const
from app.models.schema import VideoParams
from app.services import llm
from app.services import task as tm


def _series_params(**overrides):
    params = VideoParams(video_subject="beekeeping")
    params.series_enabled = True
    for name, value in overrides.items():
        setattr(params, name, value)
    return params


class TestSeriesOutline(unittest.TestCase):
    def test_parse_tolerates_fences_and_prose(self):
        parsed = llm._parse_series_outline('sure:\n```json\n["one", "two"]\n```')
        self.assertEqual(["one", "two"], parsed)

    def test_parse_rejects_non_string_items(self):
        self.assertEqual([], llm._parse_series_outline('[{"title": "one"}]'))

    def test_fixed_count_truncates_and_deduplicates(self):
        response = '["a", "A", "b", "c", "d"]'
        with patch.object(llm, "_generate_response", return_value=response):
            outline = llm.generate_series_outline("beekeeping", parts=3)
        self.assertEqual(["a", "b", "c"], outline)

    def test_automatic_count_keeps_every_chapter_the_model_returned(self):
        response = '["a", "b", "c", "d", "e", "f", "g"]'
        with patch.object(llm, "_generate_response", return_value=response):
            outline = llm.generate_series_outline("beekeeping", parts=0)
        self.assertEqual(7, len(outline))

    def test_automatic_prompt_does_not_pin_a_count(self):
        prompt = llm.build_series_outline_prompt("beekeeping", parts=0)
        self.assertIn("decide yourself how many chapters", prompt)
        self.assertNotIn("return exactly", prompt)
        self.assertIn("return exactly 20 chapters", llm.build_series_outline_prompt(
            "beekeeping", parts=20
        ))

    def test_provider_error_never_becomes_a_chapter(self):
        with patch.object(llm, "_generate_response", return_value="Error: no key"):
            self.assertEqual([], llm.generate_series_outline("beekeeping"))


class TestSeriesPartParams(unittest.TestCase):
    def test_each_part_gets_its_own_subject_script_and_keywords(self):
        params = _series_params(
            video_script="a script for the whole subject",
            video_terms="bees, hive",
            custom_audio_file="/tmp/narration.mp3",
        )
        outline = ["hive setup", "harvesting honey"]

        part = tm._build_series_part_params(params, outline, 2)

        self.assertEqual("harvesting honey", part.video_subject)
        self.assertEqual("", part.video_script)
        self.assertIsNone(part.video_terms)
        self.assertIsNone(part.custom_audio_file)
        self.assertFalse(part.series_enabled)

    def test_continuity_context_names_the_neighbouring_chapters(self):
        params = _series_params(video_script_prompt="keep it funny")
        outline = ["hive setup", "harvesting honey", "selling honey"]

        prompt = tm._build_series_part_params(params, outline, 2).video_script_prompt

        self.assertIn("part 2 of 3", prompt)
        self.assertIn("hive setup", prompt)
        self.assertIn("selling honey", prompt)
        # The user's own requirements survive, after the context that gets
        # truncated first.
        self.assertTrue(prompt.endswith("keep it funny"))

    def test_continuity_off_leaves_the_user_prompt_untouched(self):
        params = _series_params(
            video_script_prompt="keep it funny", series_continuity=False
        )
        part = tm._build_series_part_params(params, ["a", "b"], 1)
        self.assertEqual("keep it funny", part.video_script_prompt)


class TestRunSeries(unittest.TestCase):
    def setUp(self):
        self.states = []
        state_patcher = patch.object(tm.sm, "state")
        self.state = state_patcher.start()
        self.state.update_task.side_effect = lambda task_id, **kwargs: self.states.append(
            (task_id, kwargs)
        )
        patch.object(tm, "save_script_data").start()
        self.addCleanup(patch.stopall)

    def test_runs_one_pipeline_per_chapter_and_collects_the_videos(self):
        params = _series_params(series_outline=["one", "two", "three"])
        calls = []

        def fake_pipeline(task_id, part_params, **kwargs):
            calls.append((task_id, part_params.video_subject))
            return {"videos": [f"{task_id}/final-1.mp4"], "script": "text"}

        with patch.object(tm, "_run_pipeline", side_effect=fake_pipeline):
            result = tm._run_series("task-1", params)

        self.assertEqual(
            [
                ("task-1/part-01", "one"),
                ("task-1/part-02", "two"),
                ("task-1/part-03", "three"),
            ],
            calls,
        )
        self.assertEqual(3, len(result["videos"]))
        self.assertEqual(["one", "two", "three"], result["series_outline"])
        self.assertIsNone(result["warnings"])
        self.assertEqual(
            const.TASK_STATE_COMPLETE, self.states[-1][1]["state"]
        )

    def test_a_failed_part_is_reported_but_the_others_still_render(self):
        params = _series_params(series_outline=["one", "two"])

        def fake_pipeline(task_id, part_params, **kwargs):
            if part_params.video_subject == "one":
                return {"state": const.TASK_STATE_FAILED, "error": "no materials"}
            return {"videos": ["final-1.mp4"], "script": "text"}

        with patch.object(tm, "_run_pipeline", side_effect=fake_pipeline):
            result = tm._run_series("task-1", params)

        self.assertEqual(["final-1.mp4"], result["videos"])
        self.assertEqual(1, len(result["warnings"]))
        self.assertEqual("series_part_failed", result["warnings"][0]["code"])
        self.assertEqual(1, result["warnings"][0]["part"])

    def test_every_part_failing_fails_the_series(self):
        params = _series_params(series_outline=["one"])
        failure = {"state": const.TASK_STATE_FAILED, "error": "boom"}

        with patch.object(tm, "_run_pipeline", return_value=failure):
            result = tm._run_series("task-1", params)

        self.assertEqual(const.TASK_STATE_FAILED, result["state"])
        self.assertEqual("series", result["failed_stage"])

    def test_an_empty_outline_is_planned_from_the_subject(self):
        params = _series_params(series_parts=0)

        with patch.object(
            tm.llm, "generate_series_outline", return_value=["one", "two"]
        ) as planner:
            with patch.object(
                tm, "_run_pipeline", return_value={"videos": ["v.mp4"], "script": "s"}
            ):
                result = tm._run_series("task-1", params)

        planner.assert_called_once()
        self.assertEqual(0, planner.call_args.kwargs["parts"])
        self.assertEqual(["one", "two"], result["series_outline"])

    def test_a_series_that_cannot_be_planned_fails_before_any_pipeline_runs(self):
        params = _series_params()

        with patch.object(tm.llm, "generate_series_outline", return_value=[]):
            with patch.object(tm, "_run_pipeline") as pipeline:
                result = tm._run_series("task-1", params)

        pipeline.assert_not_called()
        self.assertEqual(const.TASK_STATE_FAILED, result["state"])


class TestStartRouting(unittest.TestCase):
    def test_series_tasks_take_the_series_driver(self):
        params = _series_params()
        with patch.object(tm, "_run_series", return_value={"videos": []}) as series:
            with patch.object(tm, "_run_pipeline") as pipeline:
                tm.start("task-1", params)
        series.assert_called_once()
        pipeline.assert_not_called()

    def test_single_video_tasks_keep_the_normal_pipeline(self):
        params = VideoParams(video_subject="beekeeping")
        with patch.object(tm, "_run_series") as series:
            with patch.object(tm, "_run_pipeline", return_value={}) as pipeline:
                tm.start("task-1", params)
        pipeline.assert_called_once()
        series.assert_not_called()

    def test_a_confirmed_loomloom_quote_cannot_be_reused_across_parts(self):
        params = _series_params()
        with patch.object(tm.sm, "state"):
            with patch.object(tm, "_run_series") as series:
                result = tm.start("task-1", params, loomloom_video_request=object())
        series.assert_not_called()
        self.assertEqual(const.TASK_STATE_FAILED, result["state"])


if __name__ == "__main__":
    unittest.main()
