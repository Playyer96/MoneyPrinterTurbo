import unittest
import json
import os
import shutil
import sys
import tempfile
from concurrent.futures import Future
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock
from uuid import uuid4

# add project root to python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.services import task as tm
from app.models.schema import MaterialInfo, VideoParams
from app.services.state import MemoryState, RedisState
from app.utils import utils

resources_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "resources")
RUN_INTEGRATION_TESTS = os.environ.get("MPT_RUN_INTEGRATION_TESTS", "").lower() in {
    "1",
    "true",
    "yes",
}


class TestTaskService(unittest.TestCase):
    def setUp(self):
        # The publish Future registry is process-level state. Clearing it between
        # tests keeps a mocked Future from leaking into later recovery tests without
        with tm._cross_post_registry_lock:
            tm._cross_post_futures.clear()

    def tearDown(self):
        with tm._cross_post_registry_lock:
            tm._cross_post_futures.clear()

    def test_is_task_busy_covers_generation_and_cross_posting(self):
        """Every delete entry point must recognise both active generation and active cross-posting."""
        busy_tasks = (
            {"state": tm.const.TASK_STATE_PROCESSING},
            {
                "state": tm.const.TASK_STATE_COMPLETE,
                "cross_post_state": tm.const.CROSS_POST_STATE_PENDING,
            },
            {
                "state": tm.const.TASK_STATE_COMPLETE,
                "cross_post_state": tm.const.CROSS_POST_STATE_PROCESSING,
            },
        )
        for task in busy_tasks:
            with self.subTest(task=task):
                self.assertTrue(tm.is_task_busy(task))

        self.assertFalse(
            tm.is_task_busy(
                {
                    "state": tm.const.TASK_STATE_COMPLETE,
                    "cross_post_state": tm.const.CROSS_POST_STATE_COMPLETE,
                }
            )
        )
        self.assertFalse(tm.is_task_busy(None))

    def test_generate_script_forwards_advanced_prompt_options(self):
        """
        The task entry point shares VideoParams with the WebUI/API. This checks that
        the advanced prompt fields still reach the LLM layer when the script is auto-generated, not only via /scripts.
        """
        params = VideoParams(
            video_subject="咖啡",
            video_script="",
            video_language="zh-CN",
            paragraph_number=2,
            video_script_prompt="语气轻松",
            custom_system_prompt="Only write short narration.",
        )

        with patch.object(
            tm.llm, "generate_script", return_value="生成的文案"
        ) as generate:
            result = tm.generate_script("task-id", params)

        self.assertEqual(result, "生成的文案")
        generate.assert_called_once_with(
            video_subject="咖啡",
            language="zh-CN",
            paragraph_number=2,
            video_script_prompt="语气轻松",
            custom_system_prompt="Only write short narration.",
        )

    def test_generate_final_videos_forwards_clip_speed_and_fit_mode(self):
        """The orchestration layer must pass clip speed and fit mode to the video service."""
        params = VideoParams(
            video_subject="test",
            video_count=1,
            video_clip_speed=1.25,
            video_fit_mode="contain",
        )

        with (
            patch.object(tm.video, "combine_videos") as combine_videos,
            patch.object(tm.video, "generate_video"),
            patch.object(tm.sm.state, "update_task"),
        ):
            tm.generate_final_videos(
                task_id="clip-speed-task",
                params=params,
                downloaded_videos=["material.mp4"],
                audio_file="audio.mp3",
                subtitle_path="",
                audio_duration=5,
            )

        self.assertEqual(combine_videos.call_args.kwargs["clip_speed"], 1.25)
        self.assertEqual(
            combine_videos.call_args.kwargs["video_fit_mode"],
            params.video_fit_mode,
        )

    def test_generate_final_videos_uses_generated_sonilo_music(self):
        """Sonilo must generate a soundtrack per combined video and pass it to the final mix."""
        params = VideoParams(
            video_subject="test",
            video_count=1,
            bgm_type="sonilo",
            sonilo_bgm_prompt="warm acoustic",
        )

        with (
            patch.object(tm.video, "combine_videos"),
            patch.object(
                tm.sonilo,
                "generate_bgm",
                side_effect=lambda **kwargs: kwargs["output_path"],
            ) as generate_bgm,
            patch.object(tm.video, "generate_video") as generate_video,
            patch.object(tm.sm.state, "update_task"),
        ):
            _, _, warnings = tm.generate_final_videos(
                task_id="sonilo-task",
                params=params,
                downloaded_videos=["material.mp4"],
                audio_file="audio.mp3",
                subtitle_path="",
                audio_duration=5,
            )

        self.assertEqual(warnings, [])
        self.assertEqual(generate_bgm.call_args.kwargs["video_duration"], 5)
        self.assertEqual(generate_bgm.call_args.kwargs["prompt"], "warm acoustic")
        self.assertTrue(
            generate_video.call_args.kwargs["bgm_file_override"].endswith(
                "sonilo-bgm-1.m4a"
            )
        )

    def test_generate_final_videos_uses_generated_elevenlabs_music(self):
        """ElevenLabs must reuse the video-music orchestration and the shared style prompt."""
        params = VideoParams(
            video_subject="test",
            video_count=1,
            bgm_type="elevenlabs",
            video_music_prompt="gentle documentary",
        )

        with (
            patch.object(tm.video, "combine_videos"),
            patch.object(
                tm.elevenlabs_music,
                "generate_bgm",
                side_effect=lambda **kwargs: kwargs["output_path"],
            ) as generate_bgm,
            patch.object(tm.video, "generate_video") as generate_video,
            patch.object(tm.sm.state, "update_task"),
        ):
            _, _, warnings = tm.generate_final_videos(
                task_id="elevenlabs-task",
                params=params,
                downloaded_videos=["material.mp4"],
                audio_file="audio.mp3",
                subtitle_path="",
                audio_duration=5,
            )

        self.assertEqual(warnings, [])
        self.assertEqual(generate_bgm.call_args.kwargs["video_duration"], 5)
        self.assertEqual(generate_bgm.call_args.kwargs["prompt"], "gentle documentary")
        self.assertTrue(
            generate_video.call_args.kwargs["bgm_file_override"].endswith(
                "elevenlabs-bgm-1.mp3"
            )
        )

    def test_generate_final_videos_falls_back_on_elevenlabs_failure(self):
        """A transient ElevenLabs failure must keep the music-less video and a structured warning."""
        params = VideoParams(video_subject="test", bgm_type="elevenlabs")

        with (
            patch.object(tm.video, "combine_videos"),
            patch.object(
                tm.elevenlabs_music,
                "generate_bgm",
                side_effect=tm.elevenlabs_music.ElevenLabsMusicError(
                    "temporary outage"
                ),
            ),
            patch.object(tm.video, "generate_video") as generate_video,
            patch.object(tm.sm.state, "update_task"),
        ):
            final_paths, _, warnings = tm.generate_final_videos(
                task_id="elevenlabs-fallback",
                params=params,
                downloaded_videos=["material.mp4"],
                audio_file="audio.mp3",
                subtitle_path="",
                audio_duration=5,
            )

        self.assertEqual(len(final_paths), 1)
        self.assertEqual(
            warnings,
            [{"code": "elevenlabs_bgm_failed", "video_index": 1}],
        )
        self.assertEqual(generate_video.call_args.kwargs["bgm_file_override"], "")

    def test_generate_final_videos_falls_back_without_bgm_on_sonilo_failure(self):
        """A third-party music failure must finish the video and surface a visible warning instead of dropping every artifact."""
        params = VideoParams(video_subject="test", bgm_type="sonilo")

        with (
            patch.object(tm.video, "combine_videos"),
            patch.object(
                tm.sonilo,
                "generate_bgm",
                side_effect=tm.sonilo.SoniloError("temporary outage"),
            ),
            patch.object(tm.video, "generate_video") as generate_video,
            patch.object(tm.sm.state, "update_task"),
        ):
            final_paths, _, warnings = tm.generate_final_videos(
                task_id="sonilo-fallback",
                params=params,
                downloaded_videos=["material.mp4"],
                audio_file="audio.mp3",
                subtitle_path="",
                audio_duration=5,
            )

        self.assertEqual(len(final_paths), 1)
        self.assertEqual(warnings, [{"code": "sonilo_bgm_failed", "video_index": 1}])
        self.assertEqual(generate_video.call_args.kwargs["bgm_file_override"], "")

    def test_generate_final_videos_skips_sonilo_when_volume_is_zero(self):
        """Zero volume must skip Sonilo generation entirely and explicitly disable leftover BGM."""
        params = VideoParams(
            video_subject="test",
            bgm_type="sonilo",
            bgm_volume=0.0,
            bgm_file="stale-custom-bgm.mp3",
        )

        with (
            patch.object(tm.video, "combine_videos"),
            patch.object(tm.sonilo, "generate_bgm") as generate_bgm,
            patch.object(tm.video, "generate_video", return_value=True) as generate,
            patch.object(tm.sm.state, "update_task"),
        ):
            final_paths, _, warnings = tm.generate_final_videos(
                task_id="sonilo-zero-volume",
                params=params,
                downloaded_videos=["material.mp4"],
                audio_file="audio.mp3",
                subtitle_path="",
                audio_duration=5,
            )

        self.assertEqual(len(final_paths), 1)
        self.assertEqual(warnings, [])
        generate_bgm.assert_not_called()
        self.assertEqual(generate.call_args.kwargs["bgm_file_override"], "")

    def test_generate_final_videos_warns_when_sonilo_mix_fails(self):
        """When Sonilo succeeds but the final mix fails, the task must keep the video and return a warning."""
        params = VideoParams(video_subject="test", bgm_type="sonilo")

        with (
            patch.object(tm.video, "combine_videos"),
            patch.object(
                tm.sonilo,
                "generate_bgm",
                side_effect=lambda **kwargs: kwargs["output_path"],
            ),
            patch.object(tm.video, "generate_video", return_value=False) as generate,
            patch.object(tm.sm.state, "update_task"),
        ):
            final_paths, _, warnings = tm.generate_final_videos(
                task_id="sonilo-mix-fallback",
                params=params,
                downloaded_videos=["material.mp4"],
                audio_file="audio.mp3",
                subtitle_path="",
                audio_duration=5,
            )

        self.assertEqual(len(final_paths), 1)
        self.assertEqual(warnings, [{"code": "sonilo_bgm_failed", "video_index": 1}])
        self.assertTrue(generate.call_args.kwargs["bgm_file_override"].endswith(".m4a"))

    def test_run_pipeline_fails_fast_when_ffmpeg_is_not_ready(self):
        """The full pipeline must confirm FFmpeg works before touching LLM/TTS/material services."""
        params = VideoParams(video_subject="test")
        state = MemoryState()
        with (
            patch.object(tm.utils, "check_ffmpeg_ready", return_value=False),
            patch.object(tm, "generate_script") as generate_script,
            patch.object(tm, "generate_audio") as generate_audio,
            patch.object(tm, "get_video_materials") as get_materials,
            patch.object(tm.sm, "state", state),
        ):
            result = tm.start("ffmpeg-missing", params)

        generate_script.assert_not_called()
        generate_audio.assert_not_called()
        get_materials.assert_not_called()
        self.assertEqual(result["state"], tm.const.TASK_STATE_FAILED)
        self.assertEqual(result["failed_stage"], "preflight")
        self.assertIn("ffmpeg", result["error"])

    def test_run_pipeline_skips_ffmpeg_check_for_script_stage(self):
        """The script stage does no audio/video work, so a missing FFmpeg must not reject it."""
        params = VideoParams(video_subject="test")
        state = MemoryState()
        with (
            patch.object(tm.utils, "check_ffmpeg_ready", return_value=False) as check,
            patch.object(tm, "generate_script", return_value="脚本") as generate_script,
            patch.object(tm.sm, "state", state),
        ):
            result = tm.start("ffmpeg-missing-script-stage", params, stop_at="script")

        check.assert_not_called()
        generate_script.assert_called_once()
        self.assertEqual(result, {"script": "脚本"})

    def test_run_pipeline_skips_ffmpeg_check_for_terms_stage(self):
        """The terms stage does not need FFmpeg either, so the probe must not run."""
        params = VideoParams(video_subject="test")
        state = MemoryState()
        with (
            patch.object(tm.utils, "check_ffmpeg_ready", return_value=False) as check,
            patch.object(tm, "generate_script", return_value="脚本"),
            patch.object(tm, "generate_terms", return_value=["term"]),
            patch.object(tm, "save_script_data"),
            patch.object(tm.sm, "state", state),
        ):
            result = tm.start("ffmpeg-missing-terms-stage", params, stop_at="terms")

        check.assert_not_called()
        self.assertEqual(result, {"script": "脚本", "terms": ["term"]})

    def test_run_pipeline_proceeds_past_ffmpeg_preflight_when_ready(self):
        """When FFmpeg is available the probe must not block script generation."""
        params = VideoParams(video_subject="test")
        state = MemoryState()
        with (
            patch.object(tm.utils, "check_ffmpeg_ready", return_value=True) as check,
            patch.object(tm, "generate_script", return_value="脚本") as generate_script,
            patch.object(tm.sm, "state", state),
        ):
            result = tm.start("ffmpeg-ready", params, stop_at="script")

        # Even though the script stage does not require FFmpeg, verify the probe is
        # skipped, matching the "only check outside script/terms" contract.
        check.assert_not_called()
        generate_script.assert_called_once()
        self.assertEqual(result, {"script": "脚本"})

    def test_start_rejects_missing_sonilo_key_before_costly_pipeline_steps(self):
        """A full task missing the Sonilo key must not call LLM, TTS or material services first."""
        params = VideoParams(video_subject="test", bgm_type="sonilo")
        state = MemoryState()
        with (
            patch.object(tm.sonilo, "is_enabled", return_value=False),
            patch.object(tm, "generate_script") as generate_script,
            patch.object(tm, "generate_audio") as generate_audio,
            patch.object(tm, "get_video_materials") as get_materials,
            patch.object(tm.sm, "state", state),
        ):
            result = tm.start("missing-sonilo-key", params)

        generate_script.assert_not_called()
        generate_audio.assert_not_called()
        get_materials.assert_not_called()
        failed_task = state.get_task("missing-sonilo-key")
        self.assertEqual(result, failed_task)
        self.assertEqual(failed_task["state"], tm.const.TASK_STATE_FAILED)
        self.assertEqual(failed_task["failed_stage"], "preflight")
        self.assertIn("API key", failed_task["error"])

    def test_start_does_not_require_sonilo_key_when_volume_is_zero(self):
        """Zero volume never uses Sonilo, so a missing key must still enter the normal pipeline."""
        params = VideoParams(
            video_subject="test",
            bgm_type="sonilo",
            bgm_volume=0.0,
        )
        state = MemoryState()
        with (
            patch.object(tm.sonilo, "is_enabled", return_value=False),
            patch.object(tm, "generate_script", return_value="") as generate_script,
            patch.object(tm.sm, "state", state),
        ):
            result = tm.start("zero-volume-without-key", params)

        generate_script.assert_called_once_with("zero-volume-without-key", params)
        self.assertEqual(result["failed_stage"], "script")

    def test_loomloom_material_failure_keeps_remote_run_id(self):
        """When a remote run fails after creation, the task state must keep the LoomLoom run ID."""
        params = VideoParams(video_subject="AI 办公", video_source="loomloom")
        settings = tm.loomloom.LoomLoomSettings(
            base_url="https://example.test/loom/v1",
            api_token="test-token",
            market_listing_id=tm.loomloom.DEFAULT_SCRIPT_MARKET_LISTING_ID,
        )
        batch = tm.loomloom.LoomLoomVideoBatch(
            input_rows=(
                {
                    "scenePrompt": "office worker",
                    "aspectRatio": "9:16",
                    "sceneIndex": "1",
                },
            ),
        )
        request = tm.loomloom.LoomLoomConfirmedVideoRequest(
            settings=settings,
            batch=batch,
            listing_version_id="version-1",
            client_request_id="mpt-video-1",
        )
        backend = MagicMock()
        backend.execute.return_value = tm.loomloom.LoomLoomExecution(
            run_id="run-1",
            transaction_id="transaction-1",
            transaction_status="running",
            listing_version_id="version-1",
        )
        backend.wait_for_run.side_effect = tm.loomloom.LoomLoomRunError(
            "remote run timeout"
        )
        state = MemoryState()
        state.update_task(
            "loomloom-material-timeout",
            state=tm.const.TASK_STATE_PROCESSING,
            progress=40,
        )

        with (
            patch.object(tm.sm, "state", state),
            patch.object(
                tm.loomloom,
                "LoomLoomVideoBackend",
                return_value=backend,
            ),
        ):
            result = tm.get_video_materials(
                "loomloom-material-timeout",
                params,
                ["office worker"],
                10,
                loomloom_video_request=request,
            )

        self.assertIsNone(result)
        failed_task = state.get_task("loomloom-material-timeout")
        self.assertEqual(failed_task["state"], tm.const.TASK_STATE_FAILED)
        self.assertEqual(failed_task["failed_stage"], "materials")
        self.assertEqual(failed_task["loomloom_run_id"], "run-1")
        self.assertEqual(failed_task["loomloom_listing_version_id"], "version-1")

    def test_loomloom_state_failure_does_not_abandon_paid_remote_run(self):
        """When the state backend is down, a billable remote run must still be awaited and downloaded."""
        params = VideoParams(video_subject="AI 办公", video_source="loomloom")
        settings = tm.loomloom.LoomLoomSettings(
            base_url="https://example.test/loom/v1",
            api_token="test-token",
            market_listing_id=tm.loomloom.DEFAULT_VIDEO_MARKET_LISTING_ID,
        )
        request = tm.loomloom.LoomLoomConfirmedVideoRequest(
            settings=settings,
            batch=tm.loomloom.LoomLoomVideoBatch(
                input_rows=(
                    {
                        "scenePrompt": "office worker",
                        "aspectRatio": "9:16",
                        "sceneIndex": "1",
                    },
                )
            ),
            listing_version_id="version-1",
            client_request_id="mpt-video-state-failure",
        )
        backend = MagicMock()
        backend.execute.return_value = tm.loomloom.LoomLoomExecution(
            run_id="paid-run-1",
            transaction_id="transaction-1",
            transaction_status="running",
            listing_version_id="version-1",
        )
        backend.download_video_results.return_value = ("clip.mp4",)
        unavailable_state = MagicMock()
        unavailable_state.patch_task.side_effect = RuntimeError("Redis unavailable")

        with (
            patch.object(tm.sm, "state", unavailable_state),
            patch.object(
                tm.loomloom,
                "LoomLoomVideoBackend",
                return_value=backend,
            ),
            patch.object(tm.time, "sleep") as sleep,
        ):
            result = tm.get_video_materials(
                "loomloom-state-failure",
                params,
                ["office worker"],
                10,
                loomloom_video_request=request,
            )

        self.assertEqual(result, ["clip.mp4"])
        self.assertEqual(
            unavailable_state.patch_task.call_count,
            tm._LOOMLOOM_STATE_WRITE_ATTEMPTS,
        )
        self.assertEqual(
            sleep.call_count,
            tm._LOOMLOOM_STATE_WRITE_ATTEMPTS - 1,
        )
        backend.wait_for_run.assert_called_once_with("paid-run-1")
        backend.download_video_results.assert_called_once()

    def test_mark_task_failed_preserves_a_specific_service_failure(self):
        """When the service layer already recorded a specific error, orchestration must not overwrite it with a generic one."""
        state = MemoryState()
        state.update_task(
            "specific-service-failure",
            state=tm.const.TASK_STATE_FAILED,
            progress=40,
            failed_stage="materials",
            error="remote run timed out",
            loomloom_run_id="run-1",
        )

        with patch.object(tm.sm, "state", state):
            result = tm._mark_task_failed(
                "specific-service-failure",
                "materials",
                "failed to prepare video materials",
            )

        self.assertEqual(result["error"], "remote run timed out")
        self.assertEqual(result["loomloom_run_id"], "run-1")

    def test_start_rejects_missing_elevenlabs_key_before_pipeline_steps(self):
        """A full task missing the ElevenLabs key must fail before any paid step."""
        params = VideoParams(video_subject="test", bgm_type="elevenlabs")
        state = MemoryState()
        with (
            patch.object(tm.elevenlabs_music, "is_enabled", return_value=False),
            patch.object(tm, "generate_script") as generate_script,
            patch.object(tm, "generate_audio") as generate_audio,
            patch.object(tm.sm, "state", state),
        ):
            result = tm.start("missing-elevenlabs-key", params)

        generate_script.assert_not_called()
        generate_audio.assert_not_called()
        self.assertEqual(result["state"], tm.const.TASK_STATE_FAILED)
        self.assertEqual(result["failed_stage"], "preflight")
        self.assertIn("ElevenLabs", result["error"])

    def test_start_rejects_free_elevenlabs_plan_before_pipeline_steps(self):
        """A confirmed free tier must not consume LLM, TTS or material quota first."""
        params = VideoParams(video_subject="test", bgm_type="elevenlabs")
        state = MemoryState()
        with (
            patch.object(tm.elevenlabs_music, "is_enabled", return_value=True),
            patch.object(
                tm.elevenlabs_music,
                "validate_generation_access",
                side_effect=(
                    tm.elevenlabs_music.ElevenLabsPaidPlanRequiredError(
                        "ElevenLabs Music API requires a paid plan"
                    )
                ),
            ) as validate_access,
            patch.object(tm, "generate_script") as generate_script,
            patch.object(tm, "generate_audio") as generate_audio,
            patch.object(tm.sm, "state", state),
        ):
            result = tm.start("free-elevenlabs-plan", params)

        validate_access.assert_called_once_with()
        generate_script.assert_not_called()
        generate_audio.assert_not_called()
        self.assertEqual(result["failed_stage"], "preflight")
        self.assertIn("paid plan", result["error"])

    def test_start_rejects_oversized_elevenlabs_prompt_before_account_check(self):
        """When API/CLI bypass the WebUI, an over-long prompt must still be rejected before expensive steps."""
        params = VideoParams(
            video_subject="test",
            bgm_type="elevenlabs",
            video_music_prompt="x" * 1001,
        )
        state = MemoryState()
        with (
            patch.object(tm.elevenlabs_music, "is_enabled", return_value=True),
            patch.object(
                tm.elevenlabs_music, "validate_generation_access"
            ) as validate_access,
            patch.object(tm, "generate_script") as generate_script,
            patch.object(tm.sm, "state", state),
        ):
            result = tm.start("oversized-elevenlabs-prompt", params)

        validate_access.assert_not_called()
        generate_script.assert_not_called()
        self.assertEqual(result["failed_stage"], "preflight")
        self.assertIn("1000", result["error"])

    def test_generate_terms_uses_script_order_mode_when_enabled(self):
        """
        The default mode is unaffected; only when the user explicitly enables
        script-ordered material matching does the task layer ask the LLM for ordered keywords and raise the count to cover more script segments.
        """
        params = VideoParams(
            video_subject="城市通勤",
            video_script="",
            match_materials_to_script=True,
        )

        with patch.object(
            tm.llm, "generate_terms", return_value=["city", "train"]
        ) as generate:
            result = tm.generate_terms("task-id", params, "先城市，再地铁")

        self.assertEqual(result, ["city", "train"])
        generate.assert_called_once_with(
            video_subject="城市通勤",
            video_script="先城市，再地铁",
            amount=8,
            match_script_order=True,
        )

    def test_start_stops_before_materials_when_term_provider_fails(self):
        """
        When the keyword provider fails the task must end immediately without downloading material.

        This covers the full error path from the task entry point, so a future fix to the
        service return type cannot be undone by orchestration turning an empty list into a truthy value and continuing.
        Audio may have been scheduled in parallel and is harmless to discard
        when terms fails — the safety property is "no material download".
        """
        params = VideoParams(
            video_subject="startup story",
            video_script="A short startup story.",
        )
        state = MemoryState()

        with (
            patch.object(
                tm.llm,
                "_generate_response",
                return_value="Error: invalid API key",
            ),
            patch.object(tm, "generate_audio") as generate_audio,
            patch.object(tm, "get_video_materials") as get_video_materials,
            patch.object(tm.sm, "state", state),
        ):
            result = tm.start("term-provider-error", params)

        get_video_materials.assert_not_called()
        failed_task = state.get_task("term-provider-error")
        self.assertEqual(result, failed_task)
        self.assertEqual(failed_task["state"], tm.const.TASK_STATE_FAILED)
        self.assertEqual(failed_task["failed_stage"], "terms")
        self.assertTrue(failed_task["error"])
        # Audio may have been submitted in parallel before terms failed; the
        # important assertion is that the pipeline stopped before any expensive
        # material step. generate_audio was either never called or was called
        # exactly once (it is discarded without progressing the pipeline).
        self.assertLessEqual(generate_audio.call_count, 1)

    def test_generate_audio_uses_custom_file_inside_task_directory(self):
        task_id = "test-custom-audio-safe"
        task_dir = utils.task_dir(task_id)
        custom_audio_file = os.path.join(task_dir, "custom-audio.mp3")
        with open(custom_audio_file, "wb") as audio:
            audio.write(b"fake audio")

        params = VideoParams(
            video_subject="custom audio",
            video_script="",
            custom_audio_file=custom_audio_file,
            voice_name="test-voice",
        )

        try:
            with (
                patch.object(tm.voice, "tts") as tts,
                patch.object(tm.voice, "get_audio_duration", return_value=7),
            ):
                audio_file, audio_duration, sub_maker = tm.generate_audio(
                    task_id, params, "script"
                )
        finally:
            shutil.rmtree(task_dir, ignore_errors=True)

        self.assertEqual(audio_file, os.path.realpath(custom_audio_file))
        self.assertEqual(audio_duration, 7)
        self.assertIsNone(sub_maker)
        tts.assert_not_called()

    def test_generate_audio_rejects_server_side_custom_file_by_default(self):
        task_id = "test-custom-audio-untrusted-server-side"
        task_dir = utils.task_dir(task_id)
        state = MemoryState()

        with tempfile.NamedTemporaryFile(suffix=".mp3") as server_audio:
            server_audio.write(b"fake audio")
            server_audio.flush()
            params = VideoParams(
                video_subject="custom audio",
                video_script="",
                custom_audio_file=server_audio.name,
                voice_name="test-voice",
            )

            try:
                with (
                    patch.object(tm.voice, "tts") as tts,
                    patch.object(tm.voice, "get_audio_duration") as get_duration,
                    patch.object(tm.sm, "state", state),
                ):
                    audio_file, audio_duration, result_sub_maker = tm.generate_audio(
                        task_id, params, "script"
                    )
            finally:
                shutil.rmtree(task_dir, ignore_errors=True)

        self.assertIsNone(audio_file)
        self.assertIsNone(audio_duration)
        self.assertIsNone(result_sub_maker)
        tts.assert_not_called()
        get_duration.assert_not_called()
        failed_task = state.get_task(task_id)
        self.assertEqual(failed_task["failed_stage"], "audio")
        self.assertIn("current task directory", failed_task["error"])

    def test_external_custom_audio_error_does_not_reveal_file_existence(self):
        task_id = "test-custom-audio-existence-oracle"
        task_dir = utils.task_dir(task_id)

        with tempfile.NamedTemporaryFile(suffix=".mp3") as server_audio:
            external_paths = [server_audio.name, f"{server_audio.name}.missing"]
            errors = []
            try:
                for external_path in external_paths:
                    with self.assertRaises(ValueError) as raised:
                        tm.resolve_custom_audio_file(task_id, external_path)
                    errors.append(str(raised.exception))
            finally:
                shutil.rmtree(task_dir, ignore_errors=True)

        self.assertEqual(errors[0], errors[1])
        self.assertIn("current task directory", errors[0])

    def test_generate_audio_accepts_server_side_custom_file_for_trusted_cli(self):
        task_id = "test-custom-audio-server-side"
        task_dir = utils.task_dir(task_id)

        with tempfile.NamedTemporaryFile(suffix=".mp3") as server_audio:
            server_audio.write(b"fake audio")
            server_audio.flush()
            params = VideoParams(
                video_subject="custom audio",
                video_script="",
                custom_audio_file=server_audio.name,
                voice_name="test-voice",
            )

            try:
                with (
                    patch.object(tm.voice, "tts") as tts,
                    patch.object(tm.voice, "get_audio_duration", return_value=6),
                ):
                    audio_file, audio_duration, result_sub_maker = tm.generate_audio(
                        task_id,
                        params,
                        "script",
                        allow_server_file_input=True,
                    )
            finally:
                shutil.rmtree(task_dir, ignore_errors=True)

        self.assertEqual(audio_file, os.path.realpath(server_audio.name))
        self.assertEqual(audio_duration, 6)
        self.assertIsNone(result_sub_maker)
        tts.assert_not_called()

    def test_generate_audio_rejects_missing_custom_file_without_tts(self):
        task_id = "test-custom-audio-missing"
        task_dir = utils.task_dir(task_id)
        missing_audio_file = os.path.join(task_dir, "missing.mp3")
        params = VideoParams(
            video_subject="custom audio",
            video_script="",
            custom_audio_file=missing_audio_file,
            voice_name="test-voice",
        )
        state = MemoryState()

        try:
            with (
                patch.object(tm.voice, "tts") as tts,
                patch.object(tm.sm, "state", state),
            ):
                audio_file, audio_duration, result_sub_maker = tm.generate_audio(
                    task_id, params, "script"
                )
        finally:
            shutil.rmtree(task_dir, ignore_errors=True)

        self.assertIsNone(audio_file)
        self.assertIsNone(audio_duration)
        self.assertIsNone(result_sub_maker)
        tts.assert_not_called()
        failed_task = state.get_task(task_id)
        self.assertEqual(failed_task["failed_stage"], "audio")
        self.assertIn("does not exist", failed_task["error"])

    def test_generate_audio_prefers_file_duration_over_sub_maker(self):
        # Every fixture deliberately makes the file duration and the SubMaker
        # duration ceil to DIFFERENT integers. If someone "simplifies" them to
        # values that share a ceil, this test can no longer tell which source
        # the implementation used - it stops discriminating, silently.
        cases = (
            # The maintainer's own reproduction numbers from the PR discussion.
            (8.4, 7.8375, 9),
            # An exact-integer file duration: proves math.ceil() is really
            # used and rules out int()+1 style code that adds a spurious
            # second. The SubMaker value ceils to 7, so 8 can only come
            # from the file.
            (8.0, 6.2, 8),
            # File duration shorter than the SubMaker value: the only case
            # where this change makes audio_duration smaller than before (the
            # old code returned 8). The contract is "the file wins", not
            # "the larger value wins".
            (5.0, 7.8375, 5),
        )

        for file_duration, sub_maker_duration, expected in cases:
            with self.subTest(file_duration=file_duration):
                task_id = f"test-tts-audio-priority-{uuid4().hex}"
                task_dir = utils.task_dir(task_id)
                audio_path = os.path.join(task_dir, "audio.mp3")
                params = VideoParams(
                    video_subject="tts audio",
                    video_script="",
                    voice_name="test-voice",
                )
                sub_maker = MagicMock()

                def fake_duration(target, _file=file_duration, _sub=sub_maker_duration):
                    # Dispatch on argument type, never on call order: a
                    # sequence side_effect would still pass against an
                    # implementation that measured the SubMaker first, which
                    # is exactly the regression this test exists to catch.
                    return _file if isinstance(target, str) else _sub

                try:
                    with (
                        patch.object(tm.voice, "tts", return_value=sub_maker) as tts,
                        patch.object(
                            tm.voice, "get_audio_duration", side_effect=fake_duration
                        ) as get_duration,
                    ):
                        audio_file, audio_duration, result_sub_maker = tm.generate_audio(
                            task_id, params, "script"
                        )
                finally:
                    shutil.rmtree(task_dir, ignore_errors=True)

                self.assertEqual(audio_file, audio_path)
                self.assertEqual(audio_duration, expected)
                # Asserting the value alone would still pass an
                # implementation returning 9.0; the type assertion pins the
                # other side of the rounding contract, so a refactor cannot
                # drop math.ceil() and pass the float straight through.
                self.assertIsInstance(audio_duration, int)
                self.assertIs(result_sub_maker, sub_maker)
                tts.assert_called_once()
                # When file measurement succeeds the SubMaker must not be
                # measured at all: exactly one call, and that call's argument
                # is the audio file path. Both assertions together are what
                # prove the priority order.
                self.assertEqual(len(get_duration.call_args_list), 1)
                self.assertEqual(get_duration.call_args_list[0].args[0], audio_path)

    def test_generate_audio_falls_back_to_sub_maker_when_file_duration_is_zero(self):
        task_id = "test-tts-audio-fallback"
        task_dir = utils.task_dir(task_id)
        audio_path = os.path.join(task_dir, "audio.mp3")
        params = VideoParams(
            video_subject="tts audio",
            video_script="",
            voice_name="test-voice",
        )
        sub_maker = MagicMock()

        def fake_duration(target):
            # voice.get_audio_duration() returns 0.0 when file measurement
            # fails (missing file or decode error); only then may the
            # SubMaker word-boundary duration be used.
            return 0.0 if isinstance(target, str) else 7.8375

        try:
            with (
                patch.object(tm.voice, "tts", return_value=sub_maker),
                patch.object(
                    tm.voice, "get_audio_duration", side_effect=fake_duration
                ) as get_duration,
            ):
                audio_file, audio_duration, result_sub_maker = tm.generate_audio(
                    task_id, params, "script"
                )
        finally:
            shutil.rmtree(task_dir, ignore_errors=True)

        self.assertEqual(audio_file, audio_path)
        self.assertEqual(audio_duration, 8)
        self.assertIsInstance(audio_duration, int)
        self.assertIs(result_sub_maker, sub_maker)
        self.assertEqual(len(get_duration.call_args_list), 2)
        self.assertEqual(get_duration.call_args_list[0].args[0], audio_path)
        self.assertIs(get_duration.call_args_list[1].args[0], sub_maker)

    def test_generate_audio_fails_when_file_and_sub_maker_durations_are_zero(self):
        # This change replaces the source of audio_duration, so the
        # pre-existing zero-duration guard must be proven to still fire
        # rather than be bypassed by the new file-measurement branch.
        task_id = "test-tts-audio-zero-duration"
        task_dir = utils.task_dir(task_id)
        params = VideoParams(
            video_subject="tts audio",
            video_script="",
            voice_name="test-voice",
        )
        sub_maker = MagicMock()

        try:
            with (
                patch.object(tm.voice, "tts", return_value=sub_maker),
                patch.object(tm.voice, "get_audio_duration", return_value=0.0),
                # generate_audio's zero-duration guard calls mark_task_failed
                # from within its home module (pipeline/stages.py), not
                # through the tm re-export, so the patch target must be the
                # stages module itself.
                patch.object(tm.stages, "mark_task_failed") as mark_task_failed,
            ):
                audio_file, audio_duration, result_sub_maker = tm.generate_audio(
                    task_id, params, "script"
                )
        finally:
            shutil.rmtree(task_dir, ignore_errors=True)

        self.assertIsNone(audio_file)
        self.assertIsNone(audio_duration)
        self.assertIsNone(result_sub_maker)
        mark_task_failed.assert_called_once_with(
            task_id, "audio", "generated audio duration is zero"
        )

    def test_generate_audio_skips_file_probe_for_non_real_word_providers(self):
        # Non-real-word providers (Gemini, SiliconFlow, MiniMax, ...) populate
        # sub_maker.duration with the actual rendered audio length, so the
        # file probe is pure overhead. The skip is the largest wall-clock
        # win for those providers on every audio stage (~500ms-1s each).
        task_id = "test-tts-skip-file-probe"
        task_dir = utils.task_dir(task_id)
        audio_path = os.path.join(task_dir, "audio.mp3")
        params = VideoParams(
            video_subject="tts audio",
            video_script="",
            voice_name="test-voice",
        )
        # Real MagicMock autovivifies _has_real_word_timestamps as truthy,
        # which routes to the file-probe path; explicitly set it to False
        # to test the non-real-word branch.
        sub_maker = MagicMock()
        sub_maker._has_real_word_timestamps = False
        sub_maker.duration = 7.4  # would round to 7; force a non-integer to catch math.ceil regressions

        try:
            with (
                patch.object(tm.voice, "tts", return_value=sub_maker) as tts,
                patch.object(
                    tm.voice, "get_audio_duration"
                ) as get_duration,
            ):
                audio_file, audio_duration, result_sub_maker = tm.generate_audio(
                    task_id, params, "script"
                )
        finally:
            shutil.rmtree(task_dir, ignore_errors=True)

        self.assertEqual(audio_file, audio_path)
        self.assertEqual(audio_duration, 8)  # ceil(7.4) = 8
        self.assertIs(result_sub_maker, sub_maker)
        tts.assert_called_once()
        get_duration.assert_not_called()

    def test_generate_subtitle_uses_whisper_for_custom_audio_without_sub_maker(self):
        """
        Custom audio never goes through TTS, so there is no sub_maker.
        Whisper can transcribe the audio file directly and must not be skipped early by the empty-sub_maker guard.
        """
        task_id = "test-custom-audio-whisper-subtitle"
        task_dir = utils.task_dir(task_id)
        audio_file = os.path.join(task_dir, "custom-audio.mp3")
        Path(audio_file).write_bytes(b"fake audio")
        params = VideoParams(
            video_subject="custom audio",
            video_script="Hello world.",
            subtitle_enabled=True,
            # The sentence-correction path needs an explicit mode; it must not inherit the dev machine's WebUI preference.
            subtitle_display_mode="sentence",
        )

        def fake_whisper_create(audio_file, subtitle_file, word_level=False):
            self.assertFalse(word_level)
            Path(subtitle_file).write_text(
                "1\n00:00:00,000 --> 00:00:01,000\nHello world.\n\n",
                encoding="utf-8",
            )

        try:
            with (
                patch.object(
                    tm.config,
                    "app",
                    dict(tm.config.app, subtitle_provider="whisper"),
                ),
                patch.object(
                    tm.subtitle, "create", side_effect=fake_whisper_create
                ) as create,
                patch.object(tm.subtitle, "correct") as correct,
            ):
                subtitle_path = tm.generate_subtitle(
                    task_id=task_id,
                    params=params,
                    video_script="Hello world.",
                    sub_maker=None,
                    audio_file=audio_file,
                )
        finally:
            shutil.rmtree(task_dir, ignore_errors=True)

        self.assertTrue(subtitle_path.endswith("subtitle.srt"))
        create.assert_called_once_with(
            audio_file=audio_file,
            subtitle_file=subtitle_path,
            word_level=False,
        )
        correct.assert_called_once_with(
            subtitle_file=subtitle_path, video_script="Hello world."
        )

    def test_generate_subtitle_word_modes_use_word_level_srt_and_align_words(self):
        """
        Word display modes still ask Whisper for one SRT row per word --
        the legacy MoviePy renderer (still used whenever intro/outro
        overlays are enabled) reads the SRT directly via ``SubtitlesClip``
        and needs word-level rows to group correctly; forcing sentence-level
        SRT here broke that path entirely (a real regression an earlier
        version of this fix introduced and this test now pins against).
        ``correct()`` (sentence-vs-script matching) is skipped for word-level
        SRT, exactly as before. The ``.words.json`` sidecar is written and
        re-aligned to the script tokens either way, for the newer ASS
        fast-path renderer.
        """
        task_id = "test-custom-audio-whisper-word-subtitle"
        task_dir = utils.task_dir(task_id)
        audio_file = os.path.join(task_dir, "custom-audio.mp3")
        Path(audio_file).write_bytes(b"fake audio")
        params = VideoParams(
            video_subject="custom audio",
            video_script="Hello world.",
            subtitle_enabled=True,
            subtitle_display_mode="word_by_word",
        )

        def fake_whisper_create(audio_file, subtitle_file, word_level=False):
            self.assertTrue(word_level)
            Path(subtitle_file).write_text(
                "1\n00:00:00,000 --> 00:00:00,400\nHello\n\n"
                "2\n00:00:00,500 --> 00:00:01,000\nworld\n\n",
                encoding="utf-8",
            )
            words_json = os.path.splitext(subtitle_file)[0] + ".words.json"
            Path(words_json).write_text(
                '{"version":1,"words":[{"w":"hello","s":0.0,"e":0.4},'
                '{"w":"world","s":0.5,"e":1.0}]}',
                encoding="utf-8",
            )

        try:
            with (
                patch.object(
                    tm.config,
                    "app",
                    dict(tm.config.app, subtitle_provider="whisper"),
                ),
                patch.object(
                    tm.subtitle, "create", side_effect=fake_whisper_create
                ) as create,
                patch.object(tm.subtitle, "correct") as correct,
            ):
                subtitle_path = tm.generate_subtitle(
                    task_id=task_id,
                    params=params,
                    video_script="Hello world.",
                    sub_maker=None,
                    audio_file=audio_file,
                )
                words_json = os.path.splitext(subtitle_path)[0] + ".words.json"
                aligned = json.loads(Path(words_json).read_text(encoding="utf-8"))
        finally:
            shutil.rmtree(task_dir, ignore_errors=True)

        self.assertTrue(subtitle_path.endswith("subtitle.srt"))
        create.assert_called_once_with(
            audio_file=audio_file,
            subtitle_file=subtitle_path,
            word_level=True,
        )
        correct.assert_not_called()
        self.assertTrue(aligned["aligned_to_script"])
        self.assertEqual([w["w"] for w in aligned["words"]], ["Hello", "world."])
        self.assertEqual(aligned["words"][1]["s"], 0.5)

    def test_generate_subtitle_skips_edge_provider_without_sub_maker(self):
        """
        Edge subtitles depend on the sub_maker timeline returned by TTS.
        When custom audio has no such object, skip and never produce an untrustworthy timeline.
        """
        task_id = "test-custom-audio-edge-no-submaker"
        task_dir = utils.task_dir(task_id)
        audio_file = os.path.join(task_dir, "custom-audio.mp3")
        Path(audio_file).write_bytes(b"fake audio")
        params = VideoParams(
            video_subject="custom audio",
            video_script="Hello world.",
            subtitle_enabled=True,
        )

        try:
            with (
                patch.object(
                    tm.config,
                    "app",
                    dict(tm.config.app, subtitle_provider="edge"),
                ),
                patch.object(tm.voice, "create_subtitle") as create_subtitle,
                patch.object(tm.subtitle, "create") as whisper_create,
            ):
                subtitle_path = tm.generate_subtitle(
                    task_id=task_id,
                    params=params,
                    video_script="Hello world.",
                    sub_maker=None,
                    audio_file=audio_file,
                )
        finally:
            shutil.rmtree(task_dir, ignore_errors=True)

        self.assertEqual(subtitle_path, "")
        create_subtitle.assert_not_called()
        whisper_create.assert_not_called()

    def test_generate_subtitle_does_not_fallback_to_whisper_when_edge_fails(self):
        """
        When Edge produces no subtitle file the result stays subtitle-less; Whisper must not be downloaded automatically.

        This can happen when the TTS timeline cannot be matched to the script. An automatic
        fallback would make users who never chose Whisper download a multi-GB model, so verify Whisper is never called.

        The sub_maker here must be explicitly marked as real word-level
        timestamps; otherwise it would be detected as a Gemini-style estimated
        timeline and auto-routed to Whisper.
        """
        task_id = "test-edge-subtitle-without-output"
        task_dir = utils.task_dir(task_id)
        params = VideoParams(
            video_subject="edge subtitle",
            video_script="Hello world.",
            subtitle_enabled=True,
        )
        sub_maker = tm.voice.mark_real_word_timestamps(tm.voice.SubMaker())

        try:
            with (
                patch.object(
                    tm.config,
                    "app",
                    dict(tm.config.app, subtitle_provider="edge"),
                ),
                patch.object(tm.voice, "create_subtitle") as create_subtitle,
                patch.object(tm.subtitle, "create") as whisper_create,
                patch.object(tm.subtitle, "correct") as whisper_correct,
            ):
                subtitle_path = tm.generate_subtitle(
                    task_id=task_id,
                    params=params,
                    video_script="Hello world.",
                    sub_maker=sub_maker,
                    audio_file=os.path.join(task_dir, "audio.mp3"),
                )
        finally:
            shutil.rmtree(task_dir, ignore_errors=True)

        self.assertEqual(subtitle_path, "")
        create_subtitle.assert_called_once()
        whisper_create.assert_not_called()
        whisper_correct.assert_not_called()

    def test_start_returns_each_intermediate_result(self):
        """
        The API's script, terms, audio, subtitle and materials modes share one task
        pipeline. Every early stop point must return its product without running later stages.
        """
        expected_results = {
            "script": {"script": "generated script"},
            "terms": {
                "script": "generated script",
                "terms": ["coffee", "morning"],
            },
            "audio": {"audio_file": "audio.mp3", "audio_duration": 5},
            "subtitle": {"subtitle_path": "subtitle.srt"},
            "materials": {"materials": ["clip.mp4"]},
        }

        for stop_at, expected in expected_results.items():
            with self.subTest(stop_at=stop_at):
                params = VideoParams(video_subject="Coffee")
                with (
                    patch.object(
                        tm, "generate_script", return_value="generated script"
                    ),
                    patch.object(
                        tm,
                        "generate_terms",
                        return_value=["coffee", "morning"],
                    ),
                    patch.object(tm, "save_script_data"),
                    patch.object(
                        tm,
                        "generate_audio",
                        return_value=("audio.mp3", 5, object()),
                    ),
                    patch.object(
                        tm,
                        "generate_subtitle",
                        return_value="subtitle.srt",
                    ),
                    patch.object(
                        tm,
                        "get_video_materials",
                        return_value=["clip.mp4"],
                    ),
                    patch.object(tm, "generate_final_videos") as generate_final,
                    patch.object(tm.sm.state, "update_task"),
                ):
                    result = tm.start(
                        f"intermediate-{stop_at}", params, stop_at=stop_at
                    )

                self.assertEqual(result, expected)
                generate_final.assert_not_called()

    def test_start_forwards_trusted_server_file_flag_to_audio_stage(self):
        params = VideoParams(video_subject="CLI custom audio")

        with (
            patch.object(tm.utils, "check_ffmpeg_ready", return_value=True),
            patch.object(tm, "generate_script", return_value="generated script"),
            patch.object(tm, "generate_terms", return_value=["audio"]),
            patch.object(tm, "save_script_data"),
            patch.object(
                tm,
                "generate_audio",
                return_value=("audio.mp3", 5, None),
            ) as generate_audio,
            patch.object(tm.sm.state, "update_task"),
        ):
            result = tm.start(
                "trusted-cli-audio",
                params,
                stop_at="audio",
                allow_server_file_input=True,
            )

        self.assertEqual(result, {"audio_file": "audio.mp3", "audio_duration": 5})
        generate_audio.assert_called_once_with(
            "trusted-cli-audio",
            params,
            "generated script",
            voice_preview=None,
            allow_server_file_input=True,
        )

    def test_start_completes_video_without_cross_posting(self):
        """
        A full task must complete reliably when auto-publish is unconfigured and write every
        intermediate product into the final state. This also covers the string concat-mode conversion the API may send.
        """
        params = VideoParams(video_subject="Coffee")
        params.video_concat_mode = "sequential"

        with (
            patch.object(tm, "generate_script", return_value="generated script"),
            patch.object(tm, "generate_terms", return_value=["coffee"]),
            patch.object(tm, "save_script_data"),
            patch.object(
                tm,
                "generate_audio",
                return_value=("audio.mp3", 5, object()),
            ),
            patch.object(tm, "generate_subtitle", return_value="subtitle.srt"),
            patch.object(
                tm,
                "get_video_materials",
                return_value=["clip.mp4"],
            ),
            patch.object(
                tm,
                "generate_final_videos",
                return_value=(["final.mp4"], ["combined.mp4"], []),
            ),
            patch.object(
                tm.upload_post.upload_post_service,
                "is_configured",
                return_value=False,
            ),
            patch.object(tm.upload_post, "cross_post_video") as cross_post,
            patch.object(tm.sm.state, "update_task") as update_task,
        ):
            result = tm.start("complete-video", params)

        self.assertEqual(result["videos"], ["final.mp4"])
        self.assertEqual(result["combined_videos"], ["combined.mp4"])
        self.assertEqual(result["cross_post_results"], None)
        self.assertEqual(params.video_concat_mode, tm.VideoConcatMode.sequential)
        cross_post.assert_not_called()
        update_task.assert_called_with(
            "complete-video",
            state=tm.const.TASK_STATE_COMPLETE,
            progress=100,
            **result,
        )

    def test_start_marks_pipeline_failures(self):
        """
        A missing audio, material or final video must put the task into the failed state; an
        incomplete task is never reported as complete. The three scenarios share one mock and only swap the failing stage.
        """
        failure_cases = {
            "audio": (
                (None, None, None),
                ["clip.mp4"],
                (["final.mp4"], ["combined.mp4"], []),
            ),
            "materials": (
                ("audio.mp3", 5, object()),
                None,
                (["final.mp4"], ["combined.mp4"], []),
            ),
            "video": (("audio.mp3", 5, object()), ["clip.mp4"], ([], [], [])),
        }

        for stage, failure_results in failure_cases.items():
            with self.subTest(stage=stage):
                audio_result, materials_result, videos_result = failure_results
                params = VideoParams(video_subject="Coffee")
                state = MemoryState()
                with (
                    patch.object(
                        tm, "generate_script", return_value="generated script"
                    ),
                    patch.object(tm, "generate_terms", return_value=["coffee"]),
                    patch.object(tm, "save_script_data"),
                    patch.object(tm, "generate_audio", return_value=audio_result),
                    patch.object(tm, "generate_subtitle", return_value="subtitle.srt"),
                    patch.object(
                        tm,
                        "get_video_materials",
                        return_value=materials_result,
                    ),
                    patch.object(
                        tm,
                        "generate_final_videos",
                        return_value=videos_result,
                    ),
                    patch.object(tm.sm, "state", state),
                ):
                    result = tm.start(f"failed-{stage}", params)

                failed_task = state.get_task(f"failed-{stage}")
                self.assertEqual(result, failed_task)
                self.assertEqual(failed_task["state"], tm.const.TASK_STATE_FAILED)
                self.assertEqual(failed_task["failed_stage"], stage)
                self.assertTrue(failed_task["error"])

    def test_start_records_unexpected_pipeline_exception(self):
        """An unexpected exception must also end the task and expose the original exception type and message to the API."""
        params = VideoParams(video_subject="Coffee")
        state = MemoryState()

        with (
            patch.object(
                tm,
                "generate_script",
                side_effect=RuntimeError("provider connection reset"),
            ),
            patch.object(tm.sm, "state", state),
        ):
            result = tm.start("unexpected-failure", params)

        failed_task = state.get_task("unexpected-failure")
        self.assertEqual(result, failed_task)
        self.assertEqual(failed_task["state"], tm.const.TASK_STATE_FAILED)
        self.assertEqual(failed_task["failed_stage"], "pipeline")
        self.assertEqual(
            failed_task["error"],
            "RuntimeError: provider connection reset",
        )

    def test_start_generates_youtube_metadata_for_each_cross_post(self):
        """
        Auto-publishing to YouTube generates metadata once but passes the same fields to every
        output, keeping each upload's independent success or failure in the task result.
        """
        params = VideoParams(
            video_subject="Coffee",
            video_language="en",
        )
        metadata = {
            "title": "Morning Coffee",
            "caption": "A better morning.",
            "hashtags": ["coffee", "shorts"],
        }
        service = tm.upload_post.upload_post_service
        state = MemoryState()

        def run_immediately(function, *args):
            future = Future()
            try:
                function(*args)
            except Exception as exc:
                future.set_exception(exc)
            else:
                future.set_result(None)
            return future

        with (
            patch.object(tm, "generate_script", return_value="generated script"),
            patch.object(tm, "generate_terms", return_value=["coffee"]),
            patch.object(tm, "save_script_data"),
            patch.object(
                tm,
                "generate_audio",
                return_value=("audio.mp3", 5, object()),
            ),
            patch.object(tm, "generate_subtitle", return_value="subtitle.srt"),
            patch.object(
                tm,
                "get_video_materials",
                return_value=["clip.mp4"],
            ),
            patch.object(
                tm,
                "generate_final_videos",
                return_value=(
                    ["final-1.mp4", "final-2.mp4"],
                    ["combined-1.mp4", "combined-2.mp4"],
                    [],
                ),
            ),
            patch.object(service, "is_configured", return_value=True),
            patch.object(type(service), "auto_upload", new_callable=PropertyMock, return_value=True),
            patch.object(type(service), "platforms", new_callable=PropertyMock, return_value=["youtube"]),
            patch.object(type(service), "youtube_privacy_status", new_callable=PropertyMock, return_value="unlisted"),
            patch.object(
                tm.llm,
                "generate_social_metadata",
                return_value=metadata,
            ) as generate_metadata,
            patch.object(
                tm.upload_post,
                "cross_post_video",
                side_effect=[
                    {"success": True},
                    {"success": False, "error": "upload failed"},
                ],
            ) as cross_post,
            patch.object(tm.sm, "state", state),
            patch.object(
                tm._cross_post_executor,
                "submit",
                side_effect=run_immediately,
            ),
        ):
            result = tm.start("youtube-cross-post", params)

        generate_metadata.assert_called_once_with(
            video_subject="Coffee",
            video_script="generated script",
            language="en",
            platform="youtube_shorts",
        )
        expected_extra = {
            "youtube_title": "Morning Coffee",
            "youtube_description": "A better morning.",
            "tags": ["coffee", "shorts"],
            "privacyStatus": "unlisted",
            "containsSyntheticMedia": True,
        }
        self.assertEqual(cross_post.call_count, 2)
        for call in cross_post.call_args_list:
            self.assertEqual(call.kwargs["youtube_extra"], expected_extra)
            self.assertEqual(call.kwargs["platforms"], ["youtube"])

        # start() returns the stable snapshot at video completion; background publish results come from the task query.
        self.assertEqual(result["cross_post_state"], tm.const.CROSS_POST_STATE_PENDING)
        self.assertIsNone(result["cross_post_results"])
        published_task = state.get_task("youtube-cross-post")
        self.assertEqual(published_task["state"], tm.const.TASK_STATE_COMPLETE)
        self.assertEqual(
            published_task["cross_post_state"], tm.const.CROSS_POST_STATE_FAILED
        )
        self.assertEqual(
            published_task["cross_post_results"],
            [
                {"success": True},
                {"success": False, "error": "upload failed"},
            ],
        )
        self.assertEqual(published_task["cross_post_error"], "upload failed")

    def test_start_returns_before_cross_post_worker_runs(self):
        """Video completion only submits the publish job; it must not upload synchronously in the generation thread."""
        params = VideoParams(video_subject="Coffee")
        service = tm.upload_post.upload_post_service
        state = MemoryState()
        submitted = []

        def capture_submission(function, *args):
            submitted.append((function, args))
            return MagicMock(spec=Future)

        with (
            patch.object(tm, "generate_script", return_value="generated script"),
            patch.object(tm, "generate_terms", return_value=["coffee"]),
            patch.object(tm, "save_script_data"),
            patch.object(
                tm,
                "generate_audio",
                return_value=("audio.mp3", 5, object()),
            ),
            patch.object(tm, "generate_subtitle", return_value="subtitle.srt"),
            patch.object(tm, "get_video_materials", return_value=["clip.mp4"]),
            patch.object(
                tm,
                "generate_final_videos",
                return_value=(["final.mp4"], ["combined.mp4"], []),
            ),
            patch.object(service, "is_configured", return_value=True),
            patch.object(type(service), "auto_upload", new_callable=PropertyMock, return_value=True),
            patch.object(type(service), "platforms", new_callable=PropertyMock, return_value=["tiktok"]),
            patch.object(type(service), "youtube_privacy_status", new_callable=PropertyMock, return_value="private"),
            patch.object(tm.upload_post, "cross_post_video") as cross_post,
            patch.object(tm.sm, "state", state),
            patch.object(
                tm._cross_post_executor,
                "submit",
                side_effect=capture_submission,
            ) as submit,
        ):
            result = tm.start("deferred-cross-post", params)

        submit.assert_called_once()
        cross_post.assert_not_called()
        self.assertEqual(result["videos"], ["final.mp4"])
        self.assertEqual(result["cross_post_state"], tm.const.CROSS_POST_STATE_PENDING)
        completed_task = state.get_task("deferred-cross-post")
        self.assertEqual(completed_task["state"], tm.const.TASK_STATE_COMPLETE)
        self.assertEqual(completed_task["progress"], 100)

        worker, worker_args = submitted[0]
        with (
            patch.object(tm.sm, "state", state),
            patch.object(
                tm.upload_post,
                "cross_post_video",
                return_value={"success": True, "request_id": "upload-1"},
            ),
            patch.object(
                tm.upload_post.upload_post_service,
                "poll_status",
                return_value={"success": True, "platform_statuses": {}},
            ),
        ):
            worker(*worker_args)

        published_task = state.get_task("deferred-cross-post")
        self.assertEqual(published_task["videos"], ["final.mp4"])
        self.assertEqual(
            published_task["cross_post_state"], tm.const.CROSS_POST_STATE_COMPLETE
        )

    def test_cross_post_worker_failure_does_not_change_video_completion(self):
        """A publish-thread exception may only update the publish state, never the completed video result."""
        state = MemoryState()
        state.update_task(
            "cross-post-worker-failure",
            state=tm.const.TASK_STATE_COMPLETE,
            progress=100,
            videos=["final.mp4"],
            cross_post_state=tm.const.CROSS_POST_STATE_PENDING,
        )

        with (
            patch.object(tm.sm, "state", state),
            patch.object(
                tm.llm,
                "generate_social_metadata",
                side_effect=RuntimeError("metadata provider unavailable"),
            ),
            patch.object(tm.upload_post, "cross_post_video") as cross_post,
        ):
            tm._run_cross_post(
                "cross-post-worker-failure",
                ("final.mp4",),
                "Coffee",
                "A short coffee story.",
                "en",
                ("youtube",),
                "private",
            )

        cross_post.assert_not_called()
        task = state.get_task("cross-post-worker-failure")
        self.assertEqual(task["state"], tm.const.TASK_STATE_COMPLETE)
        self.assertEqual(task["videos"], ["final.mp4"])
        self.assertEqual(task["cross_post_state"], tm.const.CROSS_POST_STATE_FAILED)
        self.assertIn("metadata provider unavailable", task["cross_post_error"])

    def test_start_returns_cross_post_scheduling_failure(self):
        """A synchronous scheduling failure must show in both the task state and the start() snapshot."""
        params = VideoParams(video_subject="Coffee")
        service = tm.upload_post.upload_post_service
        state = MemoryState()

        with (
            patch.object(tm, "generate_script", return_value="generated script"),
            patch.object(tm, "generate_terms", return_value=["coffee"]),
            patch.object(tm, "save_script_data"),
            patch.object(
                tm,
                "generate_audio",
                return_value=("audio.mp3", 5, object()),
            ),
            patch.object(tm, "generate_subtitle", return_value="subtitle.srt"),
            patch.object(tm, "get_video_materials", return_value=["clip.mp4"]),
            patch.object(
                tm,
                "generate_final_videos",
                return_value=(["final.mp4"], ["combined.mp4"], []),
            ),
            patch.object(service, "is_configured", return_value=True),
            patch.object(type(service), "auto_upload", new_callable=PropertyMock, return_value=True),
            patch.object(type(service), "platforms", new_callable=PropertyMock, return_value=["tiktok"]),
            patch.object(type(service), "youtube_privacy_status", new_callable=PropertyMock, return_value="private"),
            patch.object(tm.sm, "state", state),
            patch.object(tm._cross_post_slots, "acquire", return_value=False),
            patch.object(tm._cross_post_executor, "submit") as submit,
        ):
            result = tm.start("cross-post-queue-full-result", params)

        submit.assert_not_called()
        self.assertEqual(result["cross_post_state"], tm.const.CROSS_POST_STATE_FAILED)
        self.assertIn("queue is full", result["cross_post_error"])
        persisted_task = state.get_task("cross-post-queue-full-result")
        self.assertEqual(
            persisted_task["cross_post_state"], tm.const.CROSS_POST_STATE_FAILED
        )
        self.assertEqual(
            persisted_task["cross_post_error"],
            result["cross_post_error"],
        )

    def test_cross_post_schedule_failure_is_recorded_separately(self):
        """When the pool rejects new work the output must be kept and a queryable publish error provided."""
        state = MemoryState()
        slots = MagicMock()
        slots.acquire.return_value = True
        state.update_task(
            "cross-post-schedule-failure",
            state=tm.const.TASK_STATE_COMPLETE,
            progress=100,
            videos=["final.mp4"],
            cross_post_state=tm.const.CROSS_POST_STATE_PENDING,
        )

        with (
            patch.object(tm.sm, "state", state),
            patch.object(tm, "_cross_post_slots", slots),
            patch.object(
                tm._cross_post_executor,
                "submit",
                side_effect=RuntimeError("executor is shutting down"),
            ),
        ):
            scheduling_error = tm._schedule_cross_post(
                task_id="cross-post-schedule-failure",
                video_paths=["final.mp4"],
                params=VideoParams(video_subject="Coffee"),
                video_script="A short coffee story.",
                platforms=["tiktok"],
                youtube_privacy_status="private",
            )

        slots.release.assert_called_once_with()
        self.assertIn("executor is shutting down", scheduling_error)
        task = state.get_task("cross-post-schedule-failure")
        self.assertEqual(task["state"], tm.const.TASK_STATE_COMPLETE)
        self.assertEqual(task["videos"], ["final.mp4"])
        self.assertEqual(task["cross_post_state"], tm.const.CROSS_POST_STATE_FAILED)
        self.assertIn("executor is shutting down", task["cross_post_error"])

    def test_cross_post_worker_always_releases_queue_slot(self):
        """Capacity must be returned when the publish job exits abnormally so later publishes are not rejected forever."""
        slots = MagicMock()
        state = MemoryState()
        state.update_task(
            "task-id",
            state=tm.const.TASK_STATE_COMPLETE,
            progress=100,
            cross_post_state=tm.const.CROSS_POST_STATE_PENDING,
        )

        with (
            patch.object(tm, "_cross_post_slots", slots),
            patch.object(tm.sm, "state", state),
            patch.object(
                tm,
                "_run_cross_post",
                side_effect=RuntimeError("worker crashed"),
            ),
        ):
            tm._run_cross_post_with_slot("task-id")

        slots.release.assert_called_once_with()
        task = state.get_task("task-id")
        self.assertEqual(task["cross_post_state"], tm.const.CROSS_POST_STATE_FAILED)
        self.assertIn("worker crashed", task["cross_post_error"])

    def test_cross_post_state_backend_failure_is_logged_and_skips_upload(self):
        """A failed first state write must neither exit silently nor keep consuming publish quota."""
        state = MagicMock()
        state.patch_task.side_effect = RuntimeError("redis unavailable")

        with (
            patch.object(tm.sm, "state", state),
            patch.object(tm.upload_post, "cross_post_video") as cross_post,
            patch.object(tm.logger, "exception") as log_exception,
            patch.object(tm.time, "sleep") as sleep,
        ):
            tm._run_cross_post(
                "state-backend-failure",
                ("final.mp4",),
                "Coffee",
                "A short coffee story.",
                "en",
                ("tiktok",),
                "private",
            )

        cross_post.assert_not_called()
        self.assertEqual(state.patch_task.call_count, 6)
        self.assertEqual(sleep.call_count, 4)
        self.assertEqual(log_exception.call_count, 2)
        self.assertTrue(
            all(
                "redis unavailable" in call.args[0]
                for call in log_exception.call_args_list
            )
        )

    def test_cross_post_state_update_retries_transient_backend_failure(self):
        """After one transient state-backend failure publishing must continue and eventually save the completed state."""

        class FlakyMemoryState(MemoryState):
            def __init__(self):
                super().__init__()
                self.patch_calls = 0

            def patch_task(self, task_id, **kwargs):
                self.patch_calls += 1
                if self.patch_calls == 1:
                    raise RuntimeError("temporary redis outage")
                return super().patch_task(task_id, **kwargs)

        state = FlakyMemoryState()
        state.update_task(
            "transient-state-failure",
            state=tm.const.TASK_STATE_COMPLETE,
            progress=100,
            videos=["final.mp4"],
            cross_post_state=tm.const.CROSS_POST_STATE_PENDING,
        )

        with (
            patch.object(tm.sm, "state", state),
            patch.object(
                tm.upload_post,
                "cross_post_video",
                return_value={"success": True, "request_id": "upload-1"},
            ) as cross_post,
            patch.object(
                tm.upload_post.upload_post_service,
                "poll_status",
                return_value={"success": True, "platform_statuses": {}},
            ),
            patch.object(tm.time, "sleep") as sleep,
        ):
            tm._run_cross_post(
                "transient-state-failure",
                ("final.mp4",),
                "Coffee",
                "A short coffee story.",
                "en",
                ("tiktok",),
                "private",
            )

        sleep.assert_called_once_with(tm._CROSS_POST_STATE_RETRY_DELAY_SECONDS)
        cross_post.assert_called_once()
        task = state.get_task("transient-state-failure")
        self.assertEqual(task["cross_post_state"], tm.const.CROSS_POST_STATE_COMPLETE)
        self.assertIsNone(task["cross_post_error"])

    def test_cross_post_generates_caption_for_non_youtube_platforms(self):
        """
        TikTok/Instagram publishing must also generate social copy once and use the caption as
        the publish title shared by every output instead of sending the raw subject.
        """
        metadata = {
            "title": "Coffee Hook",
            "caption": "Watch this coffee ritual.",
            "hashtags": ["#coffee"],
        }
        state = MemoryState()
        cases = {
            "tiktok-first": (("tiktok", "instagram"), "tiktok"),
            "instagram-first": (("instagram", "tiktok"), "instagram_reels"),
        }

        for case_name, (platforms, expected_platform) in cases.items():
            with self.subTest(case=case_name):
                task_id = f"caption-{case_name}"
                state.update_task(
                    task_id,
                    state=tm.const.TASK_STATE_COMPLETE,
                    progress=100,
                    videos=["final.mp4"],
                    cross_post_state=tm.const.CROSS_POST_STATE_PENDING,
                )
                with (
                    patch.object(tm.sm, "state", state),
                    patch.object(
                        tm.llm,
                        "generate_social_metadata",
                        return_value=metadata,
                    ) as generate_metadata,
                    patch.object(
                        tm.upload_post,
                        "cross_post_video",
                        return_value={"success": True},
                    ) as cross_post,
                ):
                    tm._run_cross_post(
                        task_id,
                        ("final.mp4",),
                        "Coffee",
                        "A short coffee story.",
                        "en",
                        platforms,
                        "private",
                    )

                generate_metadata.assert_called_once_with(
                    video_subject="Coffee",
                    video_script="A short coffee story.",
                    language="en",
                    platform=expected_platform,
                )
                cross_post.assert_called_once()
                call = cross_post.call_args
                self.assertEqual(call.kwargs["title"], "Watch this coffee ritual.")
                self.assertEqual(call.kwargs["platforms"], list(platforms))
                self.assertIsNone(call.kwargs["youtube_extra"])
                task = state.get_task(task_id)
                self.assertEqual(
                    task["cross_post_state"], tm.const.CROSS_POST_STATE_COMPLETE
                )

    def test_cross_post_shares_metadata_between_youtube_fields_and_title(self):
        """YouTube-specific fields and the shared publish title must come from the same metadata call."""
        metadata = {
            "title": "Morning Coffee",
            "caption": "A better morning.",
            "hashtags": ["#coffee", "#shorts"],
        }
        state = MemoryState()
        state.update_task(
            "shared-youtube-metadata",
            state=tm.const.TASK_STATE_COMPLETE,
            progress=100,
            videos=["final-1.mp4", "final-2.mp4"],
            cross_post_state=tm.const.CROSS_POST_STATE_PENDING,
        )

        with (
            patch.object(tm.sm, "state", state),
            patch.object(
                tm.llm,
                "generate_social_metadata",
                return_value=metadata,
            ) as generate_metadata,
            patch.object(
                tm.upload_post,
                "cross_post_video",
                return_value={"success": True},
            ) as cross_post,
        ):
            tm._run_cross_post(
                "shared-youtube-metadata",
                ("final-1.mp4", "final-2.mp4"),
                "Coffee",
                "A short coffee story.",
                "en",
                ("youtube",),
                "unlisted",
            )

        generate_metadata.assert_called_once_with(
            video_subject="Coffee",
            video_script="A short coffee story.",
            language="en",
            platform="youtube_shorts",
        )
        expected_extra = {
            "youtube_title": "Morning Coffee",
            "youtube_description": "A better morning.",
            "tags": ["#coffee", "#shorts"],
            "privacyStatus": "unlisted",
            "containsSyntheticMedia": True,
        }
        self.assertEqual(cross_post.call_count, 2)
        for call in cross_post.call_args_list:
            self.assertEqual(call.kwargs["title"], "A better morning.")
            self.assertEqual(call.kwargs["youtube_extra"], expected_extra)

    def test_cross_post_empty_metadata_degrades_to_fallback_title(self):
        """Missing or empty metadata falls back step by step, ending with the old generic title."""
        state = MemoryState()
        cases = {
            "legacy-string": ({}, "", "Check out this video! #shorts #viral"),
            "title-over-subject": (
                {"title": "Fallback Title", "caption": ""},
                "Coffee",
                "Fallback Title",
            ),
        }

        for case_name, (metadata, subject, expected_title) in cases.items():
            with self.subTest(case=case_name):
                task_id = f"fallback-title-{case_name}"
                state.update_task(
                    task_id,
                    state=tm.const.TASK_STATE_COMPLETE,
                    progress=100,
                    videos=["final.mp4"],
                    cross_post_state=tm.const.CROSS_POST_STATE_PENDING,
                )
                with (
                    patch.object(tm.sm, "state", state),
                    patch.object(
                        tm.llm,
                        "generate_social_metadata",
                        return_value=metadata,
                    ) as generate_metadata,
                    patch.object(
                        tm.upload_post,
                        "cross_post_video",
                        return_value={"success": True},
                    ) as cross_post,
                ):
                    tm._run_cross_post(
                        task_id,
                        ("final.mp4",),
                        subject,
                        "",
                        "",
                        ("tiktok",),
                        "private",
                    )

                generate_metadata.assert_called_once()
                cross_post.assert_called_once()
                self.assertEqual(cross_post.call_args.kwargs["title"], expected_title)

    def test_recover_interrupted_cross_posts_preserves_active_future(self):
        """Startup recovery only handles leftover state; publish jobs still owned by this process must not be touched."""
        state = MemoryState()
        for task_id in (
            "stale-pending",
            "active-processing",
            "inactive-current-owner",
            "remote-processing",
            "already-complete",
        ):
            cross_post_state = {
                "stale-pending": tm.const.CROSS_POST_STATE_PENDING,
                "active-processing": tm.const.CROSS_POST_STATE_PROCESSING,
                "inactive-current-owner": tm.const.CROSS_POST_STATE_PROCESSING,
                "remote-processing": tm.const.CROSS_POST_STATE_PROCESSING,
                "already-complete": tm.const.CROSS_POST_STATE_COMPLETE,
            }[task_id]
            state.update_task(
                task_id,
                state=tm.const.TASK_STATE_COMPLETE,
                progress=100,
                videos=["final.mp4"],
                cross_post_state=cross_post_state,
                cross_post_owner=(
                    "another-host:123:remote"
                    if task_id == "remote-processing"
                    else (
                        tm._cross_post_process_owner
                        if task_id == "inactive-current-owner"
                        else None
                    )
                ),
            )

        active_future = Future()
        tm._register_cross_post_future("active-processing", active_future)
        with patch.object(tm.sm, "state", state):
            recovered = tm.recover_interrupted_cross_posts(page_size=1)

        self.assertEqual(recovered, 2)
        stale_task = state.get_task("stale-pending")
        self.assertEqual(
            stale_task["cross_post_state"], tm.const.CROSS_POST_STATE_FAILED
        )
        self.assertEqual(
            stale_task["cross_post_error"], tm._INTERRUPTED_CROSS_POST_ERROR
        )
        self.assertEqual(
            state.get_task("active-processing")["cross_post_state"],
            tm.const.CROSS_POST_STATE_PROCESSING,
        )
        self.assertEqual(
            state.get_task("inactive-current-owner")["cross_post_state"],
            tm.const.CROSS_POST_STATE_FAILED,
        )
        self.assertEqual(
            state.get_task("remote-processing")["cross_post_state"],
            tm.const.CROSS_POST_STATE_PROCESSING,
        )
        self.assertEqual(
            state.get_task("already-complete")["cross_post_state"],
            tm.const.CROSS_POST_STATE_COMPLETE,
        )
        active_future.set_result(None)

    def test_cross_post_owner_uses_future_registry_for_current_process(self):
        """With no active Future in this process, both old and new owners with the same PID count as interrupted."""
        stale_owner = f"{tm.socket.gethostname()}:{tm.os.getpid()}:old-instance"

        self.assertFalse(tm._is_cross_post_owner_alive(stale_owner))
        self.assertFalse(tm._is_cross_post_owner_alive(tm._cross_post_process_owner))

    def test_cross_post_owner_detection_handles_process_boundaries(self):
        """Owner detection must cover legacy records, other hosts and local process-probe error boundaries."""
        hostname = tm.socket.gethostname()

        self.assertFalse(tm._is_cross_post_owner_alive(None))
        self.assertFalse(tm._is_cross_post_owner_alive("invalid-owner"))
        self.assertTrue(tm._is_cross_post_owner_alive("another-host:123:instance"))

        with (
            patch.object(tm.os, "name", "posix"),
            patch.object(tm.os, "kill", side_effect=ProcessLookupError),
        ):
            self.assertFalse(
                tm._is_cross_post_owner_alive(f"{hostname}:987654:dead-instance")
            )
        with (
            patch.object(tm.os, "name", "posix"),
            patch.object(tm.os, "kill", side_effect=PermissionError),
        ):
            self.assertTrue(
                tm._is_cross_post_owner_alive(f"{hostname}:987654:restricted")
            )
        with (
            patch.object(tm.os, "name", "posix"),
            patch.object(tm.os, "kill", side_effect=OSError("inspection failed")),
            patch.object(tm.logger, "warning") as log_warning,
        ):
            self.assertTrue(tm._is_cross_post_owner_alive(f"{hostname}:987654:unknown"))
        self.assertIn("inspection failed", log_warning.call_args.args[0])

        with (
            patch.object(tm.os, "name", "nt"),
            patch.object(tm, "_is_windows_process_alive", return_value=True) as probe,
        ):
            self.assertTrue(tm._is_cross_post_owner_alive(f"{hostname}:987654:windows"))
        probe.assert_called_once_with(987654)

    @unittest.skipUnless(os.name == "nt", "Windows process API test")
    def test_windows_process_probe_is_read_only_and_detects_liveness(self):
        """Windows CI must really exercise the read-only process probe with no fallback to os.kill."""
        self.assertTrue(tm._is_windows_process_alive(os.getpid()))
        self.assertFalse(tm._is_windows_process_alive(2_147_483_647))

    def test_cross_post_terminal_check_converts_active_state_to_failure(self):
        """When the worker has ended but the state is still active, the final callback must write the failed terminal state."""
        state = MemoryState()
        state.update_task(
            "unfinished-cross-post",
            state=tm.const.TASK_STATE_COMPLETE,
            progress=100,
            videos=["final.mp4"],
            cross_post_state=tm.const.CROSS_POST_STATE_PROCESSING,
        )

        with patch.object(tm.sm, "state", state):
            tm._ensure_cross_post_terminal_state("unfinished-cross-post")

        task = state.get_task("unfinished-cross-post")
        self.assertEqual(task["videos"], ["final.mp4"])
        self.assertEqual(task["cross_post_state"], tm.const.CROSS_POST_STATE_FAILED)
        self.assertIn("without persisting", task["cross_post_error"])

    def test_cross_post_recovery_reports_state_backend_failure(self):
        """Startup recovery must return None when reading state fails so a later WebUI rerun can retry."""
        state = MagicMock()
        state.get_all_tasks.side_effect = RuntimeError("redis unavailable")

        with (
            patch.object(tm.sm, "state", state),
            patch.object(tm.logger, "exception") as log_exception,
        ):
            recovered = tm.recover_interrupted_cross_posts()

        self.assertIsNone(recovered)
        self.assertIn("redis unavailable", log_exception.call_args.args[0])

    def test_cancelled_cross_post_future_releases_slot_and_records_failure(self):
        """A cancelled queued Future must also release capacity and write the failed terminal state."""
        state = MemoryState()
        state.update_task(
            "cancelled-cross-post",
            state=tm.const.TASK_STATE_COMPLETE,
            progress=100,
            cross_post_state=tm.const.CROSS_POST_STATE_PENDING,
        )
        slots = MagicMock()
        future = Future()
        tm._register_cross_post_future("cancelled-cross-post", future)
        self.assertTrue(future.cancel())

        with (
            patch.object(tm.sm, "state", state),
            patch.object(tm, "_cross_post_slots", slots),
        ):
            tm._finalize_cross_post_future("cancelled-cross-post", future)

        slots.release.assert_called_once_with()
        self.assertFalse(tm._is_cross_post_active_in_process("cancelled-cross-post"))
        task = state.get_task("cancelled-cross-post")
        self.assertEqual(task["cross_post_state"], tm.const.CROSS_POST_STATE_FAILED)
        self.assertIn("cancelled", task["cross_post_error"])

    @unittest.skipUnless(
        os.getenv("MPT_TEST_REDIS_HOST"),
        "MPT_TEST_REDIS_HOST not set",
    )
    def test_real_redis_recovers_interrupted_cross_post_state(self):
        """Leftover publish state in a real Redis must keep the videos and enter the failed terminal state after recovery."""
        state = RedisState(
            host=os.environ["MPT_TEST_REDIS_HOST"],
            port=int(os.getenv("MPT_TEST_REDIS_PORT", "6379")),
            db=int(os.getenv("MPT_TEST_REDIS_DB", "15")),
        )
        task_id = f"ci-cross-post-recovery-{uuid4()}"
        state.update_task(
            task_id,
            state=tm.const.TASK_STATE_COMPLETE,
            progress=100,
            videos=["final.mp4"],
            cross_post_state=tm.const.CROSS_POST_STATE_PROCESSING,
            cross_post_owner="",
        )

        try:
            with patch.object(tm.sm, "state", state):
                recovered = tm.recover_interrupted_cross_posts(page_size=10)

            self.assertGreaterEqual(recovered, 1)
            task = state.get_task(task_id)
            self.assertEqual(task["videos"], ["final.mp4"])
            self.assertEqual(task["cross_post_state"], tm.const.CROSS_POST_STATE_FAILED)
            self.assertEqual(task["cross_post_error"], tm._INTERRUPTED_CROSS_POST_ERROR)
        finally:
            state.delete_task(task_id)

    def test_cross_post_future_exception_is_observed(self):
        """Exceptions raised by the pool itself must be logged, not left in a Future nobody reads."""
        future = Future()
        future.set_exception(RuntimeError("executor worker failed"))

        with patch.object(tm.logger, "error") as log_error:
            tm._finalize_cross_post_future("future-failure", future)

        log_error.assert_called_once()
        self.assertIn("executor worker failed", log_error.call_args.args[0])

    def test_cross_post_queue_full_rejects_only_publishing(self):
        """When the publish queue is full the output must be kept and nothing more submitted to the pool."""
        state = MemoryState()
        state.update_task(
            "cross-post-queue-full",
            state=tm.const.TASK_STATE_COMPLETE,
            progress=100,
            videos=["final.mp4"],
            cross_post_state=tm.const.CROSS_POST_STATE_PENDING,
        )

        with (
            patch.object(tm.sm, "state", state),
            patch.object(
                tm._cross_post_slots,
                "acquire",
                return_value=False,
            ),
            patch.object(tm._cross_post_executor, "submit") as submit,
        ):
            scheduling_error = tm._schedule_cross_post(
                task_id="cross-post-queue-full",
                video_paths=["final.mp4"],
                params=VideoParams(video_subject="Coffee"),
                video_script="A short coffee story.",
                platforms=["tiktok"],
                youtube_privacy_status="private",
            )

        submit.assert_not_called()
        self.assertIn("queue is full", scheduling_error)
        task = state.get_task("cross-post-queue-full")
        self.assertEqual(task["state"], tm.const.TASK_STATE_COMPLETE)
        self.assertEqual(task["videos"], ["final.mp4"])
        self.assertEqual(task["cross_post_state"], tm.const.CROSS_POST_STATE_FAILED)
        self.assertIn("queue is full", task["cross_post_error"])

    @unittest.skipUnless(
        RUN_INTEGRATION_TESTS,
        "MPT_RUN_INTEGRATION_TESTS not set",
    )
    def test_task_local_materials(self):
        task_id = "00000000-0000-0000-0000-000000000000"
        video_materials = []
        for i in range(1, 4):
            video_materials.append(
                MaterialInfo(
                    provider="local",
                    url=os.path.join(resources_dir, f"{i}.png"),
                    duration=0,
                )
            )

        params = VideoParams(
            video_subject="金钱的作用",
            video_script="金钱不仅是交换媒介，更是社会资源的分配工具。它能满足基本生存需求，如食物和住房，也能提供教育、医疗等提升生活品质的机会。拥有足够的金钱意味着更多选择权，比如职业自由或创业可能。但金钱的作用也有边界，它无法直接购买幸福、健康或真诚的人际关系。过度追逐财富可能导致价值观扭曲，忽视精神层面的需求。理想的状态是理性看待金钱，将其作为实现目标的工具而非终极目的。",
            video_terms="money importance, wealth and society, financial freedom, money and happiness, role of money",
            video_aspect="9:16",
            video_concat_mode="random",
            video_transition_mode="None",
            video_clip_duration=3,
            video_count=1,
            video_source="local",
            video_materials=video_materials,
            video_language="",
            voice_name="zh-CN-XiaoxiaoNeural-Female",
            voice_volume=1.0,
            voice_rate=1.0,
            bgm_type="random",
            bgm_file="",
            bgm_volume=0.2,
            subtitle_enabled=True,
            subtitle_position="bottom",
            custom_position=70.0,
            font_name="MicrosoftYaHeiBold.ttc",
            text_fore_color="#FFFFFF",
            text_background_color=True,
            font_size=60,
            stroke_color="#000000",
            stroke_width=1.5,
            n_threads=2,
            paragraph_number=1,
        )
        result = tm.start(task_id=task_id, params=params)
        print(result)


class TestScheduleManualCrossPost(unittest.TestCase):
    def _seed_complete_task(self, state, task_id, **overrides):
        defaults = dict(
            state=tm.const.TASK_STATE_COMPLETE,
            progress=100,
            videos=["final.mp4"],
            script="A coffee story.",
        )
        defaults.update(overrides)
        state.update_task(task_id, **defaults)

    def test_returns_404_when_task_unknown(self):
        state = MemoryState()
        with (
            patch.object(tm.upload_post.upload_post_service, "is_configured", return_value=True),
            patch.object(tm.sm, "state", state),
        ):
            scheduled, error, code = tm.schedule_manual_cross_post("missing")

        self.assertFalse(scheduled)
        self.assertEqual(code, 404)
        self.assertIn("not found", error or "")

    def test_returns_400_when_video_incomplete(self):
        state = MemoryState()
        state.update_task(
            "incomplete",
            state=tm.const.TASK_STATE_PROCESSING,
            progress=50,
            videos=[],
        )
        with (
            patch.object(tm.upload_post.upload_post_service, "is_configured", return_value=True),
            patch.object(tm.sm, "state", state),
        ):
            scheduled, error, code = tm.schedule_manual_cross_post("incomplete")

        self.assertFalse(scheduled)
        self.assertEqual(code, 400)

    def test_returns_409_when_cross_post_active(self):
        state = MemoryState()
        self._seed_complete_task(
            state, "busy", cross_post_state=tm.const.CROSS_POST_STATE_PROCESSING,
        )
        with (
            patch.object(tm.upload_post.upload_post_service, "is_configured", return_value=True),
            patch.object(tm.sm, "state", state),
        ):
            scheduled, error, code = tm.schedule_manual_cross_post("busy")

        self.assertFalse(scheduled)
        self.assertEqual(code, 409)

    def test_returns_400_when_not_configured(self):
        state = MemoryState()
        self._seed_complete_task(state, "unconfigured")
        with (
            patch.object(tm.upload_post.upload_post_service, "is_configured", return_value=False),
            patch.object(tm.sm, "state", state),
        ):
            scheduled, error, code = tm.schedule_manual_cross_post("unconfigured")

        self.assertFalse(scheduled)
        self.assertEqual(code, 400)

    def test_succeeds_and_reuses_schedule_helper(self):
        state = MemoryState()
        self._seed_complete_task(state, "ready")
        with (
            patch.object(tm.upload_post.upload_post_service, "is_configured", return_value=True),
            patch.object(
                type(tm.upload_post.upload_post_service),
                "platforms",
                new_callable=PropertyMock,
                return_value=["tiktok", "instagram"],
            ),
            patch.object(tm.sm, "state", state),
            patch.object(tm, "_schedule_cross_post", return_value=None) as schedule,
        ):
            scheduled, error, code = tm.schedule_manual_cross_post("ready")

        self.assertTrue(scheduled)
        self.assertEqual(code, 202)
        schedule.assert_called_once()
        self.assertEqual(schedule.call_args.kwargs["task_id"], "ready")
        self.assertEqual(
            schedule.call_args.kwargs["platforms"], ["tiktok", "instagram"],
        )
        task = state.get_task("ready")
        self.assertEqual(task["cross_post_state"], tm.const.CROSS_POST_STATE_PENDING)
        self.assertTrue(task["cross_post_owner"])


class TestCrossPostPolling(unittest.TestCase):
    def _seed(self, state, task_id):
        state.update_task(
            task_id,
            state=tm.const.TASK_STATE_COMPLETE,
            progress=100,
            videos=["final.mp4"],
            cross_post_state=tm.const.CROSS_POST_STATE_PENDING,
            script="A coffee story.",
        )

    def test_poll_status_is_invoked_after_successful_upload(self):
        """Upload-Post's 'accepted' response is not terminal; polling folds the
        real per-platform outcome into the result."""
        state = MemoryState()
        self._seed(state, "poll-after-upload")

        with (
            patch.object(tm.sm, "state", state),
            patch.object(tm.llm, "generate_social_metadata", return_value={"title": "T", "caption": "C", "hashtags": []}),
            patch.object(
                tm.upload_post,
                "cross_post_video",
                return_value={"success": True, "request_id": "req-1"},
            ),
            patch.object(
                tm.upload_post.upload_post_service,
                "poll_status",
                return_value={"success": True, "platform_statuses": {"tiktok": "ok"}},
            ) as poll,
        ):
            tm._run_cross_post(
                "poll-after-upload", ("final.mp4",), "Coffee", "script", "en",
                ("tiktok",), "public",
            )

        poll.assert_called_once_with("req-1")
        task = state.get_task("poll-after-upload")
        self.assertEqual(
            task["cross_post_results"][0]["platform_statuses"], {"tiktok": "ok"},
        )
        self.assertEqual(task["cross_post_state"], tm.const.CROSS_POST_STATE_COMPLETE)

    def test_poll_timeout_does_not_leave_task_in_processing_state(self):
        """A polled timeout must still produce a terminal state, not orphan PROCESSING."""
        state = MemoryState()
        self._seed(state, "poll-timeout")

        with (
            patch.object(tm.sm, "state", state),
            patch.object(tm.llm, "generate_social_metadata", return_value={"title": "T", "caption": "C", "hashtags": []}),
            patch.object(
                tm.upload_post,
                "cross_post_video",
                return_value={"success": True, "request_id": "req-2"},
            ),
            patch.object(
                tm.upload_post.upload_post_service,
                "poll_status",
                return_value={"success": False, "error": "Upload-Post did not finish within 1800s"},
            ),
        ):
            tm._run_cross_post(
                "poll-timeout", ("final.mp4",), "Coffee", "script", "en",
                ("tiktok",), "public",
            )

        task = state.get_task("poll-timeout")
        self.assertEqual(task["cross_post_state"], tm.const.CROSS_POST_STATE_FAILED)
        self.assertIn("did not finish", task["cross_post_error"])


if __name__ == "__main__":
    unittest.main()
