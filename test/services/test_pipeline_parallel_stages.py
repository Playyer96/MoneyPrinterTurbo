import threading
from unittest.mock import patch

from app.models.schema import VideoParams
from app.services import task as tm


def test_full_pipeline_overlaps_subtitles_and_materials():
    rendezvous = threading.Barrier(2, timeout=1)

    def generate_subtitle(*_args, **_kwargs):
        rendezvous.wait()
        return "subtitle.srt"

    def get_materials(*_args, **_kwargs):
        rendezvous.wait()
        return ["clip.mp4"]

    params = VideoParams(video_subject="benchmark")
    with (
        patch.object(tm.utils, "check_ffmpeg_ready", return_value=True),
        patch.object(tm, "generate_script", return_value="script"),
        patch.object(tm, "generate_terms", return_value=["term"]),
        patch.object(tm, "save_script_data"),
        patch.object(
            tm,
            "generate_audio",
            return_value=("audio.mp3", 5, object()),
        ),
        patch.object(tm, "generate_subtitle", side_effect=generate_subtitle),
        patch.object(tm, "get_video_materials", side_effect=get_materials),
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
        patch.object(tm.sm.state, "update_task"),
    ):
        result = tm.start("parallel-stages", params)

    assert result["videos"] == ["final.mp4"]


def test_terms_and_audio_run_in_parallel():
    """terms and audio both depend only on the script and must overlap.

    Both stages block on a 2-party barrier. If they ran sequentially the first
    one would block forever (or until the 1s timeout fires), so passing
    proves the executor submits them concurrently.
    """
    rendezvous = threading.Barrier(2, timeout=1)

    def generate_terms(*_args, **_kwargs):
        rendezvous.wait()
        return ["term"]

    def generate_audio(*_args, **_kwargs):
        rendezvous.wait()
        return ("audio.mp3", 5, object())

    params = VideoParams(video_subject="benchmark", video_source="pixabay")
    with (
        patch.object(tm.utils, "check_ffmpeg_ready", return_value=True),
        patch.object(tm, "generate_script", return_value="script"),
        patch.object(tm, "generate_terms", side_effect=generate_terms),
        patch.object(tm, "save_script_data"),
        patch.object(tm, "generate_audio", side_effect=generate_audio),
        patch.object(tm, "generate_subtitle", return_value="subtitle.srt"),
        patch.object(tm, "get_video_materials", return_value=["clip.mp4"]),
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
        patch.object(tm.sm.state, "update_task"),
    ):
        result = tm.start("terms-audio-parallel", params)

    assert result["videos"] == ["final.mp4"]


def test_local_source_skips_terms_stage():
    """Local material sources have nothing to download and should skip terms."""
    generate_terms_calls = []

    def fake_generate_terms(*_args, **_kwargs):
        generate_terms_calls.append(1)
        return ["term"]

    params = VideoParams(video_subject="benchmark", video_source="local")
    with (
        patch.object(tm.utils, "check_ffmpeg_ready", return_value=True),
        patch.object(tm, "generate_script", return_value="script"),
        patch.object(tm, "generate_terms", side_effect=fake_generate_terms),
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
        patch.object(
            tm.upload_post.upload_post_service,
            "is_configured",
            return_value=False,
        ),
        patch.object(tm.sm.state, "update_task"),
    ):
        result = tm.start("local-source", params)

    assert result["videos"] == ["final.mp4"]
    assert generate_terms_calls == []
