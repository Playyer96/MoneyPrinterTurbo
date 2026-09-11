"""Single-video pipeline.

Drives one ``VideoParams`` request through preflight, script, terms, audio,
subtitle, materials, and final-video stages, then schedules optional
cross-platform publishing. A series of chapters runs this driver once per
chapter inside ``app.services.pipeline.series``.
"""

from __future__ import annotations

from loguru import logger

from app.config import config
from app.models import const
from app.models.schema import VideoConcatMode, VideoParams
from app.services import (
    bgm as bgm_service,
    loomloom,
    metaso_minimax,
    ofox,
    volcengine_seedance,
)
from app.services import upload_post
from app.services import state as sm
from app.services.pipeline import stages
from app.utils import utils

# Stage helpers are imported for their module-level constants and helper
# utilities (music-prompt lookup, provider metadata). The actual call sites
# go through ``app.services.task.<name>`` so tests that patch ``tm.<name>``
# keep intercepting stage calls. Doing the lookup at call time (not at
# import time) also avoids a partial-module dance during circular imports
# between this package and ``app.services.task``.
from app.services.pipeline.stages import (  # noqa: F401  - re-exported for callers that import these names from here
    _VIDEO_MUSIC_PROVIDERS,
    get_video_music_prompt,
)


# Stages get a slice of the 0-100% progress range; final video composition
# owns the rest. These boundaries match the existing WebUI poll cadence so a
# single task and a series chapter both report progress with the same shape.
_PROGRESS_AFTER_PREFLIGHT = 5
_PROGRESS_AFTER_SCRIPT = 10
_PROGRESS_AFTER_TERMS = 20
_PROGRESS_AFTER_AUDIO = 30
_PROGRESS_AFTER_SUBTITLE = 40
_PROGRESS_AFTER_MATERIALS = 50


def run_single_video(
    task_id,
    params: VideoParams,
    stop_at: str = "video",
    voice_preview: dict | None = None,
    loomloom_video_request: loomloom.LoomLoomConfirmedVideoRequest | None = None,
    allow_server_file_input: bool = False,
):
    """Run the pipeline for one video end-to-end.

    ``allow_server_file_input`` is for local CLI use only. HTTP API and WebUI
    must keep the default value so custom audio is always constrained to the
    current task directory.
    """
    # Look up stage helpers through ``app.services.task`` so tests that patch
    # ``tm.generate_script`` (and the other stage names) keep intercepting
    # calls. The task module re-exports these names from ``pipeline.stages``;
    # calling them via the task module lets monkey-patched mocks win without
    # duplicating the patch list here.
    from app.services import task as _task

    logger.info(f"start single-video task: {task_id}, stop_at: {stop_at}")
    sm.state.update_task(task_id, state=const.TASK_STATE_PROCESSING, progress=_PROGRESS_AFTER_PREFLIGHT)

    preflight_failure = _check_preflight(task_id, params, stop_at)
    if preflight_failure is not None:
        return preflight_failure

    # 1. Generate script
    video_script = _task.generate_script(task_id, params)
    if not video_script or "Error: " in video_script:
        error = (
            video_script.removeprefix("Error: ").strip()
            if isinstance(video_script, str) and "Error: " in video_script
            else "failed to generate video script"
        )
        return stages.mark_task_failed(task_id, "script", error)

    sm.state.update_task(task_id, state=const.TASK_STATE_PROCESSING, progress=_PROGRESS_AFTER_SCRIPT)

    if stop_at == "script":
        sm.state.update_task(
            task_id, state=const.TASK_STATE_COMPLETE, progress=100, script=video_script
        )
        return {"script": video_script}

    # 2. Generate terms
    video_terms = ""
    if params.video_source != "local":
        video_terms = _task.generate_terms(task_id, params, video_script)
        if not video_terms:
            return stages.mark_task_failed(
                task_id,
                "terms",
                "failed to generate video search terms",
            )

    _task.save_script_data(task_id, video_script, video_terms, params)

    if stop_at == "terms":
        sm.state.update_task(
            task_id, state=const.TASK_STATE_COMPLETE, progress=100, terms=video_terms
        )
        return {"script": video_script, "terms": video_terms}

    sm.state.update_task(task_id, state=const.TASK_STATE_PROCESSING, progress=_PROGRESS_AFTER_TERMS)

    # 3. Generate audio
    audio_file, audio_duration, sub_maker = _task.generate_audio(
        task_id,
        params,
        video_script,
        voice_preview=voice_preview,
        allow_server_file_input=allow_server_file_input,
    )
    if not audio_file:
        return stages.mark_task_failed(
            task_id,
            "audio",
            "failed to prepare narration audio",
        )

    sm.state.update_task(task_id, state=const.TASK_STATE_PROCESSING, progress=_PROGRESS_AFTER_AUDIO)

    if stop_at == "audio":
        sm.state.update_task(
            task_id,
            state=const.TASK_STATE_COMPLETE,
            progress=100,
            audio_file=audio_file,
        )
        return {"audio_file": audio_file, "audio_duration": audio_duration}

    # 4. Generate subtitle
    subtitle_path = _task.generate_subtitle(
        task_id, params, video_script, sub_maker, audio_file
    )

    if stop_at == "subtitle":
        sm.state.update_task(
            task_id,
            state=const.TASK_STATE_COMPLETE,
            progress=100,
            subtitle_path=subtitle_path,
        )
        return {"subtitle_path": subtitle_path}

    sm.state.update_task(task_id, state=const.TASK_STATE_PROCESSING, progress=_PROGRESS_AFTER_SUBTITLE)

    # 5. Get video materials
    downloaded_videos = _task.get_video_materials(
        task_id,
        params,
        video_terms,
        audio_duration,
        loomloom_video_request=loomloom_video_request,
    )
    if not downloaded_videos:
        return stages.mark_task_failed(
            task_id,
            "materials",
            "failed to prepare video materials",
        )

    if stop_at == "materials":
        sm.state.update_task(
            task_id,
            state=const.TASK_STATE_COMPLETE,
            progress=100,
            materials=downloaded_videos,
        )
        return {"materials": downloaded_videos}

    sm.state.update_task(task_id, state=const.TASK_STATE_PROCESSING, progress=_PROGRESS_AFTER_MATERIALS)

    # Only the full video generation pipeline needs to process video concat mode;
    # this prevents /subtitle and /audio requests from accessing non-existent fields.
    if type(params.video_concat_mode) is str:
        params.video_concat_mode = VideoConcatMode(params.video_concat_mode)

    # 6. Generate final videos
    final_video_paths, combined_video_paths, generation_warnings = (
        _task.generate_final_videos(
            task_id,
            params,
            downloaded_videos,
            audio_file,
            subtitle_path,
            audio_duration,
        )
    )

    if not final_video_paths:
        return stages.mark_task_failed(
            task_id,
            "video",
            "failed to generate final video",
        )

    logger.success(
        f"task {task_id} finished, generated {len(final_video_paths)} videos."
    )

    # 7. Complete video generation first, then submit cross-platform publishing
    # on demand. Third-party uploads can take several minutes and should not
    # block video result return or retroactively affect already-generated output.
    cross_post_enabled = (
        upload_post.upload_post_service.is_configured()
        and upload_post.upload_post_service.auto_upload
    )
    platforms = (
        list(upload_post.upload_post_service.platforms) if cross_post_enabled else []
    )
    should_cross_post = cross_post_enabled and bool(platforms)
    if cross_post_enabled and not platforms:
        logger.warning(
            f"skip cross-post because no platforms are configured, task_id: {task_id}"
        )
    cross_post_state = const.CROSS_POST_STATE_PENDING if should_cross_post else None

    kwargs = {
        "videos": final_video_paths,
        "combined_videos": combined_video_paths,
        "script": video_script,
        "terms": video_terms,
        "audio_file": audio_file,
        "audio_duration": audio_duration,
        "subtitle_path": subtitle_path,
        "materials": downloaded_videos,
        "cross_post_state": cross_post_state,
        "cross_post_results": None,
        "cross_post_error": None,
        "cross_post_owner": _cross_post_owner_field(should_cross_post),
        "warnings": generation_warnings or None,
    }
    sm.state.update_task(
        task_id, state=const.TASK_STATE_COMPLETE, progress=100, **kwargs
    )

    if should_cross_post:
        scheduling_error = _schedule_cross_post(
            task_id=task_id,
            video_paths=final_video_paths,
            params=params,
            video_script=video_script,
            platforms=platforms,
            youtube_privacy_status=(
                upload_post.upload_post_service.youtube_privacy_status
            ),
        )
        # Queue full or thread pool shutdown are synchronously known scheduling
        # failures. Task state has already been updated by the scheduling
        # function; correct the return snapshot synchronously to avoid the
        # caller receiving a pending state inconsistent with subsequent queries.
        if scheduling_error:
            kwargs["cross_post_state"] = const.CROSS_POST_STATE_FAILED
            kwargs["cross_post_error"] = scheduling_error
            kwargs["cross_post_owner"] = None

    return kwargs


def _cross_post_owner_field(should_cross_post: bool) -> str | None:
    """Tag the current process as the owner of any pending cross-post job.

    Imported lazily from ``app.services.task`` to avoid a circular import
    between this module and the cross-post helpers that the task module
    owns today. Returning ``None`` when cross-posting is disabled keeps the
    field out of persisted state.
    """
    if not should_cross_post:
        return None
    from app.services.task import _cross_post_process_owner
    return _cross_post_process_owner


def _schedule_cross_post(
    task_id: str,
    video_paths: list[str],
    params: VideoParams,
    video_script: str,
    platforms: list[str],
    youtube_privacy_status: str,
) -> str | None:
    """Submit a background publishing task; returns None on success, or a queryable error reason on scheduling failure."""
    # Lazy import: cross-post lives in ``app.services.task`` to keep all
    # recovery / Future-registry code in one place. Importing it here keeps
    # this module focused on the single-video pipeline without a circular
    # top-level dependency.
    from app.services.task import _schedule_cross_post as _impl
    return _impl(
        task_id=task_id,
        video_paths=video_paths,
        params=params,
        video_script=video_script,
        platforms=platforms,
        youtube_privacy_status=youtube_privacy_status,
    )


def _check_preflight(task_id: str, params: VideoParams, stop_at: str):
    """Validate provider keys, FFmpeg readiness, and music prompt length before consuming quotas.

    Returns ``None`` when preflight passes, or a failure dict that should be
    returned to the caller without further work.
    """
    if (
        stop_at in {"materials", "video"}
        and params.video_source == "volcengine_seedance"
        and not volcengine_seedance.is_enabled()
    ):
        return stages.mark_task_failed(
            task_id,
            "preflight",
            "Volcano Engine Seedance requires an Ark API key",
        )

    if (
        stop_at in {"materials", "video"}
        and params.video_source == "ofox"
        and not ofox.is_enabled()
    ):
        return stages.mark_task_failed(
            task_id,
            "preflight",
            "OFox video generation requires an OFox API key",
        )

    if (
        stop_at in {"materials", "video"}
        and params.video_source == "metaso_minimax"
        and not metaso_minimax.is_enabled()
    ):
        return stages.mark_task_failed(
            task_id,
            "preflight",
            "Metaso MiniMax requires an API key",
        )

    if (
        stop_at in {"materials", "video"}
        and params.video_source == "openai_image"
        and not _is_openai_image_enabled()
    ):
        return stages.mark_task_failed(
            task_id,
            "preflight",
            "OpenAI image source requires openai_image_base_url and "
            "openai_image_model in config.toml (openai_image_api_keys is "
            "optional for local gateways that need no auth)",
        )

    # Only the full video generation pipeline needs a video music provider.
    # Block incomplete tasks missing a key early, before consuming LLM, TTS,
    # and material service quotas; intermediate product endpoints can still
    # be used independently.
    video_music_provider = _VIDEO_MUSIC_PROVIDERS.get(params.bgm_type)
    video_music_enabled = (
        stop_at == "video"
        and video_music_provider is not None
        and bgm_service.should_use_bgm(params.bgm_type, params.bgm_volume)
    )
    if video_music_enabled:
        service = video_music_provider["service"]
        display_name = video_music_provider["display_name"]
        if not service.is_enabled():
            return stages.mark_task_failed(
                task_id,
                "preflight",
                f"{display_name} background music requires an API key",
            )

        # WebUI limits input length, but API, CLI, and historical tasks can
        # bypass front-end controls. Re-validate against provider limits before
        # generating script, voice, and materials, so rejection by a third party
        # only happens after full video synthesis would have been attempted.
        # Service layer keeps the same validation as a last line of defense
        # for direct calls.
        music_prompt = get_video_music_prompt(params)
        max_prompt_length = int(getattr(service, "MAX_PROMPT_LENGTH", 0) or 0)
        if max_prompt_length and len(music_prompt) > max_prompt_length:
            return stages.mark_task_failed(
                task_id,
                "preflight",
                (f"{display_name} music prompt exceeds {max_prompt_length} characters"),
            )

        # Providers may optionally offer non-billed account pre-checks. The
        # check function should only throw deterministic errors; when network
        # fluctuations or permission scope cannot be confirmed, the service
        # layer logs a warning and proceeds with actual generation.
        validate_access = getattr(service, "validate_generation_access", None)
        if callable(validate_access):
            try:
                validate_access()
            except video_music_provider["error_type"] as exc:
                return stages.mark_task_failed(task_id, "preflight", str(exc))

    # Only script/terms intermediate products do not need FFmpeg (they generate
    # no audio or video). API, CLI, and WebUI all execute tasks through this
    # shared entry point, so probe here once rather than duplicating checks at
    # each entry, ensuring consistent behavior across all three paths. Placed
    # after the music-key check to preserve that check's original "fail first"
    # order and error messages.
    if stop_at not in ("script", "terms") and not utils.check_ffmpeg_ready():
        return stages.mark_task_failed(
            task_id,
            "preflight",
            "ffmpeg is not available; install ffmpeg or set app.ffmpeg_path "
            "in config.toml to a working ffmpeg executable",
        )

    return None


def _is_openai_image_enabled() -> bool:
    """Wrap material.is_openai_image_enabled to keep the preflight signature flat."""
    from app.services import material
    return material.is_openai_image_enabled(
        config.snapshot_config_with_pending(config.app)
    )
