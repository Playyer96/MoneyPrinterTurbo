"""Video generation pipelines.

Exposes ``start`` as the public entry point that dispatches to the right
pipeline:

- ``app.services.pipeline.series`` for multi-chapter series tasks.
- ``app.services.pipeline.single`` for one-off single-video tasks.

Both pipelines share ``app.services.pipeline.stages`` so the per-stage logic
(LLM, TTS, materials, BGM, subtitles) lives in one place and each pipeline
only owns its own orchestration, progress curve, and persistence shape.
"""

from __future__ import annotations

from loguru import logger

from app.models import const
from app.models.schema import VideoParams
from app.services import loomloom
from app.services.pipeline import stages


def start(
    task_id,
    params: VideoParams,
    stop_at: str = "video",
    voice_preview: dict | None = None,
    loomloom_video_request: loomloom.LoomLoomConfirmedVideoRequest | None = None,
    allow_server_file_input: bool = False,
):
    """
    Execute the task pipeline and ensure unexpected exceptions also convert to queryable failure states.

    ``allow_server_file_input`` is for local CLI use only. HTTP API and WebUI
    must keep the default value so custom audio is always constrained to the
    current task directory.
    """
    # Route through ``app.services.task`` so tests that patch ``tm._run_series``
    # or ``tm._run_pipeline`` keep intercepting these calls. The task module
    # re-exports both names from this package; the indirect lookup preserves
    # monkey-patching without forcing callers to import a private attribute.
    from app.services import task as _task

    try:
        if params.series_enabled:
            if loomloom_video_request is not None:
                return stages.mark_task_failed(
                    task_id,
                    "preflight",
                    "series mode cannot reuse a confirmed LoomLoom video quote; "
                    "each part needs its own quote",
                )
            if voice_preview:
                # The preview was rendered for a single script; every part
                # narrates its own.
                logger.warning("series mode ignores the reusable voice preview")
            return _task._run_series(
                task_id,
                params,
                stop_at=stop_at,
                allow_server_file_input=allow_server_file_input,
            )

        return _task._run_pipeline(
            task_id,
            params,
            stop_at=stop_at,
            voice_preview=voice_preview,
            loomloom_video_request=loomloom_video_request,
            allow_server_file_input=allow_server_file_input,
        )
    except Exception as exc:
        logger.exception(
            f"unexpected task pipeline failure, task_id: {task_id}, error: {exc}"
        )
        return stages.mark_task_failed(
            task_id,
            "pipeline",
            f"{type(exc).__name__}: {exc}",
        )


__all__ = [
    "start",
    "stages",
    "const",
]
