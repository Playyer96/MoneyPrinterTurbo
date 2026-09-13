"""Task entry point and cross-platform publishing helpers.

The single-video and series pipelines live under ``app.services.pipeline``
(``pipeline/stages.py`` for shared per-stage logic, ``pipeline/single.py``
for a single video, ``pipeline/series.py`` for a multi-chapter series,
``pipeline/__init__.py`` for the ``start`` dispatcher). This module keeps
two responsibilities that do not belong inside either pipeline:

- Re-exports so external callers (``app.controllers.v1.video``,
  ``app.services.webui_task``, ``cli.py``, ``webui/Main.py``) keep
  importing the task module as a single entry point for ``start``,
  ``is_task_busy``, ``const``, and the individual pipeline stages.
- Cross-platform publishing: thread pool, Future registry, startup
  recovery, and per-platform state writes. This runs after video
  generation completes, so it is orthogonal to the single/series split.

Pipeline internals call back into this module's re-exported names (instead
of calling ``pipeline.stages`` directly) so that patching ``tm.generate_script``
et al. in tests keeps intercepting pipeline calls, the way it did before the
single/series split.
"""

from __future__ import annotations

import os
import socket
import threading
import time
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from functools import partial
from uuid import uuid4

from loguru import logger

from app.config import config
from app.models import const
from app.models.schema import VideoConcatMode, VideoParams  # noqa: F401
from app.services import (  # noqa: F401
    bgm as bgm_service,
    elevenlabs_music,
    llm,
    loomloom,
    material,
    metaso_minimax,
    ofox,
    sonilo,
    subtitle,
    task_artifacts,
    twelvelabs,
    video,
    volcengine_seedance,
    voice,
)
from app.services import upload_post
from app.services import state as sm
from app.services.pipeline import stages
from app.services.pipeline import series as _series_pipeline
from app.services.pipeline import single as _single_pipeline
from app.services.pipeline import start as start
from app.utils import file_security, utils  # noqa: F401

# ---------------------------------------------------------------------------
# Re-exports for backward compatibility.
#
# Earlier code (and existing tests) imported these as ``tm.<name>``. Routing
# everything through ``app.services.pipeline`` keeps one canonical home for
# the stage logic while this module stays the public entry point that
# callers already import. The ``from app.services import ...`` block above
# also keeps every service module reachable as ``tm.<module>`` for tests
# that patch attributes on the module itself (``patch.object(tm.voice, ...)``),
# which works regardless of aliasing since modules are shared singletons.
# ---------------------------------------------------------------------------
generate_script = stages.generate_script
generate_terms = stages.generate_terms
generate_audio = stages.generate_audio
generate_subtitle = stages.generate_subtitle
generate_final_videos = stages.generate_final_videos
get_video_materials = stages.get_video_materials
save_script_data = stages.save_script_data
resolve_custom_audio_file = stages.resolve_custom_audio_file
get_video_music_prompt = stages.get_video_music_prompt
mark_task_failed = stages.mark_task_failed
_mark_task_failed = stages.mark_task_failed
_run_pipeline = _single_pipeline.run_single_video
_run_series = _series_pipeline.run_series_video
_resolve_series_outline = _series_pipeline.resolve_series_outline
_build_series_part_prompt = _series_pipeline.build_series_part_prompt
_build_series_part_params = _series_pipeline.build_series_part_params
_VIDEO_MUSIC_PROVIDERS = stages._VIDEO_MUSIC_PROVIDERS
_LOOMLOOM_STATE_WRITE_ATTEMPTS = stages._LOOMLOOM_STATE_WRITE_ATTEMPTS
_LOOMLOOM_STATE_RETRY_DELAY_SECONDS = stages._LOOMLOOM_STATE_RETRY_DELAY_SECONDS


def is_task_busy(task: dict | None) -> bool:
    """Return True while the task is still generating or publishing; shared by every delete entry point."""
    if not task:
        return False

    state = task.get("state")
    try:
        state = int(state)
    except (TypeError, ValueError):
        pass

    # Both video generation and cross-platform publishing can keep reading
    # the task directory. Treating both as busy avoids API/WebUI disagreeing
    # on whether a delete is allowed.
    return (
        state == const.TASK_STATE_PROCESSING
        or task.get("cross_post_state") in _ACTIVE_CROSS_POST_STATES
    )


# ---------------------------------------------------------------------------
# Cross-platform publishing helpers.
#
# These do not belong to either the single-video or the series pipeline, so
# they stay here. Anything that wants to schedule a cross-post must call
# ``_schedule_cross_post``; the Future registry below is the source of truth
# for "is there a still-running cross-post job for this task in this process".
# ---------------------------------------------------------------------------

# Cross-post requests can take several minutes to complete, so they must not
# occupy video-generation concurrency slots. A fixed-size thread pool keeps
# publishing throughput manageable while letting video products transition
# to completed immediately after generation.
_cross_post_executor = ThreadPoolExecutor(
    max_workers=2,
    thread_name_prefix="mpt-cross-post",
)
_cross_post_max_pending_tasks = max(
    1,
    int(config.app.get("upload_post_max_pending_tasks", 10)),
)
_cross_post_slots = threading.BoundedSemaphore(_cross_post_max_pending_tasks)
_cross_post_registry_lock = threading.RLock()
_cross_post_futures: dict[str, Future] = {}
_cross_post_process_owner = f"{socket.gethostname()}:{os.getpid()}:{uuid4().hex}"
_ACTIVE_CROSS_POST_STATES = {
    const.CROSS_POST_STATE_PENDING,
    const.CROSS_POST_STATE_PROCESSING,
}
_CROSS_POST_STATE_WRITE_ATTEMPTS = 3
_CROSS_POST_STATE_RETRY_DELAY_SECONDS = 0.1
_INTERRUPTED_CROSS_POST_ERROR = (
    "cross-posting was interrupted before the process completed"
)
# Map upload-post platform ids to the social platform names llm.py accepts.
_CROSS_POST_SOCIAL_PLATFORMS = {
    "tiktok": "tiktok",
    "instagram": "instagram_reels",
    "facebook": "facebook_reels",
}


def _register_cross_post_future(task_id: str, future: Future) -> None:
    """Register the cross-post Future held by the current process, for startup recovery and tests to check actual running state."""
    with _cross_post_registry_lock:
        _cross_post_futures[task_id] = future


def _unregister_cross_post_future(task_id: str, future: Future | None = None) -> None:
    """Remove only the matching Future, so old callbacks cannot delete newer work registered for the same task."""
    with _cross_post_registry_lock:
        current = _cross_post_futures.get(task_id)
        if current is None or (future is not None and current is not future):
            return
        _cross_post_futures.pop(task_id, None)


def _is_cross_post_active_in_process(task_id: str) -> bool:
    """Return True if the current process still holds an unfinished cross-post task."""
    with _cross_post_registry_lock:
        future = _cross_post_futures.get(task_id)
        return future is not None and not future.done()


def _is_windows_process_alive(process_id: int) -> bool:
    """Check process state via read-only Win32 API, avoiding os.kill which could incorrectly terminate the process."""
    import ctypes

    process_query_limited_information = 0x1000
    still_active = 259
    error_access_denied = 5
    error_invalid_parameter = 87
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    # ctypes treats undeclared return values as 32-bit int by default. Windows
    # 64-bit process handles can be truncated, so Win32 function signatures
    # must be declared explicitly before calling.
    kernel32.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
    kernel32.OpenProcess.restype = ctypes.c_void_p
    kernel32.GetExitCodeProcess.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_ulong),
    ]
    kernel32.GetExitCodeProcess.restype = ctypes.c_int
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    kernel32.CloseHandle.restype = ctypes.c_int
    handle = kernel32.OpenProcess(
        process_query_limited_information,
        False,
        process_id,
    )
    if not handle:
        error_code = ctypes.get_last_error()
        if error_code == error_invalid_parameter:
            return False
        if error_code == error_access_denied:
            # When the process exists but the current user lacks query permission,
            # conservatively treat it as alive to avoid incorrectly reclaiming
            # publishing tasks running under other accounts.
            return True
        logger.warning(
            "failed to open cross-post owner process on Windows, "
            f"process_id: {process_id}, error_code: {error_code}"
        )
        return True

    try:
        exit_code = ctypes.c_ulong()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            error_code = ctypes.get_last_error()
            logger.warning(
                "failed to read cross-post owner process state on Windows, "
                f"process_id: {process_id}, error_code: {error_code}"
            )
            return True
        return exit_code.value == still_active
    finally:
        kernel32.CloseHandle(handle)


def _is_cross_post_owner_alive(owner: str | None) -> bool:
    """Return True if the local process for a persisted cross-post task still exists."""
    if not owner:
        return False

    try:
        hostname, process_id_text, _ = owner.split(":", 2)
        process_id = int(process_id_text)
    except (TypeError, ValueError):
        logger.warning(f"invalid cross-post owner metadata: {owner}")
        return False

    # Cannot reliably detect processes on other hosts. In multi-host deployments
    # with shared Redis, conservatively treat them as still running to avoid
    # the current node deleting video files that another node is reading.
    if hostname != socket.gethostname():
        return True

    # Whether real publishing work remains in the current process is accurately
    # tracked by the Future registry. Reaching here means the registry has no
    # corresponding Future; treat it as interrupted even if the owner matches
    # the current process exactly. This covers cases where terminal state writes
    # keep failing and the Future has already ended.
    if process_id == os.getpid():
        return False

    # Windows os.kill(pid, 0) has different semantics than POSIX and can
    # directly terminate the target process. Use a Win32 API that requests
    # only query permission, without sending any signal to the target process.
    if os.name == "nt":
        return _is_windows_process_alive(process_id)

    try:
        os.kill(process_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError as exc:
        logger.warning(
            f"failed to inspect cross-post owner process, owner: {owner}, error: {exc}"
        )
        return True
    return True


def _patch_cross_post_state(task_id: str, **kwargs) -> bool | None:
    """Safely update publishing fields; retry briefly on transient state-backend failures."""
    for attempt in range(1, _CROSS_POST_STATE_WRITE_ATTEMPTS + 1):
        try:
            return sm.state.patch_task(task_id, **kwargs)
        except Exception as exc:
            # Redis brief disconnections should not leave tasks permanently stuck in
            # pending/processing. Publishing state write frequency is very low; fixed
            # retry count and short wait here cover transient failures while avoiding
            # infinite blocking of background threads. Last failure retains full stack
            # trace for diagnosis.
            if attempt >= _CROSS_POST_STATE_WRITE_ATTEMPTS:
                logger.exception(
                    f"failed to update cross-post state after retries, "
                    f"task_id: {task_id}, fields: {', '.join(kwargs)}, "
                    f"attempts: {attempt}, error: {exc}"
                )
                return None

            logger.warning(
                f"retry cross-post state update, task_id: {task_id}, "
                f"fields: {', '.join(kwargs)}, attempt: {attempt}, error: {exc}"
            )
            time.sleep(_CROSS_POST_STATE_RETRY_DELAY_SECONDS)

    return None


def _record_cross_post_failure(
    task_id: str,
    error: Exception,
    results: list[dict] | None = None,
) -> None:
    """Best-effort persistence of publishing failure; logs retain diagnostic info when state backend is unavailable."""
    updated = _patch_cross_post_state(
        task_id,
        cross_post_state=const.CROSS_POST_STATE_FAILED,
        cross_post_results=results or None,
        cross_post_error=str(error),
        cross_post_owner=None,
    )
    if updated is False:
        logger.warning(f"discard cross-post failure for missing task: {task_id}")


def _ensure_cross_post_terminal_state(task_id: str) -> None:
    """Converge tasks still in an active state to failed after the Future ends."""
    try:
        task = sm.state.get_task(task_id)
    except Exception as exc:
        # This is already the final callback of the Future; there is no
        # subsequent synchronous caller that can handle exceptions. After the
        # state backend recovers, the next process startup will still handle
        # lingering states through the recovery logic.
        logger.exception(
            f"failed to verify final cross-post state, task_id: {task_id}, error: {exc}"
        )
        return

    if not task or task.get("cross_post_state") not in _ACTIVE_CROSS_POST_STATES:
        return

    logger.warning(
        f"cross-post worker ended without terminal state, task_id: {task_id}, "
        f"state: {task.get('cross_post_state')}"
    )
    _record_cross_post_failure(
        task_id,
        RuntimeError("cross-post worker ended without persisting a terminal state"),
        task.get("cross_post_results"),
    )


def recover_interrupted_cross_posts(page_size: int = 100) -> int | None:
    """
    Mark publishing tasks that cannot be recovered after a process restart as failed.

    Cross-platform publishing uses a thread pool within the current process, not a
    persistent task queue. When the process starts, any pending/processing entries
    remaining in Redis will not automatically resume; if left as running, users
    will be permanently unable to delete tasks. Here we paginate through state,
    processing only active records that have no corresponding Future in the current
    process, and preserving already-generated video results.
    """
    recovered = 0
    page = 1

    while True:
        try:
            tasks, total = sm.state.get_all_tasks(page, page_size)
        except Exception as exc:
            logger.exception(f"failed to recover interrupted cross-post tasks: {exc}")
            return None

        for task in tasks:
            task_id = str(task.get("task_id") or "")
            if (
                not task_id
                or task.get("cross_post_state") not in _ACTIVE_CROSS_POST_STATES
                or _is_cross_post_active_in_process(task_id)
                or _is_cross_post_owner_alive(task.get("cross_post_owner"))
            ):
                continue

            updated = _patch_cross_post_state(
                task_id,
                cross_post_state=const.CROSS_POST_STATE_FAILED,
                cross_post_error=_INTERRUPTED_CROSS_POST_ERROR,
                cross_post_owner=None,
            )
            if updated is True:
                recovered += 1

        if page * page_size >= total or not tasks:
            break
        page += 1

    if recovered:
        logger.warning(f"recovered interrupted cross-post tasks: {recovered}")
    return recovered


def _run_cross_post(
    task_id: str,
    video_paths: tuple[str, ...],
    video_subject: str,
    video_script: str,
    video_language: str,
    platforms: tuple[str, ...],
    youtube_privacy_status: str,
) -> None:
    """Execute cross-platform publishing in the background, only supplementing task fields related to publishing."""
    results = []
    try:
        state_updated = _patch_cross_post_state(
            task_id,
            cross_post_state=const.CROSS_POST_STATE_PROCESSING,
            cross_post_error=None,
            cross_post_owner=_cross_post_process_owner,
        )
        if state_updated is not True:
            # False means the task was deleted; None means the state backend is
            # temporarily unavailable. Neither case should continue calling
            # third-party APIs, otherwise users cannot query or control the publish.
            if state_updated is False:
                logger.warning(f"skip cross-post for missing task: {task_id}")
            else:
                _record_cross_post_failure(
                    task_id,
                    RuntimeError("failed to persist cross-post processing state"),
                )
            return

        logger.info(
            f"cross-post started, task_id: {task_id}, platforms: {', '.join(platforms)}"
        )
        youtube_extra = None
        post_title = video_subject or "Check out this video! #shorts #viral"
        if platforms:
            has_youtube = any(platform.startswith("youtube") for platform in platforms)
            social_platform = "youtube_shorts"
            if not has_youtube:
                first = (platforms[0] or "").strip().lower()
                # llm.py resolves unknown ids to its default platform.
                social_platform = _CROSS_POST_SOCIAL_PLATFORMS.get(first, first)
            metadata = llm.generate_social_metadata(
                video_subject=video_subject,
                video_script=video_script,
                language=video_language or "",
                platform=social_platform,
            )
            if has_youtube:
                youtube_extra = {
                    "youtube_title": metadata.get("title", video_subject),
                    "youtube_description": metadata.get("caption", ""),
                    "tags": metadata.get("hashtags", []),
                    "privacyStatus": youtube_privacy_status,
                    "containsSyntheticMedia": True,
                }
            post_title = (
                metadata.get("caption")
                or metadata.get("title")
                or video_subject
                or "Check out this video! #shorts #viral"
            )

        for video_path in video_paths:
            result = upload_post.cross_post_video(
                video_path=video_path,
                title=post_title,
                platforms=list(platforms),
                youtube_extra=youtube_extra,
            )
            if not isinstance(result, dict):
                result = {
                    "success": False,
                    "error": "Upload-Post returned an invalid response",
                }
            elif result.get("success") and result.get("request_id"):
                # The initial response only means "Upload-Post accepted the
                # file"; actual TikTok/Instagram/YouTube publishing happens
                # async. Poll for the real terminal outcome so cross_post_results
                # reflects what the platforms actually reported, not just intake.
                poll_result = upload_post.upload_post_service.poll_status(
                    result["request_id"]
                )
                if isinstance(poll_result, dict):
                    result = {**result, **poll_result}
            results.append(result)

        failures = [result for result in results if not result.get("success")]
        if failures:
            error_messages = [
                str(
                    result.get("error")
                    or result.get("message")
                    or "unknown upload error"
                )
                for result in failures
            ]
            cross_post_state = const.CROSS_POST_STATE_FAILED
            cross_post_error = "; ".join(error_messages)
            logger.warning(
                f"cross-post completed with failures, task_id: {task_id}, "
                f"failed: {len(failures)}, total: {len(results)}"
            )
        else:
            cross_post_state = const.CROSS_POST_STATE_COMPLETE
            cross_post_error = None
            logger.success(
                f"cross-post completed, task_id: {task_id}, videos: {len(results)}"
            )

        state_updated = _patch_cross_post_state(
            task_id,
            cross_post_state=cross_post_state,
            cross_post_results=results,
            cross_post_error=cross_post_error,
            cross_post_owner=None,
        )
        if state_updated is False:
            logger.warning(f"discard cross-post result for missing task: {task_id}")
        elif state_updated is None:
            # Upload has finished but results were not persisted; cannot leave
            # processing state. Failure state write goes through finite retries
            # again, at minimum giving the caller a clear terminal state.
            _record_cross_post_failure(
                task_id,
                RuntimeError("failed to persist final cross-post result"),
                results,
            )
    except Exception as exc:
        # Publishing failure only affects publishing state; it must not
        # retroactively overwrite a completed video task. The original
        # exception text is written to task state so API callers can
        # locate the issue without accessing server-side logs.
        logger.exception(f"cross-post failed, task_id: {task_id}, error: {exc}")
        _record_cross_post_failure(task_id, exc, results)


def _run_cross_post_with_slot(*args) -> None:
    """Execute publishing task, ensuring slot is returned on success, failure, or exception."""
    try:
        _run_cross_post(*args)
    except Exception as exc:
        # _run_cross_post handles expected exceptions; this is the last line of
        # defense, preventing exceptions from new future logic from being saved
        # only in an unread Future.
        task_id = str(args[0]) if args else "unknown"
        logger.exception(f"cross-post worker crashed, task_id: {task_id}, error: {exc}")
        if args:
            _record_cross_post_failure(task_id, exc)
    finally:
        _cross_post_slots.release()


def _finalize_cross_post_future(task_id: str, future: Future) -> None:
    """Clean up Future registry and ensure cancellation, exception, and state-write failures all converge."""
    _unregister_cross_post_future(task_id, future)

    try:
        error = future.exception()
    except CancelledError:
        logger.warning(f"cross-post future was cancelled, task_id: {task_id}")
        # When a Future is cancelled before execution starts, the worker's
        # finally does not run, so slot return and persistent state update
        # to failed must be done in the callback.
        _cross_post_slots.release()
        _record_cross_post_failure(
            task_id,
            RuntimeError("cross-post job was cancelled before execution"),
        )
        return
    except Exception as exc:
        logger.exception(
            f"failed to inspect cross-post future, task_id: {task_id}, error: {exc}"
        )
        _ensure_cross_post_terminal_state(task_id)
        return

    if error is not None:
        logger.error(
            f"cross-post future failed, task_id: {task_id}, "
            f"error: {type(error).__name__}: {error}"
        )

    _ensure_cross_post_terminal_state(task_id)


def _schedule_cross_post(
    task_id: str,
    video_paths: list[str],
    params: VideoParams,
    video_script: str,
    platforms: list[str],
    youtube_privacy_status: str,
) -> str | None:
    """Submit a background publishing task; returns None on success, or a queryable error reason on scheduling failure."""
    if not _cross_post_slots.acquire(blocking=False):
        error = "cross-post queue is full; publishing was skipped"
        logger.warning(
            f"skip cross-post because queue is full, task_id: {task_id}, "
            f"capacity: {_cross_post_max_pending_tasks}"
        )
        _patch_cross_post_state(
            task_id,
            cross_post_state=const.CROSS_POST_STATE_FAILED,
            cross_post_error=error,
            cross_post_owner=None,
        )
        return error

    try:
        future = _cross_post_executor.submit(
            _run_cross_post_with_slot,
            task_id,
            tuple(video_paths),
            params.video_subject or "",
            video_script,
            params.video_language or "",
            tuple(platforms),
            youtube_privacy_status,
        )
        _register_cross_post_future(task_id, future)
        future.add_done_callback(partial(_finalize_cross_post_future, task_id))
    except RuntimeError as exc:
        _unregister_cross_post_future(task_id)
        _cross_post_slots.release()
        logger.exception(
            f"failed to schedule cross-post, task_id: {task_id}, error: {exc}"
        )
        _patch_cross_post_state(
            task_id,
            cross_post_state=const.CROSS_POST_STATE_FAILED,
            cross_post_error=f"failed to schedule cross-post: {exc}",
            cross_post_owner=None,
        )
        return f"failed to schedule cross-post: {exc}"

    return None


def schedule_manual_cross_post(
    task_id: str,
    platforms: list[str] | None = None,
    force: bool = False,
) -> tuple[bool, str | None, int]:
    """
    Schedule a cross-post for an already-completed task outside the normal
    auto-upload flow (the WebUI's manual "Publish" button, or the
    POST .../publish HTTP endpoint).

    Returns ``(scheduled, error, status_code)`` so both callers can map the
    outcome to a response without re-implementing these checks. ``force``
    only overrides an active state that recovery would also treat as
    orphaned (no Future in this process and no live owner process); it never
    lets two real cross-post jobs run for the same task at once.
    """
    if not upload_post.upload_post_service.is_configured():
        return False, "Upload-Post is not configured", 400

    task = sm.state.get_task(task_id)
    if not task:
        return False, "task not found", 404

    if task.get("state") != const.TASK_STATE_COMPLETE or not task.get("videos"):
        return False, "video generation is not complete", 400

    if task.get("cross_post_state") in _ACTIVE_CROSS_POST_STATES:
        orphaned = not _is_cross_post_active_in_process(
            task_id
        ) and not _is_cross_post_owner_alive(task.get("cross_post_owner"))
        if not (force and orphaned):
            return False, "cross-post is already active for this task", 409

    resolved_platforms = (
        list(platforms) if platforms else list(upload_post.upload_post_service.platforms)
    )
    if not resolved_platforms:
        return False, "no platforms selected", 400

    updated = _patch_cross_post_state(
        task_id,
        cross_post_state=const.CROSS_POST_STATE_PENDING,
        cross_post_results=None,
        cross_post_error=None,
        cross_post_owner=_cross_post_process_owner,
    )
    if updated is not True:
        if updated is False:
            return False, "task not found", 404
        return False, "failed to update task state", 500

    video_script = str(task.get("script") or "")
    # ponytail: video_subject is never persisted to task state (it only lives
    # transiently during generation, or inside script.json's params blob,
    # which has no reader today); truncate the script as a stand-in title
    # hint. Add a task_artifacts reader for the original subject if this
    # produces noticeably worse captions in practice.
    video_subject = video_script.strip()[:80] or task_id

    scheduling_error = _schedule_cross_post(
        task_id=task_id,
        video_paths=list(task.get("videos") or []),
        params=VideoParams(video_subject=video_subject),
        video_script=video_script,
        platforms=resolved_platforms,
        youtube_privacy_status=upload_post.upload_post_service.youtube_privacy_status,
    )
    if scheduling_error:
        return False, scheduling_error, 429

    return True, None, 202
