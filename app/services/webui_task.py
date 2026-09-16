import threading
from collections import deque

from loguru import logger

from app.config import config
from app.controllers.manager.memory_manager import InMemoryTaskManager
from app.models import const
from app.models.schema import VideoParams
from app.services import state as sm
from app.services import task as tm
from app.services.loomloom import LoomLoomConfirmedVideoRequest
from app.utils.logging_utils import format_log_record


_task_manager = InMemoryTaskManager(
    # ponytail: two workers overlap network/CPU stages. Increase only after
    # measuring GPU memory; OmniVoice itself keeps one model request at a time.
    max_concurrent_tasks=max(1, int(config.app.get("webui_max_concurrent_tasks", 2))),
    max_queued_tasks=max(1, int(config.app.get("max_queued_tasks", 100))),
)
_task_logs: dict[str, deque[str]] = {}
# ponytail: this supports reconnect recovery for only the latest submitted
# task. Add a browser-owned session token when cross-session recovery matters.
_last_submitted_task_id = ""
_task_logs_lock = threading.RLock()
_MAX_LOG_TASKS = 20
_MAX_LOG_RECORDS_PER_TASK = 1000
# Streamlit cannot update components from a worker thread. Fragment polling
# every 0.5 seconds keeps logs near real-time without expensive browser refreshes.
TASK_LOG_REFRESH_INTERVAL_SECONDS = 0.5


def _append_task_log(task_id: str, message: str) -> None:
    """Keep bounded per-task logs for safe Streamlit Fragment polling."""
    with _task_logs_lock:
        records = _task_logs.get(task_id)
        if records is None:
            # Keep only recent task logs so a long-running WebUI does not grow
            # without bound. Dict insertion order makes discarding the oldest safe.
            if len(_task_logs) >= _MAX_LOG_TASKS:
                oldest_task_id = next(iter(_task_logs))
                _task_logs.pop(oldest_task_id, None)
            records = deque(maxlen=_MAX_LOG_RECORDS_PER_TASK)
            _task_logs[task_id] = records
        records.append(message.rstrip())


def get_last_submitted_task_id() -> str:
    """Last task submitted from the WebUI, used to restore the task panel."""
    return _last_submitted_task_id


def get_task_logs(task_id: str) -> list[str]:
    """Return a log snapshot without holding the worker-thread lock while rendering."""
    with _task_logs_lock:
        return list(_task_logs.get(task_id, ()))


def _run_generation(
    task_id: str,
    params: VideoParams,
    capture_logs: bool,
    voice_preview: dict | None = None,
    loomloom_video_request: LoomLoomConfirmedVideoRequest | None = None,
    runtime_config_snapshot: dict[int, dict] | None = None,
) -> dict:
    """
    Run the existing video pipeline in a worker thread.

    Loguru sinks are process-wide, so filter to the current worker thread. That
    keeps concurrent task logs separate and never accesses Streamlit session
    state from a background thread.
    """
    log_handler_id = None
    worker_thread_id = threading.get_ident()
    try:
        if capture_logs:
            log_handler_id = logger.add(
                lambda message: _append_task_log(task_id, str(message)),
                level="DEBUG",
                format=format_log_record,
                colorize=False,
                filter=lambda record: record["thread"].id == worker_thread_id,
            )

        # Each task reads its submitted settings while new WebUI changes remain
        # available for later tasks. This removes global pipeline serialization.
        with config.use_runtime_config_snapshot(
            runtime_config_snapshot or config.capture_runtime_config()
        ):
            return tm.start(
                task_id=task_id,
                params=params,
                voice_preview=voice_preview,
                loomloom_video_request=loomloom_video_request,
            )
    except Exception as exc:
        # tm.start records pipeline failures. Protect the WebUI wrapper too so a
        # worker exception always leaves a terminal state rather than "processing".
        error = f"{type(exc).__name__}: {exc}"
        failure = {
            "task_id": task_id,
            "state": const.TASK_STATE_FAILED,
            "progress": 0,
            "failed_stage": "webui_worker",
            "error": error,
        }
        sm.state.update_task(
            task_id,
            state=failure["state"],
            progress=failure["progress"],
            failed_stage=failure["failed_stage"],
            error=failure["error"],
        )
        logger.exception(
            f"unexpected WebUI generation worker failure, "
            f"task_id={task_id}, error={exc}"
        )
        return failure
    finally:
        if log_handler_id is not None:
            try:
                logger.remove(log_handler_id)
            except ValueError:
                logger.debug(
                    f"WebUI task log handler already removed: task_id={task_id}"
                )


def submit_generation(
    task_id: str,
    params: VideoParams,
    capture_logs: bool = True,
    voice_preview: dict | None = None,
    loomloom_video_request: LoomLoomConfirmedVideoRequest | None = None,
) -> None:
    """
    Register and submit a WebUI generation task, then return immediately.

    Write state before starting the worker so a rerun or reconnect can query it
    without relying on placeholders retained by the previous page render.
    """
    global _last_submitted_task_id

    task_params = params.model_copy(deep=True)
    # Copy the preview envelope so later page reruns cannot replace cached fields
    # for a task that is already in the background queue.
    voice_preview_snapshot = dict(voice_preview) if voice_preview else None
    # The confirmed request is immutable process-local data. Its API key never
    # enters VideoParams, task state, logs, or persisted history.
    loomloom_request_snapshot = loomloom_video_request
    runtime_config_snapshot = config.capture_runtime_config()
    sm.state.update_task(
        task_id,
        state=const.TASK_STATE_PROCESSING,
        progress=0,
        video_subject=task_params.video_subject or task_params.video_script or task_id,
    )
    try:
        _task_manager.add_task(
            _run_generation,
            task_id=task_id,
            params=task_params,
            capture_logs=capture_logs,
            voice_preview=voice_preview_snapshot,
            loomloom_video_request=loomloom_request_snapshot,
            runtime_config_snapshot=runtime_config_snapshot,
        )
        _last_submitted_task_id = task_id
    except Exception as exc:
        # Scheduling failure must be queryable like a pipeline failure; otherwise
        # the task manager can show "processing" forever. Keep the exception type
        # for quick diagnosis from Docker or local logs.
        error = f"{type(exc).__name__}: {exc}"
        sm.state.update_task(
            task_id,
            state=const.TASK_STATE_FAILED,
            progress=0,
            failed_stage="scheduling",
            error=error,
        )
        logger.exception(
            f"failed to submit WebUI generation task, task_id={task_id}, error={exc}"
        )
        raise
