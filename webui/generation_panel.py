"""The generation task lifecycle, separate from the large form page."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import streamlit as st
from loguru import logger

from app.config import config
from app.models import const
from app.services import state as sm
from app.services import webui_task


def render_logs(task_id: str) -> None:
    """Render a worker-safe snapshot of one task's logs."""
    if config.ui.get("hide_log", False):
        return
    records = webui_task.get_task_logs(task_id)
    if records:
        st.code("\n".join(records), height=320)


def render_running_task(
    task_id: str,
    *,
    tr: Callable[[str], str],
    normalize_state: Callable[[Any], Any],
    remove_active_task: Callable[[str], None],
    render_snapshot: Callable[[str, dict | None], None],
) -> None:
    """Refresh an in-progress task and switch to static output when it ends."""
    try:
        task = sm.state.get_task(task_id)
    except Exception as exc:
        logger.exception(
            f"failed to query WebUI generation task: task_id={task_id}, error={exc}"
        )
        st.error(tr("Video Generation Failed"))
        return

    if normalize_state((task or {}).get("state")) in {
        const.TASK_STATE_COMPLETE,
        const.TASK_STATE_FAILED,
    }:
        remove_active_task(task_id)
        st.rerun(scope="app")
    render_snapshot(task_id, task)


def recover_task_id(*, normalize_state: Callable[[Any], Any]) -> str:
    """Recover the latest task after a Streamlit reconnect."""
    task_id = webui_task.get_last_submitted_task_id()
    if not task_id:
        return ""
    try:
        task = sm.state.get_task(task_id)
    except Exception as exc:
        logger.exception(
            f"failed to recover WebUI generation task: task_id={task_id}, error={exc}"
        )
        return ""
    if not task:
        return ""
    st.session_state["current_generation_task_id"] = task_id
    if normalize_state(task.get("state")) != const.TASK_STATE_PROCESSING:
        st.session_state["handled_generation_task_id"] = task_id
    return task_id


def render_current_task(
    *,
    tr: Callable[[str], str],
    normalize_state: Callable[[Any], Any],
    remove_active_task: Callable[[str], None],
    render_snapshot: Callable[[str, dict | None], None],
    render_running: Callable[[str], None],
    recover_task: Callable[[], str],
) -> None:
    """Render the current session task or recover it after reconnect."""
    task_id = st.session_state.get("current_generation_task_id", "") or recover_task()
    if not task_id:
        return
    try:
        task = sm.state.get_task(task_id)
    except Exception as exc:
        logger.exception(
            f"failed to query current WebUI task: task_id={task_id}, error={exc}"
        )
        st.error(tr("Video Generation Failed"))
        return

    if normalize_state((task or {}).get("state")) in {
        const.TASK_STATE_COMPLETE,
        const.TASK_STATE_FAILED,
    }:
        remove_active_task(task_id)
        render_snapshot(task_id, task)
        return
    render_running(task_id)
