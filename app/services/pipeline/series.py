"""Series (multi-chapter) pipeline.

A series turns one ``VideoParams`` subject into an ordered set of chapters.
This module plans the chapter outline, then runs the single-video pipeline
once per chapter. The parent task's progress covers the whole series, so a
parent stuck at one step while a chapter runs means the WebUI sees a flat
bar; intra-chapter updates feed the parent to keep the bar moving.
"""

from __future__ import annotations

from loguru import logger

from app.models import const
from app.models.schema import VideoParams
from app.services import llm
from app.services import state as sm
from app.services.pipeline import stages

# Series owns 5% on plan, then 90% split evenly across the chapters, with a
# 5% tail for assembling the parent task record. The 5% head and tail match
# the single-video pipeline's preflight / finalize band so the two pipelines
# share the same end-of-task progress shape.
_SERIES_HEAD_PROGRESS = 5
_SERIES_BODY_PROGRESS = 90
_SERIES_TAIL_PROGRESS = 5


def resolve_series_outline(params: VideoParams) -> list[str]:
    """Return the chapter subjects for a series task, planning them if needed."""
    outline = [
        chapter.strip() for chapter in (params.series_outline or []) if chapter.strip()
    ]
    if outline:
        # An outline confirmed in the WebUI wins over the count input: the user
        # already saw and edited the chapters they want.
        logger.info(f"using the confirmed series outline: {len(outline)} chapters")
        return outline[: const.MAX_SERIES_PARTS]

    return llm.generate_series_outline(
        video_subject=params.video_subject or params.video_script,
        parts=params.series_parts,
        language=params.video_language,
        video_script_prompt=params.video_script_prompt,
    )


def build_series_part_prompt(params: VideoParams, outline: list[str], index: int) -> str:
    """Tell one part where it sits in the series, so the chapters do not overlap."""
    total = len(outline)
    lines = [
        f'This is part {index} of {total} in a video series about '
        f'"{params.video_subject}".',
        f"This part covers only: {outline[index - 1]}",
    ]
    if index > 1:
        lines.append(f"The previous part covered: {outline[index - 2]}")
    if index < total:
        lines.append(f"The next part will cover: {outline[index]}")
    lines.append(
        "Do not repeat what the other parts cover, and do not summarize the "
        "whole series."
    )
    if index < total:
        lines.append(
            "Resolve this part's immediate problem, then make the final sentence "
            "a specific bridge or cliffhanger that clearly says the story continues "
            "in the next part. Do not use a generic 'stay tuned' sign-off."
        )
    else:
        lines.append(
            "Resolve the series' central problems and end with a definitive final "
            "sentence. Do not promise another part."
        )

    context = "\n".join(lines)
    user_prompt = (params.video_script_prompt or "").strip()
    if not user_prompt:
        return context

    # Series context goes first: the script service truncates the tail at
    # MAX_SCRIPT_PROMPT_LENGTH, and losing the context would break the arc.
    return f"{context}\n\n{user_prompt}"


def build_series_part_params(
    params: VideoParams, outline: list[str], index: int
) -> VideoParams:
    part_params = params.model_copy(deep=True)
    part_params.series_enabled = False
    part_params.video_subject = outline[index - 1]
    # Each part writes its own script and keywords. Reusing the series-level
    # ones would give every chapter the same narration and the same footage.
    part_params.video_script = ""
    part_params.video_terms = None
    if params.series_continuity:
        part_params.video_script_prompt = build_series_part_prompt(
            params, outline, index
        )
    if params.custom_audio_file:
        # One narration file cannot serve several chapters.
        logger.warning(
            "series mode ignores the custom audio file; each part narrates its "
            "own script"
        )
        part_params.custom_audio_file = None
    return part_params


def _series_body_progress(completed: int, total: int) -> int:
    """How far through the series the parent should look after ``completed`` chapters.

    Uses integer arithmetic on the 0-90% body band; the tail (90-100%) only
    lights up after all parts have finished, so the parent bar always moves
    in 5%-head, 90%-body, 5%-tail steps regardless of chapter count.
    """
    if total <= 0:
        return _SERIES_HEAD_PROGRESS + _SERIES_BODY_PROGRESS
    return _SERIES_HEAD_PROGRESS + int(_SERIES_BODY_PROGRESS * completed / total)


def run_series_video(
    task_id,
    params: VideoParams,
    stop_at: str = "video",
    allow_server_file_input: bool = False,
):
    """
    Run the single-video pipeline once per chapter and collect the results.

    Parts run under nested task ids, so each chapter keeps its own directory,
    script, and final videos inside the parent task directory. The parent
    task progress mirrors the chapter-by-chapter series body band so the UI
    does not see a stuck bar while a long chapter is rendering.
    """
    outline = resolve_series_outline(params)
    if not outline:
        return stages.mark_task_failed(task_id, "series", "failed to plan the video series")

    total = len(outline)
    logger.info(f"start series task: {task_id}, parts: {total}")
    # ``total_parts`` is written once at the start so the WebUI can render
    # "chapter N/M" without having to read the outline; ``current_part``
    # is updated at the top of every chapter so the panel follows along as
    # a chapter starts. Both are independent of ``progress`` so a stalled
    # network call inside one chapter does not freeze the indicator.
    sm.state.update_task(
        task_id,
        state=const.TASK_STATE_PROCESSING,
        progress=_SERIES_HEAD_PROGRESS,
        series_outline=outline,
        total_parts=total,
    )

    # Look the single-video runner up through ``app.services.task`` rather
    # than importing it directly here. ``task._run_pipeline`` is an alias
    # re-exported from this package's single module; routing through ``task``
    # means tests that patch ``tm._run_pipeline`` keep intercepting the inner
    # pipeline call (the same name they patched before the refactor).
    from app.services import task as _task

    parts: list[dict] = []
    videos: list[str] = []
    scripts: list[str] = []
    warnings: list[dict] = []
    for index, chapter in enumerate(outline, start=1):
        part_task_id = f"{task_id}/part-{index:02d}"
        logger.info(f"\n\n## series part {index}/{total}: {chapter}")
        # Announce the chapter before invoking the inner pipeline so the
        # WebUI's "chapter X/Y" indicator updates as soon as the chapter
        # starts, not after it finishes. Without this, the indicator would
        # only advance on completion and lag a full chapter behind.
        sm.state.update_task(
            task_id,
            current_part=index,
            current_chapter=chapter,
        )
        result = _task._run_pipeline(
            part_task_id,
            build_series_part_params(params, outline, index),
            stop_at=stop_at,
            allow_server_file_input=allow_server_file_input,
        )
        if result.get("state") == const.TASK_STATE_FAILED:
            # One failed chapter must not throw away the parts that did render.
            logger.error(
                f"series part failed: task_id={task_id}, part={index}, "
                f"error={result.get('error')}"
            )
            warnings.append(
                {
                    "code": "series_part_failed",
                    "part": index,
                    "subject": chapter,
                    "error": result.get("error"),
                }
            )
        else:
            videos.extend(result.get("videos") or [])
            scripts.append(result.get("script") or "")
            parts.append(
                {
                    "part": index,
                    "subject": chapter,
                    "task_id": part_task_id,
                    "videos": result.get("videos") or [],
                }
            )

        # Update the parent after each chapter instead of waiting on the inner
        # pipeline, so a long chapter still leaves a moving progress bar in
        # the WebUI. ``index`` is 1-based; the body band caps at 95% before
        # the tail finalize step.
        sm.state.update_task(
            task_id,
            state=const.TASK_STATE_PROCESSING,
            progress=_series_body_progress(index, total),
        )

    if not parts:
        return stages.mark_task_failed(task_id, "series", "every part of the series failed")

    video_script = "\n\n".join(script for script in scripts if script)
    # The parent directory keeps its own script.json, so a series appears in the
    # task history and stays restorable just like a single video task.
    stages.save_script_data(task_id, video_script, [], params)

    kwargs = {
        "videos": videos,
        "script": video_script,
        "series_outline": outline,
        "series_parts": parts,
        "warnings": warnings or None,
    }
    sm.state.update_task(
        task_id,
        state=const.TASK_STATE_COMPLETE,
        progress=_SERIES_HEAD_PROGRESS + _SERIES_BODY_PROGRESS + _SERIES_TAIL_PROGRESS,
        **kwargs,
    )
    logger.success(
        f"series task {task_id} finished, {len(parts)}/{total} parts, "
        f"{len(videos)} videos."
    )
    return kwargs
