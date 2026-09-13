"""Shared pipeline stages used by both single-video and series pipelines.

Each stage is a single responsibility (script, terms, audio, subtitle, materials,
final video) and writes its own progress to the task state. Both pipelines call
these stages in the same order so a series chapter and a single video share the
same per-stage progress curve.
"""

from __future__ import annotations

import math
import os
import re
import time
from os import path

from loguru import logger

from app.config import config
from app.models import const
from app.models.schema import VideoConcatMode, VideoParams
from app.services import (
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
from app.services import state as sm
from app.utils import file_security, utils

# Video music providers only need to implement ``is_enabled`` and
# ``generate_bgm``. Provider differences are isolated to file extensions,
# domain exceptions, and WebUI warning codes; task orchestration, zero-volume
# short-circuit, and failure fallback all reuse the same path to avoid
# duplicating these flows for each new provider.
_VIDEO_MUSIC_PROVIDERS = {
    "sonilo": {
        "service": sonilo,
        "error_type": sonilo.SoniloError,
        "suffix": ".m4a",
        "warning_code": "sonilo_bgm_failed",
        "display_name": "Sonilo",
    },
    "elevenlabs": {
        "service": elevenlabs_music,
        "error_type": elevenlabs_music.ElevenLabsMusicError,
        "suffix": ".mp3",
        "warning_code": "elevenlabs_bgm_failed",
        "display_name": "ElevenLabs",
    },
}

# LoomLoom paid-run state writes go through finite retries. A failure on the
# last attempt returns None so the caller can decide to keep polling without
# letting a Redis blip swallow a billable remote run.
_LOOMLOOM_STATE_WRITE_ATTEMPTS = 3
_LOOMLOOM_STATE_RETRY_DELAY_SECONDS = 0.1


def get_video_music_prompt(params: VideoParams) -> str:
    """
    Read the prompt actually used by the current video music provider.

    New tasks use the provider-agnostic field; old Sonilo CLI params and
    historical tasks may only have ``sonilo_bgm_prompt``, so only fall back
    to the old field when the Sonilo-agnostic field is empty.
    """
    prompt = str(params.video_music_prompt or "").strip()
    if params.bgm_type == "sonilo" and not prompt:
        prompt = str(params.sonilo_bgm_prompt or "").strip()
    return prompt


def mark_task_failed(
    task_id: str,
    stage: str,
    error: str,
    details: dict | None = None,
) -> dict:
    """Record structured failure info and preserve progress reached before the task failed."""
    existing_task = None
    try:
        existing_task = sm.state.get_task(task_id)
    except Exception as exc:
        logger.warning(f"failed to read task state before failure update: {exc}")

    # Concrete service functions usually have more accurate error reasons than
    # the orchestration layer. Subsequent empty-result checks must not overwrite
    # them with generic messages, or API callers would still only see vague info.
    if (
        existing_task
        and existing_task.get("state") == const.TASK_STATE_FAILED
        and existing_task.get("error")
    ):
        return existing_task

    message = str(error or "unknown task error").strip()
    progress = int((existing_task or {}).get("progress", 0) or 0)
    logger.error(f"task failed, task_id: {task_id}, stage: {stage}, error: {message}")
    failure = {
        "task_id": task_id,
        "state": const.TASK_STATE_FAILED,
        "progress": progress,
        "failed_stage": stage,
        "error": message,
    }
    # Some external tasks already created remote IDs useful for recovery or
    # troubleshooting. Failure state needs to preserve these non-sensitive fields
    # but must not allow callers to overwrite the unified state, progress, and
    # error structure.
    failure_details = {
        key: value for key, value in dict(details or {}).items() if key not in failure
    }
    failure.update(failure_details)
    sm.state.update_task(
        task_id,
        state=failure["state"],
        progress=failure["progress"],
        failed_stage=failure["failed_stage"],
        error=failure["error"],
        **failure_details,
    )
    return failure


# Back-compat alias so existing callers (tests, services) that imported
# ``_mark_task_failed`` from the old ``app.services.task`` module keep working.
_mark_task_failed = mark_task_failed


def generate_script(task_id, params):
    logger.info("\n\n## generating video script")
    video_script = params.video_script.strip()
    if not video_script:
        video_script = llm.generate_script(
            video_subject=params.video_subject,
            language=params.video_language,
            paragraph_number=params.paragraph_number,
            video_script_prompt=params.video_script_prompt,
            custom_system_prompt=params.custom_system_prompt,
        )
    else:
        logger.debug(f"video script: \n{video_script}")

    if not video_script:
        mark_task_failed(task_id, "script", "failed to generate video script")
        return None

    return video_script


def generate_terms(task_id, params, video_script):
    logger.info("\n\n## generating video terms")
    video_terms = params.video_terms
    if not video_terms:
        # When material matching follows script order, keywords themselves must
        # also be generated in script narrative order; otherwise even sequential
        # download and sequential concatenation can only reuse one set of global
        # theme keywords, failing to fix "footage appearing before its content".
        video_terms = llm.generate_terms(
            video_subject=params.video_subject,
            video_script=utils.remove_pause_tags(video_script),
            amount=8 if params.match_materials_to_script else 5,
            match_script_order=params.match_materials_to_script,
        )
    else:
        if isinstance(video_terms, str):
            video_terms = [term.strip() for term in re.split(r"[,，]", video_terms)]
        elif isinstance(video_terms, list):
            video_terms = [term.strip() for term in video_terms]
        else:
            raise ValueError("video_terms must be a string or a list of strings.")

        logger.debug(f"video terms: {utils.to_json(video_terms)}")

    if not video_terms:
        mark_task_failed(
            task_id,
            "terms",
            "failed to generate video search terms",
        )
        return None

    # Optional TwelveLabs Marengo semantic reranking: when disabled returns
    # original order with no side effects. In sequential matching mode keyword
    # order IS the script narrative order and must be preserved, so skip.
    if not params.match_materials_to_script:
        video_terms = twelvelabs.rerank_terms_by_subject(
            video_subject=params.video_subject,
            search_terms=video_terms,
        )

    return video_terms


def save_script_data(task_id, video_script, video_terms, params):
    script_data = {
        "script": video_script,
        "search_terms": video_terms,
        "params": params,
    }
    task_artifacts.write_script_data(task_id, script_data)


def resolve_custom_audio_file(
    task_id: str,
    custom_audio_file: str | None,
    *,
    allow_server_file_input: bool = False,
) -> str:
    requested_file = (custom_audio_file or "").strip()
    if not requested_file:
        return ""

    task_dir = utils.task_dir(task_id)
    try:
        return file_security.resolve_path_within_directory(
            task_dir,
            requested_file,
        )
    except ValueError as exc:
        task_dir_error = exc

    # A missing path that otherwise stays inside the task directory is safe to
    # report precisely. Paths outside that boundary use the same generic error
    # regardless of whether they exist, so callers cannot probe the host filesystem.
    if str(task_dir_error) == "file does not exist":
        raise task_dir_error

    # HTTP requests and other untrusted callers must never turn a submitted path
    # into a server-side file read. WebUI uploads already live in the task directory;
    # only the local CLI explicitly opts into resolving files elsewhere on the host.
    if not allow_server_file_input:
        raise ValueError(
            "custom audio file must be stored within the current task directory"
        ) from task_dir_error

    server_audio_file = path.realpath(
        requested_file
        if path.isabs(requested_file)
        else path.join(utils.root_dir(), requested_file)
    )
    if not path.isabs(requested_file):
        project_root = path.realpath(utils.root_dir())
        try:
            if path.commonpath([project_root, server_audio_file]) != project_root:
                raise ValueError(
                    "relative custom audio paths must stay within the project directory"
                )
        except ValueError as exc:
            raise ValueError(
                "custom audio file must be task-local or an existing server-side file"
            ) from exc

    if not path.isfile(server_audio_file):
        raise ValueError(
            "custom audio file does not exist or is not a file"
        ) from task_dir_error

    return server_audio_file


def _resolve_reusable_voice_preview(
    task_id: str,
    params,
    video_script: str,
    voice_preview: dict | None,
) -> tuple[str, float, object] | None:
    """
    Validate and parse the full voice preview cache submitted by the WebUI.

    This payload is not a public API parameter and can only come from the WebUI
    in the current process. Even so, background tasks re-verify the script and
    all voice parameters, and restrict audio to the current task directory;
    any mismatch falls back to plain TTS to prevent stale previews from
    contaminating the final output.
    """
    if not voice_preview:
        return None

    expected_values = {
        "script": str(video_script or "").strip(),
        "voice_name": params.voice_name,
        "voice_rate": float(params.voice_rate),
        "voice_volume": float(params.voice_volume),
    }
    if not math.isclose(float(params.voice_volume), 1.0) or any(
        voice_preview.get(key) != value for key, value in expected_values.items()
    ):
        logger.info(
            f"skip stale voice preview cache, task_id: {task_id}, "
            "reason: voice parameters changed"
        )
        return None

    preview_file = path.realpath(str(voice_preview.get("audio_file") or ""))
    task_root = path.realpath(utils.task_dir(task_id))
    try:
        preview_is_task_local = path.commonpath([task_root, preview_file]) == task_root
    except ValueError:
        preview_is_task_local = False

    duration = voice_preview.get("duration")
    sub_maker = voice_preview.get("sub_maker")
    if (
        not preview_is_task_local
        or not path.isfile(preview_file)
        or not isinstance(duration, (int, float))
        or not math.isfinite(duration)
        or duration <= 0
        or sub_maker is None
    ):
        logger.warning(
            f"skip invalid voice preview cache, task_id: {task_id}, "
            f"audio_file: {preview_file or '<empty>'}"
        )
        return None

    logger.info(
        f"using full voice preview audio, task_id: {task_id}, duration: {duration:.2f}s"
    )
    return preview_file, math.ceil(duration), sub_maker


def generate_audio(
    task_id,
    params,
    video_script,
    voice_preview=None,
    *,
    allow_server_file_input: bool = False,
):
    """
    Generate audio for the video script.
    If a custom audio file is provided, it will be used directly.
    There will be no subtitle maker object returned in this case.
    Otherwise, TTS will be used to generate the audio.
    Returns:
        - audio_file: path to the generated or provided audio file
        - audio_duration: duration of the audio in seconds
        - sub_maker: subtitle maker object if TTS is used, None otherwise
    """
    logger.info("\n\n## generating audio")
    # The /audio and /subtitle request models do not include custom_audio_file;
    # handle it here for compatibility so direct callers do not get AttributeError.
    requested_custom_audio_file = getattr(params, "custom_audio_file", None)
    try:
        custom_audio_file = resolve_custom_audio_file(
            task_id,
            requested_custom_audio_file,
            allow_server_file_input=allow_server_file_input,
        )
    except ValueError as exc:
        mark_task_failed(
            task_id,
            "audio",
            f"invalid custom audio file: {exc}",
        )
        return None, None, None

    if not custom_audio_file:
        reusable_preview = _resolve_reusable_voice_preview(
            task_id,
            params,
            video_script,
            voice_preview,
        )
        if reusable_preview:
            return reusable_preview

        logger.info("no custom audio file provided, using TTS to generate audio.")
        audio_file = path.join(utils.task_dir(task_id), "audio.mp3")
        sub_maker = voice.tts(
            text=video_script,
            voice_name=voice.parse_voice_name(params.voice_name),
            voice_rate=params.voice_rate,
            voice_file=audio_file,
        )
        if sub_maker is None:
            # The real failure reason (e.g. quota, auth, bad voice) is already
            # logged by the voice provider. Surface a hint pointing the user
            # at the application logs so they can distinguish quota / auth /
            # connectivity failures from "wrong voice name".
            mark_task_failed(
                task_id,
                "audio",
                "failed to synthesize audio; verify the selected voice, "
                "TTS API key, and quota. See the application log for the "
                "exact provider error.",
            )
            return None, None, None
        # Measure audio length. Non-real-word providers (Gemini, SiliconFlow,
        # MiniMax, ...) populate sub_maker.duration from the actual rendered
        # audio length, so the file probe is pure overhead. Edge TTS and
        # Azure v2 return word boundaries instead, leaving a fixed audio tail
        # (~0.88s for Edge) past the last boundary that only the file probe
        # exposes. The probe still runs for those two providers because the
        # under-count sizes paid generate_bgm() calls, is reported as
        # audio_duration to the API/WebUI, and under-sources
        # download_videos() material, scaled by video_count.
        audio_duration = 0.0
        if sub_maker is not None and not voice.has_real_word_timestamps(sub_maker):
            audio_duration = getattr(sub_maker, "duration", 0.0) or 0.0
        if audio_duration <= 0:
            audio_duration = voice.get_audio_duration(audio_file)
        if audio_duration <= 0:
            audio_duration = voice.get_audio_duration(sub_maker)
        audio_duration = math.ceil(audio_duration)
        if audio_duration == 0:
            mark_task_failed(task_id, "audio", "generated audio duration is zero")
            return None, None, None
        return audio_file, audio_duration, sub_maker
    else:
        logger.info(f"using custom audio file: {custom_audio_file}")
        audio_duration = voice.get_audio_duration(custom_audio_file)
        if audio_duration == 0:
            mark_task_failed(
                task_id,
                "audio",
                "custom audio duration is zero",
            )
            return None, None, None
        return custom_audio_file, audio_duration, None


def generate_subtitle(task_id, params, video_script, sub_maker, audio_file):
    """
    Generate subtitle for the video script.
    If subtitle generation is disabled or no subtitle maker is provided, it will return an empty string.
    Otherwise, it will generate the subtitle using the specified provider.
    Returns:
        - subtitle_path: path to the generated subtitle file
    """
    logger.info("\n\n## generating subtitle")
    if not params.subtitle_enabled:
        return ""

    subtitle_path = path.join(utils.task_dir(task_id), "subtitle.srt")
    subtitle_provider = config.app.get("subtitle_provider", "edge").strip().lower()
    logger.info(f"\n\n## generating subtitle, provider: {subtitle_provider}")

    if not subtitle_provider:
        logger.info("subtitle provider is empty, skip subtitle generation")
        return ""

    if sub_maker is None and subtitle_provider != "whisper":
        # Custom audio does not go through TTS, so there is no sub_maker
        # timeline from Edge/Azure TTS. Only Whisper can transcribe subtitles
        # directly from audio files; other subtitle providers keep their
        # original behavior to avoid generating incorrect empty timelines.
        logger.warning(
            "subtitle maker is missing, skip subtitle generation for provider: "
            f"{subtitle_provider}"
        )
        return ""

    # The regular path favors precise Whisper alignment for estimated TTS
    # timelines. Docker's fast-pipeline mode already has the complete script and
    # uses the TTS estimate directly, avoiding a redundant transcription pass.
    if (
        subtitle_provider == "edge"
        and not voice.has_real_word_timestamps(sub_maker)
        and os.environ.get("MPT_SKIP_PIPELINE_PREFLIGHT") != "1"
    ):
        logger.warning(
            "TTS provider did not return real word-level timestamps; "
            "falling back to whisper for accurate subtitle alignment"
        )
        subtitle_provider = "whisper"

    display_mode = getattr(params, "subtitle_display_mode", "sentence")
    is_word_level = display_mode != "sentence"

    if subtitle_provider == "edge":
        voice.create_subtitle(
            text=video_script,
            sub_maker=sub_maker,
            subtitle_file=subtitle_path,
            word_level=is_word_level,
        )
        if not os.path.exists(subtitle_path):
            # Edge subtitles occasionally fail to produce output when the timeline
            # cannot match the script. Auto-switching to Whisper is not done here
            # because the first failure would download a GB-scale model without
            # user knowledge. Only explicitly configured Whisper is allowed to load
            # the model; Edge failures leave a subtitle-less video and log the
            # reason, avoiding unexpected network and disk overhead.
            logger.warning(
                "edge subtitle generation did not produce a subtitle file; "
                "skip subtitles without falling back to whisper"
            )
            return ""

    if subtitle_provider == "whisper":
        subtitle.create(
            audio_file=audio_file,
            subtitle_file=subtitle_path,
            word_level=is_word_level,
        )
        if not is_word_level:
            logger.info("\n\n## correcting subtitle")
            subtitle.correct(subtitle_file=subtitle_path, video_script=video_script)

    subtitle_lines = subtitle.file_to_subtitles(subtitle_path)
    if not subtitle_lines:
        logger.warning(f"subtitle file is invalid: {subtitle_path}")
        return ""

    return subtitle_path


def get_video_materials(
    task_id,
    params,
    video_terms,
    audio_duration,
    loomloom_video_request: loomloom.LoomLoomConfirmedVideoRequest | None = None,
):
    if params.video_source == "local":
        logger.info("\n\n## preprocess local materials")
        materials = video.preprocess_video(
            materials=params.video_materials, clip_duration=params.video_clip_duration
        )
        if not materials:
            mark_task_failed(
                task_id,
                "materials",
                "no valid local video materials were found",
            )
            return None
        return [material_info.url for material_info in materials]
    elif params.video_source == "loomloom":
        if not isinstance(
            loomloom_video_request, loomloom.LoomLoomConfirmedVideoRequest
        ):
            mark_task_failed(
                task_id,
                "materials",
                "LoomLoom video generation requires a confirmed quote",
            )
            return None

        request = loomloom_video_request
        logger.info(
            "\n\n## generating "
            f"{len(request.batch.input_rows)} video materials with LoomLoom"
        )
        run_id = ""
        try:
            request.validate()
            backend = loomloom.LoomLoomVideoBackend(request.settings)
            execution = backend.execute(
                request.batch,
                client_request_id=request.client_request_id,
                listing_version_id=request.listing_version_id,
                confirm=True,
            )
            run_id = execution.run_id
            # execute returning means the paid task has been accepted by the remote.
            # The run ID must be written to the process log first, so that even if
            # Redis or other state backend becomes unavailable later, operators can
            # still locate the task on the platform side via the logs; the unique
            # identifier must not exist only in a local variable.
            logger.info(
                "LoomLoom paid video run created: "
                f"task_id={task_id}, run_id={run_id}, "
                f"listing_version_id={request.listing_version_id}"
            )
            # As soon as a paid task is created, immediately record the remote ID.
            # Even if subsequent polling times out, logs and task state can still
            # help users or platform support locate and recover already-generated
            # artifacts. State backend failures can only reduce observability;
            # they cannot interrupt already-billing remote tasks and artifact downloads.
            _record_loomloom_run_reference(
                task_id=task_id,
                run_id=run_id,
                listing_version_id=request.listing_version_id,
            )
            backend.wait_for_run(run_id)
            return list(
                backend.download_video_results(
                    run_id,
                    utils.task_dir(task_id),
                )
            )
        except (loomloom.LoomLoomError, ValueError) as exc:
            mark_task_failed(
                task_id,
                "materials",
                str(exc),
                details={
                    "loomloom_run_id": run_id,
                    "loomloom_listing_version_id": request.listing_version_id,
                },
            )
            return None
    else:
        logger.info(f"\n\n## downloading videos from {params.video_source}")
        # Sequential matching mode only applies when the user explicitly enables it.
        # Here we force material downloads to poll in keyword order to prevent
        # early keywords from downloading too much material and pushing later
        # script topics off the final timeline.
        try:
            downloaded_videos = material.download_videos(
                task_id=task_id,
                search_terms=video_terms,
                source=params.video_source,
                video_aspect=params.video_aspect,
                video_concat_mode=(
                    VideoConcatMode.sequential
                    if params.match_materials_to_script
                    else params.video_concat_mode
                ),
                audio_duration=audio_duration * params.video_count,
                max_clip_duration=params.video_clip_duration,
                match_script_order=params.match_materials_to_script,
            )
        except volcengine_seedance.VolcEngineSeedanceError as exc:
            # Both unconfirmed state and "generated but download failed" correspond
            # to a remote task recoverable from the Ark console. Write failure state
            # uniformly from the task_id carried in the exception, avoiding
            # different exception branches each maintaining their own recovery info
            # and missing it again in future extensions.
            remote_task_id = str(getattr(exc, "task_id", "") or "").strip()
            details = (
                {"volcengine_seedance_task_id": remote_task_id}
                if remote_task_id
                else None
            )
            mark_task_failed(
                task_id,
                "materials",
                str(exc),
                details=details,
            )
            return None
        except ofox.OFoxError as exc:
            # Same recovery semantics as Ark: both unconfirmed state and "generated
            # but download failed" correspond to a remote task recoverable from the
            # OFox console; write failure state uniformly from the task_id in the exception.
            remote_task_id = str(getattr(exc, "task_id", "") or "").strip()
            details = (
                {"ofox_task_id": remote_task_id} if remote_task_id else None
            )
            mark_task_failed(
                task_id,
                "materials",
                str(exc),
                details=details,
            )
            return None
        except metaso_minimax.MetasoMiniMaxError as exc:
            # Metaso tasks use different recovery entry points and field names than
            # Ark tasks and cannot be merged into a single vague remote_task_id.
            # Keeping the explicit Provider prefix allows API, WebUI, and ops logs
            # to directly locate the corresponding platform.
            remote_task_id = str(getattr(exc, "task_id", "") or "").strip()
            details = (
                {"metaso_minimax_task_id": remote_task_id} if remote_task_id else None
            )
            mark_task_failed(
                task_id,
                "materials",
                str(exc),
                details=details,
            )
            return None
        if not downloaded_videos:
            mark_task_failed(
                task_id,
                "materials",
                f"failed to download video materials from {params.video_source}",
            )
            return None
        return downloaded_videos


def _record_loomloom_run_reference(
    *, task_id: str, run_id: str, listing_version_id: str
) -> bool | None:
    """
    Best-effort persistence of a created paid LoomLoom Run; do not let state
    failures interrupt the remote task.

    Returns True on success, False if the task record no longer exists, and
    None if the state backend remains unavailable after finite retries.
    Callers should continue polling and downloading regardless of which result
    they receive, because execute has already produced external billing side
    effects; stopping the local flow only makes artifacts harder to recover.
    """
    fields = {
        "loomloom_run_id": run_id,
        "loomloom_listing_version_id": listing_version_id,
    }
    for attempt in range(1, _LOOMLOOM_STATE_WRITE_ATTEMPTS + 1):
        try:
            updated = sm.state.patch_task(task_id, **fields)
        except Exception as exc:
            if attempt >= _LOOMLOOM_STATE_WRITE_ATTEMPTS:
                logger.exception(
                    "failed to persist LoomLoom paid run after retries: "
                    f"task_id={task_id}, run_id={run_id}, attempts={attempt}, "
                    f"error={exc}"
                )
                return None
            logger.warning(
                "retry LoomLoom paid run state update: "
                f"task_id={task_id}, run_id={run_id}, attempt={attempt}, "
                f"error={exc}"
            )
            time.sleep(_LOOMLOOM_STATE_RETRY_DELAY_SECONDS)
            continue

        if updated is False:
            logger.warning(
                "could not persist LoomLoom paid run because task is missing: "
                f"task_id={task_id}, run_id={run_id}"
            )
        return updated

    return None


def generate_final_videos(
    task_id, params, downloaded_videos, audio_file, subtitle_path, audio_duration
):
    final_video_paths = []
    combined_video_paths = []
    warnings = []
    video_music_provider = _VIDEO_MUSIC_PROVIDERS.get(params.bgm_type)
    video_music_requested = (
        video_music_provider is not None
        and bgm_service.should_use_bgm(params.bgm_type, params.bgm_volume)
    )
    # Multi-video generation shuffles materials by default for variety; but
    # "match materials to script order" seeks timeline stability and
    # explainability, so when enabled all outputs use sequential concatenation.
    if params.match_materials_to_script:
        video_concat_mode = VideoConcatMode.sequential
    elif params.video_count == 1:
        video_concat_mode = params.video_concat_mode
    else:
        video_concat_mode = VideoConcatMode.random
    video_transition_mode = params.video_transition_mode

    _progress = 50
    # Cross-part clip dedup: track which (source, start, end) tuples have
    # already been used so each part of a multi-video task shows different
    # scenes instead of recycling the same handful of clips. combine_videos
    # mutates this set in place with the clips it consumed.
    used_clip_fingerprints: set = set()
    for i in range(params.video_count):
        index = i + 1
        combined_video_path = path.join(
            utils.task_dir(task_id), f"combined-{index}.mp4"
        )
        logger.info(f"\n\n## combining video: {index} => {combined_video_path}")
        video.combine_videos(
            combined_video_path=combined_video_path,
            video_paths=downloaded_videos,
            audio_file=audio_file,
            video_aspect=params.video_aspect,
            video_fit_mode=params.video_fit_mode,
            video_concat_mode=video_concat_mode,
            video_transition_mode=video_transition_mode,
            max_clip_duration=params.video_clip_duration,
            threads=params.n_threads,
            clip_speed=params.video_clip_speed,
            exclude_clip_fingerprints=used_clip_fingerprints,
            part_index=i,
            task_id=task_id,
        )

        _progress += 50 / params.video_count / 2
        sm.state.update_task(task_id, progress=_progress)

        final_video_path = path.join(utils.task_dir(task_id), f"final-{index}.mp4")

        # In video music mode, explicitly disable default BGM parsing first to
        # avoid stale bgm_file from old tasks being reused. Only generate a
        # proxy and call the paid API when volume is greater than 0; zero
        # volume is uniformly skipped.
        bgm_file_override = "" if video_music_provider else None
        if video_music_requested:
            service = video_music_provider["service"]
            display_name = video_music_provider["display_name"]
            warning_code = video_music_provider["warning_code"]
            generated_bgm_path = path.join(
                utils.task_dir(task_id),
                (f"{params.bgm_type}-bgm-{index}{video_music_provider['suffix']}"),
            )
            try:
                service.generate_bgm(
                    video_path=combined_video_path,
                    output_path=generated_bgm_path,
                    video_duration=audio_duration,
                    prompt=get_video_music_prompt(params),
                )
                bgm_file_override = generated_bgm_path
            except video_music_provider["error_type"] as exc:
                # When video, narration, and subtitles are all generated, a
                # third-party music temporary failure should not waste the whole
                # task. The current video explicitly disables BGM and returns
                # the degraded result to WebUI to alert the user.
                logger.warning(
                    f"{display_name} BGM generation failed: task_id={task_id}, "
                    f"video_index={index}, error={exc}"
                )
                bgm_file_override = ""
                warnings.append({"code": warning_code, "video_index": index})

        logger.info(f"\n\n## generating video: {index} => {final_video_path}")
        bgm_mix_succeeded = video.generate_video(
            video_path=combined_video_path,
            audio_path=audio_file,
            subtitle_path=subtitle_path,
            output_file=final_video_path,
            params=params,
            bgm_file_override=bgm_file_override,
        )
        if (
            video_music_provider is not None
            and bgm_file_override
            and not bgm_mix_succeeded
        ):
            # Third party returned successfully and passed FFmpeg validation,
            # but MoviePy's final mixing may still fail due to runtime environment.
            # Video service keeps the non-BGM output; when API generation fails
            # the override is empty, so no duplicate warning is appended.
            warnings.append(
                {
                    "code": video_music_provider["warning_code"],
                    "video_index": index,
                }
            )

        _progress += 50 / params.video_count / 2
        sm.state.update_task(task_id, progress=_progress)

        final_video_paths.append(final_video_path)
        combined_video_paths.append(combined_video_path)

    return final_video_paths, combined_video_paths, warnings
