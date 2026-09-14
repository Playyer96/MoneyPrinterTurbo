"""Material download dispatchers.

Owns ``download_videos`` (the public entry point) plus the five
``_download_videos_*_on_demand`` runners that gate the paid AI providers.
Stock providers route through ``_download_candidate_pool``; paid providers
generate per-keyword until ``audio_duration`` is covered and stop.

Test-patching note: this module looks up provider functions through the
``app.services.material`` package's current bindings (via
``sys.modules``) so monkey-patched names on ``material.search_videos_pexels``
and similar keep intercepting calls without test changes.
"""

from __future__ import annotations

import math
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, List

from loguru import logger

from app.config import config
from app.models.schema import MaterialInfo, VideoAspect, VideoConcatMode
from app.services.material._shared import (
    _MATERIAL_DOWNLOAD_WORKERS,
    _redact_request_error,
)
from app.services.material.cache import _search_terms_parallel, _search_videos_with_cache
from app.utils import utils


def _package():
    """Return the running ``app.services.material`` package so call-time
    monkey-patches on its public surface still reach this dispatcher."""
    return sys.modules["app.services.material"]


def download_videos(
    task_id: str,
    search_terms: List[str],
    source: str = "pexels",
    video_aspect: VideoAspect = VideoAspect.portrait,
    video_concat_mode: VideoConcatMode = VideoConcatMode.random,
    audio_duration: float = 0.0,
    max_clip_duration: int = 5,
    match_script_order: bool = False,
) -> List[str]:
    """Dispatch material acquisition to the right provider and return local file paths."""
    pkg = _package()
    provider = "pexels"
    remote_search_videos = pkg.search_videos_pexels
    if source == "pixabay":
        provider = "pixabay"
        remote_search_videos = pkg.search_videos_pixabay
    elif source == "coverr":
        provider = "coverr"
        remote_search_videos = pkg.search_videos_coverr

    def search_videos(
        search_term: str,
        minimum_duration: int,
        video_aspect: VideoAspect,
    ) -> List[MaterialInfo]:
        return _search_videos_with_cache(
            provider=provider,
            search_videos=remote_search_videos,
            search_term=search_term,
            minimum_duration=minimum_duration,
            video_aspect=video_aspect,
        )

    material_directory = config.app.get("material_directory", "").strip()
    if material_directory == "task":
        material_directory = utils.task_dir(task_id)
    elif material_directory and not os.path.isdir(material_directory):
        material_directory = ""

    if source == "wavespeed":
        return _download_videos_wavespeed_on_demand(
            task_id=task_id,
            search_terms=search_terms,
            video_aspect=video_aspect,
            audio_duration=audio_duration,
            max_clip_duration=max_clip_duration,
            material_directory=material_directory,
        )
    if source == "volcengine_seedance":
        return _download_videos_seedance_on_demand(
            task_id=task_id,
            search_terms=search_terms,
            video_aspect=video_aspect,
            audio_duration=audio_duration,
            max_clip_duration=max_clip_duration,
            material_directory=material_directory,
        )
    if source == "ofox":
        return _download_videos_ofox_on_demand(
            task_id=task_id,
            search_terms=search_terms,
            video_aspect=video_aspect,
            audio_duration=audio_duration,
            max_clip_duration=max_clip_duration,
            material_directory=material_directory,
        )
    if source == "metaso_minimax":
        return _download_videos_metaso_minimax_on_demand(
            task_id=task_id,
            search_terms=search_terms,
            video_aspect=video_aspect,
            audio_duration=audio_duration,
            max_clip_duration=max_clip_duration,
            material_directory=material_directory,
        )
    if source == "openai_image":
        return _download_videos_openai_image_on_demand(
            task_id=task_id,
            search_terms=search_terms,
            video_aspect=video_aspect,
            audio_duration=audio_duration,
            max_clip_duration=max_clip_duration,
            material_directory=material_directory,
        )

    if match_script_order:
        return _download_videos_by_script_order(
            task_id=task_id,
            search_terms=search_terms,
            search_videos=search_videos,
            video_aspect=video_aspect,
            audio_duration=audio_duration,
            max_clip_duration=max_clip_duration,
            material_directory=material_directory,
        )

    valid_video_items = []
    valid_video_urls = []
    found_duration = 0.0
    for term, video_items in _search_terms_parallel(
        search_terms=search_terms,
        search_videos=search_videos,
        minimum_duration=max_clip_duration,
        video_aspect=video_aspect,
    ):
        logger.info(f"found {len(video_items)} videos for '{term}'")
        for item in video_items:
            if item.url not in valid_video_urls:
                valid_video_items.append(item)
                valid_video_urls.append(item.url)
                found_duration += item.duration
    logger.info(
        f"found total videos: {len(valid_video_items)}, required duration: {audio_duration} seconds, found duration: {found_duration} seconds"
    )
    concat_mode_value = getattr(video_concat_mode, "value", video_concat_mode)
    if concat_mode_value == VideoConcatMode.random.value:
        random.shuffle(valid_video_items)
    return pkg._download_candidate_pool(
        task_id=task_id,
        candidate_pool=valid_video_items,
        audio_duration=audio_duration,
        max_clip_duration=max_clip_duration,
        material_directory=material_directory,
    )


def _download_videos_wavespeed_on_demand(
    *,
    task_id: str,
    search_terms: List[str],
    video_aspect: VideoAspect,
    audio_duration: float,
    max_clip_duration: int,
    material_directory: str,
) -> List[str]:
    """Per-keyword WaveSpeed generation; stop when audio_duration is covered."""
    pkg = _package()

    video_paths: List[str] = []
    material_sources: list[dict[str, Any]] = []
    total_duration = 0.0
    max_possible_duration = 0.0
    stop_submitting = False
    next_term_idx = 0

    def _next_search_term() -> str | None:
        nonlocal next_term_idx
        if next_term_idx >= len(search_terms):
            return None
        term = search_terms[next_term_idx]
        next_term_idx += 1
        return term

    def _submit_gen(term: str):
        nonlocal max_possible_duration
        max_possible_duration += max_clip_duration
        return (
            term,
            executor.submit(
                pkg.generate_videos_wavespeed,
                search_term=term,
                minimum_duration=max_clip_duration,
                video_aspect=video_aspect,
            ),
        )

    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="mpt-wspd") as executor:
        dl_pending: list[tuple[str, MaterialInfo, Any]] = []
        gen_state = None
        first_term = _next_search_term()
        if first_term is not None:
            gen_state = _submit_gen(first_term)
        while gen_state is not None or dl_pending:
            if gen_state is not None and gen_state[1].done():
                term, gen_future = gen_state
                gen_state = None
                video_items: list[MaterialInfo] = []
                try:
                    video_items = gen_future.result()
                except pkg.WaveSpeedUnconfirmedTaskError as e:
                    logger.error(
                        "stop submitting new wavespeed tasks, the last submitted "
                        f"task is unconfirmed: prediction_id={e.prediction_id or 'unknown'}, "
                        f"detail={e}"
                    )
                    stop_submitting = True
                except Exception as e:
                    logger.error(
                        f"wavespeed generation failed for {term!r}: "
                        f"{type(e).__name__}: {e}"
                    )
                if not video_items:
                    max_possible_duration -= max_clip_duration
                if video_items:
                    item = video_items[0]
                    dl_pending.append(
                        (
                            term,
                            item,
                            executor.submit(
                                pkg._save_generated_video_with_retry,
                                item.url,
                                material_directory,
                                "wavespeed",
                            ),
                        )
                    )
                if (
                    not stop_submitting
                    and (total_duration + max_possible_duration) < audio_duration
                ):
                    next_term = _next_search_term()
                    if next_term is not None:
                        gen_state = _submit_gen(next_term)
            if dl_pending and dl_pending[0][2].done():
                term, item, dl_future = dl_pending.pop(0)
                try:
                    saved_video_path = dl_future.result() or ""
                except Exception as e:
                    logger.error(
                        f"wavespeed download failed for {term!r}: "
                        f"{type(e).__name__}: {e}"
                    )
                    saved_video_path = ""
                if saved_video_path:
                    logger.info(f"video saved: {saved_video_path}")
                    video_paths.append(saved_video_path)
                    try:
                        material_sources.append(
                            pkg._material_source_record(item, saved_video_path)
                        )
                    except Exception as source_error:
                        logger.warning(
                            "failed to prepare material source record: "
                            f"provider=wavespeed, "
                            f"error={type(source_error).__name__}, detail={source_error}"
                        )
                    total_duration += min(max_clip_duration, item.duration)
                    max_possible_duration -= max_clip_duration
                    if total_duration >= audio_duration:
                        gen_state = None
                        break
            if gen_state is not None or dl_pending:
                time.sleep(0.01)
        for term, item, dl_future in dl_pending:
            try:
                saved_video_path = dl_future.result() or ""
                if saved_video_path:
                    video_paths.append(saved_video_path)
                    material_sources.append(
                        pkg._material_source_record(item, saved_video_path)
                    )
            except Exception:
                pass
    logger.success(f"generated and downloaded {len(video_paths)} videos")
    pkg._persist_material_sources(task_id, material_sources)
    return video_paths


def _download_videos_seedance_on_demand(
    *,
    task_id: str,
    search_terms: List[str],
    video_aspect: VideoAspect,
    audio_duration: float,
    max_clip_duration: int,
    material_directory: str,
) -> List[str]:
    """Sequential Seedance generation; stop once audio_duration is covered."""
    from app.services import volcengine_seedance
    pkg = _package()

    video_paths: List[str] = []
    material_sources: list[dict[str, Any]] = []
    try:
        required_duration = float(audio_duration)
    except (TypeError, ValueError) as exc:
        raise volcengine_seedance.VolcEngineSeedanceError(
            "Seedance audio duration must be a finite number"
        ) from exc
    if not math.isfinite(required_duration):
        raise volcengine_seedance.VolcEngineSeedanceError(
            "Seedance audio duration must be a finite number"
        )
    if required_duration <= 0:
        logger.warning(
            "skip Seedance paid generation because required audio duration is "
            f"not positive: duration={required_duration}"
        )
        pkg._persist_material_sources(task_id, material_sources)
        return video_paths
    try:
        clip_duration = int(max_clip_duration)
    except (TypeError, ValueError, OverflowError) as exc:
        raise volcengine_seedance.VolcEngineSeedanceError(
            "Seedance clip duration must be a positive integer"
        ) from exc
    if clip_duration <= 0:
        raise volcengine_seedance.VolcEngineSeedanceError(
            "Seedance clip duration must be a positive integer"
        )
    total_duration = 0.0
    for search_term in search_terms:
        try:
            video_items = volcengine_seedance.generate_videos(
                search_term=search_term,
                minimum_duration=clip_duration,
                video_aspect=video_aspect,
            )
        except volcengine_seedance.VolcEngineSeedanceUnconfirmedTaskError as exc:
            logger.error(
                "stop submitting new Seedance tasks because the last paid task "
                f"is unconfirmed: task_id={exc.task_id or 'unknown'}, detail={exc}"
            )
            pkg._persist_material_sources(task_id, material_sources)
            raise
        except volcengine_seedance.VolcEngineSeedanceError as exc:
            logger.error(f"Seedance generation failed before completion: {exc}")
            pkg._persist_material_sources(task_id, material_sources)
            raise
        for item in video_items:
            saved_video_path = pkg._save_generated_video_with_retry(
                item.url, material_directory, "volcengine_seedance"
            )
            if not saved_video_path:
                source_info = (
                    item.source_info if isinstance(item.source_info, dict) else {}
                )
                remote_task_id = str(source_info.get("asset_id") or "").strip()
                pkg._persist_material_sources(task_id, material_sources)
                raise volcengine_seedance.VolcEngineSeedanceDownloadError(
                    "Seedance generated a paid video but the result could not be "
                    f"downloaded: id={remote_task_id or 'unknown'}",
                    task_id=remote_task_id,
                )
            logger.info(f"video saved: {saved_video_path}")
            video_paths.append(saved_video_path)
            try:
                material_sources.append(pkg._material_source_record(item, saved_video_path))
            except Exception as source_error:
                logger.warning(
                    "failed to prepare generated material source record: "
                    f"provider=volcengine_seedance, "
                    f"error={type(source_error).__name__}, detail={source_error}"
                )
            total_duration += min(clip_duration, item.duration)
            if total_duration >= required_duration:
                break
        if total_duration >= required_duration:
            logger.info(
                "generated Seedance materials cover the required duration; stop "
                f"submitting paid tasks: generated={total_duration:.1f}s, "
                f"required={required_duration:.1f}s"
            )
            break
    logger.success(
        f"generated and downloaded {len(video_paths)} Volcano Engine Seedance videos"
    )
    pkg._persist_material_sources(task_id, material_sources)
    return video_paths


def _download_videos_ofox_on_demand(
    *,
    task_id: str,
    search_terms: List[str],
    video_aspect: VideoAspect,
    audio_duration: float,
    max_clip_duration: int,
    material_directory: str,
) -> List[str]:
    """Sequential OFox generation; stop once audio_duration is covered."""
    from app.services import ofox
    pkg = _package()

    video_paths: List[str] = []
    material_sources: list[dict[str, Any]] = []
    try:
        required_duration = float(audio_duration)
    except (TypeError, ValueError) as exc:
        raise ofox.OFoxError("OFox audio duration must be a finite number") from exc
    if not math.isfinite(required_duration):
        raise ofox.OFoxError("OFox audio duration must be a finite number")
    if required_duration <= 0:
        logger.warning(
            "skip OFox paid generation because required audio duration is "
            f"not positive: duration={required_duration}"
        )
        pkg._persist_material_sources(task_id, material_sources)
        return video_paths
    try:
        clip_duration = int(max_clip_duration)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ofox.OFoxError("OFox clip duration must be a positive integer") from exc
    if clip_duration <= 0:
        raise ofox.OFoxError("OFox clip duration must be a positive integer")
    total_duration = 0.0
    for search_term in search_terms:
        try:
            video_items = ofox.generate_videos(
                search_term=search_term,
                minimum_duration=clip_duration,
                video_aspect=video_aspect,
            )
        except ofox.OFoxUnconfirmedTaskError as exc:
            logger.error(
                "stop submitting new OFox tasks because the last paid task "
                f"is unconfirmed: task_id={exc.task_id or 'unknown'}, detail={exc}"
            )
            pkg._persist_material_sources(task_id, material_sources)
            raise
        except ofox.OFoxError as exc:
            logger.error(f"OFox generation failed before completion: {exc}")
            pkg._persist_material_sources(task_id, material_sources)
            raise
        for item in video_items:
            saved_video_path = pkg._save_generated_video_with_retry(
                item.url, material_directory, "ofox"
            )
            if not saved_video_path:
                source_info = (
                    item.source_info if isinstance(item.source_info, dict) else {}
                )
                remote_task_id = str(source_info.get("asset_id") or "").strip()
                pkg._persist_material_sources(task_id, material_sources)
                raise ofox.OFoxDownloadError(
                    "OFox generated a paid video but the result could not be "
                    f"downloaded: id={remote_task_id or 'unknown'}",
                    task_id=remote_task_id,
                )
            logger.info(f"video saved: {saved_video_path}")
            video_paths.append(saved_video_path)
            try:
                material_sources.append(pkg._material_source_record(item, saved_video_path))
            except Exception as source_error:
                logger.warning(
                    "failed to prepare generated material source record: "
                    f"provider=ofox, "
                    f"error={type(source_error).__name__}, detail={source_error}"
                )
            total_duration += min(clip_duration, item.duration)
            if total_duration >= required_duration:
                break
        if total_duration >= required_duration:
            logger.info(
                "generated OFox materials cover the required duration; stop "
                f"submitting paid tasks: generated={total_duration:.1f}s, "
                f"required={required_duration:.1f}s"
            )
            break
    logger.success(f"generated and downloaded {len(video_paths)} OFox videos")
    pkg._persist_material_sources(task_id, material_sources)
    return video_paths


def _download_videos_metaso_minimax_on_demand(
    *,
    task_id: str,
    search_terms: List[str],
    video_aspect: VideoAspect,
    audio_duration: float,
    max_clip_duration: int,
    material_directory: str,
) -> List[str]:
    """Sequential Metaso MiniMax generation; stop once audio_duration is covered."""
    from app.services import metaso_minimax
    pkg = _package()

    video_paths: List[str] = []
    material_sources: list[dict[str, Any]] = []
    try:
        required_duration = float(audio_duration)
    except (TypeError, ValueError) as exc:
        raise metaso_minimax.MetasoMiniMaxError(
            "Metaso MiniMax audio duration must be a finite number"
        ) from exc
    if not math.isfinite(required_duration):
        raise metaso_minimax.MetasoMiniMaxError(
            "Metaso MiniMax audio duration must be a finite number"
        )
    if required_duration <= 0:
        logger.warning(
            "skip Metaso MiniMax paid generation because required audio duration "
            f"is not positive: duration={required_duration}"
        )
        pkg._persist_material_sources(task_id, material_sources)
        return video_paths
    try:
        clip_duration = int(max_clip_duration)
    except (TypeError, ValueError, OverflowError) as exc:
        raise metaso_minimax.MetasoMiniMaxError(
            "Metaso MiniMax clip duration must be a positive integer"
        ) from exc
    if clip_duration <= 0:
        raise metaso_minimax.MetasoMiniMaxError(
            "Metaso MiniMax clip duration must be a positive integer"
        )
    total_duration = 0.0
    for search_term in search_terms:
        try:
            video_items = metaso_minimax.generate_videos(
                search_term=search_term,
                minimum_duration=clip_duration,
                video_aspect=video_aspect,
            )
        except metaso_minimax.MetasoMiniMaxUnconfirmedTaskError as exc:
            logger.error(
                "stop submitting new Metaso MiniMax tasks because the last paid "
                f"task is unconfirmed: task_id={exc.task_id or 'unknown'}, "
                f"detail={exc}"
            )
            pkg._persist_material_sources(task_id, material_sources)
            raise
        except metaso_minimax.MetasoMiniMaxError as exc:
            logger.error(f"Metaso MiniMax generation failed before completion: {exc}")
            pkg._persist_material_sources(task_id, material_sources)
            raise
        for item in video_items:
            saved_video_path = pkg._save_generated_video_with_retry(
                item.url, material_directory, "metaso_minimax"
            )
            if not saved_video_path:
                source_info = (
                    item.source_info if isinstance(item.source_info, dict) else {}
                )
                remote_task_id = str(source_info.get("asset_id") or "").strip()
                pkg._persist_material_sources(task_id, material_sources)
                raise metaso_minimax.MetasoMiniMaxDownloadError(
                    "Metaso MiniMax generated a paid video but the result could "
                    f"not be downloaded: id={remote_task_id or 'unknown'}",
                    task_id=remote_task_id,
                )
            logger.info(f"video saved: {saved_video_path}")
            video_paths.append(saved_video_path)
            try:
                material_sources.append(pkg._material_source_record(item, saved_video_path))
            except Exception as source_error:
                logger.warning(
                    "failed to prepare generated material source record: "
                    f"provider=metaso_minimax, error={type(source_error).__name__}, "
                    f"detail={source_error}"
                )
            total_duration += min(clip_duration, item.duration)
            if total_duration >= required_duration:
                break
        if total_duration >= required_duration:
            logger.info(
                "generated Metaso MiniMax materials cover the required duration; "
                f"stop submitting paid tasks: generated={total_duration:.1f}s, "
                f"required={required_duration:.1f}s"
            )
            break
    logger.success(f"generated and downloaded {len(video_paths)} Metaso MiniMax videos")
    pkg._persist_material_sources(task_id, material_sources)
    return video_paths


def _download_videos_openai_image_on_demand(
    *,
    task_id: str,
    search_terms: List[str],
    video_aspect: VideoAspect,
    audio_duration: float,
    max_clip_duration: int,
    material_directory: str,
) -> List[str]:
    """Per-keyword OpenAI image generation; render each to mp4 then stop on coverage."""
    pkg = _package()

    if not material_directory:
        material_directory = utils.task_dir(task_id)
    video_paths: List[str] = []
    material_sources: list[dict[str, Any]] = []
    total_duration = 0.0
    try:
        required_duration = float(audio_duration)
    except (TypeError, ValueError):
        required_duration = 0.0
    if required_duration <= 0:
        logger.warning(
            "skip openai image generation because required audio duration is "
            f"not positive: duration={audio_duration}"
        )
        pkg._persist_material_sources(task_id, material_sources)
        return video_paths
    for search_term in search_terms:
        items = pkg.generate_images_openai(
            search_term=search_term,
            minimum_duration=max_clip_duration,
            video_aspect=video_aspect,
            save_dir=material_directory,
        )
        for item in items:
            video_file = pkg._render_openai_image_video(item.url, max_clip_duration)
            if not video_file:
                continue
            logger.info(f"image material rendered: {video_file}")
            video_paths.append(video_file)
            try:
                material_sources.append(pkg._material_source_record(item, video_file))
            except Exception as source_error:
                logger.warning(
                    "failed to prepare generated material source record: "
                    f"provider=openai_image, "
                    f"error={type(source_error).__name__}, detail={source_error}"
                )
            total_duration += min(max_clip_duration, item.duration)
            if total_duration >= required_duration:
                break
        if total_duration >= required_duration:
            logger.info(
                "generated image materials cover the required duration, stop "
                f"generating more images: generated={total_duration:.1f}s, "
                f"required={required_duration:.1f}s"
            )
            break
    logger.success(f"generated and rendered {len(video_paths)} image materials")
    pkg._persist_material_sources(task_id, material_sources)
    return video_paths


def _download_videos_by_script_order(
    task_id: str,
    search_terms: List[str],
    search_videos,
    video_aspect: VideoAspect,
    audio_duration: float,
    max_clip_duration: int,
    material_directory: str,
) -> List[str]:
    """Round-robin download so the first keyword doesn't hog the timeline."""
    pkg = _package()
    logger.info("downloading videos with script-order material matching")
    candidate_groups = []
    valid_video_urls = set()
    found_duration = 0.0
    for term, video_items in _search_terms_parallel(
        search_terms=search_terms,
        search_videos=search_videos,
        minimum_duration=max_clip_duration,
        video_aspect=video_aspect,
    ):
        logger.info(f"found {len(video_items)} videos for '{term}'")
        term_items = []
        for item in video_items:
            if item.url in valid_video_urls:
                continue
            term_items.append(item)
            valid_video_urls.add(item.url)
            found_duration += item.duration
        if term_items:
            candidate_groups.append((term, term_items))
    logger.info(
        f"found total ordered video candidates: {sum(len(items) for _, items in candidate_groups)}, "
        f"required duration: {audio_duration} seconds, found duration: {found_duration} seconds"
    )
    video_paths = []
    material_sources: list[dict[str, Any]] = []
    total_duration = 0.0
    candidate_index = 0
    while candidate_groups and total_duration <= audio_duration:
        items_this_round: list[tuple[str, MaterialInfo]] = []
        for search_term, term_items in candidate_groups:
            if candidate_index < len(term_items):
                items_this_round.append((search_term, term_items[candidate_index]))
        if not items_this_round:
            break
        remaining = max(0.0, audio_duration - total_duration)
        items_needed = min(
            len(items_this_round),
            max(1, math.ceil(remaining / max_clip_duration)),
        )
        items_to_download = items_this_round[:items_needed]
        round_workers = min(len(items_to_download), _MATERIAL_DOWNLOAD_WORKERS)
        round_paths: list[str] = [""] * len(items_to_download)
        with ThreadPoolExecutor(
            max_workers=round_workers, thread_name_prefix="mpt-matord"
        ) as executor:
            future_to_index = {
                executor.submit(pkg.save_video, item.url, material_directory): idx
                for idx, (_, item) in enumerate(items_to_download)
            }
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    round_paths[idx] = future.result() or ""
                except Exception as exc:
                    logger.error(
                        "failed to download ordered material video: "
                        f"error={type(exc).__name__}, "
                        f"detail={_redact_request_error(exc, items_to_download[idx][1].url)}"
                    )
                    round_paths[idx] = ""
        for (search_term, item), saved_video_path in zip(
            items_to_download, round_paths
        ):
            if not saved_video_path:
                continue
            logger.info(
                f"downloaded ordered {item.provider} video for {search_term!r}: "
                f"path={saved_video_path}"
            )
            video_paths.append(saved_video_path)
            try:
                material_sources.append(
                    pkg._material_source_record(item, saved_video_path)
                )
            except Exception as source_error:
                logger.warning(
                    "failed to prepare ordered material source record: "
                    f"provider={item.provider}, "
                    f"error={type(source_error).__name__}, "
                    f"detail={source_error}"
                )
            total_duration += min(max_clip_duration, item.duration)
            if total_duration > audio_duration:
                logger.info(
                    f"total duration of downloaded videos: {total_duration} seconds, skip downloading more"
                )
                break
        candidate_index += 1
    logger.success(f"downloaded {len(video_paths)} ordered videos")
    pkg._persist_material_sources(task_id, material_sources)
    return video_paths
