"""Search-result caching and parallel search orchestration.

The three stock providers (Pexels, Pixabay, Coverr) all flow through
``_search_videos_with_cache`` so a 24-hour material cache short-circuits
duplicate API calls. ``_search_terms_parallel`` fans one search per
keyword out across a small thread pool so multi-keyword tasks don't
serialize I/O-bound HTTP requests.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, List

from loguru import logger

from app.models.schema import MaterialInfo, VideoAspect
from app.services import material_cache
from app.services.material._shared import (
    _MATERIAL_SEARCH_WORKERS,
    _filter_materials_by_aspect,
)


def _search_videos_with_cache(
    provider: str,
    search_videos: Callable[..., List[MaterialInfo]],
    search_term: str,
    minimum_duration: int,
    video_aspect: VideoAspect,
) -> List[MaterialInfo]:
    """Wrap a stock provider's search with a 24h cache + per-key lock."""
    cache_args = {
        "provider": provider,
        "search_term": search_term,
        "minimum_duration": minimum_duration,
        "video_aspect": video_aspect,
    }

    def load_cache_safely() -> List[MaterialInfo] | None:
        try:
            return material_cache.load_material_search_cache(**cache_args)
        except Exception as exc:
            logger.warning(
                "material search cache read failed, continue with remote search: "
                f"provider={provider}, error={type(exc).__name__}, detail={exc}"
            )
            return None

    def load_matching_cache() -> tuple[List[MaterialInfo] | None, int]:
        cached_items = load_cache_safely()
        if cached_items is None:
            return None, 0
        filtered_cached_items = _filter_materials_by_aspect(
            cached_items,
            video_aspect,
        )
        ignored_count = len(cached_items) - len(filtered_cached_items)
        if ignored_count:
            return None, ignored_count
        return filtered_cached_items, 0

    cached_items, ignored_count = load_matching_cache()
    if cached_items is not None:
        return cached_items
    if ignored_count:
        logger.info(
            "material search cache contains mismatched orientations, "
            f"refresh from provider: provider={provider}, term={search_term!r}, "
            f"ignored={ignored_count}"
        )

    cache_lock = material_cache.get_material_search_cache_lock(**cache_args)
    with cache_lock:
        cached_items, _ = load_matching_cache()
        if cached_items is not None:
            return cached_items
        items = search_videos(
            search_term=search_term,
            minimum_duration=minimum_duration,
            video_aspect=video_aspect,
        )
        for item in items:
            if isinstance(item.source_info, dict):
                item.source_info = dict(item.source_info)
                item.source_info["search_term"] = search_term
        if items:
            try:
                material_cache.save_material_search_cache(
                    **cache_args,
                    items=items,
                )
            except Exception as exc:
                logger.warning(
                    "material search cache write failed, use remote results: "
                    f"provider={provider}, error={type(exc).__name__}, detail={exc}"
                )
        return items


def _search_terms_parallel(
    search_terms: List[str],
    search_videos: Callable[..., List[MaterialInfo]],
    minimum_duration: int,
    video_aspect: VideoAspect,
) -> List[tuple[str, List[MaterialInfo]]]:
    """Run per-keyword stock searches concurrently; return (term, items) pairs."""
    if not search_terms:
        return []
    workers = min(len(search_terms), _MATERIAL_SEARCH_WORKERS)
    results: list[tuple[str, List[MaterialInfo]]] = []
    with ThreadPoolExecutor(
        max_workers=workers, thread_name_prefix="mpt-search"
    ) as executor:
        futures = {
            executor.submit(
                search_videos,
                search_term=term,
                minimum_duration=minimum_duration,
                video_aspect=video_aspect,
            ): term
            for term in search_terms
        }
        for future in as_completed(futures):
            term = futures[future]
            try:
                results.append((term, future.result()))
            except Exception as exc:
                logger.warning(
                    f"search failed for {term!r}: {type(exc).__name__}: {exc}"
                )
                results.append((term, []))
    return results
