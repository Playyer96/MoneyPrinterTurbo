"""Video material cache statistics, preview, and cleanup service."""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from typing import Iterator

from loguru import logger

from app.utils import utils


# Online materials use the MD5 of the URL as a stable filename. The cache
# manager accepts only this naming format, preventing accidental deletion of
# user-placed videos, documentation, or other business files in the directory.
_VIDEO_CACHE_FILE_PATTERN = re.compile(r"^vid-[0-9a-f]{32}\.mp4$")
_SECONDS_PER_DAY = 24 * 60 * 60


@dataclass(frozen=True)
class VideoCacheStats:
    """Lightweight statistics for the cache directory, filesystem metadata only."""

    file_count: int = 0
    total_size: int = 0
    oldest_mtime_ns: int | None = None
    newest_mtime_ns: int | None = None


@dataclass(frozen=True)
class VideoCacheCleanupResult:
    """Result of a single cleanup run; partial file deletion failures are allowed."""

    deleted_count: int = 0
    deleted_size: int = 0
    failed_count: int = 0


@dataclass(frozen=True)
class _VideoCacheEntry:
    """Minimal file info captured during scan; avoids opening or parsing videos during cleanup."""

    path: str
    name: str
    size: int
    mtime_ns: int


def video_cache_dir() -> str:
    """Return the project's managed default video cache directory."""

    return os.path.realpath(utils.storage_dir("cache_videos"))


def _iter_video_cache_entries() -> Iterator[_VideoCacheEntry]:
    """
    Sequentially scan the top level of the default cache directory.

    ``os.scandir`` is used to reuse the metadata returned during directory
    traversal when the cache reaches tens of thousands of files, avoiding an
    additional file-type query after ``Path.iterdir``. No recursion, no video
    files are opened, and no FFmpeg is called, so the time cost scales with
    file count rather than total video capacity.
    """

    cache_dir = video_cache_dir()
    try:
        entries = os.scandir(cache_dir)
    except FileNotFoundError:
        return
    except OSError as exc:
        logger.warning(
            f"failed to scan video cache directory: path={cache_dir}, error={exc}"
        )
        return

    with entries:
        for entry in entries:
            if not _VIDEO_CACHE_FILE_PATTERN.fullmatch(entry.name):
                continue

            try:
                # Do not follow symlinks; prevents the cleanup logic from
                # crossing outside the default cache directory boundary.
                if not entry.is_file(follow_symlinks=False):
                    continue
                stat_result = entry.stat(follow_symlinks=False)
            except OSError as exc:
                logger.warning(
                    f"failed to inspect video cache file: file={entry.name}, error={exc}"
                )
                continue

            yield _VideoCacheEntry(
                path=entry.path,
                name=entry.name,
                size=stat_result.st_size,
                mtime_ns=stat_result.st_mtime_ns,
            )


def _is_cleanup_candidate(
    entry: _VideoCacheEntry,
    max_age_days: int | None,
    now_ns: int,
) -> bool:
    if max_age_days is None:
        return True
    return entry.mtime_ns < now_ns - max_age_days * _SECONDS_PER_DAY * 1e9


def _validate_max_age_days(max_age_days: int | None) -> None:
    """Stably reject invalid cleanup parameters even when the cache directory is empty."""
    if max_age_days is None:
        return
    if (
        isinstance(max_age_days, bool)
        or not isinstance(max_age_days, int)
        or max_age_days <= 0
    ):
        raise ValueError("max_age_days must be a positive integer or None")


def get_video_cache_stats(max_age_days: int | None = None) -> VideoCacheStats:
    """
    Return stats for all cache, or a preview of cleanable cache older than max_age_days.

    ``max_age_days=None`` means all cache. The stats process only reads directory
    entry size and modification time, never the video content, so even a very large
    total cache capacity produces I/O that is not proportional to capacity.
    """

    _validate_max_age_days(max_age_days)
    now_ns = int(time.time() * 1e9)
    file_count = 0
    total_size = 0
    oldest_mtime_ns = None
    newest_mtime_ns = None

    for entry in _iter_video_cache_entries():
        if not _is_cleanup_candidate(entry, max_age_days, now_ns):
            continue
        file_count += 1
        total_size += entry.size
        oldest_mtime_ns = (
            entry.mtime_ns if oldest_mtime_ns is None else min(oldest_mtime_ns, entry.mtime_ns)
        )
        newest_mtime_ns = (
            entry.mtime_ns if newest_mtime_ns is None else max(newest_mtime_ns, entry.mtime_ns)
        )

    return VideoCacheStats(
        file_count=file_count,
        total_size=total_size,
        oldest_mtime_ns=oldest_mtime_ns,
        newest_mtime_ns=newest_mtime_ns,
    )


def clean_video_cache(max_age_days: int | None = None) -> VideoCacheCleanupResult:
    """
    Clean the default video cache and return a summary suitable for display.

    Significant time may elapse between page preview and the actual cleanup click,
    so the run must rescan and re-evaluate rather than reusing an old candidate
    list. Deletion uses per-file fault tolerance: when a single file is locked or
    lacks permissions, log a warning and continue, preventing one bad file among
    hundreds from aborting the entire cleanup.
    """

    _validate_max_age_days(max_age_days)
    now_ns = int(time.time() * 1e9)
    logger.info(
        f"start cleaning video cache: max_age_days={max_age_days}"
    )

    candidate_count = 0
    candidate_size = 0
    deleted_count = 0
    deleted_size = 0
    failed_count = 0
    cache_dir = video_cache_dir()

    # Scan and delete in one pass; no full candidate list is held in memory.
    # Even if the directory grows to hundreds of thousands of files, the extra
    # memory used during cleanup stays constant. A single `now` is used to
    # prevent the cutoff from drifting unpredictably during a long cleanup.
    for entry in _iter_video_cache_entries():
        if not _is_cleanup_candidate(entry, max_age_days, now_ns):
            continue
        candidate_count += 1
        candidate_size += entry.size
        try:
            # entry.path comes from the top-level scandir of the default directory;
            # revalidate the parent directory and filename before deletion to prevent
            # future scan-logic changes from accidentally expanding the deletable range.
            if (
                os.path.realpath(os.path.dirname(entry.path)) != cache_dir
                or not _VIDEO_CACHE_FILE_PATTERN.fullmatch(entry.name)
                or os.path.islink(entry.path)
            ):
                raise ValueError("cache file is outside the managed directory")
            os.unlink(entry.path)
            deleted_count += 1
            deleted_size += entry.size
        except (OSError, ValueError) as exc:
            failed_count += 1
            logger.warning(
                f"failed to delete video cache file: file={entry.name}, error={exc}"
            )

    logger.info(
        "finished cleaning video cache: "
        f"candidates={candidate_count}, candidate_bytes={candidate_size}, "
        f"deleted={deleted_count}, deleted_bytes={deleted_size}, failed={failed_count}"
    )
    return VideoCacheCleanupResult(
        deleted_count=deleted_count,
        deleted_size=deleted_size,
        failed_count=failed_count,
    )
