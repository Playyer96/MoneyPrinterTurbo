"""视频素材缓存的统计、预览和清理服务。"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from typing import Iterator

from loguru import logger

from app.utils import utils


# 在线素材使用 URL 的 MD5 作为稳定文件名。缓存管理只接受该命名格式，避免把
# 用户误放到目录中的视频、说明文件或其它业务文件当作缓存删除。
_VIDEO_CACHE_FILE_PATTERN = re.compile(r"^vid-[0-9a-f]{32}\.mp4$")


@dataclass(frozen=True)
class VideoCacheStats:
    """缓存目录的轻量统计结果，只包含文件系统元数据。"""

    file_count: int = 0
    total_size: int = 0
    oldest_mtime: float | None = None
    newest_mtime: float | None = None


def collect_cache_stats(root: str) -> VideoCacheStats:
    """扫描缓存目录并返回最旧/最新文件的修改时间。
    使用 ``os.path.getmtime`` 取秒级精度。
    """
    oldest: float | None = None
    newest: float | None = None
    total_size = 0
    count = 0
    for path in _iter_cache_files(root):
        mtime = os.path.getmtime(path)
        size = os.path.getsize(path)
        count += 1
        total_size += size
        if oldest is None or mtime < oldest:
            oldest = mtime
        if newest is None or mtime > newest:
            newest = mtime
    logger.info(f"扫描完成，共发现 {count} 个缓存文件")
    return VideoCacheStats(
        file_count=count,
        total_size=total_size,
        oldest_mtime=oldest,
        newest_mtime=newest,
    )