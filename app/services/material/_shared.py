"""Cross-cutting helpers for the material services.

Holds everything the source modules and the dispatcher reuse: API key
rotation, TLS verification, aspect-ratio filtering, secret redaction,
material source records, persistence into task artifacts, the
download-with-retry helper, and the on-demand generic runner that
the five AI provider runners delegate to.
"""

from __future__ import annotations

import base64
import io
import math
import os
import random
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, List
from urllib.parse import quote_plus, urlsplit, urlunsplit

import requests
from loguru import logger
from moviepy.video.io.VideoFileClip import VideoFileClip
from PIL import Image, UnidentifiedImageError

from app.config import config
from app.models.schema import MaterialInfo, VideoAspect
from app.services import (
    material_cache,
    metaso_minimax,
    ofox,
    task_artifacts,
    volcengine_seedance,
)
from app.utils import utils


# API key rotation: shared by every provider that consumes an api_key.
_api_key_counter = 0
_api_key_lock = threading.Lock()


def get_api_key(cfg_key: str):
    api_keys = config.app.get(cfg_key)
    if not api_keys:
        raise ValueError(
            f"\n\n##### {cfg_key} is not set #####\n\n"
            f"Please set it in the config.toml file: {config.config_file}\n"
        )

    if isinstance(api_keys, str):
        return api_keys

    global _api_key_counter
    with _api_key_lock:
        _api_key_counter += 1
        return api_keys[_api_key_counter % len(api_keys)]


class _OpenAIImageDecodeError(ValueError):
    """Raised when an OpenAI-compatible endpoint returns bytes that PIL cannot decode."""


def _safe_public_url(value: Any) -> str | None:
    """Keep only publicly displayable HTTP(S) page URLs and strip query params + credentials."""
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = urlsplit(value.strip())
    except ValueError:
        return None
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
    ):
        return None
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))


def _creator_info(value: Any) -> dict[str, str] | None:
    """Extract a unified public creator block from different provider shapes."""
    if isinstance(value, str) and value.strip():
        return {"name": value.strip()}
    if not isinstance(value, dict):
        return None
    creator: dict[str, str] = {}
    creator_id = value.get("id")
    creator_name = value.get("name") or value.get("username")
    creator_page = _safe_public_url(
        value.get("url") or value.get("profile_url") or value.get("profile_page")
    )
    if creator_id is not None:
        creator["id"] = str(creator_id)
    if creator_name:
        creator["name"] = str(creator_name)
    if creator_page:
        creator["profile_page"] = creator_page
    return creator or None


def _material_source_record(item: MaterialInfo, local_path: str) -> dict[str, Any]:
    """Build a minimal, allow-listed source record for a downloaded material."""
    source = item.source_info if isinstance(item.source_info, dict) else {}
    record: dict[str, Any] = {
        "provider": str(item.provider or source.get("provider") or ""),
        "local_file": Path(local_path).name,
        "duration": int(item.duration),
    }
    search_term = source.get("search_term")
    asset_id = source.get("asset_id")
    source_page = _safe_public_url(source.get("source_page"))
    if isinstance(search_term, str) and search_term.strip():
        record["search_term"] = search_term.strip()
    if asset_id not in (None, ""):
        record["asset_id"] = str(asset_id)
    if source_page:
        record["source_page"] = source_page
    creator = _creator_info(source.get("creator"))
    if creator:
        record["creator"] = creator
    raw_rendition = source.get("rendition")
    if isinstance(raw_rendition, dict):
        rendition = {}
        for field in ("id", "width", "height"):
            value = raw_rendition.get(field)
            if value not in (None, ""):
                rendition[field] = str(value) if field == "id" else value
        if rendition:
            record["rendition"] = rendition
    return record


def _persist_material_sources(
    task_id: str,
    material_sources: list[dict[str, Any]],
) -> None:
    """Best-effort append of successful material sources to the task record."""
    import sys

    pkg = sys.modules["app.services.material"]
    try:
        saved = pkg.task_artifacts.patch_script_data(
            task_id,
            material_sources=material_sources,
        )
        if saved:
            logger.info(
                f"saved material source records: "
                f"task_id={task_id}, count={len(material_sources)}"
            )
    except Exception as exc:
        logger.warning(
            "failed to persist material source records: "
            f"task_id={task_id}, error={type(exc).__name__}, detail={exc}"
        )


def _get_tls_verify() -> bool:
    tls_verify = config.app.get("tls_verify", True)
    if isinstance(tls_verify, str):
        tls_verify = tls_verify.strip().lower() not in ("0", "false", "no", "off")
    if not tls_verify:
        logger.warning(
            "TLS certificate verification is disabled by config.app.tls_verify=false. "
            "Only use this in trusted proxy environments."
        )
    return bool(tls_verify)


def _redact_secret(message: str, secret: str) -> str:
    """Strip a secret (and its URL-encoded form) from a log message."""
    safe_message = str(message)
    if not secret:
        return safe_message
    safe_message = safe_message.replace(secret, "***")
    encoded_secret = quote_plus(secret)
    if encoded_secret != secret:
        safe_message = safe_message.replace(encoded_secret, "***")
    return safe_message


def _redact_request_error(error: Exception, *secrets: str) -> str:
    """Redact both API keys and proxy credentials from a request error message."""
    safe_message = str(error)
    for secret in secrets:
        safe_message = _redact_secret(safe_message, str(secret or ""))
    for proxy_url in config.proxy.values():
        safe_message = _redact_secret(safe_message, str(proxy_url))
    return safe_message


def _matches_video_aspect(
    width: Any,
    height: Any,
    video_aspect: VideoAspect,
    *,
    is_vertical: Any = None,
) -> bool:
    """Decide whether a (width, height) tuple matches the target aspect, with an is_vertical fallback."""
    aspect = VideoAspect(video_aspect)
    try:
        normalized_width = int(float(width))
        normalized_height = int(float(height))
    except (TypeError, ValueError):
        normalized_width = 0
        normalized_height = 0
    if normalized_width > 0 and normalized_height > 0:
        if aspect == VideoAspect.portrait:
            return normalized_height > normalized_width
        if aspect == VideoAspect.landscape:
            return normalized_width > normalized_height
        return normalized_width == normalized_height
    if isinstance(is_vertical, bool) and aspect != VideoAspect.square:
        return is_vertical == (aspect == VideoAspect.portrait)
    return False


def _filter_materials_by_aspect(
    items: List[MaterialInfo],
    video_aspect: VideoAspect,
) -> List[MaterialInfo]:
    """Drop cached items that don't match the requested aspect (legacy caches may include mixed orientations)."""
    aspect = VideoAspect(video_aspect)
    if aspect == VideoAspect.square:
        return list(items)
    filtered_items = []
    for item in items:
        source_info = item.source_info if isinstance(item.source_info, dict) else {}
        rendition = source_info.get("rendition")
        rendition = rendition if isinstance(rendition, dict) else {}
        if _matches_video_aspect(
            rendition.get("width"),
            rendition.get("height"),
            aspect,
        ):
            filtered_items.append(item)
    return filtered_items


# Stock-API material searches are independent HTTP GETs against the same
# provider. They used to run serially; with N keywords at ~0.5s per call
# that's N*0.5s of wall clock wasted. 4 workers stays below the typical
# per-host rate limit; raise when an operator knows their provider allows
# more concurrency. ponytail: small pool, bump with operator-supplied evidence.
_MATERIAL_SEARCH_WORKERS = 4
# Stock-API material downloads were historically serial: a 5-keyword task
# made 5+ HTTP GETs one at a time. 3 workers stay below the typical per-host
# rate limit; raise when an operator knows their provider allows more
# concurrency. ponytail: small pool, bump with operator-supplied evidence.
_MATERIAL_DOWNLOAD_WORKERS = 3


def _save_material_item(item: MaterialInfo, save_dir: str):
    """Wrap save_video so a failure logs once and the executor can stay dumb."""
    import sys

    pkg = sys.modules["app.services.material"]
    try:
        path = pkg.save_video(video_url=item.url, save_dir=save_dir)
        return item, path
    except Exception as exc:
        logger.error(
            "failed to download material video: "
            f"provider={item.provider}, error={type(exc).__name__}, "
            f"detail={_redact_request_error(exc, item.url)}"
        )
        return item, ""


def _download_candidate_pool(
    *,
    task_id: str,
    candidate_pool: List[MaterialInfo],
    audio_duration: float,
    max_clip_duration: int,
    material_directory: str,
) -> List[str]:
    """Download a filtered stock pool in parallel until audio_duration is met."""
    import sys

    pkg = sys.modules["app.services.material"]
    if not candidate_pool:
        logger.success("downloaded 0 videos")
        pkg._persist_material_sources(task_id, [])
        return []
    n_clips_needed = max(1, math.ceil(audio_duration / max_clip_duration))
    capped_pool = candidate_pool[
        : n_clips_needed * 2 + _MATERIAL_DOWNLOAD_WORKERS
    ]
    logger.info(
        "downloading material pool in parallel: "
        f"candidates={len(capped_pool)}, "
        f"workers={_MATERIAL_DOWNLOAD_WORKERS}"
    )
    video_paths: List[str] = []
    material_sources: list[dict[str, Any]] = []
    total_duration = 0.0
    with ThreadPoolExecutor(
        max_workers=_MATERIAL_DOWNLOAD_WORKERS,
        thread_name_prefix="mpt-mat",
    ) as executor:
        future_to_item = {
            executor.submit(pkg._save_material_item, item, material_directory): item
            for item in capped_pool
        }
        for future in as_completed(future_to_item):
            item = future_to_item[future]
            try:
                _, saved_video_path = future.result()
            except Exception:
                continue
            if not saved_video_path:
                continue
            logger.info(f"video saved: {saved_video_path}")
            video_paths.append(saved_video_path)
            try:
                material_sources.append(
                    pkg._material_source_record(item, saved_video_path)
                )
            except Exception as source_error:
                logger.warning(
                    "failed to prepare material source record: "
                    f"provider={item.provider}, "
                    f"error={type(source_error).__name__}, detail={source_error}"
                )
            seconds = min(max_clip_duration, item.duration)
            total_duration += seconds
            if total_duration > audio_duration:
                logger.info(
                    f"total duration of downloaded videos: {total_duration} seconds, skip downloading more"
                )
                break
    logger.success(f"downloaded {len(video_paths)} videos")
    pkg._persist_material_sources(task_id, material_sources)
    return video_paths


# Concrete save_video + retry helper shared by every AI source. The retry
# caps are defined next to the WaveSpeed constants so the same module
# owns its retry knobs; the implementation lives here so all providers
# share it.
WAVESPEED_MAX_DOWNLOAD_RETRIES = 2
WAVESPEED_RETRY_BASE_SECONDS = 1.0


def _save_generated_video_with_retry(
    video_url: str, save_dir: str, provider: str
) -> str:
    """Download a paid generation with bounded retries on the same URL."""
    for attempt in range(WAVESPEED_MAX_DOWNLOAD_RETRIES + 1):
        try:
            saved_video_path = _package_save_video(video_url=video_url, save_dir=save_dir)
            if saved_video_path:
                return saved_video_path
            failure_detail = "empty result"
        except Exception as e:
            failure_detail = (
                f"error={type(e).__name__}, "
                f"detail={_redact_request_error(e, video_url)}"
            )
        if attempt >= WAVESPEED_MAX_DOWNLOAD_RETRIES:
            break
        delay = WAVESPEED_RETRY_BASE_SECONDS * (attempt + 1)
        logger.warning(
            "failed to download generated video, retry the same url: "
            f"provider={provider}, "
            f"attempt={attempt + 1}/{WAVESPEED_MAX_DOWNLOAD_RETRIES}, "
            f"{failure_detail}, retry_in={delay:.1f}s"
        )
        time.sleep(delay)
    logger.error(
        "failed to download generated video after "
        f"{WAVESPEED_MAX_DOWNLOAD_RETRIES + 1} attempts: "
        f"provider={provider}, {failure_detail}"
    )
    return ""


def save_video(video_url: str, save_dir: str = "") -> str:
    """Download a video to disk under a content-hashed filename and validate it with MoviePy."""
    import sys

    pkg = sys.modules["app.services.material"]
    if not save_dir:
        save_dir = utils.storage_dir("cache_videos")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    url_without_query = video_url.split("?")[0]
    url_hash = utils.md5(url_without_query)
    video_id = f"vid-{url_hash}"
    video_path = f"{save_dir}/{video_id}.mp4"
    if os.path.exists(video_path) and os.path.getsize(video_path) > 0:
        logger.info(f"video already exists: {video_path}")
        return video_path
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
    }
    with open(video_path, "wb") as f:
        f.write(
            pkg.requests.get(
                video_url,
                headers=headers,
                proxies=config.proxy,
                verify=_get_tls_verify(),
                timeout=(60, 240),
            ).content
        )
    if os.path.exists(video_path) and os.path.getsize(video_path) > 0:
        clip = None
        try:
            clip = pkg.VideoFileClip(video_path)
            duration = clip.duration
            fps = clip.fps
            if duration > 0 and fps > 0:
                return video_path
        except Exception as e:
            logger.warning(f"invalid video file: {video_path} => {str(e)}")
            try:
                os.remove(video_path)
            except Exception as remove_error:
                logger.warning(
                    f"failed to remove invalid video file: {video_path}, error: {str(remove_error)}"
                )
        finally:
            if clip is not None:
                try:
                    clip.close()
                except Exception as close_error:
                    logger.warning(
                        f"failed to close video clip: {video_path}, error: {str(close_error)}"
                    )
    return ""


def _package_save_video(video_url: str, save_dir: str = "") -> str:
    """Call the package-level ``save_video`` so monkey-patches intercept retries.

    Lives next to ``_save_generated_video_with_retry`` so the retry loop
    sees the same ``save_video`` binding callers can patch on
    ``app.services.material``.
    """
    import sys

    return sys.modules["app.services.material"].save_video(video_url, save_dir)


def _is_cloudflare_challenge(response: requests.Response) -> bool:
    """Detect Cloudflare's HTML challenge page so callers don't treat it as JSON."""
    headers = getattr(response, "headers", {}) or {}
    if str(headers.get("cf-mitigated", "")).lower() == "challenge":
        return True
    content_type = str(headers.get("content-type", "")).lower()
    if "text/html" not in content_type:
        return False
    body = str(getattr(response, "text", "")).lower()
    return "just a moment" in body or "/cdn-cgi/challenge-platform/" in body


# OpenAI image constants are imported by the openai_image module; defined
# here so every source module that does HTTP can share the retry knobs.
OPENAI_IMAGE_ENDPOINT_PATH = "images/generations"
OPENAI_IMAGE_DEFAULT_SIZES = {
    VideoAspect.portrait: "1024x1536",
    VideoAspect.landscape: "1536x1024",
    VideoAspect.square: "1024x1024",
}
OPENAI_IMAGE_RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
OPENAI_IMAGE_KEY_ERROR_STATUS_CODES = frozenset({401, 403})
OPENAI_IMAGE_MAX_ATTEMPTS = 3
OPENAI_IMAGE_RETRY_BACKOFF_SECONDS = (5, 15, 30)
OPENAI_IMAGE_REQUEST_TIMEOUT = (30, 300)
OPENAI_IMAGE_MAX_DOWNLOAD_ATTEMPTS = 3
OPENAI_IMAGE_DOWNLOAD_BACKOFF_SECONDS = 2
