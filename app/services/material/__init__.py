"""Material services — split into provider modules.

Public surface re-exported here so every existing import path
(``from app.services import material`` / ``material.search_videos_pexels``
etc.) keeps working. The provider modules live under
``app.services.material.sources``; the dispatcher lives in
``app.services.material.dispatch``; the search cache lives in
``app.services.material.cache``; shared helpers live in
``app.services.material._shared``.

Test patching note: ``patch.object(material, "search_videos_pexels", ...)``
swaps the binding in this package's namespace, so the dispatcher
fetches the provider functions through ``sys.modules["app.services.material"]``
at call time and sees the patched version. The private names that tests
also reach for (``_search_videos_with_cache``, ``_save_material_item``,
``_download_candidate_pool``, ``_download_videos_*_on_demand``,
``_material_source_record``, ``_persist_material_sources``, ``_redact_secret``,
``_redact_request_error``, ``_safe_public_url``, ``_creator_info``,
``_get_tls_verify``, ``_matches_video_aspect``, ``_filter_materials_by_aspect``,
``_is_cloudflare_challenge``, ``_save_generated_video_with_retry``,
``_render_openai_image_video``, ``_OpenAIImageDecodeError``,
``_MATERIAL_DOWNLOAD_WORKERS``) are re-exported below for the same reason.
"""

from __future__ import annotations

import time

import requests
from loguru import logger
from moviepy.video.io.VideoFileClip import VideoFileClip

from app.models.schema import MaterialInfo, VideoAspect
from app.services import material_cache, task_artifacts, video
from app.services.material._shared import (
    OPENAI_IMAGE_DEFAULT_SIZES,
    OPENAI_IMAGE_DOWNLOAD_BACKOFF_SECONDS,
    OPENAI_IMAGE_ENDPOINT_PATH,
    OPENAI_IMAGE_KEY_ERROR_STATUS_CODES,
    OPENAI_IMAGE_MAX_ATTEMPTS,
    OPENAI_IMAGE_MAX_DOWNLOAD_ATTEMPTS,
    OPENAI_IMAGE_RETRY_BACKOFF_SECONDS,
    OPENAI_IMAGE_RETRYABLE_STATUS_CODES,
    OPENAI_IMAGE_REQUEST_TIMEOUT,
    WAVESPEED_MAX_DOWNLOAD_RETRIES,
    WAVESPEED_RETRY_BASE_SECONDS,
    _MATERIAL_DOWNLOAD_WORKERS,
    _MATERIAL_SEARCH_WORKERS,
    _OpenAIImageDecodeError,
    _creator_info,
    _download_candidate_pool,
    _filter_materials_by_aspect,
    _get_tls_verify,
    _is_cloudflare_challenge,
    _matches_video_aspect,
    _material_source_record,
    _persist_material_sources,
    _redact_request_error,
    _redact_secret,
    _safe_public_url,
    _save_generated_video_with_retry,
    _save_material_item,
    get_api_key,
    save_video,
)
from app.services.material.sources.wavespeed import (
    WAVESPEED_MAX_POLL_RETRIES,
    WAVESPEED_RUN_TIMEOUT_SECONDS,
)
from app.services.material.cache import (
    _search_terms_parallel,
    _search_videos_with_cache,
)
from app.services.material.dispatch import (
    _download_videos_by_script_order,
    _download_videos_metaso_minimax_on_demand,
    _download_videos_ofox_on_demand,
    _download_videos_openai_image_on_demand,
    _download_videos_seedance_on_demand,
    _download_videos_wavespeed_on_demand,
    download_videos,
)
from app.services.material.sources.coverr import search_videos_coverr
from app.services.material.sources.openai_image import (
    _render_openai_image_video,
    generate_images_openai,
    is_openai_image_enabled,
)
from app.services.material.sources.pexels import search_videos_pexels
from app.services.material.sources.pixabay import search_videos_pixabay
from app.services.material.sources.wavespeed import (
    WaveSpeedUnconfirmedTaskError,
    generate_videos_wavespeed,
)


__all__ = [
    # public API
    "get_api_key",
    "save_video",
    "is_openai_image_enabled",
    "download_videos",
    "search_videos_pexels",
    "search_videos_pixabay",
    "search_videos_coverr",
    "generate_videos_wavespeed",
    "generate_images_openai",
    "WaveSpeedUnconfirmedTaskError",
    "MaterialInfo",
    "VideoAspect",
    "VideoFileClip",
    "WAVESPEED_MAX_POLL_RETRIES",
    "WAVESPEED_RUN_TIMEOUT_SECONDS",
    "WAVESPEED_MAX_DOWNLOAD_RETRIES",
    "WAVESPEED_RETRY_BASE_SECONDS",
    "OPENAI_IMAGE_ENDPOINT_PATH",
    "OPENAI_IMAGE_DEFAULT_SIZES",
    "OPENAI_IMAGE_RETRYABLE_STATUS_CODES",
    "OPENAI_IMAGE_KEY_ERROR_STATUS_CODES",
    "OPENAI_IMAGE_MAX_ATTEMPTS",
    "OPENAI_IMAGE_RETRY_BACKOFF_SECONDS",
    "OPENAI_IMAGE_REQUEST_TIMEOUT",
    "OPENAI_IMAGE_MAX_DOWNLOAD_ATTEMPTS",
    "OPENAI_IMAGE_DOWNLOAD_BACKOFF_SECONDS",
    "logger",
    "material_cache",
    "requests",
    "task_artifacts",
    "time",
    # internal helpers and constants used by tests or sibling modules
    "_OpenAIImageDecodeError",
    "_search_videos_with_cache",
    "_search_terms_parallel",
    "_save_material_item",
    "_download_candidate_pool",
    "_download_videos_by_script_order",
    "_download_videos_wavespeed_on_demand",
    "_download_videos_seedance_on_demand",
    "_download_videos_ofox_on_demand",
    "_download_videos_metaso_minimax_on_demand",
    "_download_videos_openai_image_on_demand",
    "_render_openai_image_video",
    "_save_generated_video_with_retry",
    "_material_source_record",
    "_persist_material_sources",
    "_redact_request_error",
    "_redact_secret",
    "_safe_public_url",
    "_creator_info",
    "_get_tls_verify",
    "_matches_video_aspect",
    "_filter_materials_by_aspect",
    "_is_cloudflare_challenge",
    "_MATERIAL_DOWNLOAD_WORKERS",
    "_MATERIAL_SEARCH_WORKERS",
]
