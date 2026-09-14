"""OpenAI-compatible text-to-image material provider.

Pay-per-image. Each call submits an image generation request, downloads
the bytes, normalizes to PNG, and returns a single MaterialInfo whose
``url`` is the local PNG path. The dispatcher hands each PNG to
``video.render_image_zoom_video`` so the rest of the pipeline treats it
like a local material clip.
"""

from __future__ import annotations

import base64
import io
import os
import time
import uuid
from typing import Any, List

import requests
from loguru import logger
from PIL import Image, UnidentifiedImageError

from app.config import config
from app.models.schema import MaterialInfo, VideoAspect
from app.services.material._shared import (
    OPENAI_IMAGE_DEFAULT_SIZES,
    OPENAI_IMAGE_DOWNLOAD_BACKOFF_SECONDS,
    OPENAI_IMAGE_ENDPOINT_PATH,
    OPENAI_IMAGE_KEY_ERROR_STATUS_CODES,
    OPENAI_IMAGE_MAX_ATTEMPTS,
    OPENAI_IMAGE_MAX_DOWNLOAD_ATTEMPTS,
    OPENAI_IMAGE_RETRYABLE_STATUS_CODES,
    OPENAI_IMAGE_RETRY_BACKOFF_SECONDS,
    OPENAI_IMAGE_REQUEST_TIMEOUT,
    _get_tls_verify,
    _redact_request_error,
    _redact_secret,
    get_api_key,
)
from app.utils import utils


class _OpenAIImageDecodeError(ValueError):
    """Raised when an OpenAI-compatible endpoint returns bytes PIL cannot decode."""


def is_openai_image_enabled(app_config: dict | None = None) -> bool:
    """Whether the OpenAI image source has the minimum required config."""
    app_config = config.app if app_config is None else app_config
    return bool(
        str(app_config.get("openai_image_base_url", "") or "").strip()
        and str(app_config.get("openai_image_model", "") or "").strip()
    )


def _openai_image_endpoint() -> tuple[str, str]:
    """Resolve the OpenAI image endpoint URL + model name from config."""
    base_url = (
        str(config.app.get("openai_image_base_url", "") or "").strip().rstrip("/")
    )
    model = str(config.app.get("openai_image_model", "") or "").strip()
    if not base_url:
        raise ValueError(
            "\n\n##### openai_image_base_url is not set #####\n\n"
            f"Please set it in the config.toml file: {config.config_file}\n"
        )
    if not model:
        raise ValueError(
            "\n\n##### openai_image_model is not set #####\n\n"
            f"Please set it in the config.toml file: {config.config_file}\n"
        )
    return f"{base_url}/{OPENAI_IMAGE_ENDPOINT_PATH}", model


def _openai_image_size(video_aspect: VideoAspect) -> str:
    """Resolve the requested image size, honouring explicit overrides."""
    configured = str(config.app.get("openai_image_size", "") or "").strip()
    if configured:
        return configured
    return OPENAI_IMAGE_DEFAULT_SIZES.get(VideoAspect(video_aspect), "1024x1024")


def _openai_image_prompt(search_term: str) -> str:
    """Apply the optional ``openai_image_prompt_template`` to the search term."""
    template = str(config.app.get("openai_image_prompt_template", "") or "").strip()
    if not template or "{term}" not in template:
        return search_term
    try:
        return template.replace("{term}", search_term)
    except Exception:
        return search_term


def _response_json_safely(response: Any) -> Any:
    """Read response.json() and return None on any parsing failure."""
    try:
        return response.json()
    except Exception:
        return None


def _openai_image_response_message(body: Any) -> str:
    """Pull a readable error message out of an OpenAI-compatible response body."""
    if not isinstance(body, dict):
        return str(body or "")[:300]
    error = body.get("error")
    if isinstance(error, dict):
        return str(error.get("message") or "")[:300]
    if error is not None:
        return str(error)[:300]
    return str(body.get("message") or "")[:300]


def _openai_image_http_failure(response: Any, status: int, api_key: str) -> str:
    """Format an HTTP error response into a redacted, loggable description."""
    message = _openai_image_response_message(_response_json_safely(response))
    if not message:
        message = str(getattr(response, "text", "") or "")[:300]
    return f"HTTP {status}: {_redact_secret(message, api_key)}"


def _openai_image_download_bytes(
    image_url: str,
    api_key: str,
) -> tuple[bytes | None, str]:
    """Download already-generated image bytes with retries on the same URL."""
    failure_detail = "no download attempt was made"
    for attempt in range(1, OPENAI_IMAGE_MAX_DOWNLOAD_ATTEMPTS + 1):
        try:
            response = requests.get(
                image_url,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/115.0.0.0 Safari/537.36"
                },
                proxies=config.proxy,
                verify=_get_tls_verify(),
                timeout=(30, 120),
            )
            if response.status_code == 200 and response.content:
                return response.content, ""
            failure_detail = f"HTTP {response.status_code} while downloading image"
        except Exception as e:
            failure_detail = (
                f"error={type(e).__name__}, detail={_redact_request_error(e, api_key)}"
            )
        if attempt < OPENAI_IMAGE_MAX_DOWNLOAD_ATTEMPTS:
            logger.warning(
                "generated image download failed, retrying the same url: "
                f"attempt={attempt}/{OPENAI_IMAGE_MAX_DOWNLOAD_ATTEMPTS}, "
                f"{failure_detail}"
            )
            time.sleep(OPENAI_IMAGE_DOWNLOAD_BACKOFF_SECONDS)
    return None, failure_detail


def _parse_openai_image_response(
    response: Any,
    api_key: str,
) -> tuple[bytes | None, str]:
    """Extract url-or-b64 image bytes from a /images/generations response."""
    body = _response_json_safely(response)
    data = body.get("data") if isinstance(body, dict) else None
    if not isinstance(data, list) or not data:
        return None, _redact_secret(_openai_image_response_message(body), api_key)
    entry = data[0]
    if not isinstance(entry, dict):
        return None, "invalid image data entry"
    b64_payload = entry.get("b64_json")
    if b64_payload:
        try:
            return base64.b64decode(b64_payload), ""
        except Exception as e:
            return None, f"invalid b64_json payload: {type(e).__name__}"
    image_url = entry.get("url")
    if isinstance(image_url, str) and image_url.startswith(("http://", "https://")):
        return _openai_image_download_bytes(image_url, api_key)
    return None, "image response has neither url nor b64_json"


def _request_openai_image(endpoint: str, payload: dict) -> tuple[bytes | None, str]:
    """POST /images/generations with backoff retries and key rotation."""
    api_keys = config.app.get("openai_image_api_keys")
    if isinstance(api_keys, (list, tuple)):
        configured_keys = [k for k in api_keys if str(k or "").strip()]
    elif str(api_keys or "").strip():
        configured_keys = [api_keys]
    else:
        configured_keys = []
    failure_detail = "no request attempt was made"
    for attempt in range(1, OPENAI_IMAGE_MAX_ATTEMPTS + 1):
        api_key = get_api_key("openai_image_api_keys") if configured_keys else ""
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        retryable = False
        try:
            response = requests.post(
                endpoint,
                json=payload,
                headers=headers,
                proxies=config.proxy,
                verify=_get_tls_verify(),
                timeout=OPENAI_IMAGE_REQUEST_TIMEOUT,
            )
        except requests.exceptions.ConnectTimeout as e:
            failure_detail = (
                f"connect timeout: detail={_redact_request_error(e, api_key)}"
            )
            retryable = True
        except Exception as e:
            failure_detail = (
                f"unconfirmed request error (no retry to avoid double billing): "
                f"{type(e).__name__}, detail={_redact_request_error(e, api_key)}"
            )
        else:
            status = int(getattr(response, "status_code", 200) or 200)
            if status in OPENAI_IMAGE_KEY_ERROR_STATUS_CODES:
                failure_detail = _openai_image_http_failure(response, status, api_key)
                retryable = len(configured_keys) > 1
            elif status in OPENAI_IMAGE_RETRYABLE_STATUS_CODES:
                failure_detail = _openai_image_http_failure(response, status, api_key)
                retryable = True
            elif status >= 400:
                return None, _openai_image_http_failure(response, status, api_key)
            else:
                image_bytes, parse_error = _parse_openai_image_response(
                    response, api_key
                )
                if image_bytes is not None:
                    return image_bytes, ""
                failure_detail = parse_error
        if retryable and attempt < OPENAI_IMAGE_MAX_ATTEMPTS:
            backoff_seconds = OPENAI_IMAGE_RETRY_BACKOFF_SECONDS[
                min(attempt - 1, len(OPENAI_IMAGE_RETRY_BACKOFF_SECONDS) - 1)
            ]
            logger.warning(
                "openai image request failed, retrying: "
                f"attempt={attempt}/{OPENAI_IMAGE_MAX_ATTEMPTS}, "
                f"next_retry_in={backoff_seconds}s, detail={failure_detail}"
            )
            time.sleep(backoff_seconds)
            continue
        return None, failure_detail
    return None, failure_detail


def _save_openai_image_file(
    image_bytes: bytes,
    save_dir: str,
) -> tuple[str, int, int]:
    """Decode image bytes, normalize to PNG, write to disk, return (path, width, height)."""
    if not save_dir:
        save_dir = utils.storage_dir("cache_images", create=True)
    elif not os.path.isdir(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    image_path = os.path.join(save_dir, f"openai-image-{uuid.uuid4().hex[:12]}.png")
    try:
        image = Image.open(io.BytesIO(image_bytes))
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise _OpenAIImageDecodeError(f"{type(exc).__name__}: {exc}") from exc
    with image:
        try:
            image.load()
        except (OSError, SyntaxError, ValueError) as exc:
            raise _OpenAIImageDecodeError(f"{type(exc).__name__}: {exc}") from exc
        if image.mode not in ("RGB", "RGBA", "L", "LA", "P"):
            image = image.convert("RGB")
        image.save(image_path, format="PNG")
        width, height = image.size
    return image_path, width, height


def generate_images_openai(
    search_term: str,
    minimum_duration: int,
    video_aspect: VideoAspect = VideoAspect.portrait,
    save_dir: str = "",
) -> List[MaterialInfo]:
    """Generate one image via the configured OpenAI-compatible endpoint."""
    import sys

    pkg = sys.modules["app.services.material"]
    aspect = VideoAspect(video_aspect)
    clip_duration = max(int(minimum_duration), 1)
    endpoint, model = _openai_image_endpoint()
    image_size = _openai_image_size(aspect)
    payload = {
        "model": model,
        "prompt": _openai_image_prompt(search_term),
        "n": 1,
        "size": image_size,
    }
    pkg.logger.info(
        f"generating image via openai-compatible endpoint: model={model}, "
        f"term={search_term!r}, size={image_size}"
    )
    image_bytes, failure_detail = _request_openai_image(endpoint, payload)
    if image_bytes is None:
        pkg.logger.error(
            f"openai image generation failed: term={search_term!r}, "
            f"detail={failure_detail}"
        )
        return []
    try:
        image_path, width, height = _save_openai_image_file(image_bytes, save_dir)
    except _OpenAIImageDecodeError as e:
        pkg.logger.error(
            "openai image response is not a decodable image, skipping term: "
            f"term={search_term!r}, error={type(e).__name__}, detail={e}"
        )
        return []
    item = MaterialInfo()
    item.provider = "openai_image"
    item.url = image_path
    item.duration = clip_duration
    item.source_info = {
        "provider": "openai_image",
        "search_term": search_term,
        "rendition": {
            "id": None,
            "width": width,
            "height": height,
        },
    }
    return [item]


def _render_openai_image_video(image_path: str, clip_duration: int) -> str:
    """Render one PNG to a slow-zoom mp4 clip via ``video.render_image_zoom_video``."""
    import sys

    pkg = sys.modules["app.services.material"]
    try:
        return pkg.video.render_image_zoom_video(image_path, clip_duration)
    except Exception as e:
        pkg.logger.error(
            "failed to render generated image as a video clip: "
            f"image={image_path}, error={type(e).__name__}, detail={e}"
        )
        return ""
