"""Coverr stock video search provider."""

from __future__ import annotations

from typing import List
from urllib.parse import urlencode

import requests
from loguru import logger

from app.config import config
from app.models.schema import MaterialInfo, VideoAspect
from app.services.material._shared import (
    _creator_info,
    _get_tls_verify,
    _matches_video_aspect,
    _redact_request_error,
    _safe_public_url,
    get_api_key,
)


def search_videos_coverr(
    search_term: str,
    minimum_duration: int,
    video_aspect: VideoAspect = VideoAspect.portrait,
) -> List[MaterialInfo]:
    """Coverr (https://coverr.co) - free HD/4K stock videos.

    Coverr API notes (based on official docs at api.coverr.co/docs/):
      - auth: Authorization: Bearer <api_key>
      - search endpoint: GET /videos?query=..., response {"hits": [...], ...}
      - add ?urls=true to also return mp4 download links in the search response
      - URLs are signed JWTs (tied to API key, no expiry)
      - Coverr supports is_vertical filter, but response is still validated
        locally via max_width/max_height or is_vertical
      - duration may come as number or string; both accepted
    """
    aspect = VideoAspect(video_aspect)
    api_key = get_api_key("coverr_api_keys")
    headers = {"Authorization": f"Bearer {api_key}"}
    params = {
        "query": search_term,
        "page_size": 20,
        "urls": "true",
        "sort": "popular",
    }
    if aspect == VideoAspect.portrait:
        params["filter"] = "is_vertical:true"
    elif aspect == VideoAspect.landscape:
        params["filter"] = "is_vertical:false"
    query_url = f"https://api.coverr.co/videos?{urlencode(params)}"
    logger.info(f"searching videos on coverr: term={search_term!r}")
    try:
        r = requests.get(
            query_url,
            headers=headers,
            proxies=config.proxy,
            verify=_get_tls_verify(),
            timeout=(30, 60),
        )
        response = r.json()
        video_items: List[MaterialInfo] = []
        if not isinstance(response, dict) or "hits" not in response:
            logger.error("coverr video search returned an unsupported response")
            return video_items
        for v in response["hits"]:
            try:
                duration = int(float(v.get("duration") or 0))
            except (TypeError, ValueError):
                continue
            if duration < minimum_duration:
                continue
            video_id = v.get("id")
            mp4_download_url = (v.get("urls") or {}).get("mp4_download")
            if not video_id or not mp4_download_url:
                continue
            if aspect != VideoAspect.square and not _matches_video_aspect(
                v.get("max_width"),
                v.get("max_height"),
                aspect,
                is_vertical=v.get("is_vertical"),
            ):
                continue
            item = MaterialInfo()
            item.provider = "coverr"
            item.url = mp4_download_url
            item.duration = duration
            item.source_info = {
                "provider": "coverr",
                "search_term": search_term,
                "asset_id": str(video_id),
                "source_page": _safe_public_url(v.get("canonical_url") or v.get("url")),
                "creator": _creator_info(v.get("creator") or v.get("author")),
                "rendition": {
                    "id": "mp4_download",
                    "width": v.get("max_width"),
                    "height": v.get("max_height"),
                },
            }
            video_items.append(item)
        return video_items
    except Exception as e:
        logger.error(
            "coverr video search failed: "
            f"error={type(e).__name__}, detail={_redact_request_error(e, api_key)}"
        )
    return []
