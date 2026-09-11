"""
Upload-Post API integration for cross-posting videos to TikTok, Instagram and YouTube Shorts.

Docs: https://docs.upload-post.com
"""
import os
import time
from typing import Optional

import requests
from loguru import logger
from app.config import config


class UploadPostService:
    API_BASE = "https://api.upload-post.com"

    @property
    def api_key(self) -> str:
        return config.app.get("upload_post_api_key", "")

    @property
    def username(self) -> str:
        return config.app.get("upload_post_username", "")

    @property
    def enabled(self) -> bool:
        return config.app.get("upload_post_enabled", False)

    @property
    def platforms(self) -> list:
        return config.app.get("upload_post_platforms", ["tiktok", "instagram"])

    @property
    def auto_upload(self) -> bool:
        return config.app.get("upload_post_auto_upload", False)

    @property
    def youtube_privacy_status(self) -> str:
        return config.app.get("upload_post_youtube_privacy_status", "public")

    @property
    def max_attempts(self) -> int:
        return max(1, int(config.app.get("upload_post_max_attempts", 3)))

    @property
    def retry_base_seconds(self) -> float:
        return max(0.0, float(config.app.get("upload_post_retry_base_seconds", 5)))

    @property
    def request_timeout_seconds(self) -> int:
        return max(1, int(config.app.get("upload_post_request_timeout_seconds", 300)))

    @property
    def poll_interval_seconds(self) -> float:
        return max(1.0, float(config.app.get("upload_post_poll_interval_seconds", 10)))

    @property
    def poll_timeout_seconds(self) -> float:
        return max(0.0, float(config.app.get("upload_post_poll_timeout_seconds", 1800)))

    def is_configured(self) -> bool:
        return bool(self.api_key and self.username and self.enabled)

    @staticmethod
    def _base_fields(username: str, title: str, privacy_level: str, platforms: list) -> list:
        data = [
            ('user', username),
            ('title', title[:2200]),
            ('privacy_level', privacy_level),
        ]
        for platform in platforms:
            data.append(('platform[]', platform))
        return data

    @staticmethod
    def _youtube_extra_fields(youtube_extra: dict) -> list:
        data = []
        if "youtube_title" in youtube_extra:
            data.append(('youtube_title', youtube_extra["youtube_title"][:100]))
        if "youtube_description" in youtube_extra:
            data.append(('youtube_description', youtube_extra["youtube_description"]))
        for tag in youtube_extra.get("tags", []):
            data.append(('tags[]', tag))
        data.append(('privacyStatus', youtube_extra.get("privacyStatus", "public")))
        # Always declared true: every video this app produces is AI-narrated/
        # AI-assembled, so there is no "authentic" branch to opt out of.
        data.append(('containsSyntheticMedia', "true"))
        return data

    def _sleep_before_retry(self, attempt: int) -> None:
        delay = self.retry_base_seconds * (2 ** (attempt - 1))
        if delay > 0:
            time.sleep(delay)

    def _post_with_retry(self, url: str, headers: dict, data: list, video_file) -> requests.Response:
        """
        POST with retry+backoff, reusing the caller's open file handle.

        Retries on network errors and HTTP 429/5xx (transient); stops
        immediately on 4xx so the caller's raise_for_status() surfaces a clean
        client error without wasting attempts. The multipart file must be
        re-seeked to the start before every attempt: the previous attempt's
        upload already advanced the handle to EOF, and re-sending it as-is
        would silently upload an empty file on retry.
        """
        max_attempts = self.max_attempts
        response = None
        for attempt in range(1, max_attempts + 1):
            video_file.seek(0)
            try:
                response = requests.post(
                    url,
                    headers=headers,
                    data=data,
                    files={'video': video_file},
                    timeout=self.request_timeout_seconds,
                )
            except requests.exceptions.RequestException:
                if attempt >= max_attempts:
                    raise
                logger.warning(
                    f"Upload-Post request failed, retrying (attempt {attempt}/{max_attempts})"
                )
                self._sleep_before_retry(attempt)
                continue

            if response.status_code == 429 or response.status_code >= 500:
                if attempt >= max_attempts:
                    # Exhausted retries: hand the response back so the caller's
                    # raise_for_status() surfaces the real status/body.
                    return response
                logger.warning(
                    f"Upload-Post returned {response.status_code}, retrying "
                    f"(attempt {attempt}/{max_attempts})"
                )
                self._sleep_before_retry(attempt)
                continue

            return response

        return response  # pragma: no cover - loop always returns or raises

    def upload_video(
        self,
        video_path: str,
        title: str,
        platforms: Optional[list] = None,
        privacy_level: str = "PUBLIC_TO_EVERYONE",
        youtube_extra: Optional[dict] = None,
    ) -> dict:
        if not self.is_configured():
            logger.warning("Upload-Post is not configured. Skipping cross-post.")
            return {"success": False, "error": "Upload-Post not configured"}

        if platforms is None:
            platforms = self.platforms

        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            return {"success": False, "error": f"Video file not found: {video_path}"}

        logger.info(f"Cross-posting video to {', '.join(platforms)} via Upload-Post...")

        try:
            with open(video_path, 'rb') as video_file:
                data = self._base_fields(self.username, title, privacy_level, platforms)

                if youtube_extra and any(p.startswith("youtube") for p in platforms):
                    data.extend(self._youtube_extra_fields(youtube_extra))

                headers = {'Authorization': f'Apikey {self.api_key}'}

                response = self._post_with_retry(
                    f"{self.API_BASE}/api/upload", headers, data, video_file,
                )

                response.raise_for_status()
                result = response.json()

                if result.get('success'):
                    logger.info(f"✅ Video cross-posted successfully! Request ID: {result.get('request_id')}")
                else:
                    logger.warning(f"Cross-post failed: {result.get('message', 'Unknown error')}")

                return result

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to cross-post video: {str(e)}")
            return {"success": False, "error": str(e)}

    def check_status(self, request_id: str) -> dict:
        """
        Check the status of an upload request.

        Args:
            request_id (str): The request ID from upload

        Returns:
            dict: Status information
        """
        try:
            headers = {
                'Authorization': f'Apikey {self.api_key}'
            }

            response = requests.get(
                f"{self.API_BASE}/api/uploadposts/status",
                params={'request_id': request_id},
                headers=headers,
                timeout=30
            )

            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to check status: {str(e)}")
            return {"success": False, "error": str(e)}

    @staticmethod
    def _extract_platform_statuses(result: dict) -> dict:
        """
        Best-effort per-platform breakdown from a status response.

        Upload-Post's exact status-endpoint schema isn't pinned down in the
        published docs, so this accepts whatever dict-shaped field is present
        under a few plausible names rather than hard-coding one — the WebUI
        and API fall back to the overall success/error when this is empty.
        """
        for key in ("platform_statuses", "results", "platforms"):
            value = result.get(key)
            if isinstance(value, dict):
                return value
        return {}

    def poll_status(
        self,
        request_id: str,
        timeout: Optional[float] = None,
        interval: Optional[float] = None,
    ) -> dict:
        """
        Poll check_status until Upload-Post reports a terminal result or the
        timeout elapses.

        Upload-Post's initial /api/upload response only means "accepted";
        real per-platform publishing can take minutes. This polls until the
        status response reports success=True or failed=True, folding in a
        best-effort platform_statuses breakdown either way. Timing out still
        returns a terminal dict (success=False) so callers never block
        indefinitely on a stuck third-party job.
        """
        timeout = self.poll_timeout_seconds if timeout is None else timeout
        interval = self.poll_interval_seconds if interval is None else interval
        deadline = time.monotonic() + timeout

        result: dict = {}
        while True:
            result = self.check_status(request_id)
            if not isinstance(result, dict):
                result = {
                    "success": False,
                    "error": "Upload-Post returned an invalid status response",
                }

            if result.get("success") is True or result.get("failed") is True:
                break

            if time.monotonic() >= deadline:
                logger.warning(
                    f"Upload-Post status poll timed out, request_id: {request_id}"
                )
                result = {
                    **result,
                    "success": False,
                    "error": result.get("error")
                    or f"Upload-Post did not finish within {int(timeout)}s",
                }
                break

            time.sleep(interval)

        result["platform_statuses"] = self._extract_platform_statuses(result)
        return result

    def test_connection(self) -> dict:
        """
        Verify the configured API key authenticates, without a real upload.

        Calls check_status with a synthetic id: an auth-shaped error means the
        key itself is rejected; any other error (e.g. "request not found")
        means the key authenticated fine and only the dummy id was unknown.
        """
        if not self.is_configured():
            return {"configured": False, "valid": False, "platforms": self.platforms}

        result = self.check_status("mpt-connection-test")
        error_text = str(result.get("error") or "").lower()
        auth_failed = any(
            token in error_text
            for token in ("401", "403", "unauthorized", "invalid api key", "forbidden")
        )
        return {
            "configured": True,
            "valid": not auth_failed,
            "platforms": self.platforms,
        }


# Singleton instance
upload_post_service = UploadPostService()


def cross_post_video(
    video_path: str,
    title: str,
    platforms: Optional[list] = None,
    youtube_extra: Optional[dict] = None,
) -> dict:
    return upload_post_service.upload_video(video_path, title, platforms, youtube_extra=youtube_extra)
