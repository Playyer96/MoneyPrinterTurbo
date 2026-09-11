import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import requests

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.services.upload_post import UploadPostService


_CONFIG_BASE = {
    "upload_post_enabled": True,
    "upload_post_api_key": "test-key",
    "upload_post_username": "testuser",
    "upload_post_platforms": ["tiktok", "instagram", "youtube"],
    "upload_post_auto_upload": True,
    "upload_post_youtube_privacy_status": "unlisted",
}


def _mock_response(success=True, status_code=200):
    r = MagicMock()
    r.json.return_value = {"success": success, "request_id": "abc123"}
    r.raise_for_status = MagicMock()
    r.status_code = status_code
    return r


def _get(data, key):
    for k, v in data:
        if k == key:
            return v
    return None


def _get_all(data, key):
    return [v for k, v in data if k == key]


def _has_key(data, key):
    return any(k == key for k, v in data)


class TestUploadPostService(unittest.TestCase):
    @patch(
        "app.services.upload_post.config.app",
        {**_CONFIG_BASE, "upload_post_enabled": False},
    )
    @patch("app.services.upload_post.requests.post")
    def test_unconfigured_service_skips_request(self, mock_post):
        """功能未启用时不能意外上传文件或消耗第三方 API 配额。"""
        result = UploadPostService().upload_video("/fake/v.mp4", "Title")

        self.assertFalse(result["success"])
        self.assertIn("not configured", result["error"])
        mock_post.assert_not_called()

    @patch("app.services.upload_post.config.app", _CONFIG_BASE)
    @patch("app.services.upload_post.os.path.exists", return_value=False)
    @patch("app.services.upload_post.requests.post")
    def test_missing_video_skips_request(self, mock_post, _exists):
        """本地成片不存在时应在发起网络请求前返回明确错误。"""
        result = UploadPostService().upload_video("/missing/v.mp4", "Title")

        self.assertFalse(result["success"])
        self.assertIn("Video file not found", result["error"])
        mock_post.assert_not_called()

    @patch("app.services.upload_post.config.app", _CONFIG_BASE)
    @patch("app.services.upload_post.os.path.exists", return_value=True)
    @patch("builtins.open", mock_open(read_data=b"fake"))
    @patch("app.services.upload_post.requests.post")
    def test_upload_request_error_returns_failure(self, mock_post, _exists):
        """网络异常需要转换为稳定结果，不能让发布失败中断视频生成任务。"""
        mock_post.side_effect = requests.exceptions.Timeout("upload timed out")

        result = UploadPostService().upload_video("/fake/v.mp4", "Title")

        self.assertFalse(result["success"])
        self.assertIn("upload timed out", result["error"])

    @patch("app.services.upload_post.config.app", _CONFIG_BASE)
    @patch("app.services.upload_post.requests.get")
    def test_check_status_returns_payload_or_network_failure(self, mock_get):
        """状态查询成功和失败应使用与上传接口一致的返回约定。"""
        response = _mock_response()
        response.json.return_value = {"success": True, "status": "processing"}
        mock_get.return_value = response
        service = UploadPostService()

        self.assertEqual(
            service.check_status("request-123"),
            {"success": True, "status": "processing"},
        )

        mock_get.side_effect = requests.exceptions.ConnectionError("offline")
        failed = service.check_status("request-123")
        self.assertFalse(failed["success"])
        self.assertIn("offline", failed["error"])


class TestUploadPostYouTubePayload(unittest.TestCase):
    @patch("app.services.upload_post.config.app", _CONFIG_BASE)
    @patch("app.services.upload_post.os.path.exists", return_value=True)
    @patch("builtins.open", mock_open(read_data=b"fake"))
    @patch("app.services.upload_post.requests.post")
    def test_youtube_fields_en_payload(self, mock_post, _exists):
        mock_post.return_value = _mock_response()
        svc = UploadPostService()

        svc.upload_video("/fake/v.mp4", "Título", youtube_extra={
            "youtube_title": "Mi Short",
            "youtube_description": "Descripción",
            "tags": ["ia", "shorts"],
            "privacyStatus": "unlisted",
        })

        data = mock_post.call_args[1]["data"]
        self.assertEqual(_get(data, "youtube_title"), "Mi Short")
        self.assertEqual(_get(data, "youtube_description"), "Descripción")
        self.assertEqual(_get_all(data, "tags[]"), ["ia", "shorts"])
        self.assertEqual(_get(data, "privacyStatus"), "unlisted")
        self.assertEqual(_get(data, "containsSyntheticMedia"), "true")

    @patch("app.services.upload_post.config.app", _CONFIG_BASE)
    @patch("app.services.upload_post.os.path.exists", return_value=True)
    @patch("builtins.open", mock_open(read_data=b"fake"))
    @patch("app.services.upload_post.requests.post")
    def test_contains_synthetic_media_siempre_true(self, mock_post, _exists):
        mock_post.return_value = _mock_response()
        svc = UploadPostService()

        svc.upload_video("/fake/v.mp4", "T", youtube_extra={"containsSyntheticMedia": False})

        data = mock_post.call_args[1]["data"]
        self.assertEqual(_get(data, "containsSyntheticMedia"), "true")

    @patch("app.services.upload_post.config.app", {
        **_CONFIG_BASE,
        "upload_post_platforms": ["tiktok", "instagram"],
    })
    @patch("app.services.upload_post.os.path.exists", return_value=True)
    @patch("builtins.open", mock_open(read_data=b"fake"))
    @patch("app.services.upload_post.requests.post")
    def test_tiktok_instagram_sin_youtube_fields(self, mock_post, _exists):
        mock_post.return_value = _mock_response()
        svc = UploadPostService()
        svc.upload_video("/fake/v.mp4", "T")

        data = mock_post.call_args[1]["data"]
        self.assertFalse(_has_key(data, "youtube_title"))
        self.assertFalse(_has_key(data, "containsSyntheticMedia"))
        self.assertFalse(_has_key(data, "privacyStatus"))

    @patch("app.services.upload_post.config.app", {
        **_CONFIG_BASE,
        "upload_post_platforms": ["tiktok"],
    })
    @patch("app.services.upload_post.os.path.exists", return_value=True)
    @patch("builtins.open", mock_open(read_data=b"fake"))
    @patch("app.services.upload_post.requests.post")
    def test_youtube_extra_ignorado_si_youtube_no_en_platforms(self, mock_post, _exists):
        mock_post.return_value = _mock_response()
        svc = UploadPostService()
        svc.upload_video("/fake/v.mp4", "T", youtube_extra={"youtube_title": "irrelevante"})

        data = mock_post.call_args[1]["data"]
        self.assertFalse(_has_key(data, "youtube_title"))

    @patch("app.services.upload_post.config.app", _CONFIG_BASE)
    @patch("app.services.upload_post.os.path.exists", return_value=True)
    @patch("builtins.open", mock_open(read_data=b"fake"))
    @patch("app.services.upload_post.requests.post")
    def test_endpoint_y_platform_format_correcto(self, mock_post, _exists):
        mock_post.return_value = _mock_response()
        svc = UploadPostService()
        svc.upload_video("/fake/v.mp4", "T")

        call_url = mock_post.call_args[0][0]
        self.assertTrue(call_url.endswith("/api/upload"), f"Endpoint incorrecto: {call_url}")

        data = mock_post.call_args[1]["data"]
        platforms = _get_all(data, "platform[]")
        self.assertIn("tiktok", platforms)
        self.assertIn("instagram", platforms)
        self.assertIn("youtube", platforms)


class TestUploadPostRetry(unittest.TestCase):
    """Verify the retry+backoff path around the upload POST."""

    @patch("app.services.upload_post.config.app", _CONFIG_BASE)
    @patch("app.services.upload_post.os.path.exists", return_value=True)
    @patch("builtins.open", mock_open(read_data=b"fake"))
    @patch("app.services.upload_post.requests.post")
    @patch("app.services.upload_post.time.sleep")
    def test_upload_retries_on_5xx_then_succeeds(self, sleep, mock_post, _exists):
        bad = _mock_response(success=False, status_code=503)
        good = _mock_response(success=True, status_code=200)
        mock_post.side_effect = [bad, bad, good]

        # 1s base × 2^attempt keeps sleep call_count verifiable; sleep is mocked
        # so the test does not actually wait.
        with patch.object(UploadPostService, "max_attempts", 3):
            with patch.object(UploadPostService, "retry_base_seconds", 1):
                result = UploadPostService().upload_video("/fake/v.mp4", "Title")

        self.assertEqual(mock_post.call_count, 3)
        self.assertTrue(result["success"])
        # Two backoff sleeps between the three attempts: 1s and 2s.
        self.assertEqual(sleep.call_count, 2)
        self.assertEqual(sleep.call_args_list[0].args, (1,))
        self.assertEqual(sleep.call_args_list[1].args, (2,))

    @patch("app.services.upload_post.config.app", _CONFIG_BASE)
    @patch("app.services.upload_post.os.path.exists", return_value=True)
    @patch("builtins.open", mock_open(read_data=b"fake"))
    @patch("app.services.upload_post.requests.post")
    @patch("app.services.upload_post.time.sleep")
    def test_upload_does_not_retry_on_4xx(self, sleep, mock_post, _exists):
        bad = _mock_response(success=False, status_code=400)
        mock_post.return_value = bad

        with patch.object(UploadPostService, "max_attempts", 3):
            with patch.object(UploadPostService, "retry_base_seconds", 1):
                # raise_for_status is a no-op MagicMock, so HTTPError path stays open.
                bad.raise_for_status.side_effect = requests.exceptions.HTTPError("400")
                result = UploadPostService().upload_video("/fake/v.mp4", "Title")

        self.assertEqual(mock_post.call_count, 1)
        sleep.assert_not_called()
        self.assertFalse(result["success"])

    @patch("app.services.upload_post.config.app", _CONFIG_BASE)
    @patch("app.services.upload_post.os.path.exists", return_value=True)
    @patch("builtins.open", mock_open(read_data=b"fake"))
    @patch("app.services.upload_post.requests.post")
    @patch("app.services.upload_post.time.sleep")
    def test_upload_retries_on_request_exception(self, sleep, mock_post, _exists):
        mock_post.side_effect = [
            requests.exceptions.ConnectionError("offline"),
            _mock_response(success=True, status_code=200),
        ]

        with patch.object(UploadPostService, "max_attempts", 3):
            with patch.object(UploadPostService, "retry_base_seconds", 1):
                result = UploadPostService().upload_video("/fake/v.mp4", "Title")

        self.assertEqual(mock_post.call_count, 2)
        self.assertEqual(sleep.call_count, 1)
        self.assertEqual(sleep.call_args.args, (1,))
        self.assertTrue(result["success"])


class TestUploadPostPollStatus(unittest.TestCase):
    @staticmethod
    def _status_response(payload, status_code=200):
        r = MagicMock()
        r.json.return_value = payload
        r.raise_for_status = MagicMock()
        r.status_code = status_code
        return r

    @patch("app.services.upload_post.config.app", _CONFIG_BASE)
    @patch("app.services.upload_post.requests.get")
    @patch("app.services.upload_post.time.sleep")
    def test_poll_status_succeeds_after_two_attempts(self, sleep, mock_get):
        mock_get.side_effect = [
            self._status_response({"success": False, "status": "processing"}),
            self._status_response({"success": True, "platform_statuses": {"tiktok": "ok"}}),
        ]

        with patch.object(UploadPostService, "poll_interval_seconds", 0):
            with patch.object(UploadPostService, "poll_timeout_seconds", 30):
                result = UploadPostService().poll_status("req-1")

        self.assertTrue(result["success"])
        self.assertEqual(result["platform_statuses"], {"tiktok": "ok"})
        sleep.assert_called_once()

    @patch("app.services.upload_post.config.app", _CONFIG_BASE)
    @patch("app.services.upload_post.requests.get")
    @patch("app.services.upload_post.time.sleep")
    def test_poll_status_times_out_when_never_terminal(self, sleep, mock_get):
        mock_get.return_value = self._status_response(
            {"success": False, "status": "processing"}
        )

        with patch.object(UploadPostService, "poll_interval_seconds", 0):
            with patch.object(UploadPostService, "poll_timeout_seconds", 0.1):
                with patch("app.services.upload_post.time.monotonic", side_effect=[0.0, 0.05, 0.2]):
                    result = UploadPostService().poll_status("req-1")

        self.assertFalse(result["success"])
        self.assertIn("did not finish", result["error"])


class TestUploadPostTestConnection(unittest.TestCase):
    @patch(
        "app.services.upload_post.config.app",
        {**_CONFIG_BASE, "upload_post_api_key": "", "upload_post_username": ""},
    )
    def test_test_connection_returns_unconfigured_when_keys_missing(self):
        result = UploadPostService().test_connection()
        self.assertFalse(result["configured"])
        self.assertFalse(result["valid"])

    @patch("app.services.upload_post.config.app", _CONFIG_BASE)
    @patch("app.services.upload_post.requests.get")
    def test_test_connection_marks_invalid_on_401(self, mock_get):
        r = MagicMock()
        r.json.return_value = {"success": False, "error": "401 unauthorized"}
        r.raise_for_status = MagicMock()
        r.status_code = 401
        mock_get.return_value = r

        result = UploadPostService().test_connection()
        self.assertTrue(result["configured"])
        self.assertFalse(result["valid"])

    @patch("app.services.upload_post.config.app", _CONFIG_BASE)
    @patch("app.services.upload_post.requests.get")
    def test_test_connection_marks_valid_on_unknown_request_id(self, mock_get):
        r = MagicMock()
        r.json.return_value = {"success": False, "error": "request not found"}
        r.raise_for_status = MagicMock()
        r.status_code = 404
        mock_get.return_value = r

        result = UploadPostService().test_connection()
        self.assertTrue(result["configured"])
        self.assertTrue(result["valid"])


class TestUploadPostPayloadHelpers(unittest.TestCase):
    def test_base_fields_include_user_title_privacy_and_platforms(self):
        data = UploadPostService._base_fields(
            "user", "title", "PUBLIC_TO_EVERYONE", ["tiktok", "instagram"],
        )
        keys = [k for k, _ in data]
        self.assertIn("user", keys)
        self.assertIn("title", keys)
        self.assertIn("privacy_level", keys)
        self.assertEqual(_get_all(data, "platform[]"), ["tiktok", "instagram"])

    def test_youtube_extra_fields_always_emit_synthetic_media_true(self):
        data = UploadPostService._youtube_extra_fields(
            {"youtube_title": "T", "tags": ["a", "b"]},
        )
        self.assertEqual(_get(data, "youtube_title"), "T")
        self.assertEqual(_get_all(data, "tags[]"), ["a", "b"])
        self.assertEqual(_get(data, "containsSyntheticMedia"), "true")
        self.assertEqual(_get(data, "privacyStatus"), "public")


if __name__ == "__main__":
    unittest.main()


class TestUploadPostServiceDynamicConfig(unittest.TestCase):
    def test_upload_post_service_dynamically_reads_config(self):
        test_app_config = {
            "upload_post_api_key": "",
            "upload_post_username": "",
            "upload_post_enabled": False,
            "upload_post_auto_upload": False,
            "upload_post_platforms": ["tiktok"],
        }
        
        with patch("app.services.upload_post.config.app", test_app_config):
            service = UploadPostService()
            self.assertFalse(service.is_configured())
            self.assertFalse(service.enabled)
            self.assertFalse(service.auto_upload)
            
            test_app_config["upload_post_enabled"] = True
            test_app_config["upload_post_auto_upload"] = True
            test_app_config["upload_post_api_key"] = "test-key"
            test_app_config["upload_post_username"] = "test-user"
            test_app_config["upload_post_platforms"] = ["tiktok", "instagram"]
            
            self.assertTrue(service.enabled)
            self.assertTrue(service.auto_upload)
            self.assertEqual(service.api_key, "test-key")
            self.assertTrue(service.is_configured())
            self.assertIn("instagram", service.platforms)
            
            test_app_config["upload_post_enabled"] = False
            self.assertFalse(service.enabled)
            self.assertFalse(service.is_configured())
