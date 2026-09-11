"""Tests for the cross-post publish / status / test-connection HTTP endpoints."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from app import asgi
from app.controllers.v1 import video as video_controller
from app.models import const
from app.services import state as sm


class TestPublishEndpoint(unittest.TestCase):
    def _request(self, task_id="pub-task"):
        return SimpleNamespace(headers={"x-task-id": "request-pub"})

    def test_publish_returns_404_when_task_unknown(self):
        # is_configured must be true first or the controller returns 400 before
        # checking the task id.
        client = TestClient(asgi.app)
        with patch.object(
            video_controller.upload_post_service.upload_post_service,
            "is_configured",
            return_value=True,
        ):
            response = client.post(
                "/api/v1/videos/missing-task/publish",
                json={"platforms": ["tiktok"]},
            )
        self.assertEqual(response.status_code, 404)
        body = response.json()
        self.assertIn("task not found", body["message"])

    def test_publish_returns_202_when_scheduled(self):
        sm.state.update_task(
            "pub-task",
            state=const.TASK_STATE_COMPLETE,
            progress=100,
            videos=["/tmp/final-1.mp4"],
            script="A coffee story.",
        )

        with patch.object(
            video_controller.tm,
            "schedule_manual_cross_post",
            return_value=(True, None, 202),
        ) as schedule:
            response = TestClient(asgi.app).post(
                "/api/v1/videos/pub-task/publish",
                json={"platforms": ["tiktok"]},
            )

        self.assertEqual(response.status_code, 202)
        self.assertEqual(
            response.json()["data"]["cross_post_state"],
            const.CROSS_POST_STATE_PENDING,
        )
        schedule.assert_called_once()

    def test_publish_returns_409_when_already_active(self):
        sm.state.update_task(
            "pub-conflict",
            state=const.TASK_STATE_COMPLETE,
            progress=100,
            videos=["/tmp/final-1.mp4"],
        )

        with patch.object(
            video_controller.tm,
            "schedule_manual_cross_post",
            return_value=(False, "cross-post is already active for this task", 409),
        ):
            response = TestClient(asgi.app).post(
                "/api/v1/videos/pub-conflict/publish",
                json={"platforms": ["tiktok"], "force": False},
            )

        self.assertEqual(response.status_code, 409)
        self.assertIn("already active", response.json()["message"])

    def test_publish_returns_400_when_not_configured(self):
        with patch.object(
            video_controller.tm,
            "schedule_manual_cross_post",
            return_value=(False, "Upload-Post is not configured", 400),
        ):
            response = TestClient(asgi.app).post(
                "/api/v1/videos/whatever/publish",
                json={},
            )

        self.assertEqual(response.status_code, 400)
        self.assertIn("not configured", response.json()["message"])


class TestCrossPostStatusEndpoint(unittest.TestCase):
    def test_returns_state_payload_when_task_known(self):
        sm.state.update_task(
            "status-task",
            state=const.TASK_STATE_COMPLETE,
            cross_post_state=const.CROSS_POST_STATE_PROCESSING,
            cross_post_results=[{"success": True, "request_id": "abc"}],
            cross_post_error=None,
        )

        response = TestClient(asgi.app).get("/api/v1/videos/status-task/cross-post")

        self.assertEqual(response.status_code, 200)
        data = response.json()["data"]
        self.assertEqual(data["task_id"], "status-task")
        self.assertEqual(data["cross_post_state"], const.CROSS_POST_STATE_PROCESSING)
        self.assertEqual(
            data["cross_post_results"], [{"success": True, "request_id": "abc"}]
        )

    def test_returns_404_when_task_unknown(self):
        response = TestClient(asgi.app).get("/api/v1/videos/nope/cross-post")
        self.assertEqual(response.status_code, 404)


class TestUploadPostTestEndpoint(unittest.TestCase):
    def test_returns_test_connection_payload(self):
        with patch.object(
            video_controller.upload_post_service.upload_post_service,
            "test_connection",
            return_value={
                "configured": True,
                "valid": True,
                "platforms": ["tiktok", "instagram"],
            },
        ):
            response = TestClient(asgi.app).get("/api/v1/upload-post/test")

        self.assertEqual(response.status_code, 200)
        data = response.json()["data"]
        self.assertTrue(data["configured"])
        self.assertTrue(data["valid"])
        self.assertEqual(data["platforms"], ["tiktok", "instagram"])

    def test_unconfigured_returns_invalid(self):
        with patch.object(
            video_controller.upload_post_service.upload_post_service,
            "test_connection",
            return_value={"configured": False, "valid": False, "platforms": []},
        ):
            response = TestClient(asgi.app).get("/api/v1/upload-post/test")

        self.assertEqual(response.status_code, 200)
        data = response.json()["data"]
        self.assertFalse(data["configured"])


if __name__ == "__main__":
    unittest.main()
