import sys
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import core.jobs as jobs
from app.factory import create_app
from core.state import state


def _active_job():
    return {
        "job_id": "active",
        "operation": "video_init",
        "status": "running",
        "stage": "working",
        "stage_label": "Working",
        "progress": 0.5,
        "current": None,
        "total": None,
        "window_index": None,
        "window_count": None,
        "frame_idx": None,
        "stage_history": [],
        "message": "working",
        "result": None,
        "error": None,
        "started_at": "now",
        "updated_at": "now",
        "completed_at": None,
    }


class OperationGuardTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(create_app())

    def tearDown(self):
        with jobs.current_job_lock:
            jobs.current_job = None
        state.video_masker = None
        state.video_frame_files = []

    def set_active_job(self):
        with jobs.current_job_lock:
            jobs.current_job = _active_job()

    def test_reset_state_returns_409_while_job_active(self):
        self.set_active_job()
        response = self.client.post("/video/reset_state")
        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["detail"], "Another operation is already running.")

    def test_add_points_returns_409_while_job_active(self):
        self.set_active_job()
        response = self.client.post(
            "/video/add_new_points_or_box",
            json={"frame_idx": 0, "obj_id": 1, "points": [[1, 2]], "labels": [1]},
        )
        self.assertEqual(response.status_code, 409)

    def test_read_only_video_info_is_not_guarded(self):
        self.set_active_job()
        state.video_frame_files = ["00000.jpg"]
        response = self.client.get("/video/info")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["num_frames"], 1)

    def test_without_active_job_endpoint_uses_normal_validation(self):
        response = self.client.post("/video/reset_state")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["error"], "Video masker not active.")


if __name__ == "__main__":
    unittest.main()
