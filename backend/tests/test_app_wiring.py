import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from support import BackendStateTestCase


class AppWiringTests(BackendStateTestCase):
    """Every route reaches its service with the application's context, including ones no other test calls."""

    def test_every_route_is_served_by_the_context_it_was_created_with(self):
        client = self.make_client(preconditions=False)
        headers = {"X-Session-ID": "absent", "X-Annotation-Revision": "0"}
        # (method, path, body, statuses a correctly wired route may answer with when nothing is loaded)
        calls = [
            ("GET", "/", None, {200}),
            ("GET", "/health", None, {200}),
            ("GET", "/status", None, {200}),
            ("GET", "/jobs/current", None, {200}),
            ("GET", "/jobs/unknown", None, {404}),
            ("POST", "/jobs/unknown/clear_result", None, {200}),
            ("POST", "/video/init_state", {"video_frames_dir": "missing"}, {200}),
            ("POST", "/video/reset_state", None, {409}),
            ("POST", "/video/add_new_points_or_box", {"frame_idx": 0, "obj_id": 1, "points": [[1, 1]], "labels": [1]}, {409}),
            ("POST", "/video/add_new_mask", {"frame_idx": 0, "obj_id": 1, "mask": [[1]]}, {409, 422}),
            ("POST", "/video/save", {"name": "x"}, {400}),
            ("POST", "/video/propagate_in_video", {}, {409}),
            ("POST", "/video/clear_all_prompts_in_frame?frame_idx=0&obj_id=1", None, {409}),
            ("POST", "/video/objects", None, {409}),
            ("PUT", "/video/objects/1", {"label": "x", "color": "#ffffff"}, {409}),
            ("POST", "/video/remove_object?obj_id=1", None, {409}),
            ("GET", "/video/info", None, {200}),
            ("GET", "/video/mask_data/0", None, {400}),
            ("GET", "/video/frame/0", None, {400}),
            ("GET", "/video/mask_frame/0", None, {404}),
            ("POST", "/tracking/load_video", {"video_path": "missing.mp4"}, {400, 404}),
            ("POST", "/tracking/track_prompt_points", {}, {409}),
            ("POST", "/tracking/track_grid", {}, {400}),
            ("POST", "/tracking/track_points", {"queries": [[0, 1, 1]]}, {400}),
            ("GET", "/tracking/results/abc", None, {404}),
        ]
        for method, path, body, expected in calls:
            with self.subTest(route=f"{method} {path}"):
                # A job started by an earlier call must not make the next one answer 409 for another reason.
                with patch.object(self.ctx.jobs, "is_active", return_value=False):
                    response = client.request(method, path, json=body, headers=headers)
                self.assertIn(response.status_code, expected, response.text)


if __name__ == "__main__":
    unittest.main()
