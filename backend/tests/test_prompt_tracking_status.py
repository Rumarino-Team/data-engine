import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.state import state
from tracking.prompt_tracking import run_prompt_tracking_job
from video.prompts import restore_video_masker_from_prompt_events


class _FakeInteractiveMasker:
    online_mode = True
    default_batch_size = 32
    offload_video_to_cpu = True
    offload_state_to_cpu = True


class _FakeTracker:
    model_name = "cotracker3_online"

    def track(self, video, *, queries, add_support_grid):
        return np.array([[[10.0, 20.0], [11.0, 21.0]]], dtype=np.float32), np.array([[True, True]])


class _FakeRestoreMasker:
    def init_state(self, *args, progress_callback=None, **kwargs):
        if progress_callback is not None:
            progress_callback(
                "loading_sam2_model",
                "Loading SAM2 model",
                0.25,
                "Loading SAM2 model weights",
            )

    def add_new_points_or_box(self, *args, **kwargs):
        return None


class PromptTrackingStatusTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        state.active_session_dir = Path(self.tmp.name)
        state.active_session_id = "session-1"
        state.video_dir = str(Path(self.tmp.name))
        state.video_frame_files = ["00000.jpg", "00001.jpg"]
        state.video_prompt_events = [
            {
                "frame_idx": 0,
                "obj_id": 1,
                "points": [[10.0, 20.0]],
                "labels": [1],
                "box": None,
                "clear_old_points": True,
            }
        ]
        state.video_masker = _FakeInteractiveMasker()
        state.tracker = _FakeTracker()
        state.video_state_epoch = 1

    def tearDown(self):
        state.active_session_dir = None
        state.active_session_id = None
        state.video_dir = None
        state.video_frame_files = []
        state.video_prompt_events = []
        state.video_masker = None
        state.tracker = None
        self.tmp.cleanup()

    def test_prompt_tracking_uses_tracking_labels_and_restore_overrides(self):
        updates = []
        restore_kwargs = []

        def capture_update(**kwargs):
            updates.append(kwargs)

        def fake_restore(**kwargs):
            restore_kwargs.append(kwargs)

        with (
            patch("tracking.prompt_tracking.update_job", side_effect=capture_update),
            patch("tracking.prompt_tracking.ensure_tracker_model", return_value=None),
            patch("tracking.prompt_tracking.should_stream_tracking", return_value=False),
            patch(
                "tracking.prompt_tracking.load_tracking_video_from_current_video_state",
                return_value=(np.zeros((2, 2, 2, 3), dtype=np.uint8), str(self.tmp.name)),
            ),
            patch("tracking.prompt_tracking.cleanup_cuda_memory", return_value=None),
            patch("tracking.prompt_tracking.restore_video_masker_from_prompt_events", side_effect=fake_restore),
            patch(
                "tracking.prompt_tracking.save_prompt_tracking_result",
                return_value={"result_id": "track-1", "summary": {}},
            ),
        ):
            result = run_prompt_tracking_job(type("Request", (), {"add_support_grid": False})())

        labels = [entry.get("stage_label") for entry in updates]
        self.assertIn("Loading tracker", labels)
        self.assertIn("Preparing tracking frames", labels)
        self.assertIn("Tracking prompt points", labels)
        self.assertIn("Restoring interactive masking state", labels)
        self.assertNotIn("Loading state.tracker", labels)
        self.assertNotIn("Loading frames", labels)
        self.assertNotIn("Restoring masking state", labels)
        self.assertTrue(restore_kwargs)
        self.assertEqual(restore_kwargs[-1]["progress_stage_override"], "restoring_masker")
        self.assertEqual(
            restore_kwargs[-1]["progress_label_override"],
            "Restoring interactive masking state",
        )
        self.assertEqual(result["tracking_result_id"], "track-1")

    def test_restore_progress_override_hides_sam2_labels(self):
        updates = []

        with (
            patch("video.prompts.update_job", side_effect=lambda **kwargs: updates.append(kwargs)),
            patch("video.prompts.svm.SAM2VideoMasker", return_value=_FakeRestoreMasker()),
        ):
            state.video_masker = None
            restore_video_masker_from_prompt_events(
                online_mode=True,
                batch_size=32,
                offload_video_to_cpu=True,
                offload_state_to_cpu=True,
                increment_epoch=False,
                progress_stage_override="restoring_masker",
                progress_label_override="Restoring interactive masking state",
                progress_message_prefix="Restoring masking state after prompt tracking",
            )

        self.assertEqual(updates[0]["stage"], "restoring_masker")
        self.assertEqual(updates[0]["stage_label"], "Restoring interactive masking state")
        self.assertNotEqual(updates[0]["stage_label"], "Loading SAM2 model")
        self.assertIn("Restoring masking state after prompt tracking", updates[0]["message"])

    def test_restore_progress_default_passes_through_labels(self):
        updates = []

        with (
            patch("video.prompts.update_job", side_effect=lambda **kwargs: updates.append(kwargs)),
            patch("video.prompts.svm.SAM2VideoMasker", return_value=_FakeRestoreMasker()),
        ):
            state.video_masker = None
            restore_video_masker_from_prompt_events(
                online_mode=True,
                batch_size=32,
                offload_video_to_cpu=True,
                offload_state_to_cpu=True,
                increment_epoch=False,
            )

        self.assertEqual(updates[0]["stage"], "loading_sam2_model")
        self.assertEqual(updates[0]["stage_label"], "Loading SAM2 model")


if __name__ == "__main__":
    unittest.main()
