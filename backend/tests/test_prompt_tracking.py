import unittest
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import Mock, patch
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from support import state
from core.annotations import SessionAnnotations
from core.errors import OperationError
from support import BackendStateTestCase
from tracking.prompt_tracking import run_prompt_tracking_job
from video.runtime import VideoMaskerRuntime


class _FakeInteractiveMasker:
    online_mode = True
    default_batch_size = 32
    offload_video_to_cpu = True
    offload_state_to_cpu = True


class _FakeTracker:
    model_name = "cotracker3_online"
    device = "cpu"
    model_resolution = (2, 2)

    def prepare_video(self, video):
        return video

    def track(self, video, *, queries, add_support_grid):
        return np.array([[[10.0, 20.0], [11.0, 21.0]]], dtype=np.float32), np.array([[True, True]])


class _FakeRestoreMasker:
    def init_state(self, *args, progress_callback=None, **kwargs):
        self.init_args = args
        self.init_kwargs = kwargs
        if progress_callback is not None:
            progress_callback(
                "loading_sam2_model",
                "Loading SAM2 model",
                0.25,
                "Loading SAM2 model weights",
            )

    def add_new_points_or_box(self, *args, **kwargs):
        return None


class PromptTrackingStatusTests(BackendStateTestCase):
    def setUp(self):
        super().setUp()
        self.tmp_dir = str(self.temp_dir())
        self.working_session(
            Path(self.tmp_dir) / "session", session_id="session-1", frames=("00000.jpg", "00001.jpg"),
            video_dir=self.tmp_dir, video_state_epoch=1, tracker=_FakeTracker(),
            video_masker=_FakeInteractiveMasker(),
            annotations=SessionAnnotations(
                objects={1: {"id": 1, "label": "Object 1", "color": "#ff0000"}},
                prompts=[{"frame_idx": 0, "obj_id": 1, "points": [[10.0, 20.0]], "labels": [1], "box": None}]),
        )
        self.job = self.job_handle()
        self.restored_masker = _FakeRestoreMasker()
        self.factory = Mock(return_value=self.restored_masker)
        self.replay = Mock()
        self.persist = Mock()
        self.runtime = VideoMaskerRuntime(
            session=state, create_masker=self.factory, replay_inputs=self.replay,
            persist_session=self.persist, release_memory=Mock(),
        )

    def run_tracking(self):
        return run_prompt_tracking_job(
            self.ctx, type("Request", (), {"add_support_grid": False})(), self.job, masker_runtime=self.runtime,
        )

    @contextmanager
    def tracking_environment(self, *, streaming=False, load_video=True):
        """Patch the tracking job's collaborators; yields the mock that publishes a result."""
        with ExitStack() as stack:
            stack.enter_context(patch("tracking.prompt_tracking.ensure_tracker_model"))
            stack.enter_context(patch("tracking.prompt_tracking.should_stream_tracking", return_value=streaming))
            if load_video:
                stack.enter_context(patch(
                    "tracking.prompt_tracking.load_tracking_video_from_current_video_state",
                    return_value=(np.zeros((2, 2, 2, 3), dtype=np.uint8), self.tmp_dir)))
            stack.enter_context(patch("tracking.prompt_tracking.cleanup_cuda_memory"))
            yield stack.enter_context(patch("tracking.prompt_tracking.save_prompt_tracking_result"))

    def test_prompt_tracking_uses_tracking_labels_and_restore_progress(self):
        def publish_after_restoration(*args, **kwargs):
            self.assertIs(state.video_masker, self.restored_masker)
            self.replay.assert_called_once_with(state)
            self.persist.assert_called_once_with(state)
            return {"result_id": "track-1", "summary": {}}

        with (
            patch("tracking.prompt_tracking.ensure_tracker_model", return_value=None),
            patch("tracking.prompt_tracking.should_stream_tracking", return_value=False),
            patch(
                "tracking.prompt_tracking.load_tracking_video_from_current_video_state",
                return_value=(np.zeros((2, 2, 2, 3), dtype=np.uint8), str(self.tmp_dir)),
            ),
            patch("tracking.prompt_tracking.cleanup_cuda_memory", return_value=None),
            patch(
                "tracking.prompt_tracking.save_prompt_tracking_result",
                side_effect=publish_after_restoration,
            ),
        ):
            result = self.run_tracking()

        labels = [call.args[1] for call in self.job.enter_stage.call_args_list]
        self.assertIn("Loading tracker", labels)
        self.assertIn("Preparing tracking frames", labels)
        self.assertIn("Tracking prompt points", labels)
        self.assertIn("Restoring interactive masking state", labels)
        self.assertNotIn("Loading state.tracker", labels)
        self.assertNotIn("Loading frames", labels)
        self.assertNotIn("Restoring masking state", labels)
        self.factory.assert_called_once()
        self.replay.assert_called_once_with(state)
        self.persist.assert_called_once_with(state)
        self.assertEqual(self.restored_masker.init_args, (self.tmp_dir,))
        self.assertEqual(self.restored_masker.init_kwargs["batch_size"], 32)
        self.assertEqual(result["tracking_result_id"], "track-1")

    def test_failed_restore_disables_predictor_without_publishing(self):
        self.factory.side_effect = RuntimeError("restore failed")
        with self.tracking_environment() as publish:
            with self.assertRaises(OperationError) as failure:
                self.run_tracking()
        self.assertEqual(failure.exception.status_code, 503)
        publish.assert_not_called()
        self.assertIsNone(state.video_masker)
        self.assertIsNone(state.tracking_video)

    def test_tracker_loading_failure_restores_masker_without_publishing(self):
        with (
            patch("tracking.prompt_tracking.ensure_tracker_model", side_effect=ValueError("unsupported tracker")),
            patch("tracking.prompt_tracking.save_prompt_tracking_result") as publish,
        ):
            with self.assertRaises(OperationError) as failure:
                self.run_tracking()

        self.assertEqual(failure.exception.status_code, 400)
        self.assertIs(state.video_masker, self.restored_masker)
        self.persist.assert_called_once_with(state)
        publish.assert_not_called()

    def test_computation_errors_keep_their_status_when_restoration_fails(self):
        for error, status in (
            (ValueError("invalid query"), 400),
            (RuntimeError("tracking failed"), 500),
            (torch.OutOfMemoryError("out of memory"), 507),
        ):
            with self.subTest(error=type(error).__name__):
                state.video_masker = _FakeInteractiveMasker()
                self.factory.side_effect = RuntimeError("restore failed")
                with (
                    self.tracking_environment() as publish,
                    patch.object(state.tracker, "track", side_effect=error),
                    self.assertLogs("video.runtime", level="ERROR"),
                ):
                    with self.assertRaises(OperationError) as failure:
                        self.run_tracking()
                self.assertEqual(failure.exception.status_code, status)
                self.assertIs(failure.exception.__cause__, error)
                self.assertIsNone(state.video_masker)
                self.assertIsNone(state.tracking_video)
                publish.assert_not_called()

    def test_inputs_are_revalidated_after_restoration_before_publication(self):
        self.replay.side_effect = lambda session: session.annotations.clear_prompts()
        with self.tracking_environment() as publish:
            with self.assertRaises(OperationError) as failure:
                self.run_tracking()

        self.assertEqual(failure.exception.status_code, 409)
        self.assertIs(state.video_masker, self.restored_masker)
        self.assertIsNone(state.tracking_video)
        publish.assert_not_called()

    def test_invalid_tracking_coverage_restores_masker_without_publishing(self):
        shapes = [
            ((1, 1, 2), (1, 1)),  # Missing frame.
            ((1, 3, 2), (1, 3)),  # Extra frame.
            ((1, 2, 2), (1, 1)),  # Visibility missing a frame.
            ((0, 2, 2), (0, 2)),  # Missing query.
            ((1, 2, 3), (1, 2)),  # Invalid coordinate shape.
        ]
        for streaming in (False, True):
            for tracks_shape, visibility_shape in shapes:
                with self.subTest(streaming=streaming, tracks=tracks_shape, visibility=visibility_shape):
                    masker = _FakeInteractiveMasker()
                    state.video_masker = masker
                    self.factory.reset_mock()
                    previous_metadata = state.session_metadata
                    state.session_metadata = {"latest_tracking_result_id": "previous-result"}
                    before_annotations = state.annotations.snapshot()
                    output = (np.zeros(tracks_shape, dtype=np.float32), np.ones(visibility_shape, dtype=bool))

                    try:
                        with (
                            self.tracking_environment(streaming=streaming) as publish,
                            patch.object(state.tracker, "track", return_value=output),
                            patch.object(state.tracker, "track_streaming", return_value=output, create=True),
                        ):
                            with self.assertRaises(OperationError) as failure:
                                self.run_tracking()
                        self.assertEqual(failure.exception.status_code, 500)
                        self.assertIn("Tracking returned", failure.exception.message)
                        publish.assert_not_called()
                        self.factory.assert_called_once()
                        self.assertIs(state.video_masker, self.restored_masker)
                        self.assertIsNone(state.tracking_video)
                        self.assertEqual(state.session_metadata["latest_tracking_result_id"], "previous-result")
                        self.assertEqual(state.annotations.snapshot(), before_annotations)
                    finally:
                        state.session_metadata = previous_metadata

    def test_unreadable_frame_rejects_tracking_before_inference(self):
        before_annotations = state.annotations.snapshot()

        with (
            self.tracking_environment(load_video=False) as publish,
            patch("video.io.cv2.imread", side_effect=[np.zeros((2, 2, 3), dtype=np.uint8), None]),
            patch.object(state.tracker, "track") as inference,
            patch.object(state, "session_metadata", {"latest_tracking_result_id": "previous-result"}),
        ):
            with self.assertRaises(OperationError) as failure:
                self.run_tracking()

            self.assertEqual(failure.exception.status_code, 400)
            self.assertIn("Unable to read frame 1:", failure.exception.message)
            inference.assert_not_called()
            publish.assert_not_called()
            self.factory.assert_called_once()
            self.assertIs(state.video_masker, self.restored_masker)
            self.assertIsNone(state.tracking_video)
            self.assertEqual(state.session_metadata["latest_tracking_result_id"], "previous-result")
            self.assertEqual(state.annotations.snapshot(), before_annotations)


if __name__ == "__main__":
    unittest.main()
