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
from support import SessionStateTestCase
from core.runtime import is_out_of_memory
from tracking.prompt_tracking import run_prompt_tracking_job, track_in_batches, track_prompt_queries
from tracking.runtime import VideoTrackerRuntime
from video.runtime import VideoMaskerRuntime


class _FakeInteractiveMasker:
    online_mode = True
    default_batch_size = 32
    offload_video_to_cpu = True
    offload_state_to_cpu = True

    def __init__(self):
        self.offload_weights = Mock()
        self.reload_weights = Mock()


class _FakeTracker:
    model_name = "cotracker3_online"
    device = "cpu"
    model_resolution = (2, 2)

    def prepare_video(self, video):
        return video

    def track(self, video, *, queries, add_support_grid):
        return np.array([[[10.0, 20.0], [11.0, 21.0]]], dtype=np.float32), np.array([[True, True]])



class PromptTrackingStatusTests(SessionStateTestCase):
    def setUp(self):
        super().setUp()
        self.tmp_dir = str(self.temp_dir())
        self.working_session(
            Path(self.tmp_dir) / "session", session_id="session-1", frames=("00000.jpg", "00001.jpg"),
            video_dir=self.tmp_dir, tracker=_FakeTracker(),
            video_masker=_FakeInteractiveMasker(),
            annotations=SessionAnnotations(
                objects={1: {"id": 1, "label": "Object 1", "color": "#ff0000"}},
                prompts=[{"frame_idx": 0, "obj_id": 1, "points": [[10.0, 20.0]], "labels": [1], "box": None}]),
        )
        self.job = self.job_handle()
        self.masker = state.video_masker
        self.factory = Mock()
        self.replay = Mock()
        self.tracker_runtime = VideoTrackerRuntime(
            session=state, create_tracker=Mock(return_value=state.tracker), release_memory=Mock(),
        )
        self.runtime = VideoMaskerRuntime(
            session=state, create_masker=self.factory, replay_inputs=self.replay, release_memory=Mock(),
        )

    def run_tracking(self):
        return run_prompt_tracking_job(
            self.ctx, type("Request", (), {"add_support_grid": False})(), self.job,
            masker_runtime=self.runtime, tracker_runtime=self.tracker_runtime,
        )

    @contextmanager
    def tracking_environment(self, *, streaming=False, load_video=True):
        """Patch the tracking job's collaborators; yields the mock that publishes a result."""
        with ExitStack() as stack:
            stack.enter_context(patch("tracking.prompt_tracking.should_stream_tracking", return_value=streaming))
            if load_video:
                stack.enter_context(patch(
                    "tracking.prompt_tracking.load_tracking_video_from_current_video_state",
                    return_value=(np.zeros((2, 2, 2, 3), dtype=np.uint8), self.tmp_dir)))
            stack.enter_context(patch("tracking.prompt_tracking.cleanup_cuda_memory"))
            yield stack.enter_context(patch("tracking.prompt_tracking.save_prompt_tracking_result"))

    def assert_masker_untouched_but_for_its_weights(self):
        """Tracking never rebuilds the predictor: same masker, no new model, no replay."""
        self.assertIs(state.video_masker, self.masker)
        self.factory.assert_not_called()
        self.replay.assert_not_called()

    def test_prompt_tracking_moves_sam2_weights_aside_and_back_before_publishing(self):
        events = []
        self.masker.offload_weights.side_effect = lambda: events.append("offload")
        self.masker.reload_weights.side_effect = lambda: events.append("reload")

        def track(*args, **kwargs):
            events.append("track")
            return np.array([[[10.0, 20.0], [11.0, 21.0]]], dtype=np.float32), np.array([[True, True]])

        def publish(*args, **kwargs):
            events.append("publish")
            return {"result_id": "track-1", "summary": {}}

        with self.tracking_environment() as saved, patch.object(state.tracker, "track", side_effect=track):
            saved.side_effect = publish
            result = self.run_tracking()

        self.assertEqual(events, ["offload", "track", "reload", "publish"])
        labels = [call.args[1] for call in self.job.enter_stage.call_args_list]
        self.assertIn("Loading tracker", labels)
        self.assertIn("Preparing tracking frames", labels)
        self.assertIn("Tracking prompt points", labels)
        self.assertIn("Restoring interactive masking state", labels)
        self.assert_masker_untouched_but_for_its_weights()
        self.assertEqual(result["tracking_result_id"], "track-1")

    def test_weights_that_cannot_return_disable_the_predictor_without_publishing(self):
        self.masker.reload_weights.side_effect = RuntimeError("restore failed")
        with self.tracking_environment() as publish, self.assertLogs("video.runtime", level="ERROR"):
            with self.assertRaises(OperationError) as failure:
                self.run_tracking()
        self.assertEqual(failure.exception.status_code, 503)
        publish.assert_not_called()
        self.assertIsNone(state.video_masker)

    def test_tracker_loading_failure_puts_the_weights_back_without_publishing(self):
        state.tracker = None
        self.tracker_runtime.create_tracker.side_effect = ValueError("unsupported tracker")
        with patch("tracking.prompt_tracking.save_prompt_tracking_result") as publish:
            with self.assertRaises(OperationError) as failure:
                self.run_tracking()

        self.assertEqual(failure.exception.status_code, 400)
        self.masker.reload_weights.assert_called_once()
        self.assert_masker_untouched_but_for_its_weights()
        publish.assert_not_called()

    def test_computation_errors_keep_their_status_when_the_weights_cannot_return(self):
        for error, status in (
            (ValueError("invalid query"), 400),
            (RuntimeError("tracking failed"), 500),
            (torch.OutOfMemoryError("out of memory"), 507),
        ):
            with self.subTest(error=type(error).__name__):
                state.video_masker = _FakeInteractiveMasker()
                state.video_masker.reload_weights.side_effect = RuntimeError("restore failed")
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
                publish.assert_not_called()

    def test_inputs_are_revalidated_after_tracking_before_publication(self):
        def edit_during_tracking(*args, **kwargs):
            state.annotations.clear_prompts()
            return np.array([[[10.0, 20.0], [11.0, 21.0]]], dtype=np.float32), np.array([[True, True]])

        with self.tracking_environment() as publish, patch.object(state.tracker, "track", side_effect=edit_during_tracking):
            with self.assertRaises(OperationError) as failure:
                self.run_tracking()

        self.assertEqual(failure.exception.status_code, 409)
        self.masker.reload_weights.assert_called_once()
        self.assert_masker_untouched_but_for_its_weights()
        publish.assert_not_called()

    def test_invalid_tracking_coverage_puts_the_weights_back_without_publishing(self):
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
                    state.video_masker = self.masker = masker
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
                        masker.reload_weights.assert_called_once()
                        self.assert_masker_untouched_but_for_its_weights()
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
            self.masker.reload_weights.assert_called_once()
            self.assert_masker_untouched_but_for_its_weights()
            self.assertEqual(state.session_metadata["latest_tracking_result_id"], "previous-result")
            self.assertEqual(state.annotations.snapshot(), before_annotations)


class BatchedTrackingTests(unittest.TestCase):
    def setUp(self):
        cleanup = patch("tracking.prompt_tracking.cleanup_cuda_memory")
        self.cleanup = cleanup.start()
        self.addCleanup(cleanup.stop)

    @staticmethod
    def rows(queries):
        return np.zeros((len(queries), 2, 2), dtype=np.float32), np.ones((len(queries), 2), dtype=bool)

    def test_out_of_memory_is_recognized_from_the_type_or_the_message(self):
        self.assertTrue(is_out_of_memory(torch.OutOfMemoryError("anything")))
        self.assertTrue(is_out_of_memory(RuntimeError("CUDA out of memory.")))
        self.assertFalse(is_out_of_memory(RuntimeError("something else")))
        self.assertFalse(is_out_of_memory(ValueError("out of memory")))

    def test_the_batch_is_halved_until_it_fits_and_every_query_is_tracked_once(self):
        sizes, progress = [], []

        def track_batch(batch):
            sizes.append(len(batch))
            if len(batch) > 2:
                raise torch.OutOfMemoryError("out of memory")
            return self.rows(batch)

        tracks, visibility = track_in_batches(
            track_batch, np.arange(24, dtype=np.float32).reshape(8, 3), on_progress=lambda *args: progress.append(args))

        self.assertEqual(sizes, [8, 4, 2, 2, 2, 2])
        self.assertEqual((tracks.shape, visibility.shape), ((8, 2, 2), (8, 2)))
        self.assertEqual(progress, [(2, 8), (4, 8), (6, 8), (8, 8)])
        self.assertEqual(self.cleanup.call_count, 2)

    def test_a_single_query_that_does_not_fit_raises_the_original_error(self):
        error = torch.OutOfMemoryError("out of memory")

        def track_batch(batch):
            raise error

        with self.assertRaises(torch.OutOfMemoryError) as caught:
            track_in_batches(track_batch, np.zeros((1, 3), dtype=np.float32), on_progress=Mock())
        self.assertIs(caught.exception, error)
        self.cleanup.assert_called_once()

    def test_other_errors_are_not_retried(self):
        track_batch = Mock(side_effect=RuntimeError("something else"))
        with self.assertRaises(RuntimeError):
            track_in_batches(track_batch, np.zeros((4, 3), dtype=np.float32), on_progress=Mock())
        track_batch.assert_called_once()
        self.cleanup.assert_not_called()

    def test_tracking_retries_once_without_the_support_grid_when_memory_runs_out(self):
        grids = []

        class Tracker:
            def prepare_video(self, video):
                return video

            def track(self, video, *, queries, add_support_grid):
                grids.append(add_support_grid)
                if add_support_grid:
                    raise RuntimeError("CUDA out of memory.")
                return BatchedTrackingTests.rows(queries)

        with patch("tracking.prompt_tracking.load_tracking_video_from_current_video_state",
                   return_value=(np.zeros((2, 2, 2, 3), dtype=np.uint8), "frames")):
            tracks, _, grid_used = track_prompt_queries(
                Mock(), Tracker(), np.zeros((1, 3), dtype=np.float32), add_support_grid=True, streaming=False,
                on_progress=Mock())

        self.assertEqual(grids, [True, False])
        self.assertFalse(grid_used)
        self.assertEqual(tracks.shape, (1, 2, 2))

    def test_without_the_support_grid_an_out_of_memory_error_is_not_retried(self):
        tracker = Mock()
        tracker.prepare_video.side_effect = lambda video: video
        tracker.track.side_effect = RuntimeError("CUDA out of memory.")
        with patch("tracking.prompt_tracking.load_tracking_video_from_current_video_state",
                   return_value=(np.zeros((2, 2, 2, 3), dtype=np.uint8), "frames")):
            with self.assertRaises(RuntimeError):
                track_prompt_queries(Mock(), tracker, np.zeros((1, 3), dtype=np.float32),
                                     add_support_grid=False, streaming=False, on_progress=Mock())
        tracker.track.assert_called_once()


if __name__ == "__main__":
    unittest.main()
