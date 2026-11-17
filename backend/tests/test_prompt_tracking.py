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
from tracking.prompt_tracking import (
    hide_before_query_frame, run_prompt_tracking_job, track_in_batches, track_prompt_queries,
)
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

    def test_a_streamed_job_shows_the_frame_it_has_reached(self):
        def track_streaming(loader, *, queries, add_support_grid, on_frames):
            on_frames(1, 2)
            on_frames(2, 2)
            return np.array([[[10.0, 20.0], [11.0, 21.0]]], dtype=np.float32), np.array([[True, True]])

        with (self.tracking_environment(streaming=True) as saved,
              patch("tracking.prompt_tracking.cot.FrameChunkLoader"),
              patch.object(state.tracker, "track_streaming", side_effect=track_streaming, create=True)):
            saved.return_value = {"result_id": "track-1", "summary": {}}
            self.run_tracking()

        shown = [(call.kwargs["current"], call.kwargs["total"], call.kwargs["message"], round(call.kwargs["progress"], 3))
                 for call in self.job.enter_stage.call_args_list if call.args[0] == "tracking_points"]
        self.assertEqual(shown[:2], [
            (1, 2, "Tracked frame 1 of 2 for 1 prompt points", 0.5),
            (2, 2, "Tracked frame 2 of 2 for 1 prompt points", 1.0),
        ])
        # Nothing but the tracking moves the bar: no progress before it, full while SAM2 is moved back.
        progress = {call.args[0]: call.kwargs["progress"] for call in self.job.enter_stage.call_args_list}
        self.assertIsNone(progress["loading_tracker"])
        self.assertIsNone(progress["loading_frames"])
        self.assertEqual(progress["restoring_masker"], 1.0)

    def test_the_published_result_hides_frames_before_the_prompt_frame(self):
        state.annotations.prompts[0].frame_idx = 1  # the fake tracker reports both frames visible

        with self.tracking_environment() as saved:
            saved.return_value = {"result_id": "track-1", "summary": {}}
            self.run_tracking()

        self.assertEqual(saved.call_args.kwargs["visibility"], [[False, True]])

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


class BatchedTrackingTests(unittest.TestCase):
    def setUp(self):
        cleanup = patch("tracking.prompt_tracking.cleanup_cuda_memory")
        self.cleanup = cleanup.start()
        self.addCleanup(cleanup.stop)

    @staticmethod
    def rows(queries):
        return np.zeros((len(queries), 2, 2), dtype=np.float32), np.ones((len(queries), 2), dtype=bool)

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

    def test_running_out_of_memory_halves_the_batch_and_never_drops_the_support_grid(self):
        grids, sizes = [], []

        class Tracker:
            def prepare_video(self, video):
                return video

            def track(self, video, *, queries, add_support_grid):
                grids.append(add_support_grid)
                sizes.append(len(queries))
                if len(queries) > 2:
                    raise RuntimeError("CUDA out of memory.")
                return BatchedTrackingTests.rows(queries)

        with patch("tracking.prompt_tracking.load_tracking_video_from_current_video_state",
                   return_value=(np.zeros((2, 2, 2, 3), dtype=np.uint8), "frames")):
            tracks, visibility = track_prompt_queries(
                Mock(), Tracker(), np.zeros((4, 3), dtype=np.float32), add_support_grid=True, streaming=False,
                on_progress=Mock())

        self.assertEqual(sizes, [4, 2, 2])
        self.assertEqual(set(grids), {True})
        self.assertEqual((tracks.shape, visibility.shape), ((4, 2, 2), (4, 2)))

    def test_a_streamed_batch_reports_frames_together_with_the_points_already_done(self):
        frames, batches = [], []

        class Tracker:
            device, model_resolution = "cpu", (2, 2)

            def track_streaming(self, loader, *, queries, add_support_grid, on_frames):
                for done in (8, 16):
                    on_frames(done, 16)
                return BatchedTrackingTests.rows(queries)

        session = Mock(video_dir="frames", video_frame_files=["0.jpg", "1.jpg"])
        with (patch("tracking.prompt_tracking.cot.FrameChunkLoader"),
              patch("tracking.prompt_tracking.DEFAULT_PROMPT_TRACK_BATCH_SIZE", 2)):
            track_prompt_queries(
                session, Tracker(), np.zeros((3, 3), dtype=np.float32), add_support_grid=True, streaming=True,
                on_progress=lambda *args: batches.append(args), on_frames=lambda *args: frames.append(args))

        # (points done before this batch, points in this batch, frames done, frames in the video)
        self.assertEqual(frames, [(0, 2, 8, 16), (0, 2, 16, 16), (2, 1, 8, 16), (2, 1, 16, 16)])
        self.assertEqual(batches, [(2, 3), (3, 3)])

    def test_a_point_is_not_visible_before_the_frame_it_was_placed_on(self):
        visibility = np.ones((3, 6), dtype=bool)
        hidden = hide_before_query_frame(visibility, [[0.0, 1.0, 1.0], [2.0, 5.0, 5.0], [6.0, 9.0, 9.0]])

        self.assertEqual(hidden.tolist(), [
            [True] * 6,
            [False, False, True, True, True, True],
            [False] * 6,
        ])
        self.assertTrue(visibility.all(), "the caller's array is left alone")


if __name__ == "__main__":
    unittest.main()
