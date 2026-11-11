import sys
import unittest
import weakref
from pathlib import Path
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.state import BackendState
from video.runtime import MaskerRestorationError, MaskerSettings, VideoMaskerRuntime


class FakeMasker:
    online_mode = False
    default_batch_size = 7
    offload_video_to_cpu = True
    offload_state_to_cpu = False

    def __init__(self):
        self.init_state = Mock()
        self.reset_state = Mock()
        self.offload_weights = Mock()
        self.reload_weights = Mock()
        self.release_state = Mock()


class VideoMaskerRuntimeTests(unittest.TestCase):
    def make_runtime(self):
        session = BackendState(video_dir="frames", active_session_id="session", video_masker=FakeMasker())
        session.annotations.create_object()
        restored = FakeMasker()
        runtime = VideoMaskerRuntime(
            session=session, create_masker=Mock(return_value=restored), replay_inputs=Mock(),
            release_memory=Mock(),
        )
        return runtime, restored

    def test_offloaded_moves_the_weights_around_the_work_and_never_rebuilds_the_state(self):
        runtime, _ = self.make_runtime()
        session = runtime.session
        masker = session.video_masker
        events = []
        masker.offload_weights.side_effect = lambda: events.append("offload")
        runtime.release_memory.side_effect = lambda: events.append("free memory")
        masker.reload_weights.side_effect = lambda: events.append("reload")

        with runtime.offloaded():
            self.assertIs(session.video_masker, masker)
            events.append("compute")

        self.assertEqual(events, ["offload", "free memory", "compute", "reload"])
        self.assertIs(session.video_masker, masker)
        masker.init_state.assert_not_called()
        runtime.create_masker.assert_not_called()
        runtime.replay_inputs.assert_not_called()

    def test_offloaded_leaves_an_absent_masker_alone(self):
        runtime, _ = self.make_runtime()
        runtime.session.video_masker = None
        with runtime.offloaded():
            pass
        runtime.create_masker.assert_not_called()
        runtime.release_memory.assert_not_called()

    def test_a_successful_computation_requires_the_weights_to_come_back(self):
        runtime, _ = self.make_runtime()
        runtime.session.video_masker.reload_weights.side_effect = RuntimeError("cannot move back")
        with self.assertLogs("video.runtime", level="ERROR"):
            with self.assertRaises(MaskerRestorationError) as caught:
                with runtime.offloaded():
                    pass
        self.assertIsInstance(caught.exception.__cause__, RuntimeError)
        self.assertIsNone(runtime.session.video_masker)

    def test_a_failed_computation_still_returns_the_weights_and_its_error_wins(self):
        runtime, _ = self.make_runtime()
        masker = runtime.session.video_masker
        failure = ValueError("computation failed")
        with self.assertRaises(ValueError) as caught:
            with runtime.offloaded():
                raise failure
        self.assertIs(caught.exception, failure)
        masker.reload_weights.assert_called_once()
        self.assertIs(runtime.session.video_masker, masker)

    def test_a_failed_return_after_a_failed_computation_keeps_the_original_error(self):
        runtime, _ = self.make_runtime()
        runtime.session.video_masker.reload_weights.side_effect = RuntimeError("cannot move back")
        failure = ValueError("computation failed")
        with self.assertLogs("video.runtime", level="ERROR"):
            with self.assertRaises(ValueError) as caught:
                with runtime.offloaded():
                    raise failure
        self.assertIs(caught.exception, failure)
        self.assertIsNone(runtime.session.video_masker)

    def test_weights_that_cannot_be_moved_off_are_put_back_and_the_error_is_raised(self):
        runtime, _ = self.make_runtime()
        masker = runtime.session.video_masker
        masker.offload_weights.side_effect = RuntimeError("cannot move off")
        with self.assertRaises(RuntimeError):
            with runtime.offloaded():
                self.fail("the computation must not run")
        masker.reload_weights.assert_called_once()
        self.assertIs(runtime.session.video_masker, masker)

    def test_initialization_can_take_the_weights_another_masker_loaded(self):
        runtime, _ = self.make_runtime()
        previous = runtime.session.video_masker
        runtime.session.video_masker = None
        runtime.initialize(MaskerSettings(True, 2, False, True), reuse=previous)
        runtime.create_masker.assert_called_once_with(progress_callback=None, reuse=previous)

    def test_restore_acts_on_the_runtimes_own_session_only(self):
        runtime, _ = self.make_runtime()
        original = runtime.session.video_masker
        other = BackendState(video_masker=FakeMasker())

        runtime.restore(MaskerSettings(True, 3, False, True))

        runtime.create_masker.assert_not_called()
        original.init_state.assert_called_once_with(
            "frames", online_mode=True, batch_size=3,
            offload_video_to_cpu=False, offload_state_to_cpu=True,
            async_loading_frames=False, progress_callback=None,
        )
        runtime.replay_inputs.assert_called_once_with(runtime.session)
        other.video_masker.init_state.assert_not_called()

    def test_initialization_builds_the_model_once(self):
        runtime, restored = self.make_runtime()
        session = runtime.session
        session.video_masker = None
        settings = MaskerSettings(True, 2, False, True)
        progress = Mock()

        runtime.initialize(settings, async_loading_frames=True, progress_callback=progress)

        self.assertIs(session.video_masker, restored)
        runtime.create_masker.assert_called_once_with(progress_callback=progress)
        restored.init_state.assert_called_once_with(
            "frames", online_mode=True, batch_size=2, offload_video_to_cpu=False,
            offload_state_to_cpu=True, async_loading_frames=True, progress_callback=progress,
        )
        runtime.replay_inputs.assert_called_once_with(session)
        runtime.initialize(settings)
        runtime.create_masker.assert_called_once()

    def test_propagating_resets_the_sessions_own_state_before_and_after_without_loading_frames_again(self):
        runtime, _ = self.make_runtime()
        session = runtime.session
        masker = session.video_masker
        events = []
        masker.reset_state.side_effect = lambda: events.append("reset")
        runtime.replay_inputs.side_effect = lambda active: events.append("replay")

        with runtime.propagating() as lent:
            self.assertIs(lent, masker)
            events.append("compute")

        self.assertEqual(events, ["reset", "replay", "compute", "reset", "replay"])
        masker.init_state.assert_not_called()
        runtime.create_masker.assert_not_called()
        self.assertIs(session.video_masker, masker)

    def test_propagating_needs_a_masker(self):
        runtime, _ = self.make_runtime()
        runtime.session.video_masker = None
        with self.assertRaises(ValueError):
            with runtime.propagating():
                self.fail("must not start without a masker")

    def test_a_state_that_cannot_be_prepared_stops_the_computation(self):
        runtime, _ = self.make_runtime()
        runtime.replay_inputs.side_effect = RuntimeError("replay failed")
        with self.assertLogs("video.runtime", level="ERROR"):
            with self.assertRaises(MaskerRestorationError):
                with runtime.propagating():
                    self.fail("the computation must not run on an unprepared state")
        self.assertIsNone(runtime.session.video_masker)

    def test_a_failed_computation_still_resets_the_state_and_its_error_wins(self):
        runtime, _ = self.make_runtime()
        masker = runtime.session.video_masker
        failure = ValueError("computation failed")
        with self.assertRaises(ValueError) as caught:
            with runtime.propagating():
                raise failure
        self.assertIs(caught.exception, failure)
        self.assertEqual(masker.reset_state.call_count, 2)
        self.assertIs(runtime.session.video_masker, masker)

    def test_a_failed_reset_after_a_failed_computation_keeps_the_original_error(self):
        runtime, _ = self.make_runtime()
        failure = ValueError("computation failed")
        with self.assertLogs("video.runtime", level="ERROR"):
            with self.assertRaises(ValueError) as caught:
                with runtime.propagating():
                    runtime.replay_inputs.side_effect = RuntimeError("replay failed")
                    raise failure
        self.assertIs(caught.exception, failure)
        self.assertIsNone(runtime.session.video_masker)

    def test_a_successful_computation_requires_the_state_to_be_put_back(self):
        runtime, _ = self.make_runtime()
        with self.assertLogs("video.runtime", level="ERROR"):
            with self.assertRaises(MaskerRestorationError):
                with runtime.propagating():
                    runtime.replay_inputs.side_effect = RuntimeError("replay failed")
        self.assertIsNone(runtime.session.video_masker)

    def test_rebuilding_inputs_disables_the_predictor_when_replay_fails(self):
        runtime, _ = self.make_runtime()
        session = runtime.session
        masker = session.video_masker
        masker.reset_state = Mock()

        runtime.rebuild_inputs()

        masker.reset_state.assert_called_once_with()
        runtime.replay_inputs.assert_called_once_with(session)
        self.assertIs(session.video_masker, masker)

        failure = RuntimeError("replay failed")
        runtime.replay_inputs.side_effect = failure
        with self.assertLogs("video.runtime", level="ERROR"):
            with self.assertRaises(MaskerRestorationError) as caught:
                runtime.rebuild_inputs()
        self.assertIs(caught.exception.__cause__, failure)
        self.assertIsNone(session.video_masker)


if __name__ == "__main__":
    unittest.main()
