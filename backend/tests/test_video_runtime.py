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


class VideoMaskerRuntimeTests(unittest.TestCase):
    def make_runtime(self):
        session = BackendState(
            video_dir="frames", active_session_id="session", video_masker=FakeMasker(), video_state_epoch=5,
        )
        session.annotations.create_object()
        restored = FakeMasker()
        runtime = VideoMaskerRuntime(
            session=session, create_masker=Mock(return_value=restored), replay_inputs=Mock(),
            persist_session=Mock(), release_memory=Mock(),
        )
        return runtime, restored

    def test_suspension_releases_weights_and_restores_in_order_with_original_settings(self):
        runtime, restored = self.make_runtime()
        session = runtime.session
        original = weakref.ref(session.video_masker)
        accepted = session.annotations.snapshot()
        events = []
        runtime.release_memory.side_effect = lambda: events.append("release")
        restored.init_state.side_effect = lambda *args, **kwargs: events.append("initialize")
        runtime.replay_inputs.side_effect = lambda active: events.append("replay")
        runtime.persist_session.side_effect = lambda active: events.append("persist")
        progress = Mock()

        with runtime.suspended(progress_callback=progress):
            self.assertIsNone(session.video_masker)
            self.assertIsNone(original())
            self.assertEqual(session.video_state_epoch, 6)
            events.append("compute")

        self.assertEqual(events, ["release", "compute", "initialize", "replay", "persist"])
        self.assertIs(session.video_masker, restored)
        self.assertEqual(session.video_state_epoch, 7)
        self.assertEqual(session.annotations.snapshot(), accepted)
        runtime.create_masker.assert_called_once_with(progress_callback=progress)
        restored.init_state.assert_called_once_with(
            "frames", online_mode=False, batch_size=7,
            offload_video_to_cpu=True, offload_state_to_cpu=False,
            async_loading_frames=False, progress_callback=progress,
        )
        runtime.replay_inputs.assert_called_once_with(session)
        runtime.persist_session.assert_called_once_with(session)

    def test_successful_computation_requires_successful_restoration(self):
        for failure_stage in ("construction", "initialization", "replay", "persistence"):
            with self.subTest(stage=failure_stage):
                runtime, restored = self.make_runtime()
                before = runtime.session.annotations.snapshot()
                failure = RuntimeError(f"{failure_stage} failed")
                collaborator = {
                    "construction": runtime.create_masker,
                    "initialization": restored.init_state,
                    "replay": runtime.replay_inputs,
                    "persistence": runtime.persist_session,
                }[failure_stage]
                collaborator.side_effect = failure

                with self.assertLogs("video.runtime", level="ERROR"):
                    with self.assertRaises(MaskerRestorationError) as caught:
                        with runtime.suspended():
                            pass

                self.assertIs(caught.exception.__cause__, failure)
                self.assertIsNone(runtime.session.video_masker)
                self.assertEqual(runtime.session.annotations.snapshot(), before)
                self.assertEqual(runtime.session.video_state_epoch, 8 if failure_stage == "persistence" else 7)

    def test_computation_error_survives_failed_restoration(self):
        runtime, _ = self.make_runtime()
        computation_error = ValueError("computation failed")
        runtime.create_masker.side_effect = RuntimeError("restoration failed")

        with self.assertLogs("video.runtime", level="ERROR"):
            with self.assertRaises(ValueError) as caught:
                with runtime.suspended():
                    raise computation_error

        self.assertIs(caught.exception, computation_error)
        self.assertIsNone(runtime.session.video_masker)

    def test_computation_error_still_restores_interactive_masker(self):
        runtime, restored = self.make_runtime()
        computation_error = ValueError("computation failed")

        with self.assertRaises(ValueError) as caught:
            with runtime.suspended():
                raise computation_error

        self.assertIs(caught.exception, computation_error)
        self.assertIs(runtime.session.video_masker, restored)
        runtime.persist_session.assert_called_once_with(runtime.session)

    def test_absent_masker_is_not_loaded_on_exit(self):
        runtime, _ = self.make_runtime()
        runtime.session.video_masker = None

        with runtime.suspended():
            pass

        runtime.create_masker.assert_not_called()
        runtime.release_memory.assert_not_called()
        self.assertEqual(runtime.session.video_state_epoch, 5)

    def test_restore_uses_explicit_session_and_can_preserve_epoch(self):
        runtime, _ = self.make_runtime()
        original = runtime.session.video_masker
        other = BackendState(video_masker=FakeMasker(), video_state_epoch=100)

        runtime.restore(MaskerSettings(True, 3, False, True), increment_epoch=False)

        runtime.create_masker.assert_not_called()
        original.init_state.assert_called_once_with(
            "frames", online_mode=True, batch_size=3,
            offload_video_to_cpu=False, offload_state_to_cpu=True,
            async_loading_frames=False, progress_callback=None,
        )
        runtime.replay_inputs.assert_called_once_with(runtime.session)
        runtime.persist_session.assert_not_called()
        self.assertEqual(runtime.session.video_state_epoch, 5)
        self.assertEqual(other.video_state_epoch, 100)
        other.video_masker.init_state.assert_not_called()


if __name__ == "__main__":
    unittest.main()
