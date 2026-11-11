import sys
import unittest
from pathlib import Path
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.state import SessionState
from tracking.runtime import VideoTrackerRuntime


def fake_tracker(model_name):
    return Mock(model_name=model_name)


class VideoTrackerRuntimeTests(unittest.TestCase):
    def make_runtime(self, tracker=None):
        session = SessionState(tracker=tracker)
        created = []

        def create(*, model_name):
            created.append(model_name)
            return fake_tracker(model_name)

        runtime = VideoTrackerRuntime(session=session, create_tracker=Mock(side_effect=create), release_memory=Mock())
        return runtime, created

    def test_the_model_is_built_once_and_then_reused(self):
        runtime, created = self.make_runtime()
        first = runtime.ensure("model-a")
        self.assertIs(runtime.session.tracker, first)
        self.assertIs(runtime.ensure("model-a"), first)
        self.assertEqual(created, ["model-a"])
        runtime.release_memory.assert_not_called()

    def test_a_loaded_tracker_for_the_same_model_is_reused_without_building(self):
        existing = fake_tracker("model-a")
        runtime, created = self.make_runtime(existing)
        self.assertIs(runtime.ensure("model-a"), existing)
        self.assertEqual(created, [])

    def test_a_different_model_releases_the_old_weights_before_building_the_new_one(self):
        events = []
        runtime, _ = self.make_runtime(fake_tracker("model-a"))
        runtime.release_memory.side_effect = lambda: events.append("release")
        runtime.create_tracker.side_effect = lambda *, model_name: events.append("build") or fake_tracker(model_name)

        replacement = runtime.ensure("model-b")

        self.assertEqual(events, ["release", "build"])
        self.assertIs(runtime.session.tracker, replacement)
        self.assertEqual(replacement.model_name, "model-b")

    def test_a_model_that_cannot_be_built_leaves_the_session_without_a_tracker(self):
        runtime, _ = self.make_runtime(fake_tracker("model-a"))
        runtime.create_tracker.side_effect = ValueError("Unsupported CoTracker model")
        with self.assertRaises(ValueError):
            runtime.ensure("model-b")
        self.assertIsNone(runtime.session.tracker)

    def test_release_frees_memory_only_when_a_tracker_was_loaded(self):
        runtime, _ = self.make_runtime(fake_tracker("model-a"))
        runtime.release()
        self.assertIsNone(runtime.session.tracker)
        runtime.release_memory.assert_called_once()
        runtime.release()
        runtime.release_memory.assert_called_once()


if __name__ == "__main__":
    unittest.main()
