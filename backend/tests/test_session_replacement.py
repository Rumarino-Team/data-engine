import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.factory import create_app
from core.errors import OperationError
from core.state import BackendState, state
from schemas.video import VideoInitStateRequest
from sessions.metadata import write_session_metadata
from video import service


class SessionReplacementTests(unittest.TestCase):
    def setUp(self):
        self.previous = state.__dict__.copy()
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.old_dir = self.root / "old"
        (self.old_dir / "frames").mkdir(parents=True)
        (self.old_dir / "masks").mkdir()
        (self.old_dir / "frames" / "00000.jpg").touch()
        self.old_masker = self.masker()
        state.__dict__.update(BackendState(
            active_session_id="old", active_session_dir=self.old_dir,
            video_dir=str(self.old_dir / "frames"), video_frame_files=["00000.jpg"],
            video_masker=self.old_masker, video_state_epoch=4,
        ).__dict__)
        state.annotations.create_object()
        write_session_metadata()
        self.before = state.annotations.snapshot()
        self.metadata = (self.old_dir / "session.json").read_bytes()
        self.source = self.root / "source"
        self.source.mkdir()
        (self.source / "00000.jpg").touch()
        self.cache = patch("video.io.SESSION_CACHE_ROOT", self.root / "candidates")
        self.cache.start()
        self.client = TestClient(create_app())

    def tearDown(self):
        self.cache.stop()
        state.__dict__.clear()
        state.__dict__.update(self.previous)
        self.tmp.cleanup()

    def masker(self):
        return Mock(online_mode=True, default_batch_size=1,
                    offload_video_to_cpu=True, offload_state_to_cpu=True)

    def run_init(self, path=None):
        return service.run_video_init_job(VideoInitStateRequest(video_frames_dir=str(path or self.source)))

    def assert_old_session(self):
        self.assertEqual(state.active_session_id, "old")
        self.assertIs(state.video_masker, self.old_masker)
        self.assertEqual(state.annotations.snapshot(), self.before)
        self.assertEqual(state.video_state_epoch, 4)
        self.assertEqual((self.old_dir / "session.json").read_bytes(), self.metadata)
        self.assertTrue((self.old_dir / "frames" / "00000.jpg").exists())
        self.old_masker.init_state.assert_not_called()

    def test_invalid_path_and_empty_directory_preserve_current_session(self):
        empty = self.root / "empty"
        empty.mkdir()
        with patch.object(service.svm, "SAM2VideoMasker") as constructor:
            for path in [self.root / "missing", empty]:
                with self.subTest(path=path), self.assertRaises(OperationError):
                    self.run_init(path)
                self.assert_old_session()
            constructor.assert_not_called()

    def test_model_and_persistence_failure_preserve_current_session(self):
        candidate = self.masker()
        candidate.init_state.side_effect = RuntimeError("model failed")
        with patch.object(service.svm, "SAM2VideoMasker", return_value=candidate):
            with self.assertRaises(RuntimeError):
                self.run_init()
        self.assert_old_session()
        with (patch.object(service.svm, "SAM2VideoMasker", return_value=self.masker()),
              patch.object(service, "write_session_metadata", side_effect=OSError("disk full"))):
            with self.assertRaises(OSError):
                self.run_init()
        self.assert_old_session()

    def test_old_snapshot_remains_visible_until_candidate_is_persisted(self):
        candidate = self.masker()
        def preparing(*args, **kwargs):
            self.assert_old_session()
            info = self.client.get("/video/info").json()
            self.assertEqual(info["session_id"], "old")
            self.assertTrue(info["predictor_ready"])
        candidate.init_state.side_effect = preparing
        def persist(extra, *, session):
            self.assert_old_session()
            self.assertIsNot(session, state)
            write_session_metadata(extra, session=session)
        with (patch.object(service.svm, "SAM2VideoMasker", return_value=candidate),
              patch.object(service, "write_session_metadata", side_effect=persist)):
            result = self.run_init()
        self.assertNotEqual(state.active_session_id, "old")
        self.assertIs(state.video_masker, candidate)
        self.assertEqual(result["state_epoch"], 5)
        self.assertEqual(json.loads((state.active_session_dir / "session.json").read_text())["session_id"],
                         state.active_session_id)
        self.assertEqual((self.old_dir / "session.json").read_bytes(), self.metadata)

    def test_saved_session_replay_failure_preserves_both_sessions_and_artifacts(self):
        saved = self.root / "saved"
        (saved / "frames").mkdir(parents=True)
        (saved / "masks" / "overrides").mkdir(parents=True)
        (saved / "frames" / "00000.jpg").touch()
        metadata = json.loads(self.metadata)
        metadata["session_id"] = "saved"
        metadata["annotations"]["prompts"] = [
            {"frame_idx": 0, "obj_id": 1, "points": [[1, 2]], "labels": [1], "box": None}]
        (saved / "session.json").write_text(json.dumps(metadata))
        orphan = saved / "masks" / "overrides" / ("0" * 32 + ".json")
        orphan.write_text("{}")
        before = (saved / "session.json").read_bytes()
        candidate = self.masker()
        candidate.add_new_points_or_box.side_effect = RuntimeError("replay failed")
        with patch.object(service.svm, "SAM2VideoMasker", return_value=candidate):
            with self.assertRaises(RuntimeError):
                self.run_init(saved)
        self.assert_old_session()
        self.assertEqual((saved / "session.json").read_bytes(), before)
        self.assertTrue(orphan.exists())

    def test_tracking_video_load_failure_preserves_annotation_session(self):
        from schemas.tracking import TrackingLoadVideoRequest
        from tracking import service as tracking
        source = self.root / "video.mp4"
        source.touch()
        with patch.object(tracking, "read_tracking_video", side_effect=ValueError("bad video")):
            with self.assertRaises(ValueError):
                asyncio.run(tracking.load_tracking_video(TrackingLoadVideoRequest(video_path=str(source))))
        self.assert_old_session()
