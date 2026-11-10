import json
import shutil
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.errors import OperationError
from core.state import BackendState
from support import state
from schemas.video import VideoInitStateRequest
from sessions.metadata import session_metadata_document
from support import BackendStateTestCase
from video import prompts, service


class SessionReplacementTests(BackendStateTestCase):
    def setUp(self):
        super().setUp()
        self.root = self.temp_dir()
        self.old_masker = self.masker()
        self.old_dir = self.working_session(
            self.root / "old", session_id="old", video_masker=self.old_masker)
        state.annotations.create_object()
        self.before = state.annotations.snapshot()
        self.old_files = self.directory_files(self.old_dir)
        self.source = self.root / "source"
        self.source.mkdir()
        (self.source / "00000.jpg").touch()
        cache = patch("video.io.SESSION_CACHE_ROOT", self.root / "candidates")
        cache.start()
        self.addCleanup(cache.stop)
        self.client = self.make_client(preconditions=False)

    def masker(self):
        return self.make_masker(offload_state_to_cpu=True)

    @staticmethod
    def directory_files(directory):
        return sorted(path.relative_to(directory).as_posix() for path in directory.rglob("*"))

    def run_init(self, path=None):
        return service.run_video_init_job(
            self.ctx, VideoInitStateRequest(video_frames_dir=str(path or self.source)), self.job_handle())

    def assert_old_session(self):
        self.assertEqual(state.active_session_id, "old")
        self.assertIs(state.video_masker, self.old_masker)
        self.assertEqual(state.annotations.snapshot(), self.before)
        self.assertEqual(self.directory_files(self.old_dir), self.old_files)
        self.assertTrue((self.old_dir / "frames" / "00000.jpg").exists())
        self.old_masker.init_state.assert_not_called()

    def test_unsupported_input_files_are_rejected_before_a_working_copy_exists(self):
        for name, expected in [("notes.txt", "Unsupported input file type"), ("frame.png", "single image file")]:
            path = self.root / name
            path.touch()
            with self.subTest(name), self.assertRaises(OperationError) as failure:
                service.initialize_video_state_from_resolved_input(
                    path, online_mode=True, batch_size=1, offload_video_to_cpu=True, offload_state_to_cpu=True,
                    async_loading_frames=False, session=BackendState(), job=self.job_handle())
            self.assertEqual(failure.exception.status_code, 400)
            self.assertIn(expected, failure.exception.message)
        self.assertFalse((self.root / "candidates").exists())

    def test_invalid_path_and_empty_directory_preserve_current_session(self):
        empty = self.root / "empty"
        empty.mkdir()
        with patch.object(prompts.svm, "SAM2VideoMasker") as constructor:
            for path in [self.root / "missing", empty]:
                with self.subTest(path=path), self.assertRaises(OperationError):
                    self.run_init(path)
                self.assert_old_session()
            constructor.assert_not_called()

    def test_model_and_late_failures_preserve_current_session(self):
        candidate = self.masker()
        candidate.init_state.side_effect = RuntimeError("model failed")
        with patch.object(prompts.svm, "SAM2VideoMasker", return_value=candidate):
            with self.assertRaises(RuntimeError):
                self.run_init()
        self.assert_old_session()
        with (patch.object(prompts.svm, "SAM2VideoMasker", return_value=self.masker()),
              patch.object(service, "content_fingerprint", side_effect=OSError("disk full"))):
            with self.assertRaises(OSError):
                self.run_init()
        self.assert_old_session()

    def test_old_snapshot_remains_visible_until_candidate_is_published(self):
        candidate = self.masker()
        def preparing(*args, **kwargs):
            self.assert_old_session()
            info = self.client.get("/video/info").json()
            self.assertEqual(info["session_id"], "old")
            self.assertTrue(info["predictor_ready"])
        candidate.init_state.side_effect = preparing
        real_sweep = service.sweep_unreferenced_artifacts
        def late_step(session):
            self.assert_old_session()
            self.assertIsNot(session, state)
            real_sweep(session)
        with (patch.object(prompts.svm, "SAM2VideoMasker", return_value=candidate),
              patch.object(service, "sweep_unreferenced_artifacts", side_effect=late_step)):
            self.run_init()
        self.assertNotEqual(state.active_session_id, "old")
        self.assertIs(state.video_masker, candidate)
        self.assertTrue((state.active_session_dir / "frames").is_dir())
        self.assertEqual(self.directory_files(self.old_dir), self.old_files)

    def adopt_as_working_copy(self):
        working = self.root / "candidates" / "current-working"
        shutil.copytree(self.old_dir, working)
        state.active_session_dir = working
        self.old_dir = working
        return working

    def test_replaced_working_directory_is_deleted_after_publication(self):
        working = self.adopt_as_working_copy()
        with patch.object(prompts.svm, "SAM2VideoMasker", return_value=self.masker()):
            self.run_init()
        self.assertNotEqual(state.active_session_dir, working)
        self.assertFalse(working.exists())
        self.assertTrue(state.active_session_dir.is_dir())

    def test_rejected_candidate_directory_is_deleted_and_current_working_copy_kept(self):
        working = self.adopt_as_working_copy()
        failing = self.masker()
        failing.init_state.side_effect = RuntimeError("model failed")
        with patch.object(prompts.svm, "SAM2VideoMasker", return_value=failing):
            with self.assertRaises(RuntimeError):
                self.run_init()
        self.assert_old_session()
        self.assertEqual([path.name for path in (self.root / "candidates").iterdir()], [working.name])

    def test_directories_outside_the_working_cache_are_never_deleted(self):
        # self.old_dir lives outside the patched cache root, as a checkpoint would.
        with patch.object(prompts.svm, "SAM2VideoMasker", return_value=self.masker()):
            self.run_init()
        self.assertTrue((self.old_dir / "frames" / "00000.jpg").exists())
        service.discard_working_session(self.source)
        self.assertTrue(self.source.exists())

    def test_saved_session_replay_failure_preserves_both_sessions_and_artifacts(self):
        saved = self.root / "saved"
        (saved / "frames").mkdir(parents=True)
        (saved / "masks" / "overrides").mkdir(parents=True)
        (saved / "frames" / "00000.jpg").touch()
        metadata = json.loads(json.dumps(session_metadata_document(state)))
        metadata.update(session_id="saved", checkpoint_id="saved-checkpoint", display_name="Saved")
        metadata["annotations"]["prompts"] = [
            {"frame_idx": 0, "obj_id": 1, "points": [[1, 2]], "labels": [1], "box": None}]
        (saved / "session.json").write_text(json.dumps(metadata))
        orphan = saved / "masks" / "overrides" / ("0" * 32 + ".json")
        orphan.write_text("{}")
        before = (saved / "session.json").read_bytes()
        candidate = self.masker()
        candidate.add_new_points_or_box.side_effect = RuntimeError("replay failed")
        with patch.object(prompts.svm, "SAM2VideoMasker", return_value=candidate):
            with self.assertRaises(RuntimeError):
                self.run_init(saved)
        self.assert_old_session()
        self.assertEqual((saved / "session.json").read_bytes(), before)
        self.assertTrue(orphan.exists())
