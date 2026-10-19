import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.factory import create_app
from core import jobs
from core.state import BackendState, state
from schemas.video import AnnotationPrecondition
from sessions.metadata import write_session_metadata
from video import service


class AnnotationPreconditionTests(unittest.TestCase):
    def setUp(self):
        self.previous = state.__dict__.copy()
        state.__dict__.update(BackendState().__dict__)
        self.tmp = tempfile.TemporaryDirectory()
        state.active_session_dir = Path(self.tmp.name)
        state.active_session_id = "current"
        state.annotations.create_object()
        state.video_masker = Mock(online_mode=True, default_batch_size=1,
                                  offload_video_to_cpu=False, offload_state_to_cpu=False)
        write_session_metadata()
        self.client = TestClient(create_app())

    def tearDown(self):
        with jobs.current_job_lock:
            jobs.current_job = None
        state.__dict__.clear()
        state.__dict__.update(self.previous)
        self.tmp.cleanup()

    def headers(self, session="current", revision=1):
        return {"X-Session-ID": session, "X-Annotation-Revision": str(revision)}

    def test_all_object_mutations_reject_stale_identity_or_revision_before_side_effects(self):
        before = state.annotations.snapshot()
        metadata = (state.active_session_dir / "session.json").read_bytes()
        for headers in [self.headers(session="previous"), self.headers(revision=0)]:
            for method, url, body in [
                ("POST", "/video/objects", None),
                ("PUT", "/video/objects/1", {"label": "changed", "color": "#123456"}),
                ("POST", "/video/remove_object?obj_id=1", None),
                ("POST", "/video/add_new_points_or_box", {"frame_idx": 0, "obj_id": 1, "points": [[1, 2]], "labels": [1]}),
                ("POST", "/video/clear_all_prompts_in_frame?frame_idx=0&obj_id=1", None),
                ("POST", "/video/reset_state", None),
            ]:
                with self.subTest(url=url, headers=headers):
                    response = self.client.request(method, url, headers=headers, json=body)
                    self.assertEqual(response.status_code, 409)
                    self.assertEqual(state.annotations.snapshot(), before)
                    self.assertEqual((state.active_session_dir / "session.json").read_bytes(), metadata)
        state.video_masker.remove_object.assert_not_called()
        state.video_masker.add_new_points_or_box.assert_not_called()
        state.video_masker.clear_all_prompts_in_frame.assert_not_called()
        state.video_masker.reset_state.assert_not_called()

    def test_preconditions_are_required_and_validated(self):
        for headers in [{}, self.headers(revision=-1), self.headers(revision="bad")]:
            self.assertEqual(self.client.post("/video/objects", headers=headers).status_code, 422)
        self.assertEqual(state.annotations.revision, 1)

    def test_two_concurrent_commands_from_same_revision_accept_only_one(self):
        barrier = threading.Barrier(2)
        def create():
            barrier.wait(timeout=5)
            return self.client.post("/video/objects", headers=self.headers()).status_code
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda _: create(), range(2)))
        self.assertEqual(sorted(results), [200, 409])
        self.assertEqual(state.annotations.revision, 2)
        self.assertEqual(state.annotations.next_object_id, 3)
        persisted = json.loads((state.active_session_dir / "session.json").read_text())
        self.assertEqual(persisted["annotations"], state.annotations.snapshot())

    def test_job_admission_waits_for_edit_persistence_and_snapshot(self):
        entered = threading.Event()
        release = threading.Event()
        admission_attempted = threading.Event()
        def persist():
            entered.set()
            self.assertTrue(release.wait(timeout=5))
            self.assertFalse(jobs.is_job_active())
            write_session_metadata()
        def admit():
            admission_attempted.set()
            return jobs._start_job("video_init", stage="init", stage_label="Init", message="init")
        with ThreadPoolExecutor(max_workers=2) as pool:
            with patch.object(service, "write_session_metadata", side_effect=persist):
                edit = pool.submit(asyncio.run, service.create_object(
                    AnnotationPrecondition(session_id="current", expected_revision=1)))
                self.assertTrue(entered.wait(timeout=5))
                admission = pool.submit(admit)
                self.assertTrue(admission_attempted.wait(timeout=5))
                try:
                    self.assertFalse(admission.done())
                finally:
                    release.set()
                self.assertEqual(edit.result(timeout=5)["annotations"]["revision"], 2)
                self.assertEqual(admission.result(timeout=5)["status"], "queued")
        self.assertEqual(self.client.post("/video/objects", headers=self.headers(revision=2)).status_code, 409)

    def test_lock_is_released_after_persistence_failure_and_snapshot_can_recover(self):
        with patch.object(service, "write_session_metadata", side_effect=OSError("disk error")):
            with self.assertRaises(OSError):
                self.client.post("/video/objects", headers=self.headers())
        # Persistence transactions are separate work: recover the actual accepted
        # in-memory state instead of replaying an uncertain create command.
        snapshot = self.client.get("/video/info").json()
        self.assertEqual(snapshot["annotations"]["revision"], 2)
        self.assertEqual(self.client.post("/video/objects", headers=self.headers()).status_code, 409)
        self.assertEqual(self.client.put("/video/objects/2", headers=self.headers(revision=2),
                         json={"label": "Recovered", "color": "#123456"}).status_code, 200)
