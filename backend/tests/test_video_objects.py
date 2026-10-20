import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import core.jobs as jobs
from app.factory import create_app
from core.state import state
from core.annotations import SessionAnnotations
from pydantic import ValidationError
from schemas.video import VideoPrompt
from annotation_delta import apply_delta, same_annotations


class VideoObjectTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.session_path = Path(self.tmp.name)
        state.active_session_dir = self.session_path
        state.active_session_id = "object-session"
        state.annotations = SessionAnnotations()
        self.client = TestClient(create_app())
        def preconditions(request):
            request.headers.setdefault("X-Session-ID", state.active_session_id or "absent")
            request.headers.setdefault("X-Annotation-Revision", str(state.annotations.revision))
        self.client.event_hooks["request"].append(preconditions)

    def tearDown(self):
        with jobs.current_job_lock:
            jobs.current_job = None
        state.active_session_dir = None
        state.active_session_id = None
        state.annotations = SessionAnnotations()
        state.video_masker = None
        self.tmp.cleanup()

    def session_objects(self):
        return sorted(json.loads((self.session_path / "session.json").read_text())["annotations"]["objects"].values(), key=lambda obj: obj["id"])

    def info(self):
        return self.client.get("/video/info").json()["annotations"]

    def test_create_allocates_defaults_and_update_preserves_id(self):
        response = self.client.post("/video/objects")
        self.assertEqual(response.status_code, 200)
        snapshot = response.json()
        self.assertEqual(snapshot["created_object_id"], 1)
        obj = snapshot["annotation_delta"]["objects"]["1"]
        self.assertEqual(obj["label"], "Object 1")
        self.assertRegex(obj["color"], r"^#[0-9a-f]{6}$")
        self.assertEqual(snapshot["session_id"], "object-session")
        self.assertEqual(snapshot["annotation_delta"]["base_revision"], 0)
        self.assertEqual(snapshot["annotation_delta"]["revision"], 1)
        self.assertEqual(snapshot["annotation_delta"]["next_object_id"], 2)
        before = self.info()
        response = self.client.put("/video/objects/1", json={"label": "  buoy ", "color": "#ff0000"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["annotation_delta"]["revision"], 2)
        self.assertEqual(self.session_objects(), [{"id": 1, "label": "buoy", "color": "#ff0000"}])
        self.assertTrue(same_annotations(apply_delta(before, response.json()["annotation_delta"]), self.info()))

    def test_update_cannot_create_or_resurrect_client_selected_id(self):
        response = self.client.put("/video/objects/99", json={"label": "buoy", "color": "#ff0000"})
        self.assertEqual(response.status_code, 404)
        self.assertEqual(state.annotations.revision, 0)
        self.assertEqual(state.annotations.next_object_id, 1)

    def test_allocation_survives_deletion_and_metadata_reload(self):
        state.annotations = SessionAnnotations(next_object_id=8)
        self.assertEqual(self.client.post("/video/objects").json()["created_object_id"], 8)
        state.video_masker = mock.Mock(online_mode=True, default_batch_size=1,
                                      offload_video_to_cpu=False, offload_state_to_cpu=False)
        self.assertEqual(self.client.post("/video/remove_object", params={"obj_id": 8}).status_code, 200)
        metadata = json.loads((self.session_path / "session.json").read_text())
        state.annotations = SessionAnnotations.model_validate(metadata["annotations"])
        self.assertEqual(state.annotations.objects, {})
        self.assertEqual(self.client.post("/video/objects").json()["created_object_id"], 9)

    def test_create_requires_session_and_is_guarded_during_jobs(self):
        state.active_session_dir = None
        self.assertEqual(self.client.post("/video/objects").status_code, 409)
        state.active_session_dir = self.session_path
        with jobs.current_job_lock:
            jobs.current_job = {"job_id": "active", "status": "running"}
        self.assertEqual(self.client.post("/video/objects").status_code, 409)
        self.assertEqual(state.annotations.next_object_id, 1)
        self.assertEqual(state.annotations.objects, {})

    def test_update_rejects_blank_label_and_bad_id(self):
        blank = self.client.put("/video/objects/1", json={"label": "   ", "color": "#ff0000"})
        self.assertEqual(blank.status_code, 422)
        bad_id = self.client.put("/video/objects/0", json={"label": "buoy", "color": "#ff0000"})
        self.assertEqual(bad_id.status_code, 422)
        self.assertEqual(state.annotations.objects, {})

    def test_update_requires_session(self):
        state.active_session_dir = None
        response = self.client.put("/video/objects/1", json={"label": "buoy", "color": "#ff0000"})
        self.assertEqual(response.status_code, 409)

    def test_update_is_guarded_while_job_active(self):
        with jobs.current_job_lock:
            jobs.current_job = {"job_id": "active", "status": "running"}
        response = self.client.put("/video/objects/1", json={"label": "buoy", "color": "#ff0000"})
        self.assertEqual(response.status_code, 409)

    def test_remove_object_drops_metadata(self):
        self.client.post("/video/objects")
        state.video_masker = mock.Mock(
            online_mode=True, default_batch_size=1, offload_video_to_cpu=False, offload_state_to_cpu=False
        )
        response = self.client.post("/video/remove_object", params={"obj_id": 1})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["annotation_delta"]["removed_objects"], [1])
        self.assertEqual(response.json()["annotation_delta"]["revision"], 2)
        state.video_masker.remove_object.assert_called_once_with(1)
        self.assertEqual(state.annotations.objects, {})
        self.assertEqual(self.session_objects(), [])

    def test_annotations_rejects_invalid_objects(self):
        with self.assertRaises(ValidationError):
            SessionAnnotations(objects={2: {"id": 2, "label": "", "color": "#00ff00"}})

    def test_relabel_response_preserves_other_objects_and_prompts(self):
        state.annotations.upsert_object({"id": 1, "label": "hand", "color": "#ff0000"})
        state.annotations.upsert_object({"id": 2, "label": "cup", "color": "#00ff00"})
        state.annotations.set_prompt(VideoPrompt(frame_idx=0, obj_id=2, points=[[1, 2]], labels=[1]))
        state.annotations.take_delta(3)
        before = state.annotations.snapshot()
        response = self.client.put("/video/objects/1", json={"label": "left hand", "color": "#ff0000"})
        snapshot = response.json()
        delta = snapshot["annotation_delta"]
        self.assertEqual(delta["revision"], 4)
        # A relabel carries only the relabeled object, never unrelated objects or prompts.
        self.assertEqual(list(delta["objects"]), ["1"])
        self.assertEqual(delta["prompts"], [])
        persisted = json.loads((self.session_path / "session.json").read_text())
        self.assertTrue(same_annotations(apply_delta(before, delta), persisted["annotations"]))
        self.assertNotIn("annotations", snapshot)
        self.assertNotIn("frame_files", snapshot)

    def test_deletion_response_contains_only_remaining_accepted_inputs(self):
        for obj_id in (1, 2):
            self.client.post("/video/objects")
            state.annotations.set_prompt(VideoPrompt(frame_idx=0, obj_id=obj_id, points=[[1, 2]], labels=[1]))
        state.video_masker = mock.Mock(
            online_mode=True, default_batch_size=1, offload_video_to_cpu=False, offload_state_to_cpu=False,
        )
        before = self.info()
        snapshot = self.client.post("/video/remove_object", params={"obj_id": 1}).json()
        self.assertEqual(snapshot["session_id"], "object-session")
        delta = snapshot["annotation_delta"]
        self.assertEqual(delta["removed_objects"], [1])
        self.assertEqual(delta["removed_prompts"], [[0, 1]])
        self.assertEqual(delta["revision"], 5)
        self.assertEqual(delta["next_object_id"], 3)
        after = apply_delta(before, delta)
        self.assertEqual(list(after["objects"]), ["2"])
        self.assertEqual([prompt["obj_id"] for prompt in after["prompts"]], [2])
        self.assertTrue(same_annotations(after, self.info()))


if __name__ == "__main__":
    unittest.main()
