"""The editing contract between a client and the backend: object lifecycle, preconditions and serialization."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import threading
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.annotations import SessionAnnotations
from core.errors import OperationError
from support import state
from pydantic import ValidationError
from schemas.video import AnnotationPrecondition, VideoPrompt
from sessions.metadata import session_metadata_document
from annotation_delta import apply_delta, same_annotations
from support import BackendStateTestCase
from video import service

VALID_UPDATE = {"label": "buoy", "color": "#ff0000"}


class PositivePointTests(unittest.TestCase):
    def test_positive_points_are_the_one_definition_of_what_tracking_follows(self):
        annotations = SessionAnnotations(
            objects={1: {"id": 1, "label": "a", "color": "#ff0000"}, 2: {"id": 2, "label": "b", "color": "#00ff00"}},
            prompts=[
                VideoPrompt(frame_idx=0, obj_id=1, points=[[1, 2], [3, 4]], labels=[0, 1]),
                VideoPrompt(frame_idx=1, obj_id=1, box=[0, 0, 5, 5]),
                VideoPrompt(frame_idx=2, obj_id=2, points=[[5, 6]], labels=[1]),
            ])
        points = annotations.positive_points()
        # Negative points and box-only prompts are skipped; indexes keep counting them.
        self.assertEqual([(p.event_idx, p.point_idx, p.obj_id, p.frame_idx, p.x, p.y) for p in points],
                         [(0, 1, 1, 0, 3.0, 4.0), (2, 0, 2, 2, 5.0, 6.0)])
        self.assertEqual(annotations.positive_prompt_keys(),
                         {"obj:1|frame:0|x:3.000|y:4.000", "obj:2|frame:2|x:5.000|y:6.000"})
        self.assertEqual({p.key for p in points}, annotations.positive_prompt_keys())


class ObjectLifecycleTests(BackendStateTestCase):
    def setUp(self):
        super().setUp()
        self.session_path = self.working_session(session_id="object-session", frames=())
        self.client = self.make_client()

    def session_objects(self):
        # What Save would write: the accepted state, built from memory.
        annotations = session_metadata_document(state)["annotations"]
        return sorted(annotations["objects"].values(), key=lambda obj: obj["id"])

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

    def test_invalid_updates_are_rejected_without_changing_state(self):
        for object_id, body, status in [
            (99, VALID_UPDATE, 404),  # an update can never create or resurrect an id
            (1, {"label": "   ", "color": "#ff0000"}, 422),
            (0, VALID_UPDATE, 422),
        ]:
            with self.subTest(object_id=object_id, body=body):
                self.assertEqual(self.client.put(f"/video/objects/{object_id}", json=body).status_code, status)
        self.assertEqual(state.annotations.objects, {})
        self.assertEqual((state.annotations.revision, state.annotations.next_object_id), (0, 1))

    def test_the_model_rejects_a_blank_label(self):
        with self.assertRaises(ValidationError):
            SessionAnnotations(objects={2: {"id": 2, "label": "", "color": "#00ff00"}})

    def test_edits_without_an_active_session_are_rejected(self):
        state.active_session_dir = None
        for method, url, body in [("post", "/video/objects", None), ("put", "/video/objects/1", VALID_UPDATE)]:
            with self.subTest(url=url):
                self.assertEqual(getattr(self.client, method)(url, json=body).status_code, 409)
        self.assertEqual(state.annotations.objects, {})

    def test_allocation_survives_deletion_and_metadata_reload(self):
        state.annotations = SessionAnnotations(next_object_id=8)
        self.assertEqual(self.client.post("/video/objects").json()["created_object_id"], 8)
        state.video_masker = self.make_masker()
        self.assertEqual(self.client.post("/video/remove_object", params={"obj_id": 8}).status_code, 200)
        metadata = session_metadata_document(state)
        state.annotations = SessionAnnotations.model_validate(metadata["annotations"])
        self.assertEqual(state.annotations.objects, {})
        self.assertEqual(self.client.post("/video/objects").json()["created_object_id"], 9)

    def test_relabel_response_carries_only_the_relabeled_object(self):
        state.annotations.upsert_object({"id": 1, "label": "hand", "color": "#ff0000"})
        state.annotations.upsert_object({"id": 2, "label": "cup", "color": "#00ff00"})
        state.annotations.set_prompt(VideoPrompt(frame_idx=0, obj_id=2, points=[[1, 2]], labels=[1]))
        state.annotations.take_delta(3)
        before = state.annotations.snapshot()
        snapshot = self.client.put("/video/objects/1", json={"label": "left hand", "color": "#ff0000"}).json()
        delta = snapshot["annotation_delta"]
        self.assertEqual(delta["revision"], 4)
        self.assertEqual(list(delta["objects"]), ["1"])
        self.assertEqual(delta["prompts"], [])
        saved = session_metadata_document(state)
        self.assertTrue(same_annotations(apply_delta(before, delta), saved["annotations"]))
        self.assertNotIn("annotations", snapshot)
        self.assertNotIn("frame_files", snapshot)

    def test_deletion_removes_the_object_and_its_prompts_everywhere(self):
        for obj_id in (1, 2):
            self.client.post("/video/objects")
            state.annotations.set_prompt(VideoPrompt(frame_idx=0, obj_id=obj_id, points=[[1, 2]], labels=[1]))
        state.video_masker = self.make_masker()
        before = self.info()
        snapshot = self.client.post("/video/remove_object", params={"obj_id": 1}).json()
        self.assertEqual(snapshot["session_id"], "object-session")
        delta = snapshot["annotation_delta"]
        self.assertEqual(delta["removed_objects"], [1])
        self.assertEqual(delta["removed_prompts"], [[0, 1]])
        self.assertEqual(delta["revision"], 5)
        self.assertEqual(delta["next_object_id"], 3)
        state.video_masker.remove_object.assert_called_once_with(1)
        after = apply_delta(before, delta)
        self.assertEqual(list(after["objects"]), ["2"])
        self.assertEqual([prompt["obj_id"] for prompt in after["prompts"]], [2])
        self.assertTrue(same_annotations(after, self.info()))
        self.assertEqual([obj["id"] for obj in self.session_objects()], [2])


class PreconditionTests(BackendStateTestCase):
    def setUp(self):
        super().setUp()
        state.active_session_dir = self.temp_dir()
        state.active_session_id = "current"
        state.annotations.create_object()
        state.video_masker = self.make_masker(offload_video_to_cpu=False)
        self.client = self.make_client(preconditions=False)

    def headers(self, session="current", revision=1):
        return {"X-Session-ID": session, "X-Annotation-Revision": str(revision)}

    def test_all_object_mutations_reject_stale_identity_or_revision_before_side_effects(self):
        before = state.annotations.snapshot()
        metadata = dict(state.session_metadata)
        for headers in [self.headers(session="previous"), self.headers(revision=0)]:
            for method, url, body in [
                ("POST", "/video/objects", None),
                ("PUT", "/video/objects/1", {"label": "changed", "color": "#123456"}),
                ("POST", "/video/remove_object?obj_id=1", None),
                ("POST", "/video/add_new_points_or_box", {"frame_idx": 0, "obj_id": 1, "points": [[1, 2]], "labels": [1]}),
                ("POST", "/video/clear_all_prompts_in_frame?frame_idx=0&obj_id=1", None),
                ("POST", "/video/reset_state", None),
                ("POST", "/video/propagate_in_video", {}),
                ("POST", "/tracking/track_prompt_points", {}),
            ]:
                with self.subTest(url=url, headers=headers):
                    response = self.client.request(method, url, headers=headers, json=body)
                    self.assertEqual(response.status_code, 409)
                    self.assertEqual(state.annotations.snapshot(), before)
                    self.assertEqual(state.session_metadata, metadata)
        state.video_masker.remove_object.assert_not_called()
        state.video_masker.add_new_points_or_box.assert_not_called()
        state.video_masker.clear_all_prompts_in_frame.assert_not_called()
        state.video_masker.reset_state.assert_not_called()

    def test_mutations_are_rejected_while_a_job_is_active_but_reads_are_not(self):
        self.activate_job()
        for method, url, body in [
            ("POST", "/video/objects", None),
            ("PUT", "/video/objects/1", {"label": "changed", "color": "#123456"}),
            ("POST", "/video/remove_object?obj_id=1", None),
            ("POST", "/video/add_new_points_or_box", {"frame_idx": 0, "obj_id": 1, "points": [[1, 2]], "labels": [1]}),
            ("POST", "/video/clear_all_prompts_in_frame?frame_idx=0&obj_id=1", None),
            ("POST", "/video/reset_state", None),
        ]:
            with self.subTest(url=url):
                response = self.client.request(method, url, headers=self.headers(), json=body)
                self.assertEqual(response.status_code, 409)
                self.assertEqual(response.json()["detail"], "Another operation is already running.")
        self.assertEqual(state.annotations.revision, 1)
        self.assertEqual(self.client.get("/video/info").status_code, 200)

    def test_preconditions_are_required_and_validated(self):
        for headers in [{}, self.headers(revision=-1), self.headers(revision="bad")]:
            self.assertEqual(self.client.post("/video/objects", headers=headers).status_code, 422)
        self.assertEqual(state.annotations.revision, 1)

    def test_job_admission_captures_inputs_and_rejects_stale_worker(self):
        state.annotations.set_prompt(VideoPrompt(frame_idx=0, obj_id=1, points=[[1, 2]], labels=[1]))
        for route, module, runner, operation in [
                ("/video/propagate_in_video", "video.propagation", "run_propagation_job", "mask_propagation"),
                ("/tracking/track_prompt_points", "tracking.prompt_tracking", "run_prompt_tracking_job", "prompt_tracking")]:
            started = {"job_id": "test", "status": "queued", "operation": operation, "message": "queued"}
            with patch.object(self.ctx.jobs, "start", return_value=started) as queue:
                response = self.client.post(route, json={}, headers=self.headers(revision=state.annotations.revision))
                self.assertEqual(response.status_code, 200)
                with patch(f"{module}.{runner}") as run:
                    queue.call_args.kwargs["worker"](self.job_handle())
                inputs = run.call_args.args[3]
                self.assertEqual(inputs["session_id"], "current")
                self.assertEqual(inputs["input_revision"], state.annotations.revision)
                state.active_session_id = "replacement"
                with self.assertRaises(OperationError) as stale:
                    queue.call_args.kwargs["worker"](self.job_handle())
                self.assertEqual(stale.exception.status_code, 409)
                state.active_session_id = "current"
        state.video_masker.init_state.assert_not_called()

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

    def test_job_admission_waits_for_an_edit_and_its_snapshot(self):
        entered = threading.Event()
        release = threading.Event()
        admission_attempted = threading.Event()

        real_update = service.annotation_update

        def hold_the_edit(*args):
            entered.set()
            self.assertTrue(release.wait(timeout=5))
            self.assertFalse(self.ctx.jobs.is_active())
            return real_update(*args)

        def admit():
            admission_attempted.set()
            self.activate_job("queued")
            return self.ctx.jobs.snapshot()
        with ThreadPoolExecutor(max_workers=2) as pool:
            with patch.object(service, "annotation_update", side_effect=hold_the_edit):
                edit = pool.submit(service.create_object, self.ctx,
                                   AnnotationPrecondition(session_id="current", expected_revision=1))
                self.assertTrue(entered.wait(timeout=5))
                admission = pool.submit(admit)
                self.assertTrue(admission_attempted.wait(timeout=5))
                try:
                    self.assertFalse(admission.done())
                finally:
                    release.set()
                self.assertEqual(edit.result(timeout=5)["annotation_delta"]["revision"], 2)
                self.assertEqual(admission.result(timeout=5)["status"], "queued")
        self.assertEqual(self.client.post("/video/objects", headers=self.headers(revision=2)).status_code, 409)

    def test_lock_is_released_after_a_failed_edit_and_snapshot_can_recover(self):
        real_create = SessionAnnotations.create_object

        def create_then_fail(annotations):
            real_create(annotations)
            raise OSError("disk error")
        with patch.object(SessionAnnotations, "create_object", create_then_fail):
            with self.assertRaises(OSError):
                self.client.post("/video/objects", headers=self.headers())
        # Rejected creates leave the accepted revision and ID high-water mark intact.
        snapshot = self.client.get("/video/info").json()
        self.assertEqual(snapshot["annotations"]["revision"], 1)
        response = self.client.post("/video/objects", headers=self.headers())
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["created_object_id"], 2)


if __name__ == "__main__":
    unittest.main()
