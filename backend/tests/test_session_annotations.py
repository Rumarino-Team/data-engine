import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
from fastapi.testclient import TestClient
from fastapi import HTTPException

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.factory import create_app
from core.annotations import SessionAnnotations
from core.state import BackendState, state
from schemas.video import VideoPrompt
from sessions.cache import release_active_session
from sessions.metadata import write_session_metadata
from video import service
from video.prompts import restore_video_masker_from_prompt_events
from sessions.metadata import interactive_state_projection


OBJECT = {"id": 1, "label": "Hand", "color": "#ff0000"}
PROMPT = {"frame_idx": 0, "obj_id": 1, "points": [[1, 2]], "labels": [1]}


class SessionAnnotationsTests(unittest.TestCase):
    def setUp(self):
        self.previous_state = state.__dict__.copy()
        state.__dict__.update(BackendState().__dict__)
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.session = self.root / "working"
        (self.session / "frames").mkdir(parents=True)
        (self.session / "masks").mkdir()
        (self.session / "frames" / "00000.jpg").touch()
        state.active_session_dir = self.session
        state.active_session_id = "annotations-session"
        state.video_dir = str(self.session / "frames")
        state.video_frame_files = ["00000.jpg"]
        state.annotations = SessionAnnotations(objects={1: OBJECT})
        self.masker = Mock(online_mode=True, default_batch_size=1,
                           offload_video_to_cpu=True, offload_state_to_cpu=True)
        self.masker.add_new_points_or_box.return_value = (0, [1], [np.ones((3, 3))])
        state.video_masker = self.masker
        self.client = TestClient(create_app())
        def preconditions(request):
            request.headers.setdefault("X-Session-ID", state.active_session_id or "absent")
            request.headers.setdefault("X-Annotation-Revision", str(state.annotations.revision))
        self.client.event_hooks["request"].append(preconditions)

    def tearDown(self):
        state.__dict__.clear()
        state.__dict__.update(self.previous_state)
        self.tmp.cleanup()

    def metadata(self):
        return json.loads((state.active_session_dir / "session.json").read_text())

    def reopen(self):
        saved_path = state.active_session_dir
        state.annotations = SessionAnnotations()
        return service.initialize_video_state_from_resolved_input(
            saved_path, online_mode=True, batch_size=1,
            offload_video_to_cpu=True, offload_state_to_cpu=True,
            async_loading_frames=False,
        )

    def test_new_session_initializes_and_persists_first_object(self):
        source = self.root / "source"
        source.mkdir()
        (source / "00000.jpg").touch()
        state.annotations = SessionAnnotations()
        with patch("video.io.SESSION_CACHE_ROOT", self.root / "cache"):
            result = service.initialize_video_state_from_resolved_input(
                source, online_mode=True, batch_size=1,
                offload_video_to_cpu=True, offload_state_to_cpu=True,
                async_loading_frames=False,
            )
        self.assertNotIn("restored_session", result)
        annotations = self.client.get("/video/info").json()["annotations"]
        self.assertEqual(list(annotations["objects"]), ["1"])
        self.assertEqual(annotations["objects"]["1"]["label"], "Object 1")
        self.assertEqual(annotations["next_object_id"], 2)
        self.assertEqual(annotations["revision"], 1)
        self.assertEqual(annotations, self.metadata()["annotations"])
        self.reopen()
        self.assertEqual(state.annotations.snapshot(), annotations)

    def test_accepted_prompt_is_persisted_and_snapshot_is_detached(self):
        with patch.object(service, "mask_logits_to_2d_bool", side_effect=lambda mask: mask > 0):
            response = self.client.post("/video/add_new_points_or_box", json=PROMPT)
        self.assertEqual(response.status_code, 200)
        snapshot = self.client.get("/video/info").json()
        self.assertEqual(snapshot["session_id"], "annotations-session")
        self.assertEqual(snapshot["frame_files"], ["00000.jpg"])
        self.assertEqual(self.client.get("/video/document").status_code, 404)
        self.assertEqual(snapshot["annotations"]["revision"], 1)
        self.assertEqual(snapshot["annotations"], self.metadata()["annotations"])
        self.assertEqual(response.json()["annotations"], snapshot["annotations"])
        snapshot["annotations"]["prompts"][0]["points"][0][0] = 999
        self.assertEqual(state.annotations.prompts[0].points, [[1, 2]])

    def seed_manifest(self):
        from video.masks import manifest_frame_payload
        from utils import write_mask_manifest
        payload = manifest_frame_payload({1: np.ones((3, 3), dtype=bool)})
        payload["input_revisions"] = {"1": state.annotations.input_revisions.get(1, 0)}
        payload["overlay_path"] = "frame_00000_masks.png"
        write_mask_manifest(self.session / "masks" / "manifest.json",
                            {"num_frames": 1, "frames": {"0": payload}})
        (self.session / "masks" / "frame_00000_masks.png").write_bytes(b"overlay")

    def add_override(self, foreground=False):
        with patch.object(service.cv2, "imread", return_value=np.zeros((3, 3, 3))):
            return self.client.post("/video/add_new_mask", json={
                "frame_idx": 0, "obj_id": 1, "mask": np.full((3, 3), foreground).tolist()})

    def test_empty_override_wins_persists_and_replays_without_frontend_cache(self):
        self.seed_manifest()
        response = self.add_override()
        self.assertEqual(response.status_code, 200)
        mask_data = response.json()["mask_data"]
        self.assertEqual(mask_data["objects"]["1"]["rle"], [])
        self.assertEqual(mask_data["sources"], {"1": "override"})
        self.assertEqual(self.client.get("/video/mask_data/0").json(), mask_data)
        self.assertEqual(self.client.get("/video/mask_frame/0").status_code, 404)
        self.assertEqual(self.metadata()["annotations"]["mask_overrides"][0]["counts"], [9])
        state.interactive_masks = {}
        self.masker.add_new_mask.reset_mock()
        self.reopen()
        self.masker.add_new_mask.assert_called_once()
        self.assertFalse(self.masker.add_new_mask.call_args.kwargs["mask"].any())
        self.assertEqual(self.client.get("/video/mask_data/0").json()["objects"], mask_data["objects"])

    def test_prompt_replaces_override_and_clear_does_not_resurrect_propagation(self):
        self.seed_manifest()
        self.add_override(True)
        with patch.object(service, "mask_logits_to_2d_bool", side_effect=lambda mask: mask > 0):
            response = self.client.post("/video/add_new_points_or_box", json=PROMPT)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(state.annotations.mask_overrides, [])
        mask_data = response.json()["mask_data"]
        self.assertEqual(mask_data["sources"], {"1": "live"})
        self.assertEqual(self.client.get("/video/mask_data/0").json(), mask_data)
        self.reopen()
        self.assertEqual(self.client.get("/video/mask_data/0").json()["objects"], mask_data["objects"])
        self.client.post("/video/clear_all_prompts_in_frame", params={"frame_idx": 0, "obj_id": 1})
        self.assertEqual(self.client.get("/video/mask_data/0").json()["objects"], {})
        self.reopen()
        self.assertEqual(self.client.get("/video/mask_data/0").json()["objects"], {})

    def test_mask_preconditions_and_validation_reject_without_runtime_mutation(self):
        request = {"frame_idx": 0, "obj_id": 1, "mask": [[True]]}
        for headers, payload, expected in [
            ({"X-Session-ID": "other"}, request, 409),
            ({"X-Annotation-Revision": "99"}, request, 409),
            ({}, {**request, "mask": [[True], []]}, 422),
            ({}, {**request, "obj_id": 7}, 404),
            ({}, {**request, "frame_idx": 2}, 404),
            ({}, request, 400),
        ]:
            with self.subTest(payload=payload, headers=headers), patch.object(service.cv2, "imread", return_value=np.zeros((3, 3, 3))):
                self.assertEqual(self.client.post("/video/add_new_mask", json=payload, headers=headers).status_code, expected)
        self.masker.add_new_mask.assert_not_called()
        self.masker.clear_all_prompts_in_frame.assert_not_called()
        self.assertEqual(state.annotations.revision, 0)

    def test_failed_override_persistence_restores_previous_mask_and_predictor(self):
        self.add_override(True)
        before = self.metadata()
        with patch.object(service, "write_session_metadata", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                self.add_override(False)
        self.assertEqual(state.annotations.snapshot(), before["annotations"])
        self.assertEqual(self.metadata(), before)
        self.assertTrue(self.masker.add_new_mask.call_args.kwargs["mask"].all())
        self.assertEqual(self.client.get("/video/mask_data/0").json()["objects"]["1"]["rle"], [[0, 9]])

    def test_input_changes_invalidate_masks_but_label_changes_do_not(self):
        state.annotations.set_prompt(VideoPrompt(**PROMPT))
        self.seed_manifest()
        self.assertEqual(self.client.get("/video/mask_data/0").json()["sources"], {"1": "manifest"})
        response = self.client.put("/video/objects/1", json={"label": "Renamed", "color": "#00ff00"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.client.get("/video/mask_data/0").json()["sources"], {"1": "manifest"})
        state.annotations.set_prompt(VideoPrompt(**{**PROMPT, "points": [[2, 2]]}))
        self.assertEqual(self.client.get("/video/mask_data/0").json()["objects"], {})
        self.assertEqual(self.client.get("/video/mask_frame/0").status_code, 404)
        state.annotations.set_prompt(VideoPrompt(**PROMPT))
        self.assertEqual(self.client.get("/video/mask_data/0").json()["objects"], {})

    def test_clear_delete_and_reset_invalidate_overrides_after_reopen(self):
        for endpoint, params in [("clear_all_prompts_in_frame", {"frame_idx": 0, "obj_id": 1}),
                                 ("reset_state", {}), ("remove_object", {"obj_id": 1})]:
            with self.subTest(endpoint=endpoint):
                self.add_override()
                response = self.client.post("/video/" + endpoint, params=params)
                self.assertEqual(response.status_code, 200)
                self.reopen()
                self.assertEqual(state.annotations.mask_overrides, [])
                self.assertEqual(self.client.get("/video/mask_data/0").json()["objects"], {})

    def test_reset_logically_invalidates_masks_when_file_cleanup_fails(self):
        self.seed_manifest()
        with patch.object(service.shutil, "rmtree"):
            self.assertEqual(self.client.post("/video/reset_state").status_code, 200)
        self.assertTrue((self.session / "masks" / "manifest.json").exists())
        self.assertEqual(self.client.get("/video/mask_data/0").json()["objects"], {})

    def test_invalid_prompts_never_reach_predictor_or_change_revision(self):
        invalid = [
            {**PROMPT, "labels": []}, {**PROMPT, "points": [[1]]},
            {**PROMPT, "labels": [2]}, {**PROMPT, "obj_id": 99},
            {**PROMPT, "box": [3, 3, 1, 1]},
        ]
        for prompt in invalid:
            with self.subTest(prompt=prompt):
                self.assertIn(self.client.post("/video/add_new_points_or_box", json=prompt).status_code,
                              (404, 422))
        self.masker.add_new_points_or_box.assert_not_called()
        self.assertEqual(state.annotations.revision, 0)

    def test_failed_computation_does_not_accept_prompt(self):
        self.masker.add_new_points_or_box.side_effect = RuntimeError("inference failed")
        with self.assertRaisesRegex(RuntimeError, "inference failed"):
            self.client.post("/video/add_new_points_or_box", json=PROMPT)
        self.assertEqual(state.annotations.prompts, [])
        self.assertEqual(state.annotations.revision, 0)

    def test_failed_inference_rebuilds_previous_accepted_prompts_and_advances_epoch(self):
        state.annotations.set_prompt(VideoPrompt(**PROMPT))
        write_session_metadata()
        before = state.annotations.snapshot()
        epoch = state.video_state_epoch
        self.masker.add_new_points_or_box.side_effect = [RuntimeError("partial inference"), (0, [1], [np.ones((3, 3))])]
        with self.assertRaisesRegex(RuntimeError, "partial inference"):
            self.client.post("/video/add_new_points_or_box", json={**PROMPT, "points": [[8, 9]]})
        self.assertEqual(state.annotations.snapshot(), before)
        self.assertEqual(self.metadata()["annotations"], before)
        self.masker.reset_state.assert_called_once()
        self.assertEqual(self.masker.add_new_points_or_box.call_args.kwargs["points"], [[1.0, 2.0]])
        self.assertEqual(state.video_state_epoch, epoch + 1)
        self.assertTrue(self.client.get("/video/info").json()["predictor_ready"])

    def test_failed_runtime_recovery_disables_predictor_and_preserves_annotations(self):
        state.annotations.set_prompt(VideoPrompt(**PROMPT))
        before = state.annotations.snapshot()
        self.masker.add_new_points_or_box.side_effect = RuntimeError("inference and replay failed")
        response = self.client.post("/video/add_new_points_or_box", json=PROMPT)
        self.assertEqual(response.status_code, 503)
        self.assertIsNone(state.video_masker)
        self.assertEqual(state.annotations.snapshot(), before)
        self.assertFalse(self.client.get("/video/info").json()["predictor_ready"])
        self.assertEqual(self.client.post("/video/add_new_points_or_box", json=PROMPT).status_code, 400)

    def test_failed_prompt_persistence_rejects_candidate_and_restores_runtime(self):
        write_session_metadata()
        before = state.annotations.snapshot()
        with patch.object(service, "mask_logits_to_2d_bool", side_effect=lambda mask: mask > 0), \
                patch("video.service.write_session_metadata", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                self.client.post("/video/add_new_points_or_box", json=PROMPT)
        self.assertEqual(state.annotations.snapshot(), before)
        self.assertEqual(self.metadata()["annotations"], before)
        self.masker.reset_state.assert_called_once()

    def test_failed_clear_and_reset_preserve_accepted_prompts(self):
        state.annotations.set_prompt(VideoPrompt(**PROMPT))
        before = state.annotations.snapshot()
        self.masker.clear_all_prompts_in_frame.side_effect = RuntimeError("clear failed")
        with self.assertRaisesRegex(RuntimeError, "clear failed"):
            self.client.post("/video/clear_all_prompts_in_frame", params={"frame_idx": 0, "obj_id": 1})
        self.assertEqual(state.annotations.snapshot(), before)
        self.masker.reset_state.side_effect = [RuntimeError("reset failed"), None]
        with self.assertRaisesRegex(RuntimeError, "reset failed"):
            self.client.post("/video/reset_state")
        self.assertEqual(state.annotations.snapshot(), before)
        self.assertEqual(self.masker.add_new_points_or_box.call_args.kwargs["points"], [[1.0, 2.0]])

    def test_failed_mask_encoding_does_not_publish_prompt_acceptance(self):
        write_session_metadata()
        before = state.annotations.snapshot()
        with patch.object(service, "mask_logits_to_2d_bool", side_effect=lambda mask: mask > 0), \
                patch.object(service, "encode_live_mask_payload", side_effect=ValueError("bad mask")):
            with self.assertRaises(ValueError):
                self.client.post("/video/add_new_points_or_box", json=PROMPT)
        self.assertEqual(state.annotations.snapshot(), before)
        self.assertEqual(self.metadata()["annotations"], before)
        self.masker.reset_state.assert_called_once()

    def test_save_without_frontend_annotations_preserves_backend_annotations(self):
        state.annotations.set_prompt(VideoPrompt(**PROMPT))
        write_session_metadata()
        with patch.object(service, "SAVED_ROOT", self.root / "saved"):
            response = self.client.post("/video/save", json={"name": "review"})
        self.assertEqual(response.status_code, 200)
        expected = state.annotations.snapshot()
        restored = self.reopen()
        self.assertEqual(state.annotations.snapshot(), expected)
        self.assertEqual(restored["restored_session"]["interactive_state"]["points"][0]["x"], 1)
        self.masker.add_new_points_or_box.assert_called_once_with(
            frame_idx=0, obj_id=1, points=[[1, 2]], labels=[1], clear_old_points=True, box=None,
        )

    def test_stale_frontend_save_cannot_replace_backend_prompts_or_restore_its_masks(self):
        state.annotations.set_prompt(VideoPrompt(**PROMPT))
        write_session_metadata()
        with patch.object(service, "SAVED_ROOT", self.root / "saved"):
            response = self.client.post("/video/save", json={"name": "review", "interactive_state": {
                "points": [{"frame_idx": 0, "obj_id": 1, "x": 99, "y": 99, "label": 1}],
                "live_masks": [{"frame_idx": 0, "obj_id": 1, "height": 1, "width": 1, "counts": [0, 1]}],
            }})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.metadata()["interactive_state"]["points"][0]["x"], 1)
        self.assertEqual(self.metadata()["interactive_state"]["live_masks"], [])
        self.assertEqual(self.metadata()["annotations"], state.annotations.snapshot())

    def test_delete_then_reopen_does_not_resurrect_inputs_or_manifest_masks(self):
        state.annotations.set_prompt(VideoPrompt(**PROMPT))
        write_session_metadata()
        manifest = {"num_frames": 1, "frames": {"0": {"objects": {"1": {"size": [1, 1], "rle": [[0, 1]]}}}}}
        (self.session / "masks" / "manifest.json").write_text(json.dumps(manifest))
        self.assertEqual(self.client.post("/video/remove_object", params={"obj_id": 1}).status_code, 200)
        # Inject an obsolete frontend cache to verify canonical precedence during loading.
        metadata = self.metadata()
        metadata["interactive_state"]["points"] = [
            {"frame_idx": 0, "obj_id": 1, "x": 1, "y": 2, "label": 1}]
        (self.session / "session.json").write_text(json.dumps(metadata))
        restored = self.reopen()["restored_session"]
        self.assertEqual(state.annotations.objects, {})
        self.assertEqual(state.annotations.next_object_id, 2)
        self.assertEqual(state.annotations.prompts, [])
        self.assertEqual(restored["interactive_state"]["points"], [])
        self.assertEqual(restored["annotations"]["next_object_id"], 2)
        self.assertEqual(self.client.get("/video/mask_data/0").json()["objects"], {})
        self.assertEqual(self.client.post("/video/objects").json()["created_object_id"], 2)

    def test_clear_and_reset_survive_reopening(self):
        for path, params in [("/video/clear_all_prompts_in_frame", {"frame_idx": 0, "obj_id": 1}),
                             ("/video/reset_state", {})]:
            with self.subTest(path=path):
                state.annotations.set_prompt(VideoPrompt(**PROMPT))
                write_session_metadata()
                response = self.client.post(path, params=params)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["annotations"], state.annotations.snapshot())
                self.reopen()
                self.assertEqual(state.annotations.prompts, [])
                self.assertEqual(self.metadata()["interactive_state"]["points"], [])
                self.assertIn(1, state.annotations.objects)

    def test_box_prompts_survive_reconstruction_without_changing_revision(self):
        state.annotations.set_prompt(VideoPrompt(frame_idx=0, obj_id=1, box=[0, 0, 2, 2]))
        expected = state.annotations.snapshot()
        restore_video_masker_from_prompt_events(
            online_mode=True, batch_size=1, offload_video_to_cpu=True, offload_state_to_cpu=True,
        )
        self.assertEqual(state.annotations.snapshot(), expected)
        self.masker.add_new_points_or_box.assert_called_once_with(
            frame_idx=0, obj_id=1, points=None, labels=None, clear_old_points=True, box=[0, 0, 2, 2],
        )
        self.reopen()
        self.assertEqual(state.annotations.snapshot(), expected)

    def test_session_without_canonical_annotations_is_rejected(self):
        (self.session / "session.json").write_text(json.dumps({"session_id": "annotations-session", "objects": [], "interactive_state": {
            "points": [{"frame_idx": 0, "obj_id": 7, "x": 1, "y": 2, "label": 1}],
        }}))
        with self.assertRaisesRegex(HTTPException, "Invalid session annotations or editor state"):
            self.reopen()

    def test_corrupt_annotations_is_rejected(self):
        (self.session / "session.json").write_text(json.dumps({"session_id": "annotations-session", "annotations": {"version": 999}}))
        with self.assertRaisesRegex(HTTPException, "Invalid session annotations or editor state"):
            self.reopen()

    def test_invalid_editor_state_is_rejected_without_partial_restore(self):
        write_session_metadata()
        metadata = self.metadata()
        metadata["interactive_state"] = {"points": [{"invalid": True}]}
        (self.session / "session.json").write_text(json.dumps(metadata))
        with self.assertRaisesRegex(HTTPException, "Invalid session annotations or editor state"):
            self.reopen()

    def test_failed_metadata_replacement_keeps_previous_complete_annotations(self):
        write_session_metadata()
        previous = self.metadata()
        state.annotations.set_prompt(VideoPrompt(**PROMPT))
        with patch("sessions.metadata.os.replace", side_effect=OSError("disk unavailable")):
            with self.assertRaises(OSError):
                write_session_metadata()
        self.assertEqual(self.metadata(), previous)
        self.assertEqual(list(self.session.glob(".session-*.json")), [])

    def test_release_clears_annotations_and_snapshot_reports_no_session(self):
        state.annotations.set_prompt(VideoPrompt(**PROMPT))
        with patch("core.runtime.cleanup_cuda_memory"):
            release_active_session()
        self.assertEqual(state.annotations.snapshot(), SessionAnnotations().snapshot())
        response = self.client.get("/video/info")
        self.assertEqual(response.status_code, 200)
        self.assertIsNone(response.json()["annotations"])
        self.assertIsNone(response.json()["session_id"])

    def test_replacing_prompts_keeps_unrelated_frames_and_projects_appends(self):
        annotations = SessionAnnotations(objects={1: OBJECT})
        annotations.set_prompt(VideoPrompt(**PROMPT))
        annotations.set_prompt(VideoPrompt(**{**PROMPT, "frame_idx": 1}))
        annotations.set_prompt(VideoPrompt(**{**PROMPT, "points": [[2, 2]], "clear_old_points": False}))
        self.assertEqual(len(annotations.interactive_points()), 3)
        annotations.set_prompt(VideoPrompt(**{**PROMPT, "points": [[3, 3]]}))
        self.assertEqual([point["x"] for point in annotations.interactive_points()], [1, 3])
        annotations.upsert_object(OBJECT)
        self.assertEqual(annotations.revision, 4)

    def test_equivalent_frontend_point_order_keeps_generated_mask_cache(self):
        state.annotations.set_prompt(VideoPrompt(**PROMPT))
        state.annotations.set_prompt(VideoPrompt(**{**PROMPT, "frame_idx": 1}))
        points = state.annotations.interactive_points()
        mask = {"frame_idx": 0, "obj_id": 1, "height": 1, "width": 1, "counts": [0, 1]}
        projection = interactive_state_projection({"points": list(reversed(points)), "live_masks": [mask]})
        self.assertEqual(projection["points"], points)
        self.assertEqual(projection["live_masks"], [])


if __name__ == "__main__":
    unittest.main()
