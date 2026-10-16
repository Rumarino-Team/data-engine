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
        snapshot["annotations"]["prompts"][0]["points"][0][0] = 999
        self.assertEqual(state.annotations.prompts[0].points, [[1, 2]])

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

    def test_clear_and_reset_survive_reopening(self):
        for path, params in [("/video/clear_all_prompts_in_frame", {"frame_idx": 0, "obj_id": 1}),
                             ("/video/reset_state", {})]:
            with self.subTest(path=path):
                state.annotations.set_prompt(VideoPrompt(**PROMPT))
                write_session_metadata()
                self.assertEqual(self.client.post(path, params=params).status_code, 200)
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
        self.assertEqual(projection["live_masks"], [mask])


if __name__ == "__main__":
    unittest.main()
