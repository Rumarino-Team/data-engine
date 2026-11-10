"""Video editing endpoints: accepting prompts and masks, failing safely, and persisting across reopen."""
from contextlib import ExitStack
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.annotations import SessionAnnotations
from core.errors import OperationError
from support import state
from schemas.video import VideoPrompt
from sessions.cache import release_active_session
from sessions.metadata import interactive_state_projection, session_metadata_document
from video import service
from video.mask_encoding import encode_live_mask_payload
from video.prompts import create_masker_runtime
from video.runtime import MaskerSettings
from annotation_delta import apply_delta, same_annotations
from support import BackendStateTestCase


OBJECT = {"id": 1, "label": "Hand", "color": "#ff0000"}
PROMPT = {"frame_idx": 0, "obj_id": 1, "points": [[1, 2]], "labels": [1]}
CLEAR = ("clear_all_prompts_in_frame", {"frame_idx": 0, "obj_id": 1})
RESET = ("reset_state", {})
REMOVE = ("remove_object", {"obj_id": 1})
NEW_POINT = {**PROMPT, "points": [[8, 9]]}


class VideoEditTests(BackendStateTestCase):
    def setUp(self):
        super().setUp()
        self.root = self.temp_dir()
        self.masker = self.make_masker()
        self.masker.add_new_points_or_box.return_value = (0, [1], [np.ones((3, 3))])
        self.client = self.make_client()
        self.start_fresh_session()

    # --- helpers -----------------------------------------------------------------------------------------

    def start_fresh_session(self):
        """A clean working session with one object; table tests call this before each case."""
        self.reset_backend_state()
        self.session = self.working_session(
            self.temp_dir() / "working", session_id="annotations-session",
            annotations=SessionAnnotations(objects={1: OBJECT}))
        self.masker.reset_mock(side_effect=True)
        state.video_masker = self.masker

    def metadata(self):
        """The document Save would write: the accepted state, built from memory."""
        document = json.loads(json.dumps(session_metadata_document(state)))
        document.pop("updated_at")
        return document

    def save_checkpoint(self, name="checkpoint"):
        with patch.object(service, "SAVED_ROOT", self.root / "saved"):
            response = self.client.post("/video/save", json={"name": name, "save_as": True})
        self.assertEqual(response.status_code, 200)
        return Path(response.json()["saved_path"])

    def reopen(self, checkpoint=None):
        """Open a checkpoint (saving the working copy first by default) as a new working copy."""
        checkpoint = checkpoint or self.save_checkpoint()
        state.annotations = SessionAnnotations()
        with patch("video.io.SESSION_CACHE_ROOT", self.root / "cache"):
            return service.initialize_video_state_from_resolved_input(
                checkpoint, online_mode=True, batch_size=1,
                offload_video_to_cpu=True, offload_state_to_cpu=True,
                async_loading_frames=False, session=state,
                job=self.job_handle(),
            )

    def rewrite_checkpoint(self, checkpoint, **changes):
        metadata = json.loads((checkpoint / "session.json").read_text())
        metadata.update(changes)
        (checkpoint / "session.json").write_text(json.dumps(metadata))

    def seed_manifest(self):
        from video.masks import manifest_frame_payload
        from utils import write_mask_manifest
        state.mask_manifest_path = str(self.session / "masks" / "manifest.json")
        payload = manifest_frame_payload({1: np.ones((3, 3), dtype=bool)})
        payload["input_revisions"] = {"1": state.annotations.input_revisions.get(1, 0)}
        payload["overlay_path"] = "frame_00000_masks.png"
        write_mask_manifest(self.session / "masks" / "manifest.json", {"num_frames": 1, "frames": {"0": payload}})
        (self.session / "masks" / "frame_00000_masks.png").write_bytes(b"overlay")

    @staticmethod
    def failing_after(owner, name):
        """Let a real step run, then fail, as an edit that breaks after changing staged state."""
        real = getattr(owner, name)

        def run(*args, **kwargs):
            real(*args, **kwargs)
            raise OSError("disk full")
        return patch.object(owner, name, run)

    def add_override(self, foreground=False):
        with patch.object(service.cv2, "imread", return_value=np.zeros((3, 3, 3))):
            return self.client.post("/video/add_new_mask", json={
                "frame_idx": 0, "obj_id": 1, "mask": np.full((3, 3), foreground).tolist()})

    def override_file(self, artifact, session=None):
        return (session or state.active_session_dir) / "masks" / "overrides" / artifact

    def override_files(self, session=None):
        directory = (session or state.active_session_dir) / "masks" / "overrides"
        return sorted(path.name for path in directory.glob("*.json"))

    def post_point(self, mask, prompt=PROMPT):
        self.masker.add_new_points_or_box.return_value = (0, [1], [mask])
        with patch.object(service, "mask_logits_to_2d_bool", side_effect=lambda value: np.asarray(value, dtype=bool)):
            return self.client.post("/video/add_new_points_or_box", json=prompt)

    # --- accepting prompts -------------------------------------------------------------------------------

    def test_accepted_prompt_is_in_the_saved_document_and_snapshot_is_detached(self):
        before = state.annotations.snapshot()
        with patch.object(service, "mask_logits_to_2d_bool", side_effect=lambda mask: mask > 0):
            response = self.client.post("/video/add_new_points_or_box", json=PROMPT)
        self.assertEqual(response.status_code, 200)
        snapshot = self.client.get("/video/info").json()
        self.assertEqual(snapshot["session_id"], "annotations-session")
        self.assertEqual(snapshot["frame_files"], ["00000.jpg"])
        self.assertEqual(self.client.get("/video/document").status_code, 404)
        self.assertEqual(snapshot["annotations"]["revision"], 1)
        self.assertEqual(snapshot["annotations"], self.metadata()["annotations"])
        self.assertNotIn("annotations", response.json())
        self.assertTrue(same_annotations(apply_delta(before, response.json()["annotation_delta"]), snapshot["annotations"]))
        snapshot["annotations"]["prompts"][0]["points"][0][0] = 999
        self.assertEqual(state.annotations.prompts[0].points, [[1, 2]])

    def test_appended_clicks_stay_one_prompt_and_replay_equivalently(self):
        with patch.object(service, "mask_logits_to_2d_bool", side_effect=lambda mask: mask > 0):
            for point in ([1, 2], [2, 2], [3, 1]):
                response = self.client.post("/video/add_new_points_or_box", json={
                    **PROMPT, "points": [point], "labels": [1], "clear_old_points": False})
                self.assertEqual(response.status_code, 200)
        # Clicks reach the predictor incrementally, but annotations hold one merged prompt.
        self.assertEqual(self.masker.add_new_points_or_box.call_args.kwargs["clear_old_points"], False)
        self.assertEqual([prompt.points for prompt in state.annotations.prompts], [[[1, 2], [2, 2], [3, 1]]])
        self.assertEqual(self.metadata()["annotations"]["prompts"][0]["labels"], [1, 1, 1])
        self.masker.add_new_points_or_box.reset_mock()
        self.reopen()
        self.masker.add_new_points_or_box.assert_called_once_with(
            frame_idx=0, obj_id=1, points=[[1, 2], [2, 2], [3, 1]], labels=[1, 1, 1],
            clear_old_points=True, box=None)

    def test_invalid_prompts_never_reach_predictor_or_change_revision(self):
        invalid = [
            {**PROMPT, "labels": []}, {**PROMPT, "points": [[1]]},
            {**PROMPT, "labels": [2]}, {**PROMPT, "obj_id": 99},
            {**PROMPT, "box": [3, 3, 1, 1]},
        ]
        for prompt in invalid:
            with self.subTest(prompt=prompt):
                self.assertIn(self.client.post("/video/add_new_points_or_box", json=prompt).status_code, (404, 422))
        self.masker.add_new_points_or_box.assert_not_called()
        self.assertEqual(state.annotations.revision, 0)

    def test_point_response_encodes_each_mask_with_matching_metadata(self):
        self.masker.propagate_in_video.return_value = {}
        compact = np.zeros((108, 192), dtype=bool)
        compact[20:80, 30:140] = True
        fragmented = np.indices((17, 19)).sum(axis=0) % 2 == 0
        for name, mask in {"compact": compact, "fragmented": fragmented, "empty": np.zeros((2, 2), dtype=bool)}.items():
            with self.subTest(name):
                body = self.post_point(mask).json()
                self.assertEqual(body["mask_data"]["objects"]["1"], encode_live_mask_payload(mask))
                self.assertEqual(body["mask_shapes"], {"1": list(mask.shape)})
                self.assertEqual(body["mask_pixel_counts"], {"1": int(mask.sum())})
                self.assertFalse(body["single_frame_fallback_used"])
        self.assertEqual(self.client.get("/video/mask_data/0").json()["objects"]["1"]["rle"], [])
        self.assertEqual(self.post_point(fragmented).json()["mask_data"]["objects"]["1"]["encoding"], "packed-bits")

    def test_empty_interactive_result_falls_back_to_a_single_frame_propagation(self):
        fallback = np.array([[True, False], [False, False]])
        self.masker.propagate_in_video.return_value = {0: {1: fallback}}
        body = self.post_point(np.zeros((2, 2), dtype=bool)).json()
        self.assertTrue(body["single_frame_fallback_used"])
        self.assertEqual(body["mask_data"]["objects"]["1"], encode_live_mask_payload(fallback))
        self.assertEqual(body["mask_pixel_counts"], {"1": 1})
        self.masker.propagate_in_video.reset_mock()
        self.assertFalse(self.post_point(np.ones((2, 2), dtype=bool)).json()["single_frame_fallback_used"])
        self.masker.propagate_in_video.assert_not_called()

    def test_every_mutation_delta_reconstructs_the_snapshot(self):
        self.masker.add_new_points_or_box.side_effect = lambda **kwargs: (0, [kwargs["obj_id"]], [np.ones((3, 3))])
        steps = [
            ("post", "/video/objects", {}),
            ("post", "/video/add_new_points_or_box", {"json": {**PROMPT, "obj_id": 2}}),
            ("post", "/video/add_new_points_or_box", {"json": {**PROMPT, "points": [[2, 2]], "clear_old_points": False}}),
            ("override", None, {}),
            ("post", "/video/add_new_points_or_box", {"json": PROMPT}),
            ("put", "/video/objects/2", {"json": {"label": "Cup", "color": "#00ff00"}}),
            ("post", "/video/clear_all_prompts_in_frame", {"params": {"frame_idx": 0, "obj_id": 1}}),
            ("post", "/video/remove_object", {"params": {"obj_id": 2}}),
            ("post", "/video/reset_state", {}),
        ]
        snapshot = self.client.get("/video/info").json()["annotations"]
        with patch.object(service, "mask_logits_to_2d_bool", side_effect=lambda mask: mask > 0):
            for method, path, kwargs in steps:
                with self.subTest(path=path, kwargs=kwargs):
                    if method == "override":
                        response = self.add_override(True)
                    else:
                        response = getattr(self.client, method)(path, **kwargs)
                    self.assertEqual(response.status_code, 200)
                    self.assertNotIn("annotations", response.json())
                    snapshot = apply_delta(snapshot, response.json()["annotation_delta"])
                    self.assertTrue(same_annotations(snapshot, self.client.get("/video/info").json()["annotations"]))

    # --- masks ---------------------------------------------------------------------------------------------

    def test_current_overlay_frame_is_served(self):
        self.seed_manifest()
        response = self.client.get("/video/mask_frame/0")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b"overlay")

    def test_empty_override_wins_and_replays_without_frontend_cache(self):
        self.seed_manifest()
        response = self.add_override()
        self.assertEqual(response.status_code, 200)
        mask_data = response.json()["mask_data"]
        self.assertEqual(mask_data["objects"]["1"]["rle"], [])
        self.assertEqual(mask_data["sources"], {"1": "override"})
        self.assertEqual(self.client.get("/video/mask_data/0").json(), mask_data)
        self.assertEqual(self.client.get("/video/mask_frame/0").status_code, 404)
        reference = self.metadata()["annotations"]["mask_overrides"][0]
        self.assertEqual(set(reference), {"frame_idx", "obj_id", "artifact"})
        self.assertEqual(json.loads(self.override_file(reference["artifact"]).read_text())["counts"], [9])
        state.interactive_mask_files = {}
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
            with self.subTest(payload=payload, headers=headers), \
                    patch.object(service.cv2, "imread", return_value=np.zeros((3, 3, 3))):
                self.assertEqual(self.client.post("/video/add_new_mask", json=payload, headers=headers).status_code, expected)
        self.masker.add_new_mask.assert_not_called()
        self.masker.clear_all_prompts_in_frame.assert_not_called()
        self.assertEqual(state.annotations.revision, 0)

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

    def test_reset_invalidates_propagated_masks(self):
        self.seed_manifest()
        self.assertEqual(self.client.get("/video/mask_data/0").json()["sources"], {"1": "manifest"})
        self.assertEqual(self.client.post("/video/reset_state").status_code, 200)
        self.assertIsNone(state.mask_manifest_path)
        self.assertEqual(self.client.get("/video/mask_data/0").json()["objects"], {})

    def test_replaced_and_cleared_overrides_reclaim_their_artifacts(self):
        self.add_override(True)
        first = state.annotations.mask_overrides[0].artifact
        self.add_override(False)
        second = state.annotations.mask_overrides[0].artifact
        self.assertEqual(self.override_files(), [second])
        self.assertNotIn(first, json.dumps(self.metadata()))
        self.client.post("/video/clear_all_prompts_in_frame", params={"frame_idx": 0, "obj_id": 1})
        self.assertEqual(self.override_files(), [])

    # --- failing safely ----------------------------------------------------------------------------------

    def test_a_failed_prompt_edit_is_rejected_and_the_predictor_rebuilt(self):
        state.annotations.set_prompt(VideoPrompt(**PROMPT))
        before = state.annotations.snapshot()
        accepted_result = (0, [1], [np.ones((3, 3))])
        failures = [
            ("inference fails", RuntimeError, [], [RuntimeError("partial inference"), accepted_result]),
            ("mask encoding fails", ValueError,
             [patch.object(service, "encode_live_mask_payload", side_effect=ValueError("bad mask"))], None),
        ]
        for name, error, patches, inference in failures:
            with self.subTest(name), ExitStack() as stack:
                self.masker.reset_mock(side_effect=True)
                for active in patches:
                    stack.enter_context(active)
                stack.enter_context(patch.object(service, "mask_logits_to_2d_bool", side_effect=lambda mask: mask > 0))
                if inference:
                    self.masker.add_new_points_or_box.side_effect = inference
                with self.assertRaises(error):
                    self.client.post("/video/add_new_points_or_box", json=NEW_POINT)
                self.assertEqual(state.annotations.snapshot(), before)
                self.assertEqual(self.metadata()["annotations"], before)
                self.masker.reset_state.assert_called_once()
                # The predictor was rebuilt from the accepted prompt, not the rejected one.
                self.assertEqual(self.masker.add_new_points_or_box.call_args.kwargs["points"], [[1.0, 2.0]])
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

    def test_failed_override_edit_restores_previous_mask_predictor_and_files(self):
        self.add_override(True)
        before = self.metadata()
        accepted_files = self.override_files()
        with patch.object(service, "effective_mask_data", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                self.add_override(False)
        self.assertEqual(state.annotations.snapshot(), before["annotations"])
        self.assertEqual(self.metadata(), before)
        self.assertEqual(self.override_files(), accepted_files)
        self.assertTrue(self.masker.add_new_mask.call_args.kwargs["mask"].all())
        self.assertEqual(self.client.get("/video/mask_data/0").json()["objects"]["1"]["rle"], [[0, 9]])

    def test_rejected_object_mutations_preserve_inputs_artifacts_and_metadata(self):
        self.add_override(True)
        before = state.annotations.snapshot()
        metadata = self.metadata()
        artifacts = self.override_files()
        for method, path, body, step in [
            ("post", "/video/objects", None, "create_object"),
            ("put", "/video/objects/1", {"label": "Rejected", "color": "#00ff00"}, "upsert_object"),
            ("post", "/video/remove_object?obj_id=1", None, "remove_object"),
        ]:
            with self.subTest(path=path), self.failing_after(SessionAnnotations, step):
                with self.assertRaises(OSError):
                    getattr(self.client, method)(path, json=body)
            self.assertEqual(state.annotations.snapshot(), before)
            self.assertEqual(self.metadata(), metadata)
            self.assertEqual(self.override_files(), artifacts)
        self.masker.reset_state.assert_called_once()
        self.assertTrue(self.masker.add_new_mask.call_args.kwargs["mask"].all())

    def test_rejected_deletion_disables_failed_runtime_but_preserves_accepted_state(self):
        self.add_override(True)
        before = state.annotations.snapshot()
        metadata = self.metadata()
        self.masker.remove_object.side_effect = RuntimeError("partial deletion")
        self.masker.reset_state.side_effect = RuntimeError("recovery failed")
        response = self.client.post("/video/remove_object?obj_id=1")
        self.assertEqual(response.status_code, 503)
        self.assertIsNone(state.video_masker)
        self.assertEqual(state.annotations.snapshot(), before)
        # The document drops the predictor's own settings once it is disabled; the accepted inputs are unchanged.
        self.assertEqual(self.metadata()["annotations"], metadata["annotations"])
        self.assertFalse(self.client.get("/video/info").json()["predictor_ready"])

    # --- persistence and reopening -----------------------------------------------------------------------

    def test_new_session_initializes_its_first_object(self):
        source = self.root / "source"
        source.mkdir()
        (source / "00000.jpg").touch()
        state.annotations = SessionAnnotations()
        with patch("video.io.SESSION_CACHE_ROOT", self.root / "cache"):
            result = service.initialize_video_state_from_resolved_input(
                source, online_mode=True, batch_size=1,
                offload_video_to_cpu=True, offload_state_to_cpu=True,
                async_loading_frames=False, session=state,
                job=self.job_handle(),
            )
        self.assertNotIn("restored_session", result)
        annotations = self.client.get("/video/info").json()["annotations"]
        self.assertEqual(list(annotations["objects"]), ["1"])
        self.assertEqual(annotations["objects"]["1"]["label"], "Object 1")
        self.assertEqual(annotations["next_object_id"], 2)
        self.assertEqual(annotations["revision"], 1)
        self.assertEqual(annotations, self.metadata()["annotations"])
        # A freshly loaded, never-saved session has nothing to lose until its first edit.
        saved = self.client.get("/video/info").json()["saved_state"]
        self.assertEqual((saved["checkpoint_id"], saved["dirty"]), (None, False))
        edit = self.client.put("/video/objects/1", json={"label": "Renamed", "color": "#ff0000"})
        self.assertTrue(edit.json()["saved_state"]["dirty"])
        self.reopen()
        self.assertEqual(state.annotations.snapshot(), self.client.get("/video/info").json()["annotations"])

    def test_selected_manifest_survives_save_move_and_reopen(self):
        self.seed_manifest()
        original = Path(state.mask_manifest_path)
        selected = original.with_name("manifest-run.json")
        selected.write_bytes(original.read_bytes())
        state.mask_manifest_path = str(selected)
        checkpoint = self.save_checkpoint("result")
        self.assertEqual(Path(state.mask_manifest_path).name, "manifest-run.json")
        self.reopen(checkpoint)
        self.assertEqual(Path(state.mask_manifest_path).name, "manifest-run.json")
        self.assertEqual(self.client.get("/video/mask_data/0").json()["sources"], {"1": "manifest"})

    def test_save_without_frontend_annotations_preserves_backend_annotations(self):
        state.annotations.set_prompt(VideoPrompt(**PROMPT))
        checkpoint = self.save_checkpoint("review")
        expected = state.annotations.snapshot()
        restored = self.reopen(checkpoint)
        self.assertEqual(state.annotations.snapshot(), expected)
        self.assertEqual(restored["restored_session"]["annotations"]["prompts"][0]["points"], [[1, 2]])
        self.masker.add_new_points_or_box.assert_called_once_with(
            frame_idx=0, obj_id=1, points=[[1, 2]], labels=[1], clear_old_points=True, box=None,
        )

    def test_save_persists_editor_preferences_and_rejects_client_supplied_inputs(self):
        state.annotations.set_prompt(VideoPrompt(**PROMPT))
        with patch.object(service, "SAVED_ROOT", self.root / "saved"):
            # Editor state carries preferences only; client copies of accepted inputs are rejected outright.
            rejected = self.client.post("/video/save", json={"name": "review", "interactive_state": {
                "points": [{"frame_idx": 0, "obj_id": 1, "x": 99, "y": 99, "label": 1}]}})
            self.assertEqual(rejected.status_code, 422)
            response = self.client.post("/video/save", json={"name": "review", "interactive_state": {
                "selected_object_id": 1, "interaction_mode": "negative", "current_frame_idx": 0}})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.metadata()["interactive_state"]["interaction_mode"], "negative")
        self.assertEqual(self.metadata()["annotations"], state.annotations.snapshot())

    def test_clear_reset_and_delete_stay_applied_after_reopen(self):
        def seed_prompt():
            state.annotations.set_prompt(VideoPrompt(**PROMPT))

        for kind, seed in [("prompt", seed_prompt), ("override", self.add_override)]:
            for endpoint, params in [CLEAR, RESET, REMOVE]:
                with self.subTest(input=kind, endpoint=endpoint):
                    self.start_fresh_session()
                    self.seed_manifest()
                    seed()
                    before = state.annotations.snapshot()
                    response = self.client.post("/video/" + endpoint, params=params)
                    self.assertEqual(response.status_code, 200)
                    self.assertTrue(same_annotations(
                        apply_delta(before, response.json()["annotation_delta"]), state.annotations.snapshot()))
                    restored = self.reopen()["restored_session"]
                    self.assertEqual(state.annotations.prompts, [])
                    self.assertEqual(state.annotations.mask_overrides, [])
                    self.assertNotIn("points", restored["interactive_state"])
                    self.assertEqual(self.client.get("/video/mask_data/0").json()["objects"], {})
                    if endpoint == "remove_object":
                        self.assertEqual(state.annotations.objects, {})
                        self.assertEqual(restored["annotations"]["next_object_id"], 2)
                        self.assertEqual(self.client.post("/video/objects").json()["created_object_id"], 2)
                    else:
                        self.assertIn(1, state.annotations.objects)

    def test_box_prompts_survive_reconstruction_without_changing_revision(self):
        state.annotations.set_prompt(VideoPrompt(frame_idx=0, obj_id=1, box=[0, 0, 2, 2]))
        expected = state.annotations.snapshot()
        create_masker_runtime(state).restore(MaskerSettings(True, 1, True, True))
        self.assertEqual(state.annotations.snapshot(), expected)
        self.masker.add_new_points_or_box.assert_called_once_with(
            frame_idx=0, obj_id=1, points=None, labels=None, clear_old_points=True, box=[0, 0, 2, 2],
        )
        self.reopen()
        self.assertEqual(state.annotations.snapshot(), expected)

    def test_reopen_sweeps_orphans_and_rejects_missing_artifacts(self):
        self.add_override(True)
        accepted = self.override_files()
        self.override_file("0" * 32 + ".json").write_text("{}")
        checkpoint = self.save_checkpoint()
        self.reopen(checkpoint)
        self.assertEqual(self.override_files(), accepted)
        self.override_file(accepted[0], checkpoint).unlink()
        with self.assertRaisesRegex(OperationError, "Mask override artifact is unavailable"):
            self.reopen(checkpoint)

    def test_invalid_checkpoints_are_rejected_without_partial_restore(self):
        corruptions = [
            ("no canonical annotations", "Invalid session annotations or editor state", lambda c: (c / "session.json").write_text(json.dumps({
                "session_id": "annotations-session", "checkpoint_id": "cp", "display_name": "Name", "objects": [],
                "interactive_state": {"points": [{"frame_idx": 0, "obj_id": 7, "x": 1, "y": 2, "label": 1}]}}))),
            ("corrupt annotations", "Invalid session annotations or editor state",
             lambda c: self.rewrite_checkpoint(c, annotations={"version": 999})),
            ("invalid editor state", "Invalid session annotations or editor state",
             lambda c: self.rewrite_checkpoint(c, interactive_state={"points": [{"invalid": True}]})),
            ("missing checkpoint identity", "checkpoint identity", lambda c: self.rewrite_checkpoint(c, checkpoint_id=None)),
        ]
        for name, message, corrupt in corruptions:
            with self.subTest(name):
                checkpoint = self.save_checkpoint(name)
                corrupt(checkpoint)
                with self.assertRaisesRegex(OperationError, message):
                    self.reopen(checkpoint)

    def test_session_metadata_survives_save_and_reopen(self):
        state.session_metadata.update({"created_at": "then", "latest_tracking_result_id": "abc"})
        state.annotations.set_prompt(VideoPrompt(**PROMPT))
        self.reopen()
        self.assertEqual(state.session_metadata["created_at"], "then")
        self.assertEqual(state.session_metadata["latest_tracking_result_id"], "abc")

    def test_release_clears_annotations_and_snapshot_reports_no_session(self):
        state.annotations.set_prompt(VideoPrompt(**PROMPT))
        with patch("core.runtime.cleanup_cuda_memory"):
            release_active_session(state)
        self.assertEqual(state.annotations.snapshot(), SessionAnnotations().snapshot())
        response = self.client.get("/video/info")
        self.assertEqual(response.status_code, 200)
        self.assertIsNone(response.json()["annotations"])
        self.assertIsNone(response.json()["session_id"])

    # --- the annotation model ---------------------------------------------------------------------------

    def test_replacing_prompts_keeps_unrelated_frames_and_merges_appends(self):
        annotations = SessionAnnotations(objects={1: OBJECT})
        annotations.set_prompt(VideoPrompt(**PROMPT))
        annotations.set_prompt(VideoPrompt(**{**PROMPT, "frame_idx": 1}))
        annotations.set_prompt(VideoPrompt(**{**PROMPT, "points": [[2, 2]]}), clear_old_points=False)
        self.assertEqual(len(annotations.prompts), 2)
        self.assertEqual(next(p for p in annotations.prompts if p.frame_idx == 0).points, [[1, 2], [2, 2]])
        annotations.set_prompt(VideoPrompt(**{**PROMPT, "points": [[3, 3]]}))
        self.assertEqual({p.frame_idx: p.points for p in annotations.prompts}, {0: [[3, 3]], 1: [[1, 2]]})
        annotations.upsert_object(OBJECT)
        self.assertEqual(annotations.revision, 4)

    def test_duplicate_prompt_keys_are_rejected(self):
        with self.assertRaises(ValueError):
            SessionAnnotations(objects={1: OBJECT}, prompts=[PROMPT, {**PROMPT, "points": [[5, 5]]}])

    def test_editor_state_projection_drops_deleted_selection(self):
        projection = interactive_state_projection({"selected_object_id": 7, "interaction_mode": "positive"}, session=state)
        self.assertEqual(projection, {"selected_object_id": None, "interaction_mode": "positive", "version": 1})


if __name__ == "__main__":
    unittest.main()
