"""Undo and redo of accepted changes.

A step restores the inputs (objects, prompts, drawn masks, the input revisions their results are stamped with), the
mask each click showed, and the tracking result reference. It never restores or removes propagated masks: those are
derived output, so undoing an old click cannot take away a propagation made after it. History lives with the session
in memory; it is never part of a checkpoint.
"""
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from annotation_delta import apply_delta, same_annotations
from core.annotations import SessionAnnotations
from core.history import HistoryState, UndoHistory
from core.jobs import capture_job_inputs
from schemas.video import VideoPrompt
from sessions.publication import publish_results
from support import SessionStateTestCase, state
from tracking.results import save_prompt_tracking_result
from utils import write_mask_manifest
from video import edits, save
from video.mask_encoding import encode_live_mask_payload  # noqa: F401  (keeps the encoder import path exercised)
from video.masks import manifest_frame_payload

BOY = {"id": 1, "label": "Boy", "color": "#112233"}
FRAMES = tuple(f"{index:05d}.jpg" for index in range(6))
SHAPE = (3, 3)


class UndoRedoTests(SessionStateTestCase):
    def setUp(self):
        super().setUp()
        self.root = self.temp_dir()
        self.masker = self.make_masker()
        self.masker.add_new_points_or_box.return_value = (0, [1], [np.ones(SHAPE)])
        self.client = self.make_client()
        self.reset_backend_state()
        self.session = self.working_session(
            self.temp_dir() / "working", session_id="undo-session", frames=FRAMES,
            annotations=SessionAnnotations(objects={1: BOY}))
        state.video_masker = self.masker

    # --- helpers ---------------------------------------------------------------------------------------------

    def click(self, *points, obj=1, frame=0):
        with patch.object(edits, "mask_logits_to_2d_bool", side_effect=lambda value: np.asarray(value, dtype=bool)):
            response = self.client.post("/video/add_new_points_or_box", json={
                "frame_idx": frame, "obj_id": obj, "points": [list(point) for point in points],
                "labels": [1] * len(points)})
        self.assertEqual(response.status_code, 200, response.text)
        return response

    def undo(self):
        return self.client.post("/video/undo")

    def redo(self):
        return self.client.post("/video/redo")

    def history(self):
        return self.client.get("/video/info").json()["history"]

    def prompts(self):
        return [(prompt.frame_idx, prompt.obj_id, prompt.points) for prompt in state.annotations.prompts]

    def write_manifest(self, stamp, frames=range(4)):
        """A propagation result: every listed frame holds a mask for object 1, made under input revision `stamp`."""
        payload = {}
        for frame in frames:
            entry = manifest_frame_payload({1: np.ones(SHAPE, dtype=bool)})
            entry["input_revisions"] = {"1": stamp}
            payload[str(frame)] = entry
        path = self.session / "masks" / f"manifest-{uuid4().hex}.json"
        write_mask_manifest(path, {"num_frames": len(FRAMES), "frames": payload})
        state.mask_manifest_path = str(path)

    def add_override(self, foreground=True):
        with patch.object(edits.cv2, "imread", return_value=np.zeros((3, 3, 3))):
            return self.client.post("/video/add_new_mask", json={
                "frame_idx": 0, "obj_id": 1, "mask": np.full(SHAPE, foreground).tolist()})

    def override_files(self, root=None):
        return sorted(path.name for path in ((root or state.active_session_dir) / "masks" / "overrides").glob("*.json"))

    def track(self):
        """A finished tracking run, accepted the way the job accepts it."""
        inputs = capture_job_inputs(state, "prompt_tracking")
        with publish_results(self.ctx, inputs, "prompt_tracking") as session:
            return save_prompt_tracking_result(
                session, model_name="test", num_points=1, num_frames=len(FRAMES), add_support_grid_used=False,
                tracking_mode="streaming", streaming_frame_threshold=256,
                points=[{"point_id": "p0", "obj_id": 1, "source_frame_idx": 0, "source_x": 1.0, "source_y": 2.0}],
                tracks=[[[1.0, 2.0]] * len(FRAMES)], visibility=[[True] * len(FRAMES)])

    # --- what becomes a step -------------------------------------------------------------------------------

    def test_each_accepted_change_is_a_step_and_its_own_response_already_counts_it(self):
        self.assertEqual(self.history(), {"undo": 0, "redo": 0, "next_undo": None, "next_redo": None})
        first = self.click((1, 2))
        self.assertEqual(first.json()["history"], {"undo": 1, "redo": 0, "next_undo": "Add points", "next_redo": None})
        created = self.client.post("/video/objects")
        self.assertEqual(created.json()["history"]["undo"], 2)
        self.assertEqual(created.json()["history"]["next_undo"], "Add object")
        renamed = self.client.put("/video/objects/1", json={"label": "Renamed", "color": "#112233"})
        self.assertEqual(renamed.json()["history"], {"undo": 3, "redo": 0, "next_undo": "Edit object", "next_redo": None})
        self.assertEqual(self.history(), renamed.json()["history"])

    # --- undo ----------------------------------------------------------------------------------------------

    def test_undo_puts_back_the_points_the_click_results_and_the_stamps(self):
        self.click((1, 2))
        first_revision = state.annotations.revision
        first_artifact = state.interactive_mask_files["0"]
        self.click((5, 6))
        second_revision = state.annotations.revision
        self.assertNotEqual(state.interactive_mask_files["0"], first_artifact)
        before = state.annotations.snapshot()
        self.masker.reset_mock()

        response = self.undo()

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["message"], "Undo: Add points")
        self.assertEqual(self.prompts(), [(0, 1, [[1.0, 2.0]])])
        self.assertEqual(state.annotations.input_revisions, {1: first_revision})
        self.assertEqual(state.interactive_mask_files["0"], first_artifact)
        self.assertEqual(state.annotations.revision, second_revision + 1)  # the counter only moves forward
        self.assertTrue(same_annotations(apply_delta(before, response.json()["annotation_delta"]), state.annotations.snapshot()))
        self.assertEqual(self.client.get("/video/mask_data/0").json()["sources"], {"1": "live"})
        # The predictor is rebuilt from the restored points.
        self.masker.reset_state.assert_called_once()
        self.masker.add_new_points_or_box.assert_called_once_with(
            frame_idx=0, obj_id=1, points=[[1.0, 2.0]], labels=[1], clear_old_points=True, box=None)
        self.assertEqual(response.json()["history"], {"undo": 1, "redo": 1, "next_undo": "Add points", "next_redo": "Add points"})

    def test_undoing_an_object_change_brings_back_the_object_its_name_and_its_masks(self):
        state.annotations.set_prompt(VideoPrompt(frame_idx=0, obj_id=1, points=[[1, 2]], labels=[1]))
        self.write_manifest(state.annotations.input_revisions[1])
        self.assertEqual(self.client.post("/video/remove_object", json=None, params={"obj_id": 1}).status_code, 200)
        self.assertEqual(self.client.get("/video/mask_data/0").json()["objects"], {})
        response = self.undo()
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(self.client.get("/video/mask_data/0").json()["sources"], {"1": "manifest"})
        self.assertEqual(self.prompts(), [(0, 1, [[1.0, 2.0]])])
        self.assertEqual(response.json()["outdated_masks"], {})

        self.client.put("/video/objects/1", json={"label": "Renamed", "color": "#112233"})
        self.assertEqual(state.annotations.objects[1].label, "Renamed")
        self.undo()
        self.assertEqual(state.annotations.objects[1].label, "Boy")

    def test_undoing_an_object_creation_does_not_reuse_its_id(self):
        self.client.post("/video/objects")
        self.assertEqual(sorted(state.annotations.objects), [1, 2])
        self.undo()
        self.assertEqual(sorted(state.annotations.objects), [1])
        self.assertEqual(self.client.post("/video/objects").json()["created_object_id"], 3)

    # --- redo ----------------------------------------------------------------------------------------------

    def test_redo_goes_forward_again_and_a_new_change_ends_it(self):
        self.click((1, 2))
        self.click((5, 6))
        self.undo()
        self.assertEqual(self.history()["redo"], 1)
        response = self.redo()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["message"], "Redo: Add points")
        self.assertEqual(self.prompts(), [(0, 1, [[5.0, 6.0]])])
        self.assertEqual(self.history(), {"undo": 2, "redo": 0, "next_undo": "Add points", "next_redo": None})
        self.undo()
        self.click((9, 9))  # something new: what could have been redone no longer can
        self.assertEqual(self.history()["redo"], 0)
        self.assertEqual(self.redo().status_code, 400)

    def test_redo_after_undoing_everything_replays_in_order(self):
        for point in [(1, 1), (2, 2), (3, 3)]:
            self.click(point)
        for _ in range(3):
            self.assertEqual(self.undo().status_code, 200)
        self.assertEqual(self.prompts(), [])
        for expected in [(1, 1), (2, 2), (3, 3)]:
            self.assertEqual(self.redo().status_code, 200)
            self.assertEqual(self.prompts(), [(0, 1, [[float(expected[0]), float(expected[1])]])])

    # --- propagated masks and undo -------------------------------------------------------------------------

    def test_undoing_an_edit_made_after_a_propagation_makes_those_masks_current_again(self):
        state.annotations.set_prompt(VideoPrompt(frame_idx=0, obj_id=1, points=[[1, 2]], labels=[1]))
        self.write_manifest(state.annotations.input_revisions[1])
        self.click((5, 6))
        self.assertEqual(self.client.get("/video/info").json()["outdated_masks"], {"1": [[1, 3]]})
        self.assertEqual(self.undo().json()["outdated_masks"], {})
        self.assertEqual(self.redo().json()["outdated_masks"], {"1": [[1, 3]]})

    def test_a_propagation_after_an_edit_survives_undoing_that_edit(self):
        state.annotations.set_prompt(VideoPrompt(frame_idx=0, obj_id=1, points=[[1, 2]], labels=[1]))
        self.write_manifest(state.annotations.input_revisions[1])
        self.click((5, 6))
        self.write_manifest(state.annotations.input_revisions[1])  # propagated again, under the edited points
        propagated = state.mask_manifest_path
        undone = self.undo()
        self.assertEqual(state.mask_manifest_path, propagated)  # undo does not take the propagation away
        for frame in range(4):
            self.assertEqual(self.client.get(f"/video/mask_data/{frame}").json()["sources"], {"1": "manifest"})
        # ...but those masks no longer match the restored points, and the timeline says so.
        self.assertEqual(undone.json()["outdated_masks"], {"1": [[0, 3]]})

    def test_a_tracking_run_is_a_step_and_a_propagation_is_not(self):
        state.annotations.set_prompt(VideoPrompt(frame_idx=0, obj_id=1, points=[[1, 2]], labels=[1]))
        steps = self.history()["undo"]
        inputs = capture_job_inputs(state, "mask_propagation")
        with publish_results(self.ctx, inputs, "mask_propagation"):
            self.write_manifest(0)
        self.assertEqual(self.history()["undo"], steps)  # a propagation is no step

        result = self.track()
        self.assertEqual(self.history(), {"undo": steps + 1, "redo": 0, "next_undo": "Track points", "next_redo": None})
        self.assertEqual(state.session_metadata["latest_tracking_result_id"], result["result_id"])
        tracking_file = state.active_session_dir / "tracking" / f"{result['result_id']}.json"

        undone = self.undo()
        self.assertEqual(undone.status_code, 200, undone.text)
        self.assertNotIn("latest_tracking_result_id", state.session_metadata)
        self.assertIsNone(undone.json()["tracking_validity"])
        self.assertTrue(tracking_file.is_file())  # kept: redo brings the result back

        redone = self.redo()
        self.assertEqual(state.session_metadata["latest_tracking_result_id"], result["result_id"])
        self.assertEqual(redone.json()["tracking_validity"]["result_id"], result["result_id"])

    # --- files ---------------------------------------------------------------------------------------------

    def test_files_undo_can_bring_back_are_kept_but_a_checkpoint_never_carries_them(self):
        self.add_override(True)
        first = state.annotations.mask_overrides[0].artifact
        self.add_override(False)
        second = state.annotations.mask_overrides[0].artifact
        self.assertEqual(self.override_files(), sorted([first, second]))
        with patch.object(save, "SAVED_ROOT", self.root / "saved"):
            response = self.client.post("/video/save", json={"name": "kept", "save_as": True})
        self.assertEqual(response.status_code, 200, response.text)
        checkpoint = Path(response.json()["saved_path"])
        self.assertEqual(self.override_files(checkpoint), [second])
        self.assertNotIn(first, json.dumps(json.loads((checkpoint / "session.json").read_text())))

    def test_undo_brings_back_a_drawn_mask_file(self):
        self.add_override(True)
        first = state.annotations.mask_overrides[0].artifact
        self.add_override(False)
        self.assertEqual(self.undo().status_code, 200)
        self.assertEqual(state.annotations.mask_overrides[0].artifact, first)
        self.assertEqual(self.client.get("/video/mask_data/0").json()["sources"], {"1": "override"})

    # --- limits --------------------------------------------------------------------------------------------

    def test_the_history_is_limited_and_the_oldest_steps_fall_off(self):
        state.history.limit = 3
        for point in [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5)]:
            self.click(point)
        self.assertEqual(self.history()["undo"], 3)
        for _ in range(3):
            self.assertEqual(self.undo().status_code, 200)
        self.assertEqual(self.prompts(), [(0, 1, [[2.0, 2.0]])])  # the two oldest steps are gone
        self.assertEqual(self.undo().status_code, 400)

    # --- refusals ------------------------------------------------------------------------------------------

    def test_nothing_to_undo_or_redo_is_refused(self):
        for response, word in ((self.undo(), "undo"), (self.redo(), "redo")):
            self.assertEqual(response.status_code, 400)
            self.assertIn(f"Nothing to {word}", response.json()["detail"])

    def test_stale_or_busy_requests_are_refused_and_change_nothing(self):
        self.click((1, 2))
        revision = state.annotations.revision
        for headers in ({"X-Session-ID": "another"}, {"X-Annotation-Revision": "999"}):
            with self.subTest(headers=headers):
                self.assertEqual(self.client.post("/video/undo", headers=headers).status_code, 409)
        self.activate_job()
        busy = self.undo()
        self.assertEqual(busy.status_code, 409)
        self.assertIn("already running", busy.json()["detail"])
        self.assertEqual(state.annotations.revision, revision)
        self.assertEqual(self.history()["undo"], 1)
        self.assertEqual(self.prompts(), [(0, 1, [[1.0, 2.0]])])

    def test_a_failed_rebuild_leaves_the_inputs_and_the_history_as_they_were(self):
        self.click((1, 2))
        self.click((5, 6))
        before = state.annotations.snapshot()
        self.masker.reset_state.side_effect = RuntimeError("gpu lost")
        failed = self.undo()
        self.assertEqual(failed.status_code, 503)
        self.assertTrue(same_annotations(before, state.annotations.snapshot()))
        self.assertEqual((len(state.history.undo), len(state.history.redo)), (2, 0))


class UndoHistoryUnitTests(unittest.TestCase):
    @staticmethod
    def step(label):
        return HistoryState(label, SessionAnnotations(), {}, {})

    def test_recording_ends_redo_and_trims_to_the_limit(self):
        history = UndoHistory(limit=2)
        for label in "abc":
            history.record(self.step(label))
        self.assertEqual([state.label for state in history.undo], ["b", "c"])
        history.redo.append(self.step("x"))
        history.record(self.step("d"))
        self.assertEqual(history.redo, [])

    def test_restore_to_never_moves_the_revision_back(self):
        earlier = SessionAnnotations(objects={1: BOY})
        later = earlier.model_copy(deep=True)
        later.set_prompt(VideoPrompt(frame_idx=0, obj_id=1, points=[[1, 2]], labels=[1]))
        later.upsert_object({**BOY, "label": "Renamed"})
        revision = later.revision
        later.restore_to(earlier)
        self.assertEqual(later.revision, revision + 1)
        self.assertEqual(later.prompts, [])
        self.assertEqual(later.objects[1].label, "Boy")
        delta = later.take_delta(revision)
        self.assertEqual(delta["removed_prompts"], [[0, 1]])
        self.assertEqual(delta["objects"]["1"]["label"], "Boy")
        self.assertEqual(delta["removed_input_revisions"], [1])


if __name__ == "__main__":
    unittest.main()
