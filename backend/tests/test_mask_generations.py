"""Which mask a frame shows when results were made under different inputs, and which stretches are marked outdated.

A mask the user already has is never hidden by a later edit. For each object a frame shows the newest of its live
result (a click) and its propagated result, and the editor is told which stretches predate the object's latest
point change, so it can mark them. Marking is information: those masks are still shown and exported.
"""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.annotations import SessionAnnotations
from schemas.video import MaskOverride, VideoPrompt
from support import SessionStateTestCase, state
from utils import load_mask_manifest, write_mask_manifest
from video import edits
from video.mask_encoding import encode_live_mask_payload
from video.masks import manifest_frame_payload, outdated_mask_ranges, retain_interactive_masks, store_mask_override

BOY = {"id": 1, "label": "Boy", "color": "#112233"}
GOAL = {"id": 2, "label": "Goal", "color": "#445566"}
FRAMES = tuple(f"{index:05d}.jpg" for index in range(6))
SHAPE = (3, 3)


def prompt(frame_idx=0, obj_id=1):
    return VideoPrompt(frame_idx=frame_idx, obj_id=obj_id, points=[[1, 2]], labels=[1])


class MaskGenerationTests(SessionStateTestCase):
    def setUp(self):
        super().setUp()
        self.masker = self.make_masker()
        self.client = self.make_client()
        self.reset_backend_state()
        self.session = self.working_session(
            self.temp_dir() / "working", session_id="generations", frames=FRAMES,
            annotations=SessionAnnotations(objects={1: BOY, 2: GOAL}))
        state.video_masker = self.masker

    # --- fixtures: results made under chosen input revisions ------------------------------------------------

    def write_manifest(self, frames):
        """A propagation result: frame -> {object: input revision it was made under}."""
        payload = {}
        for frame, stamps in frames.items():
            entry = manifest_frame_payload({obj: np.ones(SHAPE, dtype=bool) for obj in stamps})
            entry["input_revisions"] = {str(obj): stamp for obj, stamp in stamps.items()}
            payload[str(frame)] = entry
        path = self.session / "masks" / f"manifest-{uuid4().hex}.json"
        write_mask_manifest(path, {"num_frames": len(FRAMES), "frames": payload})
        state.mask_manifest_path = str(path)

    def write_live(self, frame, results):
        """A click result: object -> (input revision, has pixels)."""
        payload = {
            str(obj): {"input_revision": stamp,
                       "mask": encode_live_mask_payload(np.ones(SHAPE, dtype=bool) if pixels else np.zeros(SHAPE, dtype=bool))}
            for obj, (stamp, pixels) in results.items()
        }
        name = f"{uuid4().hex}.json"
        write_mask_manifest(self.session / "masks" / "interactive" / name, payload)
        state.interactive_mask_files[str(frame)] = name

    def edited(self, obj_id, revision):
        """As if the object's points had been edited up to this revision."""
        state.annotations.input_revisions[obj_id] = revision

    def shown(self, frame):
        return self.client.get(f"/video/mask_data/{frame}").json()

    # --- masks the user has stay ---------------------------------------------------------------------------

    def test_masks_made_under_older_inputs_are_still_shown(self):
        self.write_manifest({frame: {1: 2, 2: 2} for frame in range(4)})
        self.edited(1, 5)
        for frame in range(4):
            with self.subTest(frame=frame):
                self.assertEqual(self.shown(frame)["sources"], {"1": "manifest", "2": "manifest"})

    def test_propagating_one_segment_leaves_every_other_frame_visible_and_only_marks_the_rest(self):
        self.write_manifest({frame: {1: 1} for frame in range(6)})
        self.edited(1, 4)
        self.assertEqual(outdated_mask_ranges(state), {"1": [[0, 5]]})
        # A propagation over frames 2 and 3 replaces just those frames, under the current inputs.
        self.write_manifest({**{frame: {1: 1} for frame in (0, 1, 4, 5)}, 2: {1: 4}, 3: {1: 4}})
        for frame in range(6):
            self.assertEqual(self.shown(frame)["sources"], {"1": "manifest"})
        self.assertEqual(outdated_mask_ranges(state), {"1": [[0, 1], [4, 5]]})

    # --- the newest result wins per frame ------------------------------------------------------------------

    def test_a_frame_shows_the_newest_of_its_click_and_its_propagation(self):
        self.write_manifest({0: {1: 3}, 1: {1: 5}, 2: {1: 6}, 3: {1: 3}})
        self.edited(1, 6)
        for frame, live in {0: (5, True), 1: (5, True), 2: (5, True), 3: (5, False)}.items():
            self.write_live(frame, {1: live})
        self.assertEqual(self.shown(0)["sources"], {"1": "live"})  # the click is newer than the propagation
        self.assertEqual(self.shown(1)["sources"], {"1": "manifest"})  # a tie: the propagation ran after the click
        self.assertEqual(self.shown(2)["sources"], {"1": "manifest"})  # propagated again after the click
        empty = self.shown(3)  # a newer empty click is a decision: it hides the older propagated mask
        self.assertEqual(empty["sources"], {"1": "live"})
        self.assertEqual(empty["objects"]["1"]["rle"], [])

    def test_an_explicit_override_beats_everything(self):
        self.write_manifest({0: {1: 0}})
        self.write_live(0, {1: (99, True)})
        override = store_mask_override(state.active_session_dir, 0, 1, np.ones(SHAPE, dtype=bool))
        state.annotations.mask_overrides.append(override)
        self.assertEqual(self.shown(0)["sources"], {"1": "override"})

    # --- which stretches are marked ------------------------------------------------------------------------

    def test_outdated_stretches_follow_the_mask_that_is_shown(self):
        override = store_mask_override(state.active_session_dir, 2, 1, np.ones(SHAPE, dtype=bool))
        state.annotations.set_mask(MaskOverride(frame_idx=2, obj_id=1, artifact=override.artifact))
        latest = state.annotations.input_revisions[1]
        self.assertGreater(latest, 0)
        self.write_manifest({frame: {1: 0} for frame in range(6)})
        self.write_live(0, {1: (latest, True)})  # a click made after the latest change: current
        self.assertEqual(outdated_mask_ranges(state), {"1": [[1, 1], [3, 5]]})  # 0 is a current click, 2 is an override

    def test_a_change_to_one_object_does_not_mark_another(self):
        self.write_manifest({frame: {1: 0, 2: 0} for frame in range(3)})
        self.edited(1, 3)
        self.assertEqual(outdated_mask_ranges(state), {"1": [[0, 2]]})

    def test_objects_that_are_gone_or_have_no_masks_are_not_reported(self):
        self.write_manifest({0: {1: 0}, 1: {1: 0}})
        self.edited(1, 2)
        self.edited(2, 2)
        self.assertEqual(list(outdated_mask_ranges(state)), ["1"])  # object 2 has no masks
        state.annotations.remove_object(1)
        self.assertEqual(outdated_mask_ranges(state), {})

    def test_nothing_is_reported_for_a_session_with_no_generated_masks(self):
        self.edited(1, 4)
        self.assertEqual(outdated_mask_ranges(state), {})
        self.assertEqual(self.client.get("/video/info").json()["outdated_masks"], {})

    # --- a click keeps results only for objects prompted on that frame ------------------------------------

    def test_a_click_stores_a_result_only_for_objects_prompted_on_that_frame(self):
        state.annotations.set_prompt(prompt(0, 1))
        retain_interactive_masks(
            state, 0, [1, 2], [encode_live_mask_payload(np.ones(SHAPE, dtype=bool)), encode_live_mask_payload(np.zeros(SHAPE, dtype=bool))])
        stored = load_mask_manifest(state.active_session_dir / "masks" / "interactive" / state.interactive_mask_files["0"])
        self.assertEqual(list(stored), ["1"])

    def test_clicking_one_object_does_not_blank_another_objects_propagated_mask(self):
        self.write_manifest({0: {1: 0, 2: 0}, 1: {1: 0, 2: 0}})
        # SAM2 answers for every object on the frame; object 2 was not prompted here, so it comes back empty.
        self.masker.add_new_points_or_box.return_value = (0, [1, 2], [np.ones(SHAPE), np.zeros(SHAPE)])
        with patch.object(edits, "mask_logits_to_2d_bool", side_effect=lambda value: np.asarray(value, dtype=bool)):
            response = self.client.post("/video/add_new_points_or_box", json={
                "frame_idx": 0, "obj_id": 1, "points": [[1, 2]], "labels": [1]})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["mask_data"]["sources"], {"1": "live", "2": "manifest"})
        self.assertTrue(response.json()["mask_data"]["objects"]["2"]["rle"])  # object 2 still has its mask
        # The click moved object 1 on; object 2's masks are untouched, and so is object 1's other frame.
        self.assertEqual(response.json()["outdated_masks"], {"1": [[1, 1]]})
        self.assertEqual(self.shown(1)["sources"], {"1": "manifest", "2": "manifest"})

    # --- clearing a frame ----------------------------------------------------------------------------------

    def test_clearing_blanks_only_that_frame_and_the_rest_keep_their_masks(self):
        self.write_manifest({frame: {1: 0} for frame in range(4)})
        state.annotations.set_prompt(prompt(1, 1))
        response = self.client.post("/video/clear_all_prompts_in_frame", params={"frame_idx": 1, "obj_id": 1})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.shown(1)["objects"]["1"]["rle"], [])
        for frame in (0, 2, 3):
            self.assertEqual(self.shown(frame)["sources"], {"1": "manifest"})
            self.assertTrue(self.shown(frame)["objects"]["1"]["rle"])

    def test_clearing_a_frame_that_had_no_prompt_changes_nothing(self):
        self.write_manifest({frame: {1: 0} for frame in range(3)})
        revision = state.annotations.revision
        response = self.client.post("/video/clear_all_prompts_in_frame", params={"frame_idx": 1, "obj_id": 1})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(state.annotations.revision, revision)
        self.assertEqual(self.shown(1)["sources"], {"1": "manifest"})
        self.assertTrue(self.shown(1)["objects"]["1"]["rle"])

    def test_clearing_keeps_the_other_objects_results_on_the_frame(self):
        self.write_manifest({0: {1: 0, 2: 0}})
        state.annotations.set_prompt(prompt(0, 1))
        state.annotations.set_prompt(prompt(0, 2))
        self.write_live(0, {2: (state.annotations.input_revisions[2], True)})
        self.client.post("/video/clear_all_prompts_in_frame", params={"frame_idx": 0, "obj_id": 1})
        shown = self.shown(0)
        self.assertEqual(shown["objects"]["1"]["rle"], [])
        self.assertEqual(shown["sources"], {"1": "live", "2": "live"})
        self.assertTrue(shown["objects"]["2"]["rle"])


if __name__ == "__main__":
    unittest.main()
