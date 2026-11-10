import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.annotations import SessionAnnotations
from core.state import state
from sessions import artifacts
from sessions.metadata import write_session_metadata
from support import BackendStateTestCase
from video.masks import retain_interactive_masks

OBJECT = {"id": 1, "label": "Hand", "color": "#ff0000"}


class ArtifactLifecycleTests(BackendStateTestCase):
    def setUp(self):
        super().setUp()
        self.session = self.working_session(annotations=SessionAnnotations(objects={1: OBJECT}))
        self.masks = self.session / "masks"
        artifacts._read_manifest_overlays.cache_clear()

    def names(self, directory):
        return sorted(path.relative_to(self.session).as_posix() for path in (self.session / directory).rglob("*") if path.is_file())

    def touch(self, relative):
        path = self.session / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
        return path

    def manifest(self, name, overlays):
        frames = {str(i): {"objects": {}, "overlay_path": overlay} for i, overlay in enumerate(overlays)}
        (self.masks / name).write_text(json.dumps({"frames": frames}))

    def test_replaced_interactive_masks_are_deleted_only_after_commit(self):
        retain_interactive_masks(0, [1], [{"size": [1, 1], "rle": []}])
        write_session_metadata()
        first = state.interactive_mask_files["0"]
        self.assertEqual(self.names("masks/interactive"), [f"masks/interactive/{first}"])
        retain_interactive_masks(0, [1], [{"size": [1, 1], "rle": []}])
        # The new artifact exists beside the old one until metadata names it.
        self.assertEqual(len(self.names("masks/interactive")), 2)
        write_session_metadata()
        second = state.interactive_mask_files["0"]
        self.assertNotEqual(first, second)
        self.assertEqual(self.names("masks/interactive"), [f"masks/interactive/{second}"])

    def test_failed_commit_keeps_committed_artifacts_and_rejected_edit_reclaims_new_one(self):
        retain_interactive_masks(0, [1], [{"size": [1, 1], "rle": []}])
        write_session_metadata()
        committed = dict(state.interactive_mask_files)
        retain_interactive_masks(1, [1], [{"size": [1, 1], "rle": []}])
        with patch("sessions.metadata.os.replace", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                write_session_metadata()
        # Nothing was committed, so nothing was reclaimed.
        self.assertEqual(len(self.names("masks/interactive")), 2)
        state.interactive_mask_files = committed
        artifacts.sweep_unreferenced_artifacts(state)
        self.assertEqual(self.names("masks/interactive"), [f"masks/interactive/{committed['0']}"])

    def test_reset_removes_interactive_manifest_and_overlay_results(self):
        retain_interactive_masks(0, [1], [{"size": [1, 1], "rle": []}])
        self.manifest("manifest-a.json", ["overlays_a/f0.png"])
        self.touch("masks/overlays_a/f0.png")
        state.mask_manifest_path = str(self.masks / "manifest-a.json")
        write_session_metadata()
        self.assertEqual(len(self.names("masks")), 3)
        state.interactive_mask_files = {}
        state.mask_manifest_path = None
        write_session_metadata()
        self.assertEqual(self.names("masks"), [])

    def test_new_run_reclaims_previous_manifest_but_keeps_carried_forward_overlays(self):
        self.manifest("manifest-a.json", ["overlays_a/f0.png", "overlays_a/f1.png"])
        for name in ("f0", "f1"):
            self.touch(f"masks/overlays_a/{name}.png")
        state.mask_manifest_path = str(self.masks / "manifest-a.json")
        write_session_metadata()
        # Run B reprocessed frame 1 only and carries frame 0 forward from run A.
        self.manifest("manifest-b.json", ["overlays_a/f0.png", "overlays_b/f1.png"])
        self.touch("masks/overlays_b/f1.png")
        state.mask_manifest_path = str(self.masks / "manifest-b.json")
        write_session_metadata()
        self.assertEqual(self.names("masks"), ["masks/manifest-b.json", "masks/overlays_a/f0.png",
                                               "masks/overlays_b/f1.png"])

    def test_full_sweep_removes_leftovers_that_no_commit_ever_named(self):
        self.manifest("manifest-a.json", ["overlays_a/f0.png"])
        self.touch("masks/overlays_a/f0.png")
        self.touch("masks/overlays_a/stray.png")
        self.touch("masks/overlays_dead/f0.png")
        self.touch("masks/manifest-dead.json")
        self.touch("masks/interactive/" + "0" * 32 + ".json")
        self.touch("masks/overrides/" + "1" * 32 + ".json")
        self.touch("tracking/dead.json")
        self.touch("masks/manifest.json")
        state.mask_manifest_path = str(self.masks / "manifest-a.json")
        write_session_metadata()
        # Commits only reclaim what an earlier commit named, so leftovers survive until a sweep.
        self.assertGreater(len(self.names("masks")), 3)
        artifacts.sweep_unreferenced_artifacts(state)
        self.assertEqual(self.names("masks"), ["masks/manifest-a.json", "masks/manifest.json", "masks/overlays_a/f0.png"])
        self.assertEqual(self.names("tracking"), [])

    def test_overlay_paths_with_windows_separators_are_recognized(self):
        self.manifest("manifest-a.json", ["overlays_a\\f0.png"])
        self.touch("masks/overlays_a/f0.png")
        self.touch("masks/overlays_a/stray.png")
        state.mask_manifest_path = str(self.masks / "manifest-a.json")
        write_session_metadata()
        artifacts.sweep_unreferenced_artifacts(state)
        self.assertEqual(self.names("masks/overlays_a"), ["masks/overlays_a/f0.png"])

    def test_unreadable_manifest_keeps_every_result_file(self):
        (self.masks / "manifest-a.json").write_text("not json")
        self.manifest("manifest-old.json", ["overlays_old/f0.png"])
        self.touch("masks/overlays_old/f0.png")
        state.mask_manifest_path = str(self.masks / "manifest-a.json")
        write_session_metadata()
        artifacts.sweep_unreferenced_artifacts(state)
        self.assertEqual(len(self.names("masks")), 3)

    def test_only_owned_names_are_touched(self):
        self.touch("masks/manifest.json")
        self.touch("masks/frame_00000_masks.png")
        self.touch("notes.txt")
        write_session_metadata()
        artifacts.sweep_unreferenced_artifacts(state)
        self.assertEqual(self.names("masks"), ["masks/frame_00000_masks.png", "masks/manifest.json"])
        self.assertTrue((self.session / "notes.txt").exists())

    def test_superseded_tracking_results_are_deleted(self):
        self.touch("tracking/old.json")
        write_session_metadata({"latest_tracking_result_path": "tracking/old.json"})
        self.touch("tracking/new.json")
        write_session_metadata({"latest_tracking_result_path": "tracking/new.json"})
        self.assertEqual(self.names("tracking"), ["tracking/new.json"])

    def test_reclamation_never_fails_an_accepted_commit(self):
        self.touch("tracking/old.json")
        write_session_metadata({"latest_tracking_result_path": "tracking/old.json"})
        self.touch("tracking/new.json")
        real_unlink = Path.unlink

        def unlink(path, *args, **kwargs):
            if path.parent.name == "tracking":
                raise PermissionError("in use")
            return real_unlink(path, *args, **kwargs)
        with patch.object(Path, "unlink", unlink):
            write_session_metadata({"latest_tracking_result_path": "tracking/new.json"})
        # The commit is accepted even though the superseded file could not be removed.
        self.assertEqual(self.names("tracking"), ["tracking/new.json", "tracking/old.json"])
        self.assertEqual(json.loads((self.session / "session.json").read_text())["latest_tracking_result_path"],
                         "tracking/new.json")


if __name__ == "__main__":
    unittest.main()
