import json
import shutil
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.annotations import SessionAnnotations
from core.errors import OperationError
from core.state import state
from sessions import checkpoints
from sessions.metadata import write_session_metadata
from support import BackendStateTestCase
from video import service

OBJECT = {"id": 1, "label": "Hand", "color": "#ff0000"}


class CheckpointTests(BackendStateTestCase):
    def setUp(self):
        super().setUp()
        self.root = self.temp_dir()
        self.saved_root = self.root / "saved"
        self.working = self.working_session(
            self.root / "working", session_id="first-session", frame_bytes=b"frame",
            annotations=SessionAnnotations(objects={1: OBJECT}), video_masker=self.make_masker())
        write_session_metadata(state)
        for active in (patch.object(service, "SAVED_ROOT", self.saved_root),
                       patch("video.io.SESSION_CACHE_ROOT", self.root / "cache")):
            active.start()
            self.addCleanup(active.stop)
        self.client = self.make_client()

    def save(self, **body):
        return self.client.post("/video/save", json=body)

    def info(self):
        return self.client.get("/video/info").json()["saved_state"]

    def metadata(self, directory):
        return json.loads((directory / "session.json").read_text())

    def open_checkpoint(self, directory):
        return service.initialize_video_state_from_resolved_input(
            directory, online_mode=True, batch_size=1, offload_video_to_cpu=True,
            offload_state_to_cpu=True, async_loading_frames=False, session=state, job=self.job_handle())

    def edit(self, label):
        response = self.client.put("/video/objects/1", json={"label": label, "color": "#ff0000"})
        self.assertEqual(response.status_code, 200)

    def test_dirty_state_follows_save_edit_and_reopen(self):
        self.assertTrue(self.info()["dirty"])
        self.assertIsNone(self.info()["checkpoint_id"])
        self.assertEqual(self.save(name="Bedroom").status_code, 200)
        self.assertFalse(self.info()["dirty"])
        self.edit("Renamed")
        self.assertTrue(self.info()["dirty"])
        checkpoint = Path(self.save(name="Bedroom").json()["saved_path"])
        self.open_checkpoint(checkpoint)
        self.assertFalse(self.info()["dirty"])
        self.edit("After reopen")
        self.assertTrue(self.info()["dirty"])

    def test_checkpoint_identity_is_separate_from_display_name(self):
        body = self.save(name="  My   Bedroom: take 2 ").json()
        checkpoint = Path(body["saved_path"])
        self.assertEqual(body["name"], "My Bedroom: take 2")
        self.assertEqual(checkpoint.parent, self.saved_root)
        self.assertTrue(checkpoint.name.startswith("my-bedroom-take-2-"))
        metadata = self.metadata(checkpoint)
        self.assertEqual(metadata["display_name"], "My Bedroom: take 2")
        self.assertEqual(metadata["checkpoint_id"], body["checkpoint_id"])
        self.assertNotEqual(state.active_session_dir, checkpoint)
        self.assertEqual(Path(metadata["resolved_video_frames_dir"]), checkpoint / "frames")

    def test_first_save_and_save_as_require_a_name(self):
        self.assertEqual(self.save().status_code, 400)
        self.assertEqual(self.save(name="   ").status_code, 400)
        self.save(name="Named")
        self.assertEqual(self.save(save_as=True).status_code, 400)
        self.assertFalse(list(self.saved_root.glob(".staging-*")))

    def test_resave_replaces_same_checkpoint_and_keeps_session_editable(self):
        first = self.save(name="Bedroom").json()
        self.edit("Second")
        second = self.save().json()
        self.assertEqual((second["checkpoint_id"], second["saved_path"]), (first["checkpoint_id"], first["saved_path"]))
        self.assertEqual(self.metadata(Path(first["saved_path"]))["annotations"]["objects"]["1"]["label"], "Second")
        self.assertEqual(len(list(self.saved_root.iterdir())), 1)
        self.edit("Third")
        self.assertEqual(self.metadata(Path(first["saved_path"]))["annotations"]["objects"]["1"]["label"], "Second")
        self.assertTrue(self.info()["dirty"])

    def test_resave_with_name_renames_only_the_display_name(self):
        first = self.save(name="Bedroom").json()
        renamed = self.save(name="Kitchen").json()
        self.assertEqual(renamed["saved_path"], first["saved_path"])
        self.assertEqual(self.metadata(Path(first["saved_path"]))["display_name"], "Kitchen")
        self.assertEqual(self.info()["display_name"], "Kitchen")

    def test_save_as_creates_new_identity_and_leaves_previous_checkpoint(self):
        first = self.save(name="Bedroom").json()
        before = (Path(first["saved_path"]) / "session.json").read_bytes()
        self.edit("Variant")
        copy = self.save(name="Bedroom", save_as=True).json()
        self.assertNotEqual(copy["checkpoint_id"], first["checkpoint_id"])
        self.assertNotEqual(copy["saved_path"], first["saved_path"])
        self.assertEqual((Path(first["saved_path"]) / "session.json").read_bytes(), before)
        self.assertEqual(self.info()["checkpoint_id"], copy["checkpoint_id"])
        self.edit("Next")
        self.assertEqual(self.save().json()["saved_path"], copy["saved_path"])

    def test_opened_checkpoint_is_never_written_by_later_edits_or_results(self):
        checkpoint = Path(self.save(name="Bedroom").json()["saved_path"])
        artifacts = {path.relative_to(checkpoint): path.read_bytes() for path in checkpoint.rglob("*") if path.is_file()}
        self.open_checkpoint(checkpoint)
        working = state.active_session_dir
        self.assertNotEqual(working, checkpoint)
        self.assertEqual((working / "frames" / "00000.jpg").read_bytes(), b"frame")
        self.edit("Working change")
        (working / "masks" / "new-result.json").write_text("{}")
        write_session_metadata(state, {"latest_tracking_result_id": "abc"})
        self.assertEqual(
            {path.relative_to(checkpoint): path.read_bytes() for path in checkpoint.rglob("*") if path.is_file()},
            artifacts)
        self.assertEqual(self.metadata(checkpoint)["annotations"]["objects"]["1"]["label"], "Hand")
        self.assertEqual(state.checkpoint_dir, checkpoint)

    def test_reclaiming_working_artifacts_never_removes_the_checkpoints_copy(self):
        from video.masks import retain_interactive_masks
        retain_interactive_masks(state, 0, [1], [{"size": [1, 1], "rle": []}])
        write_session_metadata(state)
        artifact = state.interactive_mask_files["0"]
        checkpoint = Path(self.save(name="Bedroom").json()["saved_path"])
        self.open_checkpoint(checkpoint)
        working_file = state.active_session_dir / "masks" / "interactive" / artifact
        self.assertTrue(working_file.exists())
        # The working copy stops referencing the result; only its own link is reclaimed.
        state.interactive_mask_files = {}
        write_session_metadata(state)
        self.assertFalse(working_file.exists())
        self.assertTrue((checkpoint / "masks" / "interactive" / artifact).exists())
        self.assertIn(artifact, json.dumps(self.metadata(checkpoint)))

    def test_failed_save_preserves_previous_checkpoint_and_working_changes(self):
        checkpoint = Path(self.save(name="Bedroom").json()["saved_path"])
        before = (checkpoint / "session.json").read_bytes()
        self.edit("Unsaved")
        working_label = state.annotations.objects[1].label
        with patch.object(checkpoints.os, "replace", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                self.save()
        self.assertEqual((checkpoint / "session.json").read_bytes(), before)
        self.assertEqual(state.annotations.objects[1].label, working_label)
        self.assertTrue(self.info()["dirty"])
        self.assertEqual([path.name for path in self.saved_root.iterdir()], [checkpoint.name])

    def commit_result(self, frame=0):
        from video.masks import retain_interactive_masks
        retain_interactive_masks(state, frame, [1], [{"size": [1, 1], "rle": []}])
        write_session_metadata(state)
        return state.interactive_mask_files[str(frame)]

    def files(self, checkpoint):
        return sorted(path.relative_to(checkpoint).as_posix() for path in checkpoint.rglob("*") if path.is_file())

    def test_first_save_copies_frames_and_only_referenced_artifacts(self):
        kept = self.commit_result()
        (self.working / "masks" / "interactive" / ("0" * 32 + ".json")).write_text("{}")
        (self.working / "masks" / "notes.txt").write_text("scratch")
        checkpoint = Path(self.save(name="Bedroom").json()["saved_path"])
        self.assertEqual(self.files(checkpoint), ["frames/00000.jpg", f"masks/interactive/{kept}", "session.json"])
        self.assertTrue((checkpoint / "masks").is_dir())

    def test_resave_copies_only_new_files_and_deletes_superseded_ones(self):
        first = self.commit_result()
        checkpoint = Path(self.save(name="Bedroom").json()["saved_path"])
        # A sentinel proves later saves neither recopy frames nor rewrite unchanged artifacts.
        (checkpoint / "frames" / "00000.jpg").write_bytes(b"sentinel-frame")
        (checkpoint / "masks" / "interactive" / first).write_text("sentinel-artifact")
        second = self.commit_result()
        self.assertNotEqual(first, second)
        with patch.object(checkpoints, "_link_or_copy", wraps=checkpoints._link_or_copy) as link:
            self.save()
        self.assertEqual([Path(call.args[1]).name for call in link.call_args_list], [second])
        self.assertEqual(self.files(checkpoint), ["frames/00000.jpg", f"masks/interactive/{second}", "session.json"])
        self.assertEqual((checkpoint / "frames" / "00000.jpg").read_bytes(), b"sentinel-frame")
        self.assertEqual(self.metadata(checkpoint)["interactive_mask_files"], {"0": second})

    def test_resave_after_open_continues_incrementally(self):
        first = self.commit_result()
        checkpoint = Path(self.save(name="Bedroom").json()["saved_path"])
        self.open_checkpoint(checkpoint)
        write_session_metadata(state, {"latest_tracking_result_path": "tracking/new.json"})
        (state.active_session_dir / "tracking").mkdir()
        (state.active_session_dir / "tracking" / "new.json").write_text("{}")
        with patch.object(checkpoints, "_link_or_copy", wraps=checkpoints._link_or_copy) as link:
            self.save()
        self.assertEqual([Path(call.args[1]).name for call in link.call_args_list], ["new.json"])
        self.assertEqual(self.files(checkpoint), ["frames/00000.jpg", f"masks/interactive/{first}",
                                                  "session.json", "tracking/new.json"])

    def test_failed_resave_removes_added_files_and_keeps_previous_checkpoint(self):
        self.commit_result()
        checkpoint = Path(self.save(name="Bedroom").json()["saved_path"])
        before, before_files = (checkpoint / "session.json").read_bytes(), self.files(checkpoint)
        self.commit_result(1)
        self.edit("Edited alongside the new mask")
        with patch("sessions.metadata.os.replace", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                self.save()
        self.assertEqual((checkpoint / "session.json").read_bytes(), before)
        self.assertEqual(self.files(checkpoint), before_files)
        self.assertTrue(self.info()["dirty"])
        # A later save still succeeds and includes both results.
        self.save()
        self.assertEqual(len(self.metadata(checkpoint)["interactive_mask_files"]), 2)

    def test_resave_with_a_missing_working_artifact_fails_before_changing_the_checkpoint(self):
        checkpoint = Path(self.save(name="Bedroom").json()["saved_path"])
        before, before_files = (checkpoint / "session.json").read_bytes(), self.files(checkpoint)
        missing = self.commit_result(1)
        (self.working / "masks" / "interactive" / missing).unlink()
        response = self.save()
        self.assertEqual(response.status_code, 500)
        self.assertEqual((checkpoint / "session.json").read_bytes(), before)
        self.assertEqual(self.files(checkpoint), before_files)

    def test_resave_recreates_a_deleted_checkpoint_and_refuses_a_damaged_one(self):
        checkpoint = Path(self.save(name="Bedroom").json()["saved_path"])
        shutil.rmtree(checkpoint)
        self.assertEqual(self.save().status_code, 200)
        self.assertTrue((checkpoint / "session.json").is_file())
        (checkpoint / "session.json").unlink()
        self.assertEqual(self.save().status_code, 409)

    def test_failed_first_save_leaves_no_staging_or_checkpoint(self):
        self.commit_result()
        with patch.object(checkpoints.os, "replace", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                self.save(name="Bedroom")
        self.assertEqual(list(self.saved_root.iterdir()), [])
        self.assertIsNone(self.info()["checkpoint_id"])

    def test_sweep_discards_working_copies_and_staging_but_keeps_checkpoints(self):
        checkpoint = Path(self.save(name="Bedroom").json()["saved_path"])
        stale = self.saved_root / ".staging-abandoned"
        stale.mkdir()
        cache = self.root / "cache-sweep"
        (cache / "old-session").mkdir(parents=True)
        checkpoints.sweep_stale_session_storage(cache, self.saved_root)
        self.assertFalse(stale.exists())
        self.assertEqual(list(cache.iterdir()), [])
        self.assertTrue((checkpoint / "session.json").exists())

    def test_display_name_validation(self):
        with self.assertRaises(OperationError):
            checkpoints.normalize_display_name("\x00\n ")
        with self.assertRaises(OperationError):
            checkpoints.normalize_display_name("x" * 121)
        self.assertEqual(checkpoints.normalize_display_name("a\tb\n c"), "a b c")


if __name__ == "__main__":
    unittest.main()
