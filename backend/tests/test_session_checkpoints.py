import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.factory import create_app
from core.annotations import SessionAnnotations
from core.errors import OperationError
from core.state import BackendState, state
from sessions import checkpoints
from sessions.metadata import write_session_metadata
from video import service

OBJECT = {"id": 1, "label": "Hand", "color": "#ff0000"}


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.previous_state = state.__dict__.copy()
        state.__dict__.update(BackendState().__dict__)
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.saved_root = self.root / "saved"
        self.working = self.root / "working"
        (self.working / "frames").mkdir(parents=True)
        (self.working / "masks").mkdir()
        (self.working / "frames" / "00000.jpg").write_bytes(b"frame")
        state.active_session_dir = self.working
        state.active_session_id = "first-session"
        state.video_dir = str(self.working / "frames")
        state.video_frame_files = ["00000.jpg"]
        state.annotations = SessionAnnotations(objects={1: OBJECT})
        state.video_masker = Mock(online_mode=True, default_batch_size=1,
                                  offload_video_to_cpu=True, offload_state_to_cpu=True)
        write_session_metadata()
        self.patches = [patch.object(service, "SAVED_ROOT", self.saved_root),
                        patch("video.io.SESSION_CACHE_ROOT", self.root / "cache")]
        for active in self.patches:
            active.start()
        self.client = TestClient(create_app())

        def preconditions(request):
            request.headers.setdefault("X-Session-ID", state.active_session_id or "absent")
            request.headers.setdefault("X-Annotation-Revision", str(state.annotations.revision))
        self.client.event_hooks["request"].append(preconditions)

    def tearDown(self):
        for active in self.patches:
            active.stop()
        state.__dict__.clear()
        state.__dict__.update(self.previous_state)
        self.tmp.cleanup()

    def save(self, **body):
        return self.client.post("/video/save", json=body)

    def info(self):
        return self.client.get("/video/info").json()["saved_state"]

    def metadata(self, directory):
        return json.loads((directory / "session.json").read_text())

    def open_checkpoint(self, directory):
        return service.initialize_video_state_from_resolved_input(
            directory, online_mode=True, batch_size=1, offload_video_to_cpu=True,
            offload_state_to_cpu=True, async_loading_frames=False, session=state)

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
        write_session_metadata({"latest_tracking_result_id": "abc"})
        self.assertEqual(
            {path.relative_to(checkpoint): path.read_bytes() for path in checkpoint.rglob("*") if path.is_file()},
            artifacts)
        self.assertEqual(self.metadata(checkpoint)["annotations"]["objects"]["1"]["label"], "Hand")
        self.assertEqual(state.checkpoint_dir, checkpoint)

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

    def test_failed_swap_restores_the_previous_checkpoint(self):
        checkpoint = Path(self.save(name="Bedroom").json()["saved_path"])
        before = (checkpoint / "session.json").read_bytes()
        self.edit("Unsaved")
        real_replace = checkpoints.os.replace
        calls = []

        def replace(source, target):
            calls.append(Path(target).name)
            if len(calls) == 2:
                raise OSError("swap failed")
            return real_replace(source, target)
        with patch.object(checkpoints.os, "replace", side_effect=replace):
            with self.assertRaises(OSError):
                self.save()
        self.assertEqual((checkpoint / "session.json").read_bytes(), before)
        self.assertEqual([path.name for path in self.saved_root.iterdir()], [checkpoint.name])

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
