import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.annotations import SessionAnnotations
from core.errors import OperationError
from core.jobs import capture_job_inputs
from sessions import artifacts
from sessions.publication import publish_results
from support import BackendStateTestCase, state

OBJECT = {"id": 1, "label": "Hand", "color": "#ff0000"}


class PublishResultsTests(BackendStateTestCase):
    def setUp(self):
        super().setUp()
        self.session = self.working_session(session_id="s", annotations=SessionAnnotations(objects={1: OBJECT}))
        self.masks = self.session / "masks"
        artifacts._read_manifest_overlays.cache_clear()
        self.inputs = capture_job_inputs(state, "mask_propagation")
        self.previous = self.write("masks/manifest-previous.json")
        state.mask_manifest_path = str(self.previous)
        artifacts.reclaim_superseded_artifacts(state)  # the previous run is the committed baseline

    def write(self, relative):
        path = self.session / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
        return path

    def manifests(self):
        return sorted(path.name for path in self.masks.glob("manifest-*.json"))

    def test_success_names_the_new_files_and_deletes_the_superseded_ones(self):
        new = self.write("masks/manifest-new.json")
        with publish_results(self.ctx, self.inputs, "mask_propagation") as session:
            session.mask_manifest_path = str(new)
        self.assertEqual(state.mask_manifest_path, str(new))
        self.assertEqual(self.manifests(), ["manifest-new.json"])

    def test_failure_restores_references_and_sweeps_what_the_job_wrote(self):
        new = self.write("masks/manifest-new.json")
        state.session_metadata["latest_tracking_result_id"] = "kept"
        with self.assertRaises(OSError):
            with publish_results(self.ctx, self.inputs, "mask_propagation") as session:
                session.mask_manifest_path = str(new)
                session.interactive_mask_files = {"0": "x" * 32 + ".json"}
                session.session_metadata["latest_tracking_result_id"] = "changed"
                raise OSError("disk full")
        self.assertEqual(state.mask_manifest_path, str(self.previous))
        self.assertEqual(state.interactive_mask_files, {})
        self.assertEqual(state.session_metadata["latest_tracking_result_id"], "kept")
        self.assertEqual(self.manifests(), ["manifest-previous.json"])

    def test_stale_inputs_refuse_publication_and_sweep_the_jobs_files(self):
        new = self.write("masks/manifest-new.json")
        state.active_session_id = "replaced"
        with self.assertRaises(OperationError) as failure:
            with publish_results(self.ctx, self.inputs, "mask_propagation"):
                self.fail("publishing must not start for stale inputs")
        self.assertEqual(failure.exception.status_code, 409)
        self.assertFalse(new.exists())
        self.assertEqual(state.mask_manifest_path, str(self.previous))


if __name__ == "__main__":
    unittest.main()
