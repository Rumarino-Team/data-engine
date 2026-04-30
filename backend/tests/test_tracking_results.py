import tempfile
import unittest
from pathlib import Path
import sys

from fastapi import HTTPException

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.state import state
from sessions.metadata import write_session_metadata
from tracking.results import (
    load_prompt_tracking_result,
    restored_tracking_result_payload,
    save_prompt_tracking_result,
)
from utils import load_mask_manifest


class TrackingResultTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.session_path = Path(self.tmp.name)
        state.active_session_dir = self.session_path
        state.active_session_id = "session-1"
        state.active_session_saved_name = None
        state.video_frame_files = ["00000.jpg", "00001.jpg"]
        state.video_state_epoch = 1

    def tearDown(self):
        state.active_session_dir = None
        state.active_session_id = None
        state.active_session_saved_name = None
        state.video_frame_files = []
        self.tmp.cleanup()

    def test_save_and_read_tracking_result(self):
        saved = save_prompt_tracking_result(
            model_name="cotracker3_online",
            num_points=1,
            num_frames=2,
            add_support_grid_used=False,
            tracking_mode="streaming",
            streaming_frame_threshold=256,
            points=[{"point_id": "p0_0", "obj_id": 1, "source_frame_idx": 0, "source_x": 10, "source_y": 20}],
            tracks=[[[10, 20], [11, 21]]],
            visibility=[[True, True]],
        )

        result_path = self.session_path / "tracking" / f"{saved['result_id']}.json"
        self.assertTrue(result_path.exists())
        loaded = load_prompt_tracking_result(saved["result_id"])
        self.assertEqual(loaded["tracks"], [[[10, 20], [11, 21]]])
        self.assertEqual(loaded["visibility"], [[True, True]])

    def test_invalid_result_id_returns_400(self):
        with self.assertRaises(HTTPException) as context:
            load_prompt_tracking_result("../bad")
        self.assertEqual(context.exception.status_code, 400)

    def test_missing_result_returns_404(self):
        with self.assertRaises(HTTPException) as context:
            load_prompt_tracking_result("missing")
        self.assertEqual(context.exception.status_code, 404)

    def test_metadata_preserves_latest_tracking_keys(self):
        write_session_metadata(
            {
                "latest_tracking_result_id": "abc",
                "latest_tracking_result_path": "tracking/abc.json",
                "latest_tracking_result_updated_at": "now",
                "latest_tracking_result_summary": {"num_points": 1},
            }
        )
        write_session_metadata({"schema_version": 2})

        metadata = load_mask_manifest(self.session_path / "session.json")
        self.assertEqual(metadata["latest_tracking_result_id"], "abc")
        self.assertEqual(metadata["latest_tracking_result_summary"], {"num_points": 1})

    def test_restored_tracking_result_payload_requires_existing_file(self):
        tracking_dir = self.session_path / "tracking"
        tracking_dir.mkdir()
        (tracking_dir / "abc.json").write_text("{}", encoding="utf-8")
        metadata = {
            "latest_tracking_result_id": "abc",
            "latest_tracking_result_path": "tracking/abc.json",
            "latest_tracking_result_summary": {"num_points": 1},
        }

        restored = restored_tracking_result_payload(self.session_path, metadata)

        self.assertEqual(restored["result_id"], "abc")
        self.assertEqual(restored["summary"], {"num_points": 1})


if __name__ == "__main__":
    unittest.main()
