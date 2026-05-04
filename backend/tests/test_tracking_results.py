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
from tracking.guidance import TrackingGuidance, TrackedGuidancePoint, load_latest_tracking_guidance
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
        state.video_prompt_events = []
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
        self.assertEqual(loaded["version"], 2)
        self.assertEqual(loaded["state_epoch"], 1)
        self.assertEqual(loaded["tracked_prompt_keys"], ["obj:1|frame:0|x:10.000|y:20.000"])
        self.assertEqual(loaded["tracks"], [[[10, 20], [11, 21]]])
        self.assertEqual(loaded["visibility"], [[True, True]])

    def test_latest_guidance_uses_matching_tracks_with_post_tracking_prompts(self):
        state.video_prompt_events = [
            {
                "frame_idx": 0,
                "obj_id": 1,
                "points": [[10.0, 20.0]],
                "labels": [1],
                "box": None,
                "clear_old_points": True,
            },
            {
                "frame_idx": 1,
                "obj_id": 1,
                "points": [[30.0, 40.0]],
                "labels": [1],
                "box": None,
                "clear_old_points": False,
            },
        ]
        save_prompt_tracking_result(
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

        guidance = load_latest_tracking_guidance(num_frames=2)

        self.assertIsNone(guidance.skipped_reason)
        self.assertEqual(len(guidance.points), 1)
        self.assertEqual(len(state.video_prompt_events), 2)

    def test_latest_guidance_ignores_tracks_when_source_prompt_was_removed(self):
        state.video_prompt_events = [
            {
                "frame_idx": 1,
                "obj_id": 1,
                "points": [[30.0, 40.0]],
                "labels": [1],
                "box": None,
                "clear_old_points": True,
            }
        ]
        save_prompt_tracking_result(
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

        guidance = load_latest_tracking_guidance(num_frames=2)

        self.assertEqual(guidance.points, [])
        self.assertIn("No CoTracker points match", guidance.skipped_reason)

    def test_latest_guidance_skips_frame_count_mismatch(self):
        state.video_prompt_events = [
            {
                "frame_idx": 0,
                "obj_id": 1,
                "points": [[10.0, 20.0]],
                "labels": [1],
                "box": None,
                "clear_old_points": True,
            }
        ]
        save_prompt_tracking_result(
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

        guidance = load_latest_tracking_guidance(num_frames=3)

        self.assertEqual(guidance.points, [])
        self.assertIn("does not match", guidance.skipped_reason)

    def test_guidance_builds_adaptive_keyframes_filters_visibility_and_caps_points(self):
        guidance = TrackingGuidance(
            points=[
                TrackedGuidancePoint(
                    obj_id=1,
                    source_frame_idx=0,
                    tracks=[[0, 0], [10, 10], [20, 20], [30, 30], [40, 40]],
                    visibility=[True, True, False, True, True],
                ),
                TrackedGuidancePoint(
                    obj_id=1,
                    source_frame_idx=0,
                    tracks=[[1, 1], [11, 11], [21, 21], [31, 31], [41, 41]],
                    visibility=[True, True, False, True, True],
                ),
            ]
        )

        batches, seeded_count, seeded_frames = guidance.build_window_batches(
            window_start=0,
            window_end=4,
            propagation_start_frame_idx=0,
            frame_width=32,
            frame_height=32,
            keyframe_interval=2,
            max_points_per_object_per_frame=1,
        )

        self.assertEqual(set(batches.keys()), {0, 4})
        self.assertEqual(seeded_count, 2)
        self.assertEqual(seeded_frames, {0, 4})
        self.assertEqual(batches[4][1], [[31.0, 31.0]])

    def test_guidance_skips_excluded_boundary_frame_object(self):
        guidance = TrackingGuidance(
            points=[
                TrackedGuidancePoint(
                    obj_id=1,
                    source_frame_idx=2,
                    tracks=[[0, 0], [10, 10], [20, 20], [30, 30], [40, 40]],
                    visibility=[True, True, True, True, True],
                )
            ]
        )

        batches, seeded_count, seeded_frames = guidance.build_window_batches(
            window_start=2,
            window_end=4,
            propagation_start_frame_idx=0,
            frame_width=64,
            frame_height=64,
            keyframe_interval=2,
            max_points_per_object_per_frame=16,
            excluded_frame_objects={(2, 1)},
        )

        self.assertNotIn(0, batches)
        self.assertEqual(seeded_count, 1)
        self.assertEqual(seeded_frames, {4})

    def test_guidance_skips_bad_coordinate_values(self):
        guidance = TrackingGuidance(
            points=[
                TrackedGuidancePoint(
                    obj_id=1,
                    source_frame_idx=0,
                    tracks=[["bad", 0], [10, 10]],
                    visibility=[True, True],
                )
            ]
        )

        batches, seeded_count, seeded_frames = guidance.build_window_batches(
            window_start=0,
            window_end=1,
            propagation_start_frame_idx=0,
            frame_width=64,
            frame_height=64,
            keyframe_interval=1,
            max_points_per_object_per_frame=16,
        )

        self.assertEqual(seeded_count, 1)
        self.assertEqual(seeded_frames, {1})
        self.assertEqual(batches[1][1], [[10.0, 10.0]])

    def test_latest_guidance_no_result_does_not_warn(self):
        guidance = load_latest_tracking_guidance(num_frames=2)

        self.assertEqual(guidance.points, [])
        self.assertFalse(guidance.should_warn)
        self.assertIn("No CoTracker result", guidance.skipped_reason)

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
