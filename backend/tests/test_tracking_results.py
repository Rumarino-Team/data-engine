import unittest
from unittest.mock import patch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.errors import OperationError
from support import state
from core.annotations import SessionAnnotations
from sessions.metadata import session_metadata_document
from tracking.results import (
    load_prompt_tracking_result,
    restored_tracking_result_payload,
    save_prompt_tracking_result,
)
from tracking.guidance import TrackingGuidance, TrackedGuidancePoint, load_latest_tracking_guidance
from support import SessionStateTestCase, commit_session
from schemas.video import VideoPropagateRequest
from unittest.mock import Mock


class TrackingResultTests(SessionStateTestCase):
    def setUp(self):
        super().setUp()
        self.session_path = self.temp_dir()
        state.active_session_dir = self.session_path
        state.active_session_id = "session-1"
        state.video_frame_files = ["00000.jpg", "00001.jpg"]
        state.annotations = SessionAnnotations(objects={1: {"id": 1, "label": "Object 1", "color": "#ff0000"}},
            prompts=[{"frame_idx": 0, "obj_id": 1, "points": [[10, 20]], "labels": [1]}])

    def test_save_and_read_tracking_result(self):
        saved = save_prompt_tracking_result(
            state,
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
        loaded = load_prompt_tracking_result(state, saved["result_id"])
        self.assertEqual(loaded["version"], 3)
        self.assertEqual(loaded["tracked_prompt_keys"], ["obj:1|frame:0|x:10.000|y:20.000"])
        self.assertEqual(loaded["tracks"], [[[10, 20], [11, 21]]])
        self.assertEqual(loaded["visibility"], [[True, True]])

    def test_failed_result_write_keeps_the_previous_reference(self):
        kwargs = dict(model_name="cotracker3_online", num_points=1, num_frames=2,
            add_support_grid_used=False, tracking_mode="streaming", streaming_frame_threshold=256,
            points=[{"point_id": "p0", "obj_id": 1, "source_frame_idx": 0, "source_x": 10, "source_y": 20}],
            tracks=[[[10, 20], [11, 21]]], visibility=[[True, True]])
        old = save_prompt_tracking_result(state, **kwargs)
        old_metadata = dict(state.session_metadata)
        with patch("tracking.results.write_mask_manifest", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                save_prompt_tracking_result(state, **kwargs)
        self.assertEqual(state.session_metadata, old_metadata)
        self.assertEqual(load_prompt_tracking_result(state, old["result_id"])["num_points"], 1)
        # The rejected result was never written or named, so only the previous file remains.
        self.assertEqual([path.stem for path in (self.session_path / "tracking").glob("*.json")], [old["result_id"]])

    def test_a_new_result_supersedes_the_previous_file(self):
        kwargs = dict(model_name="cotracker3_online", num_points=1, num_frames=2,
            add_support_grid_used=False, tracking_mode="streaming", streaming_frame_threshold=256,
            points=[{"point_id": "p0", "obj_id": 1, "source_frame_idx": 0, "source_x": 10, "source_y": 20}],
            tracks=[[[10, 20], [11, 21]]], visibility=[[True, True]])
        first = save_prompt_tracking_result(state, **kwargs)
        second = save_prompt_tracking_result(state, **kwargs)
        self.assertNotEqual(first["result_id"], second["result_id"])
        self.assertEqual(state.session_metadata["latest_tracking_result_id"], second["result_id"])
        self.assertEqual([path.stem for path in (self.session_path / "tracking").glob("*.json")], [second["result_id"]])

    def test_same_validity_for_retrieval_guidance_and_restore(self):
        saved = save_prompt_tracking_result(state, model_name="cotracker3_online", num_points=1, num_frames=2,
            add_support_grid_used=False, tracking_mode="streaming", streaming_frame_threshold=256,
            points=[{"point_id": "p0", "obj_id": 1, "source_frame_idx": 0, "source_x": 10, "source_y": 20}],
            tracks=[[[10, 20], [11, 21]]], visibility=[[True, True]])
        state.annotations.objects[1].label = "Renamed"
        self.assertEqual(load_prompt_tracking_result(state, saved["result_id"])["num_points"], 1)
        self.assertIsNotNone(restored_tracking_result_payload(state, self.session_path, state.session_metadata))
        state.annotations.clear_prompts(0, 1)
        self.assertEqual(load_prompt_tracking_result(state, saved["result_id"])["points"], [])
        self.assertEqual(load_latest_tracking_guidance(state, num_frames=2).points, [])
        self.assertIsNone(restored_tracking_result_payload(state, self.session_path, state.session_metadata))
        state.active_session_id = "another-session"
        with self.assertRaises(OperationError) as mismatch:
            load_prompt_tracking_result(state, saved["result_id"])
        self.assertEqual(mismatch.exception.status_code, 409)

    def test_latest_guidance_uses_matching_tracks_with_post_tracking_prompts(self):
        state.annotations = SessionAnnotations(objects={1: {"id": 1, "label": "Object 1", "color": "#ff0000"}}, prompts=[
            {
                "frame_idx": 0,
                "obj_id": 1,
                "points": [[10.0, 20.0]],
                "labels": [1],
                "box": None,
            },
            {
                "frame_idx": 1,
                "obj_id": 1,
                "points": [[30.0, 40.0]],
                "labels": [1],
                "box": None,
            },
        ])
        save_prompt_tracking_result(
            state,
            model_name="cotracker3_online",
            num_points=1,
            num_frames=2,
            add_support_grid_used=True,
            tracking_mode="streaming",
            streaming_frame_threshold=256,
            points=[{"point_id": "p0_0", "obj_id": 1, "source_frame_idx": 0, "source_x": 10, "source_y": 20}],
            tracks=[[[10, 20], [11, 21]]],
            visibility=[[True, True]],
        )

        guidance = load_latest_tracking_guidance(state, num_frames=2)

        self.assertIsNone(guidance.skipped_reason)
        self.assertEqual(len(guidance.points), 1)
        self.assertEqual(len(state.annotations.prompts), 2)

    def test_latest_guidance_ignores_tracks_when_source_prompt_was_removed(self):
        state.annotations = SessionAnnotations(objects={1: {"id": 1, "label": "Object 1", "color": "#ff0000"}}, prompts=[
            {
                "frame_idx": 1,
                "obj_id": 1,
                "points": [[30.0, 40.0]],
                "labels": [1],
                "box": None,
            }
        ])
        save_prompt_tracking_result(
            state,
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

        guidance = load_latest_tracking_guidance(state, num_frames=2)

        self.assertEqual(guidance.points, [])
        self.assertIn("No CoTracker points match", guidance.skipped_reason)

    def test_latest_guidance_skips_frame_count_mismatch(self):
        state.annotations = SessionAnnotations(objects={1: {"id": 1, "label": "Object 1", "color": "#ff0000"}}, prompts=[
            {
                "frame_idx": 0,
                "obj_id": 1,
                "points": [[10.0, 20.0]],
                "labels": [1],
                "box": None,
            }
        ])
        save_prompt_tracking_result(
            state,
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

        guidance = load_latest_tracking_guidance(state, num_frames=3)

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

        batches, seeded_count, seeded_frames = guidance.build_seed_batches(
            range_start=0,
            range_end=4,
            num_video_frames=5,
            tail_frames=0,
            frame_width=32,
            frame_height=32,
            keyframe_interval=2,
            max_keyframes=100,
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

        batches, seeded_count, seeded_frames = guidance.build_seed_batches(
            range_start=2,
            range_end=4,
            num_video_frames=5,
            tail_frames=0,
            frame_width=64,
            frame_height=64,
            keyframe_interval=2,
            max_keyframes=100,
            max_points_per_object_per_frame=16,
            excluded_frame_objects={(2, 1)},
        )

        self.assertNotIn(0, batches)
        self.assertEqual(seeded_count, 1)
        self.assertEqual(seeded_frames, {4})

    def guidance_over(self, frames):
        return TrackingGuidance(points=[TrackedGuidancePoint(
            obj_id=1, source_frame_idx=0, tracks=[[10, 10]] * frames, visibility=[True] * frames)])

    def seeded_frames(self, frames, *, interval, cap, start=0):
        _, _, seeded = self.guidance_over(frames).build_seed_batches(
            range_start=start, range_end=frames - 1, num_video_frames=frames, tail_frames=0, frame_width=64, frame_height=64,
            keyframe_interval=interval, max_keyframes=cap, max_points_per_object_per_frame=16)
        return sorted(seeded)

    def test_keyframes_follow_the_interval_while_they_stay_under_the_cap(self):
        self.assertEqual(self.seeded_frames(21, interval=8, cap=16), [0, 8, 16])

    def test_a_long_range_gets_at_most_the_cap_evenly_spaced_with_both_ends(self):
        frames = self.seeded_frames(923, interval=8, cap=16)
        self.assertEqual(len(frames), 16)
        self.assertEqual((frames[0], frames[-1]), (0, 922))
        gaps = {later - earlier for earlier, later in zip(frames, frames[1:])}
        self.assertLessEqual(max(gaps) - min(gaps), 1)

    def test_the_cap_counts_from_the_start_of_a_custom_range(self):
        frames = self.seeded_frames(300, interval=1, cap=5, start=100)
        self.assertEqual(frames, [100, 150, 200, 249, 299])

    def test_a_cap_of_one_seeds_only_the_first_frame_and_a_short_range_is_unaffected(self):
        self.assertEqual(self.seeded_frames(300, interval=8, cap=1), [0])
        self.assertEqual(self.seeded_frames(3, interval=1, cap=16), [0, 1, 2])

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

        batches, seeded_count, seeded_frames = guidance.build_seed_batches(
            range_start=0,
            range_end=1,
            num_video_frames=2,
            tail_frames=0,
            frame_width=64,
            frame_height=64,
            keyframe_interval=1,
            max_keyframes=100,
            max_points_per_object_per_frame=16,
        )

        self.assertEqual(seeded_count, 1)
        self.assertEqual(seeded_frames, {1})
        self.assertEqual(batches[1][1], [[10.0, 10.0]])

    def track_and_load(self, prompts, *, grid=True, objects=(1, 2)):
        state.annotations = SessionAnnotations(
            objects={obj: {"id": obj, "label": f"Object {obj}", "color": "#ff0000"} for obj in objects}, prompts=prompts)
        points = [{"point_id": f"p{index}", "obj_id": prompt["obj_id"], "source_frame_idx": prompt["frame_idx"],
                   "source_x": x, "source_y": y}
                  for index, prompt in enumerate(prompts)
                  for (x, y), label in zip(prompt["points"], prompt["labels"]) if label == 1]
        save_prompt_tracking_result(
            state, model_name="cotracker3_online", num_points=len(points), num_frames=2, add_support_grid_used=grid,
            tracking_mode="streaming", streaming_frame_threshold=0, points=points,
            tracks=[[[0, 0], [1, 1]]] * len(points), visibility=[[True, True]] * len(points))
        return load_latest_tracking_guidance(state, num_frames=2)

    def test_guidance_leaves_out_objects_whose_definition_tracking_cannot_carry(self):
        positive = {"frame_idx": 0, "obj_id": 2, "points": [[5.0, 5.0]], "labels": [1]}
        for name, refined in [
            ("a negative point", {"frame_idx": 0, "obj_id": 1, "points": [[10.0, 20.0], [30.0, 40.0]], "labels": [1, 0]}),
            ("a box", {"frame_idx": 0, "obj_id": 1, "points": [[10.0, 20.0]], "labels": [1], "box": [0.0, 0.0, 50.0, 50.0]}),
            ("a negative point on a later frame",
             {"frame_idx": 1, "obj_id": 1, "points": [[30.0, 40.0]], "labels": [0]}),
        ]:
            with self.subTest(name):
                prompts = [refined, positive]
                if refined["frame_idx"] == 1:
                    prompts.insert(0, {"frame_idx": 0, "obj_id": 1, "points": [[10.0, 20.0]], "labels": [1]})
                guidance = self.track_and_load(prompts)
                self.assertEqual([point.obj_id for point in guidance.points], [2])
                self.assertFalse(guidance.should_warn)
                self.assertIn("Objects 1 have negative points, boxes or masks", guidance.skipped_reason)

    def test_guidance_is_skipped_with_a_reason_when_no_object_can_be_seeded(self):
        guidance = self.track_and_load(
            [{"frame_idx": 0, "obj_id": 1, "points": [[10.0, 20.0], [30.0, 40.0]], "labels": [1, 0]}])
        self.assertEqual(guidance.points, [])
        self.assertTrue(guidance.should_warn)
        self.assertIn("negative points, boxes or masks", guidance.skipped_reason)

    def test_guidance_is_skipped_when_the_result_was_tracked_without_the_support_grid(self):
        guidance = self.track_and_load(
            [{"frame_idx": 0, "obj_id": 1, "points": [[10.0, 20.0]], "labels": [1]}], grid=False)
        self.assertEqual(guidance.points, [])
        self.assertTrue(guidance.should_warn)
        self.assertIn("support grid", guidance.skipped_reason)

    def test_latest_guidance_no_result_does_not_warn(self):
        guidance = load_latest_tracking_guidance(state, num_frames=2)

        self.assertEqual(guidance.points, [])
        self.assertFalse(guidance.should_warn)
        self.assertIn("No CoTracker result", guidance.skipped_reason)

    def test_invalid_result_id_returns_400(self):
        with self.assertRaises(OperationError) as context:
            load_prompt_tracking_result(state, "../bad")
        self.assertEqual(context.exception.status_code, 400)

    def test_missing_result_returns_404(self):
        with self.assertRaises(OperationError) as context:
            load_prompt_tracking_result(state, "missing")
        self.assertEqual(context.exception.status_code, 404)

    def test_saved_metadata_document_keeps_the_latest_tracking_keys(self):
        commit_session(
            state, {
                "latest_tracking_result_id": "abc",
                "latest_tracking_result_path": "tracking/abc.json",
                "latest_tracking_result_updated_at": "now",
                "latest_tracking_result_summary": {"num_points": 1},
            }
        )
        commit_session(state, {"created_at": "then"})

        metadata = session_metadata_document(state)
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

        restored = restored_tracking_result_payload(state, self.session_path, metadata)

        self.assertIsNone(restored)


class SeedingFramesTests(unittest.TestCase):
    def test_frames_before_the_point_existed_are_not_seeded(self):
        guidance = TrackingGuidance(points=[TrackedGuidancePoint(
            obj_id=1, source_frame_idx=2, tracks=[[i, i] for i in range(5)], visibility=[True] * 5)])

        batches, seeded_count, seeded_frames = guidance.build_seed_batches(
            range_start=0, range_end=4, num_video_frames=5, tail_frames=0, frame_width=32, frame_height=32, keyframe_interval=1, max_keyframes=100,
            max_points_per_object_per_frame=1)

        self.assertEqual(sorted(seeded_frames), [2, 3, 4])
        self.assertEqual(sorted(batches), [2, 3, 4])
        self.assertEqual(seeded_count, 3)


class SeedingRulesTests(unittest.TestCase):
    def seed(self, points, *, frames=100, start=0, end=None, interval=8, excluded=None, cap=100, per_frame=16, **kwargs):
        return TrackingGuidance(points=points).build_seed_batches(
            range_start=start, range_end=frames - 1 if end is None else end, num_video_frames=frames,
            frame_width=64, frame_height=64, keyframe_interval=interval, max_keyframes=cap,
            max_points_per_object_per_frame=per_frame, excluded_frame_objects=excluded, **kwargs)

    @staticmethod
    def point(obj=1, source=0, frames=100, visible=None, x=10.0):
        flags = [True] * frames if visible is None else visible
        return TrackedGuidancePoint(obj_id=obj, source_frame_idx=source, tracks=[[x, x]] * frames, visibility=flags)

    def test_the_final_window_of_the_video_is_not_seeded_and_the_last_frame_is_not_forced(self):
        _, _, seeded = self.seed([self.point(frames=40)], frames=40)
        self.assertEqual(sorted(seeded), [0, 8, 16])  # frames 24 to 39 are the last 16
        _, _, seeded = self.seed([self.point(frames=40)], frames=40, interval=1)
        self.assertEqual(max(seeded), 23)

    def test_the_final_window_is_the_videos_not_the_propagated_ranges(self):
        _, _, seeded = self.seed([self.point(frames=200)], frames=200, end=19)
        self.assertEqual(sorted(seeded), [0, 8, 16])
        _, _, seeded = self.seed([self.point(frames=200)], frames=200, start=100, end=199, interval=1)
        self.assertEqual(max(seeded), 183)

    def test_a_video_no_longer_than_the_final_window_is_never_seeded(self):
        for frames in (1, 10, 16, 17):
            with self.subTest(frames=frames):
                batches, count, seeded = self.seed([self.point(frames=frames)], frames=frames, interval=1,
                                                   excluded={(0, 1)} if frames == 17 else None)
                self.assertEqual((batches, count, seeded), ({}, 0, set()))

    def test_the_cap_spaces_keyframes_over_the_seedable_span(self):
        _, _, seeded = self.seed([self.point(frames=116)], frames=116, interval=1, cap=3)
        self.assertEqual(sorted(seeded), [0, 50, 99])

    def test_frames_the_user_prompted_are_not_seeded_and_a_point_stops_at_the_next_prompt(self):
        # Object 1 was prompted on frames 0 and 20. Its frame-0 point speaks for frames 1 to 19 only; the
        # frame-20 point (tracked after the correction) takes over from there.
        points = [self.point(source=0, x=1.0), self.point(source=20, x=2.0)]
        batches, _, seeded = self.seed(points, interval=4, excluded={(0, 1), (20, 1)})
        self.assertEqual(sorted(seeded), [4, 8, 12, 16, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80])
        self.assertEqual(batches[16][1], [[1.0, 1.0]])
        self.assertEqual(batches[24][1], [[2.0, 2.0]])

    def test_an_untracked_later_prompt_ends_seeding_for_that_object(self):
        _, _, seeded = self.seed([self.point(source=0)], interval=8, excluded={(0, 1), (30, 1)})
        self.assertEqual(sorted(seeded), [8, 16, 24])

    def test_a_prompt_for_another_object_changes_nothing_for_this_one(self):
        _, _, seeded = self.seed([self.point(obj=1)], interval=8, end=24, excluded={(16, 2), (8, 2)})
        self.assertEqual(sorted(seeded), [0, 8, 16, 24])

    def test_an_object_is_seeded_only_where_all_its_points_in_effect_are_visible(self):
        visible = [True] * 100
        visible[8] = False
        points = [self.point(x=1.0), self.point(x=2.0, visible=visible)]
        batches, count, seeded = self.seed(points, interval=8, end=24)
        self.assertEqual(sorted(seeded), [0, 16, 24])
        self.assertEqual(batches[16][1], [[1.0, 1.0], [2.0, 2.0]])
        self.assertNotIn(8, batches)
        self.assertEqual(count, 6)

    def test_a_missing_visibility_flag_counts_as_not_visible(self):
        visible = [True] * 100
        visible[8] = None
        _, _, seeded = self.seed([self.point(visible=visible)], interval=8, end=16)
        self.assertEqual(sorted(seeded), [0, 16])

    def test_one_objects_missing_point_does_not_block_another_object(self):
        hidden = [False] * 100
        batches, _, _ = self.seed([self.point(obj=1), self.point(obj=2, visible=hidden)], interval=8, end=8)
        self.assertEqual(batches, {0: {1: [[10.0, 10.0]]}, 8: {1: [[10.0, 10.0]]}})

    def test_points_are_capped_per_object_per_frame_after_the_visibility_check(self):
        points = [self.point(x=float(index)) for index in range(5)]
        batches, count, _ = self.seed(points, interval=8, end=0, per_frame=2)
        self.assertEqual(batches[0][1], [[0.0, 0.0], [1.0, 1.0]])
        self.assertEqual(count, 2)


class SeedTrackedPointsTests(unittest.TestCase):
    def test_seeds_only_unprompted_frames_with_positive_labels_and_no_old_points_cleared(self):
        from types import SimpleNamespace
        from video.propagation import seed_tracked_points

        annotations = SessionAnnotations(objects={1: {"id": 1, "label": "Object 1", "color": "#ff0000"}},
                                         prompts=[{"frame_idx": 0, "obj_id": 1, "points": [[10, 10]], "labels": [1]}])
        guidance = TrackingGuidance(points=[TrackedGuidancePoint(
            obj_id=1, source_frame_idx=0, tracks=[[10 + i, 10 + i] for i in range(60)], visibility=[True] * 60)])
        run = SimpleNamespace(
            guidance=guidance, session=SimpleNamespace(annotations=annotations, video_frame_files=[""] * 60),
            request=VideoPropagateRequest(use_tracked_points=True, tracked_point_keyframe_interval=8),
            start_frame_idx=0, end_frame_idx=59, frame_width=64, frame_height=64,
            tracked_points_seeded=0, tracked_points_seeded_frames=set(), report=lambda *args, **kwargs: None)
        masker = Mock()

        seed_tracked_points(run, masker)

        frames = [call.kwargs["frame_idx"] for call in masker.add_new_points_or_box.call_args_list]
        self.assertEqual(frames, [8, 16, 24, 32, 40])  # not frame 0 (prompted), nothing in the last 16 frames
        for call in masker.add_new_points_or_box.call_args_list:
            self.assertEqual((call.kwargs["labels"], call.kwargs["clear_old_points"], call.kwargs["box"]),
                             ([1], False, None))
        self.assertEqual(run.tracked_points_seeded, 5)
        self.assertEqual(run.tracked_points_seeded_frames, {8, 16, 24, 32, 40})
        self.assertIsNone(guidance.skipped_reason)

    def test_when_no_frame_can_be_seeded_the_result_gets_a_reason(self):
        from types import SimpleNamespace
        from video.propagation import seed_tracked_points

        annotations = SessionAnnotations(objects={1: {"id": 1, "label": "Object 1", "color": "#ff0000"}},
                                         prompts=[{"frame_idx": 0, "obj_id": 1, "points": [[10, 10]], "labels": [1]}])
        guidance = TrackingGuidance(points=[TrackedGuidancePoint(
            obj_id=1, source_frame_idx=0, tracks=[[10, 10]] * 10, visibility=[True] * 10)])
        run = SimpleNamespace(
            guidance=guidance, session=SimpleNamespace(annotations=annotations, video_frame_files=[""] * 10),
            request=VideoPropagateRequest(use_tracked_points=True), start_frame_idx=0, end_frame_idx=9,
            frame_width=64, frame_height=64, tracked_points_seeded=0, tracked_points_seeded_frames=set(),
            report=lambda *args, **kwargs: None)
        masker = Mock()

        seed_tracked_points(run, masker)

        masker.add_new_points_or_box.assert_not_called()
        self.assertTrue(guidance.should_warn)
        self.assertIn("No frame could be seeded", guidance.skipped_reason)


if __name__ == "__main__":
    unittest.main()
