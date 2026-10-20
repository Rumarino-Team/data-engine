import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.state import BackendState
from core.annotations import SessionAnnotations
from schemas.video import VideoPropagateRequest
from utils import build_empty_mask_manifest, load_mask_manifest, write_mask_manifest
from video import propagation
from video.masks import manifest_frame_payload, accepted_mask


class PartialPropagationTests(unittest.TestCase):
    def test_replaces_only_processed_frames_and_preserves_earlier_and_later_outputs(self):
        for export_overlay_frames, explicit_override in [(True, False), (False, False), (True, True), (False, True)]:
            with self.subTest(export_overlay_frames=export_overlay_frames, explicit_override=explicit_override):
                with tempfile.TemporaryDirectory() as directory, ExitStack() as patches:
                    root = Path(directory)
                    frames = root / "frames"
                    masks = root / "masks"
                    frames.mkdir()
                    masks.mkdir()
                    for i in range(5):
                        cv2.imwrite(str(frames / f"{i:05d}.jpg"), np.zeros((8, 8, 3), dtype=np.uint8))
                        (masks / f"frame_{i:05d}_masks.png").write_bytes(f"old-{i}".encode())
                    old_frame = manifest_frame_payload({1: np.ones((8, 8), dtype=bool)})
                    manifest = build_empty_mask_manifest(
                        source_video_path=None, resolved_video_frames_dir=str(frames),
                        num_frames=5, frame_height=8, frame_width=8,
                    )
                    manifest["frames"] = {str(i): old_frame for i in range(5)}
                    write_mask_manifest(masks / "manifest.json", manifest)
                    masker = Mock(online_mode=True, default_batch_size=2,
                                  offload_video_to_cpu=True, offload_state_to_cpu=False)
                    def produce(**kwargs):
                        for index in range(kwargs["start_frame_idx"], kwargs["start_frame_idx"] + kwargs["max_frame_num_to_track"]):
                            kwargs["frame_callback"](index, {1: np.zeros((8, 8), dtype=bool)})
                    masker.propagate_in_video.side_effect = produce
                    state = BackendState(
                        video_masker=masker, video_dir=str(frames),
                        video_frame_files=[f"{i:05d}.jpg" for i in range(5)],
                        annotations=SessionAnnotations(objects={1: {"id": 1, "label": "Object 1", "color": "#ff0000"}},
                                                    prompts=[{"frame_idx": 1, "obj_id": 1, "points": [[1, 1]], "labels": [1]}]),
                        mask_manifest_path=str(masks / "manifest.json"),
                    )
                    if explicit_override:
                        state.annotations.set_mask(accepted_mask(1, 1, np.ones((8, 8), dtype=bool)))
                    state.interactive_masks = {str(i): {"1": {"mask": "old"}} for i in range(5)}
                    for name, value in {
                        "state": state, "WINDOW_FRAMES_ROOT": root / "windows",
                        "current_masks_dir": lambda: masks, "clear_window_cache": Mock(),
                        "update_job": Mock(), "write_session_metadata": Mock(),
                        "restore_video_masker_from_prompt_events": Mock(),
                    }.items():
                        patches.enter_context(patch.object(propagation, name, value))
                    # Repeat the shorter range: neither run may remove the earlier/later data.
                    for _ in range(2):
                        response = propagation.run_propagation_job(VideoPropagateRequest(
                            start_frame_idx=1, max_frame_num_to_track=2, use_tracked_points=False,
                            export_overlay_frames=export_overlay_frames,
                        ))
                        updated = load_mask_manifest(masks / "manifest.json")
                        self.assertEqual(set(updated["frames"]), {"0", "1", "2", "3", "4"})
                        for i in [0, 3, 4]:
                            self.assertEqual(updated["frames"][str(i)], old_frame)
                            self.assertEqual((masks / f"frame_{i:05d}_masks.png").read_bytes(), f"old-{i}".encode())
                        for i in [1, 2]:
                            self.assertEqual(updated["frames"][str(i)]["objects"]["1"]["rle"],
                                             [[0, 64]] if explicit_override and i == 1 else [])
                            self.assertEqual(updated["frames"][str(i)]["input_revisions"],
                                             {"1": state.annotations.input_revisions.get(1, 0)})
                            overlay = masks / f"frame_{i:05d}_masks.png"
                            if export_overlay_frames:
                                self.assertIsNotNone(cv2.imread(str(masks / updated["frames"][str(i)]["overlay_path"])))
                                self.assertEqual(overlay.read_bytes(), f"old-{i}".encode())
                            else:
                                self.assertEqual(overlay.read_bytes(), f"old-{i}".encode())
                        self.assertEqual(response["video_segments_total_frames"], 2)
                        self.assertEqual(set(state.interactive_masks), {"0", "3", "4"})
                        if explicit_override:
                            self.assertTrue(any(call.kwargs["mask"].all() for call in masker.add_new_mask.call_args_list))


if __name__ == "__main__":
    unittest.main()
