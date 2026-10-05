import base64
import sys
from pathlib import Path
import unittest
import ast
import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from video.mask_encoding import encode_mask_payload, encode_live_mask_payload
from video.masks import manifest_frame_payload


class MaskEncodingTests(unittest.TestCase):
    def test_exact_row_major_runs_and_bbox(self):
        mask = np.array([[False, True, True], [True, False, True]])
        self.assertEqual(encode_mask_payload(mask), {
            "size": [2, 3], "rle": [[1, 3], [5, 1]], "bbox": [0, 0, 3, 2]})

    def test_empty_and_full_masks(self):
        self.assertEqual(encode_mask_payload(np.zeros((2, 3), dtype=bool)), {
            "size": [2, 3], "rle": [], "bbox": [0, 0, 0, 0]})
        self.assertEqual(encode_mask_payload(np.ones((2, 3), dtype=bool)), {
            "size": [2, 3], "rle": [[0, 6]], "bbox": [0, 0, 3, 2]})

    def test_round_trip_random_fragmented_and_noncontiguous_masks(self):
        rng = np.random.default_rng(42)
        for shape in [(1, 1), (1, 40), (40, 1), (20, 30)]:
            for density in [0.0, 0.1, 0.5, 1.0]:
                mask = (rng.random(shape) < density)[:, ::-1]
                payload = encode_mask_payload(mask)
                restored = np.zeros(mask.size, dtype=bool)
                for start, length in payload["rle"]:
                    restored[start:start + length] = True
                np.testing.assert_array_equal(restored.reshape(mask.shape), mask)

    def test_saved_and_live_encodings_match(self):
        mask = np.array([[False, True], [True, False]])
        self.assertEqual(manifest_frame_payload({7: mask})["objects"]["7"], encode_mask_payload(mask))

    def test_live_fragmented_masks_pack_exact_pixels_with_zero_padding(self):
        rng = np.random.default_rng(8)
        for shape in [(17, 19), (108, 192), (40, 41)]:
            mask = (rng.random(shape) < 0.5)[:, ::-1]
            payload = encode_live_mask_payload(mask)
            self.assertEqual(payload["encoding"], "packed-bits")
            raw = np.frombuffer(base64.b64decode(payload["data"]), dtype=np.uint8)
            bits = np.unpackbits(raw, bitorder="big")
            np.testing.assert_array_equal(bits[:mask.size].reshape(shape), mask)
            self.assertFalse(bits[mask.size:].any())
            self.assertEqual(len(raw), (mask.size + 7) // 8)

    def test_live_compact_masks_keep_rle(self):
        mask = np.zeros((108, 192), dtype=bool)
        mask[20:80, 30:140] = True
        self.assertEqual(encode_live_mask_payload(mask), encode_mask_payload(mask))

    def test_invalid_dimensions(self):
        for mask in [np.array([True]), np.zeros((2, 2, 2)), np.zeros((0, 2)), np.zeros((2, 0))]:
            with self.assertRaises(ValueError):
                encode_mask_payload(mask)


class LiveMaskResponseTests(unittest.TestCase):
    """Exercise the service functions without importing the GPU/model runtime."""

    def setUp(self):
        self.mask = np.array([[True, False], [False, False]])
        self.masker = SimpleNamespace(
            online_mode=True,
            add_new_points_or_box=Mock(return_value=(0, [1], [self.mask])),
            add_new_mask=Mock(return_value=(0, [1], [self.mask])),
            propagate_in_video=Mock(return_value={}),
        )
        self.state = SimpleNamespace(video_masker=self.masker,
                                     video_frame_files=["00000.jpg"], video_state_epoch=3)
        self.record = Mock()

    def call(self, name, request):
        # Compile the actual endpoint body while substituting only runtime dependencies.
        source = Path(__file__).resolve().parents[1] / "video" / "service.py"
        module = ast.parse(source.read_text())
        function = next(node for node in module.body if isinstance(node, ast.AsyncFunctionDef) and node.name == name)
        namespace = {"np": np, "state": self.state, "logger": logging.getLogger(__name__),
                     "encode_live_mask_payload": encode_live_mask_payload,
                     "mask_logits_to_2d_bool": lambda mask: np.asarray(mask, dtype=bool),
                     "require_no_active_job": Mock(), "record_prompt_event": self.record,
                     "VideoAddPointsOrBoxRequest": object, "VideoAddMaskRequest": object}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
        return asyncio.run(namespace[name](request))

    def request(self):
        return SimpleNamespace(frame_idx=0, obj_id=1, points=[[0, 0]], labels=[1],
                               clear_old_points=True, box=None)

    def test_fragmented_point_response_uses_packed_bits(self):
        mask = np.indices((17, 19)).sum(axis=0) % 2 == 0
        self.masker.add_new_points_or_box.return_value = (0, [1], [mask])
        result = self.call("add_new_points_or_box", self.request())
        self.assertEqual(result["out_masks"][0]["encoding"], "packed-bits")
        self.assertEqual(result["mask_shapes"], {1: [17, 19]})
        self.assertEqual(result["mask_pixel_counts"], {1: int(mask.sum())})

    def test_point_response_contains_encoded_masks_and_matching_metadata(self):
        result = self.call("add_new_points_or_box", self.request())
        self.assertEqual(result["mask_encoding"], "mixed")
        self.assertEqual(result["out_masks"], [encode_mask_payload(self.mask)])
        self.assertEqual(result["mask_pixel_counts"], {1: 1})
        self.assertEqual(result["mask_shapes"], {1: [2, 2]})
        self.assertEqual(result["state_epoch"], 3)
        self.record.assert_called_once()

    def test_fallback_mask_is_encoded_after_replacing_empty_interactive_result(self):
        self.masker.add_new_points_or_box.return_value = (0, [1], [np.zeros((2, 2), dtype=bool)])
        self.masker.propagate_in_video.return_value = {0: {1: self.mask}}
        result = self.call("add_new_points_or_box", self.request())
        self.assertTrue(result["single_frame_fallback_used"])
        self.assertEqual(result["out_masks"], [encode_mask_payload(self.mask)])
        self.assertEqual(result["mask_pixel_counts"], {1: 1})

    def test_empty_result_remains_encoded_and_identifiable(self):
        self.masker.add_new_points_or_box.return_value = (0, [1], [np.zeros((2, 2), dtype=bool)])
        result = self.call("add_new_points_or_box", self.request())
        self.assertEqual(result["out_masks"][0]["rle"], [])
        self.assertEqual(result["mask_pixel_counts"], {1: 0})
        self.assertFalse(result["single_frame_fallback_used"])

    def test_add_mask_response_uses_the_same_encoding(self):
        result = self.call("add_new_mask", SimpleNamespace(frame_idx=0, obj_id=1, mask=self.mask.tolist()))
        self.assertEqual(result["mask_encoding"], "mixed")
        self.assertEqual(result["out_masks"], [encode_mask_payload(self.mask)])


if __name__ == "__main__":
    unittest.main()
