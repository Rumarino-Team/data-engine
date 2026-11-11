import base64
import sys
from pathlib import Path
import unittest
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


if __name__ == "__main__":
    unittest.main()
