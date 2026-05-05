import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path

from fastapi import HTTPException

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.state import state
from video.service import get_mask_data_window


class MaskDataWindowTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.video_dir = Path(self.temp_dir.name)
        self.manifest_path = self.video_dir / "manifest.json"
        state.video_dir = str(self.video_dir)
        state.video_frame_files = ["00000.jpg", "00001.jpg", "00002.jpg", "00003.jpg"]
        state.mask_manifest_path = str(self.manifest_path)
        self.manifest_path.write_text(
            json.dumps(
                {
                    "num_frames": 4,
                    "frames": {
                        "0": {"objects": {"1": {"size": [1, 1], "rle": [[0, 1]], "bbox": [0, 0, 1, 1]}}},
                        "2": {
                            "objects": {
                                "1": {"size": [1, 1], "rle": [[0, 1]], "bbox": [0, 0, 1, 1]},
                                "2": {"size": [1, 1], "rle": [[0, 1]], "bbox": [0, 0, 1, 1]},
                            }
                        },
                    },
                }
            )
        )

    def tearDown(self):
        state.video_dir = None
        state.video_frame_files = []
        state.mask_manifest_path = None
        self.temp_dir.cleanup()

    def test_returns_requested_range_and_omits_empty_frames(self):
        response = asyncio.run(get_mask_data_window(0, 2))

        self.assertEqual(response["start_frame_idx"], 0)
        self.assertEqual(response["end_frame_idx"], 2)
        self.assertEqual(set(response["frames"].keys()), {"0", "2"})

    def test_filters_by_object_ids(self):
        response = asyncio.run(get_mask_data_window(0, 3, object_ids="2"))

        self.assertEqual(set(response["frames"].keys()), {"2"})
        self.assertEqual(set(response["frames"]["2"]["objects"].keys()), {"2"})

    def test_includes_empty_frames_when_requested(self):
        response = asyncio.run(get_mask_data_window(1, 1, include_empty=True))

        self.assertEqual(response["frames"], {"1": {"objects": {}}})

    def test_rejects_invalid_object_ids(self):
        with self.assertRaises(HTTPException) as context:
            asyncio.run(get_mask_data_window(0, 1, object_ids="1,nope"))

        self.assertEqual(context.exception.status_code, 400)

    def test_clamps_to_manifest_bounds(self):
        response = asyncio.run(get_mask_data_window(2, 99, include_empty=True))

        self.assertEqual(response["start_frame_idx"], 2)
        self.assertEqual(response["end_frame_idx"], 3)
        self.assertEqual(set(response["frames"].keys()), {"2", "3"})

    def test_no_manifest_returns_empty_frames(self):
        self.manifest_path.unlink()
        response = asyncio.run(get_mask_data_window(0, 3))

        self.assertEqual(response["frames"], {})


if __name__ == "__main__":
    unittest.main()
