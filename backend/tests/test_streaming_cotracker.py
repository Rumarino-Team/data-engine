import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import torch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import co_tracker
from co_tracker import CoTracker, FrameChunkLoader, resize_frames_to_model


class FakeOnlineModel:
    def __init__(self, step=2, num_frames=5):
        self.step = step
        self.num_frames = num_frames
        self.calls = []

    def __call__(self, *, video_chunk, is_first_step, queries=None, add_support_grid=False):
        self.calls.append(
            {
                "shape": tuple(video_chunk.shape),
                "is_first_step": bool(is_first_step),
                "has_queries": queries is not None,
                "add_support_grid": bool(add_support_grid),
            }
        )
        if is_first_step:
            return None, None

        num_points = 1
        tracks = torch.zeros((1, self.num_frames, num_points, 2), dtype=torch.float32)
        visibility = torch.ones((1, self.num_frames, num_points), dtype=torch.bool)
        return tracks, visibility


class FakeFrameLoader:
    frame_size = (4, 4)

    def __init__(self, num_frames=5):
        self.num_frames = num_frames
        self.ranges = []

    def load_chunk(self, start_idx, end_idx_exclusive):
        self.ranges.append((start_idx, end_idx_exclusive))
        length = end_idx_exclusive - start_idx
        return torch.zeros((1, length, 3, 4, 4), dtype=torch.float32)


class FrameChunkLoaderTests(unittest.TestCase):
    def test_load_chunk_returns_cotracker_tensor_shape_and_rgb_order(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            frame_files = []
            colors_bgr = [
                (10, 20, 30),
                (40, 50, 60),
            ]
            for index, color in enumerate(colors_bgr):
                name = f"{index:05d}.png"
                frame = np.zeros((3, 4, 3), dtype=np.uint8)
                frame[:, :] = color
                cv2.imwrite(str(root / name), frame)
                frame_files.append(name)

            loader = FrameChunkLoader(root, frame_files, device="cpu", resolution=(6, 8))
            chunk = loader.load_chunk(0, 2)

            # Frames arrive resized to model resolution; the source size stays available.
            self.assertEqual(tuple(chunk.shape), (1, 2, 3, 6, 8))
            self.assertEqual(loader.frame_size, (3, 4))
            self.assertEqual(int(chunk[0, 0, 0, 0, 0].item()), 30)
            self.assertEqual(int(chunk[0, 0, 1, 0, 0].item()), 20)
            self.assertEqual(int(chunk[0, 0, 2, 0, 0].item()), 10)

    def test_overlapping_chunks_reuse_decoded_frames(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            frame_files = []
            for index in range(6):
                name = f"{index:05d}.png"
                cv2.imwrite(str(root / name), np.full((3, 4, 3), index * 10, dtype=np.uint8))
                frame_files.append(name)
            loader = FrameChunkLoader(root, frame_files, device="cpu", resolution=(3, 4))
            with patch.object(co_tracker.cv2, "imread", wraps=cv2.imread) as imread:
                first = loader.load_chunk(0, 4)
                second = loader.load_chunk(2, 6)
            self.assertEqual(imread.call_count, 6)
            self.assertTrue(torch.equal(first[0, 2:], second[0, :2]))

    def test_load_chunk_rejects_invalid_range(self):
        loader = FrameChunkLoader("/tmp", ["00000.png"], device="cpu")

        with self.assertRaises(ValueError):
            loader.load_chunk(1, 1)


class StreamingCoTrackerTests(unittest.TestCase):
    def test_track_streaming_uses_online_chunks_without_full_video_tensor(self):
        tracker = object.__new__(CoTracker)
        tracker.device = "cpu"
        tracker.model = FakeOnlineModel(step=2, num_frames=5)

        frame_loader = FakeFrameLoader(num_frames=5)
        queries = np.array([[0, 1.0, 2.0]], dtype=np.float32)

        tracks, visibility = tracker.track_streaming(
            frame_loader,
            queries=queries,
            add_support_grid=True,
        )

        self.assertEqual(tracks.shape, (1, 5, 2))
        self.assertEqual(visibility.shape, (1, 5))
        self.assertEqual(frame_loader.ranges, [(0, 4), (0, 4), (2, 5)])
        self.assertTrue(tracker.model.calls[0]["is_first_step"])
        self.assertTrue(tracker.model.calls[0]["has_queries"])
        self.assertFalse(tracker.model.calls[1]["is_first_step"])
        self.assertLessEqual(tracker.model.calls[1]["shape"][1], 4)
        self.assertLessEqual(tracker.model.calls[2]["shape"][1], 4)


class CoordinateRoundTripModel:
    """Echoes model-space queries as tracks for every frame."""
    step = 2
    interp_shape = (6, 8)

    def __init__(self):
        self.chunk_shapes = []

    def __call__(self, *, video_chunk, is_first_step, queries=None, add_support_grid=False, **_):
        self.chunk_shapes.append(tuple(video_chunk.shape))
        if is_first_step:
            self.queries = queries
            return None, None
        frames = video_chunk.shape[1]
        tracks = self.queries[:, None, :, 1:].expand(1, frames, -1, 2)
        return tracks, torch.ones(tracks.shape[:3], dtype=torch.bool)


class ModelResolutionTests(unittest.TestCase):
    def tracker(self):
        tracker = object.__new__(CoTracker)
        tracker.device = "cpu"
        tracker.model = CoordinateRoundTripModel()
        return tracker

    def test_track_feeds_model_resolution_and_returns_source_coordinates(self):
        tracker = self.tracker()
        video = np.random.default_rng(0).integers(0, 255, (4, 30, 40, 3), dtype=np.uint8)
        queries = np.array([[0, 39.0, 29.0], [1, 12.5, 7.25]], dtype=np.float32)
        tracks, _ = tracker.track(video, queries=queries, add_support_grid=False)
        self.assertTrue(all(shape[-2:] == (6, 8) for shape in tracker.model.chunk_shapes))
        np.testing.assert_allclose(tracks[:, 0], queries[:, 1:], rtol=1e-6)

    def test_prepared_video_matches_one_shot_resize_and_is_reusable(self):
        tracker = self.tracker()
        video = np.random.default_rng(1).integers(0, 255, (co_tracker.PREPARE_CHUNK_FRAMES + 3, 30, 40, 3), dtype=np.uint8)
        prepared = tracker.prepare_video(video)
        self.assertTrue(torch.equal(prepared.frames[0], resize_frames_to_model(video, (6, 8), "cpu")))
        self.assertEqual((prepared.height, prepared.width), (30, 40))
        queries = np.array([[0, 20.0, 10.0]], dtype=np.float32)
        first, _ = tracker.track(prepared, queries=queries, add_support_grid=False)
        second, _ = tracker.track(prepared, queries=queries, add_support_grid=False)
        np.testing.assert_array_equal(first, second)


if __name__ == "__main__":
    unittest.main()
