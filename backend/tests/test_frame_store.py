import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sam2_frame_store import LazyFrameStore, PreloadedFrameStore, frame_store_loading, list_frame_paths

try:
    import sam2.sam2_video_predictor as predictor_module
    from sam2.utils.misc import load_video_frames as sam2_load_video_frames
except ImportError:  # the stores themselves do not need SAM2; only the comparisons against it do
    predictor_module = None

IMAGE_SIZE = 64
FRAME_COUNT = 12


def write_frames(directory: Path, count: int = FRAME_COUNT, start: int = 0) -> list[Path]:
    """Non-square noise frames, so a resize or a channel mix-up cannot go unnoticed."""
    rng = np.random.default_rng(3)
    paths = []
    for index in range(start, start + count):
        pixels = rng.integers(0, 256, size=(40, 60, 3), dtype=np.uint8)
        path = directory / f"{index:05d}.jpg"
        Image.fromarray(pixels).save(path, quality=90)
        paths.append(path)
    return paths


def wait_until(condition, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.01)
    return condition()


class FrameStoreCase(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.paths = [str(path) for path in write_frames(self.directory)]


class FramePathTests(FrameStoreCase):
    def test_frames_are_listed_in_numeric_order_and_only_jpegs(self):
        write_frames(self.directory, 2, start=100)
        (self.directory / "notes.txt").write_text("not a frame")
        (self.directory / "00005.png").write_bytes(b"")
        names = [Path(path).name for path in list_frame_paths(str(self.directory))]
        self.assertEqual(names, [f"{index:05d}.jpg" for index in [*range(FRAME_COUNT), 100, 101]])


@unittest.skipIf(predictor_module is None, "requires SAM2")
class SameValuesAsSam2Tests(FrameStoreCase):
    """The stores replace SAM2's own loader, so every frame must be bit-identical to what it would have produced."""

    def reference(self):
        frames, height, width = sam2_load_video_frames(
            str(self.directory), IMAGE_SIZE, True, compute_device=torch.device("cpu"))
        return frames, height, width

    def test_every_frame_and_the_video_size_match(self):
        reference, height, width = self.reference()
        stores = {
            "lazy": LazyFrameStore(self.paths, IMAGE_SIZE),
            "preload": PreloadedFrameStore(self.paths, IMAGE_SIZE),
        }
        for name, store in stores.items():
            with self.subTest(store=name):
                self.assertEqual(len(store), len(reference))
                self.assertEqual((store.video_height, store.video_width), (height, width))
                for index in range(len(store)):
                    self.assertTrue(torch.equal(store[index], reference[index]), f"frame {index} differs")

    def test_values_do_not_depend_on_the_order_frames_are_read(self):
        reference, _, _ = self.reference()
        store = LazyFrameStore(self.paths, IMAGE_SIZE)
        for index in [7, 2, 11, 0, 5, 5, 6, 4, 3]:
            self.assertTrue(torch.equal(store[index], reference[index]), f"frame {index} differs")


class SequenceBehaviourTests(FrameStoreCase):
    def stores(self):
        return [LazyFrameStore(self.paths, IMAGE_SIZE), PreloadedFrameStore(self.paths, IMAGE_SIZE)]

    def test_the_preloaded_store_hands_out_frames_that_cannot_change_what_it_holds(self):
        store = PreloadedFrameStore(self.paths, IMAGE_SIZE)
        first = store[0]
        expected = first.clone()
        first.zero_()
        self.assertTrue(torch.equal(store[0], expected))


class LazyStoreTests(FrameStoreCase):
    def test_only_a_few_recent_frames_are_kept(self):
        store = LazyFrameStore(self.paths, IMAGE_SIZE, cache_frames=3, prefetch_frames=0)
        for index in range(FRAME_COUNT):
            store[index]
        self.assertEqual(list(store._cache), [9, 10, 11])

    def test_reading_in_order_decodes_the_next_frames_in_the_background(self):
        store = LazyFrameStore(self.paths, IMAGE_SIZE, cache_frames=8, prefetch_frames=4)
        store[0]
        store[1]
        self.assertTrue(wait_until(lambda: {2, 3, 4, 5} <= set(store._cache)))

    def test_a_read_ahead_that_failed_surfaces_its_error_when_the_frame_is_read(self):
        Path(self.paths[3]).write_bytes(b"not a jpeg")
        store = LazyFrameStore(self.paths, IMAGE_SIZE, cache_frames=8, prefetch_frames=4)
        store[0]
        store[1]
        self.assertTrue(wait_until(lambda: store._pending == {}))
        self.assertTrue(wait_until(lambda: {2, 4, 5} <= set(store._cache)))
        with self.assertRaises(OSError):
            store[3]
        store[4]  # the bad frame does not poison the others

    def test_concurrent_readers_get_correct_frames(self):
        store = LazyFrameStore(self.paths, IMAGE_SIZE, cache_frames=8, prefetch_frames=4)
        expected = {index: PreloadedFrameStore(self.paths, IMAGE_SIZE)[index] for index in range(FRAME_COUNT)}
        failures = []

        def read(order):
            try:
                for index in order:
                    if not torch.equal(store[index], expected[index]):
                        failures.append(index)
            except Exception as error:  # noqa: BLE001 - reported below
                failures.append(error)

        threads = [threading.Thread(target=read, args=(order,)) for order in (
            range(FRAME_COUNT), reversed(range(FRAME_COUNT)), [5, 1, 9, 2, 7] * 3)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(10)
        self.assertEqual(failures, [])


@unittest.skipIf(predictor_module is None, "requires SAM2")
class LoaderSwapTests(FrameStoreCase):
    def test_init_state_gets_a_store_only_while_the_block_runs(self):
        original = predictor_module.load_video_frames
        with frame_store_loading("lazy"):
            swapped = predictor_module.load_video_frames
            self.assertIsNot(swapped, original)
            frames, height, width = swapped(str(self.directory), IMAGE_SIZE, True)
        self.assertIsInstance(frames, LazyFrameStore)
        self.assertEqual((height, width), (40, 60))
        self.assertIs(predictor_module.load_video_frames, original)

    def test_the_loader_is_restored_when_initialization_fails(self):
        original = predictor_module.load_video_frames
        with self.assertRaises(RuntimeError), frame_store_loading("lazy"):
            raise RuntimeError("model failed")
        self.assertIs(predictor_module.load_video_frames, original)

    def test_two_states_never_initialize_at_once(self):
        entered = threading.Event()

        def other():
            with frame_store_loading("lazy"):
                entered.set()

        with frame_store_loading("lazy"):
            thread = threading.Thread(target=other)
            thread.start()
            self.assertFalse(entered.wait(0.3))
        thread.join(5)
        self.assertTrue(entered.is_set())


if __name__ == "__main__":
    unittest.main()
