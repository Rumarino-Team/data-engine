"""Frame storage for SAM2 video states.

SAM2's own loader decodes every frame of the clip, one at a time, into one float32 tensor (about 12 MiB per frame at
1024x1024) when `init_state` runs. The predictor only ever uses that store as `len(images)` and `images[frame_idx]`
(`SAM2VideoPredictor._get_image_feature`), so these classes offer the same two operations with a cheaper strategy:

- `PreloadedFrameStore` decodes every frame at init, in parallel, and keeps it as uint8 (3 MiB per frame). The float32
  frame is rebuilt from the 8-bit pixels when it is read.
- `LazyFrameStore` decodes a frame from its JPEG the first time it is asked for and keeps only a few recent ones, so
  memory does not grow with the clip.

Both return values that are bit-identical to SAM2's. SAM2 divides by 255 in float64, casts to float32, then applies the
mean and std. There are only 256 possible pixel values, so that whole chain is computed once per channel into a lookup
table and each frame is a table lookup, with no float64 copy of the frame.
"""
import os
import threading
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Literal, Optional, Sequence

import numpy as np
import torch
from PIL import Image

IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)
JPEG_SUFFIXES = {".jpg", ".jpeg"}

# SAM2 walks the clip in order, so the lazy cache only has to cover the frames around the one being worked on.
DEFAULT_CACHE_FRAMES = 8
DEFAULT_PREFETCH_FRAMES = 4
PREFETCH_WORKERS = 2
PRELOAD_WORKERS = min(8, os.cpu_count() or 1)

StoreMode = Literal["lazy", "preload"]


def list_frame_paths(frames_dir: str) -> list[str]:
    """The JPEG frames of a directory in frame order, named like SAM2 expects ("<frame_index>.jpg")."""
    folder = Path(frames_dir)
    names = [p.name for p in folder.iterdir() if p.suffix.lower() in JPEG_SUFFIXES]
    if not names:
        raise RuntimeError(f"no images found in {folder}")
    names.sort(key=lambda name: int(Path(name).stem))
    return [str(folder / name) for name in names]


def _normalization_table() -> torch.Tensor:
    """(3, 256) float32: what SAM2 turns each 8-bit value of each channel into, computed through its exact steps."""
    levels = torch.from_numpy((np.arange(256) / 255.0).astype(np.float32))  # float64 division, then the float32 cast
    mean = torch.tensor(IMAGE_MEAN, dtype=torch.float32)[:, None]
    std = torch.tensor(IMAGE_STD, dtype=torch.float32)[:, None]
    return (levels[None, :] - mean) / std


class _FrameDecoder(Sequence):
    """What both stores share: where the frames are, how to decode one, how to turn pixels into SAM2's input."""

    def __init__(self, frame_paths: list[str], image_size: int):
        self.frame_paths = frame_paths
        self.image_size = image_size
        self._table = _normalization_table()
        # The first frame also tells SAM2 the original video size, which it needs to resize output masks.
        with Image.open(frame_paths[0]) as first:
            self.video_width, self.video_height = first.size

    def __len__(self) -> int:
        return len(self.frame_paths)

    def _checked_index(self, index: int) -> int:
        if isinstance(index, slice):
            raise TypeError("Frame stores do not support slices.")
        return range(len(self))[index]  # negative indexes and IndexError like a list

    def _decode_pixels(self, index: int) -> np.ndarray:
        """(image_size, image_size, 3) uint8, resized like SAM2's loader."""
        with Image.open(self.frame_paths[index]) as image:
            return np.array(image.convert("RGB").resize((self.image_size, self.image_size)))

    def _normalize(self, pixels: np.ndarray) -> torch.Tensor:
        """(3, image_size, image_size) float32, bit-identical to SAM2's decode + normalize."""
        frame = torch.empty((3, self.image_size, self.image_size), dtype=torch.float32)
        pixels = torch.from_numpy(pixels)
        for channel in range(3):
            frame[channel] = self._table[channel][pixels[..., channel].long()]
        return frame


class PreloadedFrameStore(_FrameDecoder):
    """Every frame decoded at init (in parallel) and held as uint8; float32 frames are built when read."""

    def __init__(self, frame_paths: list[str], image_size: int, workers: int = PRELOAD_WORKERS):
        super().__init__(frame_paths, image_size)
        with ThreadPoolExecutor(max_workers=max(1, workers), thread_name_prefix="sam2-preload") as pool:
            self._pixels = list(pool.map(self._decode_pixels, range(len(frame_paths))))

    def __getitem__(self, index: int) -> torch.Tensor:
        return self._normalize(self._pixels[self._checked_index(index)])


_loader_swap_lock = threading.Lock()
_prefetch_pool: Optional[ThreadPoolExecutor] = None
_prefetch_pool_lock = threading.Lock()


def _shared_prefetch_pool() -> ThreadPoolExecutor:
    global _prefetch_pool
    with _prefetch_pool_lock:
        if _prefetch_pool is None:
            _prefetch_pool = ThreadPoolExecutor(max_workers=PREFETCH_WORKERS, thread_name_prefix="sam2-frames")
        return _prefetch_pool


class LazyFrameStore(_FrameDecoder):
    """Frames decoded from JPEG on first use; only a few recent ones are kept.

    When frames are asked for one after another (propagation), the next few are decoded in the background so the
    model never waits on the decoder. A jump to an unrelated frame (a click) decodes just that frame.
    """

    def __init__(
        self, frame_paths: list[str], image_size: int,
        cache_frames: int = DEFAULT_CACHE_FRAMES, prefetch_frames: int = DEFAULT_PREFETCH_FRAMES,
    ):
        if prefetch_frames < 0:
            raise ValueError("prefetch_frames cannot be negative.")
        if cache_frames < prefetch_frames + 2:
            raise ValueError("cache_frames must hold the prefetched frames plus the current and previous one.")
        super().__init__(frame_paths, image_size)
        self.cache_frames = cache_frames
        self.prefetch_frames = prefetch_frames
        self._cache: OrderedDict[int, torch.Tensor] = OrderedDict()
        self._pending: dict[int, Future] = {}
        self._last_index: Optional[int] = None
        self._direction = 0  # +1 or -1 while frames are being read in order, else 0
        # Re-entrant: a read-ahead that finishes instantly runs its callback in the thread that holds the lock.
        self._lock = threading.RLock()

    def __getitem__(self, index: int) -> torch.Tensor:
        index = self._checked_index(index)
        with self._lock:
            frame = self._cache.get(index)
            if frame is not None:
                self._cache.move_to_end(index)
            pending = self._pending.get(index) if frame is None else None
        if frame is None and pending is not None:
            try:
                frame = pending.result()
            except Exception:
                frame = None  # a failed read-ahead is retried below, where the error reaches the caller
        if frame is None:
            frame = self._load(index)
            self._store(index, frame)
        self._read_ahead(index)
        return frame

    def _load(self, index: int) -> torch.Tensor:
        return self._normalize(self._decode_pixels(index))

    def _store(self, index: int, frame: torch.Tensor) -> None:
        with self._lock:
            self._cache[index] = frame
            self._cache.move_to_end(index)
            while len(self._cache) > self.cache_frames:
                self._cache.popitem(last=False)

    def _read_ahead(self, index: int) -> None:
        with self._lock:
            if self._last_index is not None and index - self._last_index in (1, -1):
                self._direction = index - self._last_index
            elif index != self._last_index:
                self._direction = 0
            self._last_index = index
            if not self.prefetch_frames or not self._direction:
                return
            wanted = [
                target for target in (index + self._direction * step for step in range(1, self.prefetch_frames + 1))
                if 0 <= target < len(self) and target not in self._cache and target not in self._pending
            ]
            pool = _shared_prefetch_pool()
            for target in wanted:
                future = pool.submit(self._load, target)
                self._pending[target] = future
                future.add_done_callback(lambda done, target=target: self._finish_read_ahead(target, done))

    def _finish_read_ahead(self, index: int, future: Future) -> None:
        if not future.cancelled() and future.exception() is None:
            self._store(index, future.result())
        with self._lock:
            self._pending.pop(index, None)


@contextmanager
def frame_store_loading(
    mode: StoreMode = "lazy",
    cache_frames: int = DEFAULT_CACHE_FRAMES, prefetch_frames: int = DEFAULT_PREFETCH_FRAMES,
) -> Iterator[None]:
    """Make `SAM2VideoPredictor.init_state` use one of these stores for frame directories while the block runs.

    `init_state` calls `load_video_frames` through its module namespace, so swapping that name is enough; the predictor
    itself is untouched. The swap is process-wide, so the block holds a lock: two states never initialize at once.
    """
    import sam2.sam2_video_predictor as predictor_module

    if mode not in ("lazy", "preload"):
        raise ValueError(f"Unknown frame loading mode: {mode!r}")
    with _loader_swap_lock:
        original = predictor_module.load_video_frames
        predictor_module.load_video_frames = _store_loader(original, mode, cache_frames, prefetch_frames)
        try:
            yield
        finally:
            predictor_module.load_video_frames = original


def _store_loader(original, mode: StoreMode, cache_frames: int, prefetch_frames: int):

    def load_video_frames(video_path, image_size, offload_video_to_cpu, *args, **kwargs):
        if not isinstance(video_path, str) or not Path(video_path).is_dir():
            return original(video_path, image_size, offload_video_to_cpu, *args, **kwargs)
        paths = list_frame_paths(video_path)
        if mode == "preload":
            store = PreloadedFrameStore(paths, image_size)
        else:
            store = LazyFrameStore(paths, image_size, cache_frames, prefetch_frames)
        return store, store.video_height, store.video_width

    return load_video_frames
