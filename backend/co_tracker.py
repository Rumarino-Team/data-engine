import torch
import torch.nn.functional as F
import numpy as np
import cv2
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

from core.config import COTRACKER_REPO, COTRACKER_REVISION

VIDEO_INPUT_RESO = (384, 512) # Resolution of the input video to the model
DEFAULT_COTRACKER_MODEL = "cotracker3_online"
# Full-resolution frames uploaded per resize step; bounds peak upload memory.
PREPARE_CHUNK_FRAMES = 16


def resize_frames_to_model(frames_rgb: np.ndarray, resolution: Tuple[int, int], device) -> torch.Tensor:
    """(T, H, W, 3) uint8 RGB -> (T, 3, h, w) float32 at model resolution.

    Same float conversion and bilinear/align_corners resize the online predictor
    applies to each chunk, so feeding the result changes no model input.
    """
    chunk = torch.from_numpy(np.ascontiguousarray(frames_rgb)).to(device)
    chunk = chunk.permute(0, 3, 1, 2).float()
    return F.interpolate(chunk, tuple(resolution), mode="bilinear", align_corners=True)


@dataclass
class PreparedVideo:
    """Model-resolution frames on the tracker device, plus the source size for coordinates."""
    frames: torch.Tensor  # (1, T, 3, h, w) float32
    height: int
    width: int


class FrameChunkLoader:
    """Loads bounded frame ranges as model-resolution CoTracker tensors.

    Online windows overlap by half, so frames from the previous chunk are reused
    instead of decoded again; only one chunk of resized frames is retained.
    """

    def __init__(
        self,
        video_dir: str | Path,
        frame_files: Sequence[str],
        device: str | torch.device,
        resolution: Tuple[int, int] = VIDEO_INPUT_RESO,
    ):
        self.video_dir = Path(video_dir)
        self.frame_files = list(frame_files)
        self.device = torch.device(device)
        self.resolution = tuple(resolution)
        self._frame_shape: tuple[int, int, int] | None = None
        self._cached: dict[int, torch.Tensor] = {}

    @property
    def num_frames(self) -> int:
        return len(self.frame_files)

    @property
    def frame_size(self) -> tuple[int, int]:
        """Source (height, width); known after the first chunk is loaded."""
        if self._frame_shape is None:
            raise ValueError("Frame size is unknown until a chunk is loaded.")
        return self._frame_shape[0], self._frame_shape[1]

    def _load_frame(self, frame_idx: int) -> torch.Tensor:
        frame_path = self.video_dir / self.frame_files[frame_idx]
        frame_bgr = cv2.imread(str(frame_path))
        if frame_bgr is None:
            raise ValueError(f"Unable to read frame {frame_idx}: {frame_path}")

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if self._frame_shape is None:
            self._frame_shape = tuple(frame_rgb.shape)
        elif tuple(frame_rgb.shape) != self._frame_shape:
            raise ValueError(
                "Frame dimensions changed during tracking: "
                f"expected {self._frame_shape}, got {tuple(frame_rgb.shape)} "
                f"at frame {frame_idx} ({frame_path})."
            )
        return resize_frames_to_model(frame_rgb[None], self.resolution, self.device)[0]

    def load_chunk(self, start_idx: int, end_idx_exclusive: int) -> torch.Tensor:
        start_idx = int(start_idx)
        end_idx_exclusive = int(end_idx_exclusive)
        if start_idx < 0 or end_idx_exclusive > self.num_frames or start_idx >= end_idx_exclusive:
            raise ValueError(
                f"Invalid frame chunk range [{start_idx}, {end_idx_exclusive}) "
                f"for {self.num_frames} frames."
            )

        frames = {
            frame_idx: self._cached[frame_idx] if frame_idx in self._cached else self._load_frame(frame_idx)
            for frame_idx in range(start_idx, end_idx_exclusive)
        }
        self._cached = frames
        return torch.stack(list(frames.values()))[None]

class CoTracker:
    def __init__(self, model_name=DEFAULT_COTRACKER_MODEL):
        if model_name != DEFAULT_COTRACKER_MODEL:
            raise ValueError(f"Unsupported CoTracker model '{model_name}'. Use '{DEFAULT_COTRACKER_MODEL}'.")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float32
        self.model_name = model_name

        # Use CoTracker's online model to keep tracking memory bounded.
        # This is independent of SAM2's online mode (video masking).
        self.model = torch.hub.load(f"{COTRACKER_REPO}:{COTRACKER_REVISION}", model_name)
        self.model = self.model.to(self.device)

        # Increase support grid from default 6×6 (36 pts) to 10×10 (100 pts).
        # More support points give the attention mechanism better global context
        # for correlation, reducing drift on long sequences and with many points.
        self.model.support_grid_size = 10

    @property
    def model_resolution(self) -> Tuple[int, int]:
        return tuple(getattr(self.model, "interp_shape", VIDEO_INPUT_RESO))

    def prepare_video(self, video: np.ndarray) -> PreparedVideo:
        """Resize once on the device; reuse the result across query batches."""
        video = np.asarray(video)
        if video.ndim != 4 or video.shape[-1] != 3 or video.shape[0] == 0:
            raise ValueError(f"Expected video shape (T, H, W, 3), got {video.shape}.")
        num_frames, height, width = video.shape[:3]
        resolution = self.model_resolution
        frames = torch.empty((1, num_frames, 3, *resolution), dtype=torch.float32, device=self.device)
        for start in range(0, num_frames, PREPARE_CHUNK_FRAMES):
            end = min(num_frames, start + PREPARE_CHUNK_FRAMES)
            frames[0, start:end] = resize_frames_to_model(video[start:end], resolution, self.device)
        return PreparedVideo(frames=frames, height=int(height), width=int(width))

    # The predictor scales coordinates by (model - 1) / (source - 1). Inputs are
    # already at model resolution, so its factors are exactly 1 and the source
    # scaling happens here instead, with the same arithmetic.
    def _queries_to_model(self, queries: torch.Tensor, height: int, width: int) -> torch.Tensor:
        model_height, model_width = self.model_resolution
        queries = queries.clone()
        queries[:, :, 1:] *= queries.new_tensor([(model_width - 1) / (width - 1), (model_height - 1) / (height - 1)])
        return queries

    def _tracks_from_model(self, tracks: torch.Tensor, height: int, width: int) -> torch.Tensor:
        model_height, model_width = self.model_resolution
        return tracks * tracks.new_tensor([(width - 1) / (model_width - 1), (height - 1) / (model_height - 1)])

    def track(self, video: Union[np.ndarray, PreparedVideo], queries: Optional[np.ndarray] = None, grid_size=15, add_support_grid=True):
        """
        Tracks points in a video.

        Args:
            video: A (T, H, W, 3) RGB array, or a PreparedVideo from prepare_video() to
                   avoid resizing again for each query batch.
            queries (Optional[np.ndarray]): An array of query points of shape (N, 3) where each
                                            row is (t, x, y). If None, a grid of points is tracked.
            grid_size (int): The size of the grid for grid tracking.
            add_support_grid (bool): Whether to add a support grid for user-specified queries.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing:
                - tracks (np.ndarray): The predicted tracks of shape (N, T, 2) for each point.
                - visibility (np.ndarray): The predicted visibility of each point of shape (N, T).
        """

        # Full-resolution frames never reach the GPU as float; only model-resolution
        # frames (B, T, C, h, w) do, the layout the predictor wrapper expects.
        prepared = video if isinstance(video, PreparedVideo) else self.prepare_video(video)

        # Build query tensor if user supplied points
        queries_torch = None
        if queries is not None:
            queries = np.asarray(queries, dtype=np.float32)
            queries_torch = torch.from_numpy(queries).float()[None].to(self.device)  # (1, N, 3)
            queries_torch = self._queries_to_model(queries_torch, prepared.height, prepared.width)

        with torch.inference_mode():
            pred_tracks, pred_visibility = self._track_online(
                prepared.frames, queries_torch, grid_size, add_support_grid
            )
        pred_tracks = self._tracks_from_model(pred_tracks, prepared.height, prepared.width)

        # pred_tracks: (B, T, N, 2),  pred_visibility: (B, T, N)
        # Transpose to (N, T, 2) and (N, T) for our output convention.
        tracks_np = pred_tracks[0].permute(1, 0, 2).detach().cpu().numpy()
        visibility_np = pred_visibility[0].permute(1, 0).detach().cpu().numpy().astype(bool)

        return tracks_np, visibility_np

    def track_streaming(
        self,
        frame_loader: FrameChunkLoader,
        queries: np.ndarray,
        *,
        add_support_grid: bool = True,
    ):
        """
        Tracks query points while loading only CoTracker online chunks onto GPU.

        Returns the same shapes as track():
        - tracks: (N, T, 2)
        - visibility: (N, T)
        """
        if queries is None:
            raise ValueError("Streaming CoTracker currently requires query points.")

        queries = np.asarray(queries, dtype=np.float32)
        if queries.ndim != 2 or queries.shape[1] != 3:
            raise ValueError(f"Expected queries with shape (N, 3), got {queries.shape}.")
        if queries.shape[0] == 0:
            raise ValueError("No query points provided for streaming tracking.")

        num_frames = int(frame_loader.num_frames)
        if num_frames < 2:
            raise ValueError(
                f"Video too short ({num_frames} frames) for online tracking "
                f"(minimum 2 frames required)."
            )
        if np.any(queries[:, 0] < 0) or np.any(queries[:, 0] >= num_frames):
            raise ValueError(f"Query frame index out of bounds for {num_frames} frames.")

        queries_torch = torch.from_numpy(queries).float()[None].to(self.device)
        pred_tracks, pred_visibility, (height, width) = self._track_online_streaming(
            frame_loader,
            queries_torch,
            add_support_grid=add_support_grid,
        )
        pred_tracks = self._tracks_from_model(pred_tracks, height, width)

        tracks_np = pred_tracks[0].permute(1, 0, 2).detach().cpu().numpy()
        visibility_np = pred_visibility[0].permute(1, 0).detach().cpu().numpy().astype(bool)

        if tracks_np.shape[1] != num_frames:
            raise ValueError(
                "Streaming CoTracker returned an unexpected frame count: "
                f"expected {num_frames}, got {tracks_np.shape[1]}."
            )

        return tracks_np, visibility_np

    # ------------------------------------------------------------------
    # Online tracking — sliding window (window_len=16, step=8).
    # The predictor exposes a chunked `forward` interface:
    #   1. First call with is_first_step=True (2×step frames) → (None, None)
    #   2. Subsequent calls with step new frames → accumulated tracks.
    # ------------------------------------------------------------------
    def _track_online(self, video_torch, queries_torch, grid_size, add_support_grid):
        step = self.model.step  # 8 for cotracker3_online (window_len=16)
        T = video_torch.shape[1]

        if T < 2:
            raise ValueError(
                f"Video too short ({T} frames) for online tracking "
                f"(minimum 2 frames required)."
            )

        # Initialize online state once per video (stores query points internally).
        if queries_torch is not None:
            self.model(
                video_chunk=video_torch,
                is_first_step=True,
                queries=queries_torch,
                add_support_grid=add_support_grid,
            )
        else:
            self.model(
                video_chunk=video_torch,
                is_first_step=True,
                grid_size=grid_size,
                grid_query_frame=0,
                add_support_grid=False,
            )

        pred_tracks, pred_visibility = None, None
        process_add_support_grid = add_support_grid if queries_torch is not None else False
        # Match official online API usage from CoTracker:
        #   for ind in range(0, T - step, step):
        #       model(video_chunk=video[:, ind : ind + 2*step])
        # For short videos (T <= step), still run a single processing window.
        for ind in range(0, max(T - step, 1), step):
            pred_tracks, pred_visibility = self.model(
                video_chunk=video_torch[:, ind : ind + step * 2],
                is_first_step=False,
                add_support_grid=process_add_support_grid,
            )

        if pred_tracks is None:
            raise ValueError(
                f"Online tracking produced no output for {T} frames. "
                f"The video may be too short for the sliding window."
            )

        return pred_tracks, pred_visibility

    def _track_online_streaming(self, frame_loader, queries_torch, add_support_grid):
        step = self.model.step
        num_frames = int(frame_loader.num_frames)
        chunk_len = max(1, int(step) * 2)

        init_end = min(num_frames, chunk_len)
        init_chunk = frame_loader.load_chunk(0, init_end)
        # Source size is known once the first chunk is decoded.
        height, width = frame_loader.frame_size
        try:
            self.model(
                video_chunk=init_chunk,
                is_first_step=True,
                queries=self._queries_to_model(queries_torch, height, width),
                add_support_grid=add_support_grid,
            )
        finally:
            del init_chunk

        pred_tracks, pred_visibility = None, None
        for start_idx in range(0, max(num_frames - step, 1), step):
            end_idx = min(num_frames, start_idx + chunk_len)
            video_chunk = frame_loader.load_chunk(start_idx, end_idx)
            try:
                pred_tracks, pred_visibility = self.model(
                    video_chunk=video_chunk,
                    is_first_step=False,
                    add_support_grid=add_support_grid,
                )
            finally:
                del video_chunk

        if pred_tracks is None or pred_visibility is None:
            raise ValueError(
                f"Online tracking produced no output for {num_frames} frames. "
                f"The video may be too short for the sliding window."
            )

        return pred_tracks, pred_visibility, (height, width)

