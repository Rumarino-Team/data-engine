import torch
import numpy as np
import mediapy
from typing import List, Optional, Tuple
import colorsys
import random

VIDEO_INPUT_RESO = (384, 512) # Resolution of the input video to the model

# Generate random colormaps for visualizing different points.
def get_colors(num_colors: int) -> List[Tuple[int, int, int]]:
  """Gets colormap for points."""
  colors = []
  for i in np.arange(0.0, 360.0, 360.0 / num_colors):
    hue = i / 360.0
    lightness = (50 + np.random.rand() * 10) / 100.0
    saturation = (90 + np.random.rand() * 10) / 100.0
    color = colorsys.hls_to_rgb(hue, lightness, saturation)
    colors.append(
        (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))
    )
  random.shuffle(colors)
  return colors

def get_points_on_a_grid(
    size: int,
    extent: Tuple[float, ...],
    center: Optional[Tuple[float, ...]] = None,
    device: Optional[torch.device] = torch.device("cpu"),
):
    r"""Get a grid of points covering a rectangular region

    `get_points_on_a_grid(size, extent)` generates a :attr:`size` by
    :attr:`size` grid fo points distributed to cover a rectangular area
    specified by `extent`.

    The `extent` is a pair of integer :math:`(H,W)` specifying the height
    and width of the rectangle.

    Optionally, the :attr:`center` can be specified as a pair :math:`(c_y,c_x)`
    specifying the vertical and horizontal center coordinates. The center
    defaults to the middle of the extent.

    Points are distributed uniformly within the rectangle leaving a margin
    :math:`m=W/64` from the border.

    It returns a :math:`(1, \text{size} \times \text{size}, 2)` tensor of
    points :math:`P_{ij}=(x_i, y_i)` where

    .. math::
        P_{ij} = \left(
             c_x + m -\frac{W}{2} + \frac{W - 2m}{\text{size} - 1}\, j,~
             c_y + m -\frac{H}{2} + \frac{H - 2m}{\text{size} - 1}\, i
        \right)

    Points are returned in row-major order.

    Args:
        size (int): grid size.
        extent (tuple): height and with of the grid extent.
        center (tuple, optional): grid center.
        device (str, optional): Defaults to `"cpu"`.

    Returns:
        Tensor: grid.
    """
    if size == 1:
        return torch.tensor([extent[1] / 2, extent[0] / 2], device=device)[None, None]

    if center is None:
        center = [extent[0] / 2, extent[1] / 2]

    margin = extent[1] / 64
    range_y = (margin - extent[0] / 2 + center[0], extent[0] / 2 + center[0] - margin)
    range_x = (margin - extent[1] / 2 + center[1], extent[1] / 2 + center[1] - margin)
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(*range_y, size, device=device),
        torch.linspace(*range_x, size, device=device),
        indexing="ij",
    )
    return torch.stack([grid_x, grid_y], dim=-1).reshape(1, -1, 2)

def paint_point_track(
    frames: np.ndarray,
    point_tracks: np.ndarray,
    visibles: np.ndarray,
    colormap: Optional[List[Tuple[int, int, int]]] = None,
) -> np.ndarray:
  """Converts a sequence of points to color code video.

  Args:
    frames: [num_frames, height, width, 3], np.uint8, [0, 255]
    point_tracks: [num_points, num_frames, 2], np.float32, [0, width / height]
    visibles: [num_points, num_frames], bool
    colormap: colormap for points, each point has a different RGB color.

  Returns:
    video: [num_frames, height, width, 3], np.uint8, [0, 255]
  """
  num_points, num_frames = point_tracks.shape[0:2]
  if colormap is None:
    colormap = get_colors(num_colors=num_points)
  height, width = frames.shape[1:3]
  dot_size_as_fraction_of_min_edge = 0.015
  radius = int(round(min(height, width) * dot_size_as_fraction_of_min_edge))
  diam = radius * 2 + 1
  quadratic_y = np.square(np.arange(diam)[:, np.newaxis] - radius - 1)
  quadratic_x = np.square(np.arange(diam)[np.newaxis, :] - radius - 1)
  icon = (quadratic_y + quadratic_x) - (radius**2) / 2.0
  sharpness = 0.15
  icon = np.clip(icon / (radius * 2 * sharpness), 0, 1)
  icon = 1 - icon[:, :, np.newaxis]
  icon1 = np.pad(icon, [(0, 1), (0, 1), (0, 0)])
  icon2 = np.pad(icon, [(1, 0), (0, 1), (0, 0)])
  icon3 = np.pad(icon, [(0, 1), (1, 0), (0, 0)])
  icon4 = np.pad(icon, [(1, 0), (1, 0), (0, 0)])

  video = frames.copy()
  # Use the minimum of video frames and track frames to avoid index errors
  num_frames_to_paint = min(num_frames, frames.shape[0])
  for t in range(num_frames_to_paint):
    # Pad so that points that extend outside the image frame don't crash us
    image = np.pad(
        video[t],
        [
            (radius + 1, radius + 1),
            (radius + 1, radius + 1),
            (0, 0),
        ],
    )
    for i in range(num_points):
      # The icon is centered at the center of a pixel, but the input coordinates
      # are raster coordinates.  Therefore, to render a point at (1,1) (which
      # lies on the corner between four pixels), we need 1/4 of the icon placed
      # centered on the 0'th row, 0'th column, etc.  We need to subtract
      # 0.5 to make the fractional position come out right.
      x, y = point_tracks[i, t, :] + 0.5
      x = min(max(x, 0.0), width)
      y = min(max(y, 0.0), height)

      if visibles[i, t]:
        x1, y1 = np.floor(x).astype(np.int32), np.floor(y).astype(np.int32)
        x2, y2 = x1 + 1, y1 + 1

        # bilinear interpolation
        patch = (
            icon1 * (x2 - x) * (y2 - y)
            + icon2 * (x2 - x) * (y - y1)
            + icon3 * (x - x1) * (y2 - y)
            + icon4 * (x - x1) * (y - y1)
        )
        x_ub = x1 + 2 * radius + 2
        y_ub = y1 + 2 * radius + 2
        image[y1:y_ub, x1:x_ub, :] = (1 - patch) * image[
            y1:y_ub, x1:x_ub, :
        ] + patch * np.array(colormap[i])[np.newaxis, np.newaxis, :]

      # Remove the pad
      video[t] = image[
          radius + 1 : -radius - 1, radius + 1 : -radius - 1
      ].astype(np.uint8)
  return video

class CoTracker:
    def __init__(self, model_name="cotracker3_offline"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float32
        self.model_name = model_name
        self.is_online = "online" in model_name

        # Load the requested CoTracker model variant:
        #   • cotracker3_offline — single global attention pass over all frames,
        #     globally consistent tracks, no sliding-window boundary artefacts.
        #   • cotracker3_online  — sliding window (window_len=16, step=8), lower
        #     VRAM but may exhibit drift at window boundaries.
        # Both are independent of SAM2's online mode (video masking).
        self.model = torch.hub.load("facebookresearch/co-tracker", model_name)
        self.model = self.model.to(self.device)

        # Increase support grid from default 6×6 (36 pts) to 10×10 (100 pts).
        # More support points give the attention mechanism better global context
        # for correlation, reducing drift on long sequences and with many points.
        self.model.support_grid_size = 10

    def track(self, video: np.ndarray, queries: Optional[np.ndarray] = None, grid_size=15, add_support_grid=True):
        """
        Tracks points in a video.

        Args:
            video (np.ndarray): A video as a numpy array of shape (T, H, W, 3) in RGB format.
            queries (Optional[np.ndarray]): An array of query points of shape (N, 3) where each
                                            row is (t, x, y). If None, a grid of points is tracked.
            grid_size (int): The size of the grid for grid tracking.
            add_support_grid (bool): Whether to add a support grid for user-specified queries.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing:
                - tracks (np.ndarray): The predicted tracks of shape (N, T, 2) for each point.
                - visibility (np.ndarray): The predicted visibility of each point of shape (N, T).
        """

        # Preprocess video on CPU first to avoid holding full-resolution frames on GPU.
        # Shape: (B, T, C, H, W) — the predictor wrapper expects this layout.
        video_torch = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float().to(self.device)

        # Build query tensor if user supplied points
        queries_torch = None
        if queries is not None:
            queries = np.asarray(queries, dtype=np.float32)
            queries_torch = torch.from_numpy(queries).float()[None].to(self.device)  # (1, N, 3)

        with torch.inference_mode():
            if self.is_online:
                pred_tracks, pred_visibility = self._track_online(
                    video_torch, queries_torch, grid_size, add_support_grid
                )
            else:
                pred_tracks, pred_visibility = self._track_offline(
                    video_torch, queries_torch, grid_size, add_support_grid
                )

        # pred_tracks: (B, T, N, 2),  pred_visibility: (B, T, N)
        # Transpose to (N, T, 2) and (N, T) for our output convention.
        tracks_np = pred_tracks[0].permute(1, 0, 2).detach().cpu().numpy()
        # Always visible — caller controls rendering.
        all_visible = np.ones((tracks_np.shape[0], tracks_np.shape[1]), dtype=bool)

        return tracks_np, all_visible

    # ------------------------------------------------------------------
    # Offline tracking — single global attention pass over all T frames.
    # Supports backward_tracking for forward+backward track merging.
    # ------------------------------------------------------------------
    def _track_offline(self, video_torch, queries_torch, grid_size, add_support_grid):
        if queries_torch is not None:
            return self.model(
                video_torch,
                queries=queries_torch,
                add_support_grid=add_support_grid,
                backward_tracking=True,
            )
        else:
            return self.model(
                video_torch,
                grid_size=grid_size,
                grid_query_frame=0,
                backward_tracking=True,
            )

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

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, default="./videos/bear.mp4", help="path to a video")
    parser.add_argument("--output_path", type=str, default="./output.mp4", help="path to save the output video")
    parser.add_argument("--grid_size", type=int, default=15, help="grid size for tracking")
    args = parser.parse_args()

    # Load video
    video = mediapy.read_video(args.video_path)
    
    # Initialize tracker
    tracker = CoTracker()

    # Track points
    tracks, visibility = tracker.track(video, grid_size=args.grid_size)

    # Visualize and save video
    painted_video = paint_point_track(video, tracks, visibility)
    mediapy.write_video(args.output_path, painted_video, fps=mediapy.read_video(args.video_path).metadata.fps)
    print(f"Saved tracking video to {args.output_path}")
