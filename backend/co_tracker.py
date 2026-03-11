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
    def __init__(self, model_name="cotracker3_online"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float32
        
        self.model = torch.hub.load("facebookresearch/co-tracker", model_name)
        self.model = self.model.to(self.device)

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
                - tracks (np.ndarray): The predicted tracks of shape (T, N, 2) for each point.
                - visibility (np.ndarray): The predicted visibility of each point of shape (T, N).
        """
        
        # Preprocess video on CPU first to avoid holding full-resolution frames on GPU.
        # Shape: B, T, C, H, W
        video_torch = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float()

        # Resize for model input on CPU, then move the smaller tensor to GPU.
        video_torch_resized = torch.nn.functional.interpolate(
            video_torch[0],
            size=VIDEO_INPUT_RESO,
            mode='bilinear',
            align_corners=False,
        )
        video_torch_resized = video_torch_resized[None].to(self.device, dtype=self.dtype)

        if queries is None:
            # Grid tracking
            xy = get_points_on_a_grid(grid_size, video_torch_resized.shape[3:], device=self.device)
            queries_torch = torch.cat([torch.zeros_like(xy[:, :, :1]), xy], dim=2).to(self.device)
            add_support_grid = False
        else:
            # Point tracking
            # Scale queries to model input resolution
            H, W = video.shape[1:3]
            queries_scaled = queries.copy()
            queries_scaled[:, 1] *= VIDEO_INPUT_RESO[1] / W
            queries_scaled[:, 2] *= VIDEO_INPUT_RESO[0] / H
            
            # Convert to tensor: N, 3 -> 1, N, 3
            queries_torch = torch.tensor(queries_scaled).float()[None].to(self.device, self.dtype)
            # tyx -> txy
            queries_torch = queries_torch[:, :, [0, 2, 1]]

        # Run tracker using the model's internal forward method
        # The torch.hub model is a wrapper (CoTrackerOnlinePredictor), access the actual model
        actual_model = self.model.model if hasattr(self.model, 'model') else self.model
        
        # For online models, we need to initialize the video processing first
        actual_model.init_video_online_processing()
        
        with torch.inference_mode():
            pred_tracks, pred_visibility = actual_model(
                video=video_torch_resized,
                queries=queries_torch,
                iters=4,
                is_train=False,
                add_space_attn=add_support_grid,
                is_online=False  # We process the whole video at once, not in online mode
            )[:2]  # Get only tracks and visibility, ignore confidence and train_data

        # Scale tracks back to original video resolution
        H, W = video.shape[1:3]
        pred_tracks_scaled = pred_tracks * torch.tensor([W, H]).to(self.device) / torch.tensor([VIDEO_INPUT_RESO[1], VIDEO_INPUT_RESO[0]]).to(self.device)
        
        return pred_tracks_scaled[0].permute(1, 0, 2).detach().cpu().numpy(), pred_visibility[0].permute(1, 0).detach().cpu().numpy()

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
