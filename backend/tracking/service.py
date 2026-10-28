from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import co_tracker as cot
from core.config import DEFAULT_STREAMING_TRACK_FRAME_THRESHOLD
from core.errors import OperationError
from core.jobs import foreground_operation
from core.runtime import cleanup_cuda_memory
from core.state import BackendState, state
from schemas.tracking import TrackingGridRequest, TrackingLoadVideoRequest, TrackingPointsRequest
from sessions.paths import resolve_input_path
from video.io import load_video_frames_as_numpy

def ensure_tracker_model(model_name: str):
    if model_name != cot.DEFAULT_COTRACKER_MODEL:
        raise ValueError(f"Unsupported CoTracker model '{model_name}'. Use '{cot.DEFAULT_COTRACKER_MODEL}'.")
    if state.tracker is None or getattr(state.tracker, "model_name", None) != model_name:
        if state.tracker is not None:
            state.tracker = None
            cleanup_cuda_memory()
        state.tracker = cot.CoTracker(model_name=model_name)

def load_tracking_video_from_current_video_state() -> tuple[np.ndarray, str]:
    if state.video_dir is None or not state.video_frame_files:
        raise ValueError("Video is not initialized for tracking.")

    # Keep tracking frame indices aligned with the exact frame sequence used by masking/frontend.
    # Decoding directly from source_video_path can introduce index drift across different decoders.
    frame_video = load_video_frames_as_numpy(Path(state.video_dir), state.video_frame_files)
    return frame_video, str(state.video_dir)

def should_stream_tracking(num_frames: int) -> bool:
    threshold = int(DEFAULT_STREAMING_TRACK_FRAME_THRESHOLD)
    if threshold <= 0:
        return True
    return int(num_frames) >= threshold

def read_tracking_video(video_path: str | Path) -> np.ndarray:
    return cot.read_video_rgb(video_path)

def load_tracking_video(request: TrackingLoadVideoRequest):
    """Load a video file for tracking."""

    with foreground_operation():
        requested_model = getattr(request, "model_name", cot.DEFAULT_COTRACKER_MODEL)
        if requested_model != cot.DEFAULT_COTRACKER_MODEL:
            raise OperationError(400, f"Unsupported CoTracker model '{requested_model}'.")
        resolved_video_path = resolve_input_path(request.video_path, expect_dir=False)
        video = read_tracking_video(str(resolved_video_path))
        tracker = state.tracker
        if tracker is None or tracker.model_name != requested_model:
            tracker = cot.CoTracker(model_name=requested_model)
        candidate = BackendState(tracker=tracker, tracking_video=video,
                                 tracking_video_path=str(resolved_video_path),
                                 video_state_epoch=state.video_state_epoch + 1)
        response = {
            "message": "Video loaded successfully",
            "model_name": tracker.model_name,
            "shape": video.shape,
            "num_frames": video.shape[0],
            "resolved_video_path": str(resolved_video_path),
        }
        state.__dict__.update(candidate.__dict__)
        return response


def track_grid(request: TrackingGridRequest):
    """Track a grid of points across the video."""

    with foreground_operation():
        if state.tracker is None:
            raise OperationError(400, "Tracker not active. Call /tracking/load_video first.")

        if state.tracking_video is None:
            raise OperationError(400, "No video loaded. Call /tracking/load_video first.")

        # Run tracking
        try:
            tracks, visibility = state.tracker.track(
                state.tracking_video,
                queries=None,
                grid_size=request.grid_size,
                add_support_grid=request.add_support_grid
            )
        except torch.OutOfMemoryError as error:
            cleanup_cuda_memory()
            raise OperationError(
                507, "CUDA out of memory during tracking. Try a smaller tracking workload and rerun."
            ) from error
        except RuntimeError as error:
            if "out of memory" in str(error).lower():
                cleanup_cuda_memory()
                raise OperationError(
                    507, "CUDA out of memory during tracking. Try a smaller tracking workload and rerun."
                ) from error
            raise

        # Save visualization
        painted_video = cot.paint_point_track(state.tracking_video, tracks, visibility)

        # Save output video
        video_name = Path(state.tracking_video_path).stem
        output_dir = Path(state.tracking_video_path).parent
        timestamp = int(datetime.now().timestamp())
        mode_label = cot.DEFAULT_COTRACKER_MODEL.replace("cotracker3_", "")
        output_filename = f"{video_name}_tracked_grid_{mode_label}_{timestamp}.mp4"
        output_path = output_dir / output_filename

        fps = cot.read_video_fps(state.tracking_video_path)
        cot.write_video_rgb(output_path, painted_video, fps=fps)

        cleanup_cuda_memory()

        return {
            "message": "Grid tracking completed",
            "tracks": tracks.tolist(),
            "visibility": visibility.tolist(),
            "num_points": tracks.shape[0],
            "num_frames": tracks.shape[1],
            "output_video_path": str(output_path)
        }

def track_points(request: TrackingPointsRequest):
    """Track specific query points across the video."""

    with foreground_operation():
        if state.tracker is None:
            raise OperationError(400, "Tracker not active. Call /tracking/load_video first.")

        if state.tracking_video is None:
            raise OperationError(400, "No video loaded. Call /tracking/load_video first.")

        # Convert queries to numpy array
        queries = np.array(request.queries)

        # Run tracking
        try:
            tracks, visibility = state.tracker.track(
                state.tracking_video,
                queries=queries,
                add_support_grid=request.add_support_grid
            )
        except torch.OutOfMemoryError as error:
            cleanup_cuda_memory()
            raise OperationError(
                507, "CUDA out of memory during tracking. Try a smaller tracking workload and rerun."
            ) from error
        except RuntimeError as error:
            if "out of memory" in str(error).lower():
                cleanup_cuda_memory()
                raise OperationError(
                    507, "CUDA out of memory during tracking. Try a smaller tracking workload and rerun."
                ) from error
            raise

        # Save visualization
        painted_video = cot.paint_point_track(state.tracking_video, tracks, visibility)

        # Save output video
        video_name = Path(state.tracking_video_path).stem
        output_dir = Path(state.tracking_video_path).parent
        timestamp = int(datetime.now().timestamp())
        mode_label = cot.DEFAULT_COTRACKER_MODEL.replace("cotracker3_", "")
        output_tag = "tracked_points_support" if request.add_support_grid else "tracked_points"
        output_filename = f"{video_name}_{output_tag}_{mode_label}_{timestamp}.mp4"
        output_path = output_dir / output_filename

        fps = cot.read_video_fps(state.tracking_video_path)
        cot.write_video_rgb(output_path, painted_video, fps=fps)

        cleanup_cuda_memory()

        return {
            "message": "Point tracking completed",
            "tracks": tracks.tolist(),
            "visibility": visibility.tolist(),
            "num_points": tracks.shape[0],
            "num_frames": tracks.shape[1],
            "output_video_path": str(output_path)
        }
