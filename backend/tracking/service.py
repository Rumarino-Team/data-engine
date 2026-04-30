from datetime import datetime
from pathlib import Path
import mediapy
import numpy as np
from fastapi import HTTPException
import torch
import co_tracker as cot
from core.config import DEFAULT_STREAMING_TRACK_FRAME_THRESHOLD
from core.runtime import cleanup_cuda_memory
from core.state import state
from schemas.tracking import TrackingGridRequest, TrackingLoadVideoRequest, TrackingPointsRequest
from sessions.cache import bump_video_state_epoch, release_active_session
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

async def load_tracking_video(request: TrackingLoadVideoRequest):
    """Load a video file for tracking."""

    release_active_session(clear_tracker=False, clear_cache_session=True)
    bump_video_state_epoch()
    
    requested_model = getattr(request, "model_name", cot.DEFAULT_COTRACKER_MODEL)
    try:
        ensure_tracker_model(requested_model)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    
    resolved_video_path = resolve_input_path(request.video_path, expect_dir=False)
    state.tracking_video_path = str(resolved_video_path)
    
    # Load video using mediapy
    state.tracking_video = mediapy.read_video(state.tracking_video_path)
    
    return {
        "message": "Video loaded successfully",
        "model_name": state.tracker.model_name,
        "shape": state.tracking_video.shape,
        "num_frames": state.tracking_video.shape[0],
        "resolved_video_path": state.tracking_video_path
    }

async def track_grid(request: TrackingGridRequest):
    """Track a grid of points across the video."""
    
    if state.tracker is None:
        return {"error": "Tracker not active. Call /tracking/load_video first."}
    
    if state.tracking_video is None:
        return {"error": "No video loaded. Call /tracking/load_video first."}
    
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
        raise HTTPException(
            status_code=507,
            detail="CUDA out of memory during tracking. Try a smaller tracking workload and rerun.",
        ) from error
    except RuntimeError as error:
        if "out of memory" in str(error).lower():
            cleanup_cuda_memory()
            raise HTTPException(
                status_code=507,
                detail="CUDA out of memory during tracking. Try a smaller tracking workload and rerun.",
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
    
    fps = 30  # Default fps
    try:
        video_metadata = mediapy.read_video(state.tracking_video_path)
        if hasattr(video_metadata, 'metadata') and video_metadata.metadata and hasattr(video_metadata.metadata, 'fps'):
            fps = video_metadata.metadata.fps
    except:
        pass
    
    mediapy.write_video(str(output_path), painted_video, fps=fps)

    cleanup_cuda_memory()
    
    return {
        "message": "Grid tracking completed",
        "tracks": tracks.tolist(),
        "visibility": visibility.tolist(),
        "num_points": tracks.shape[0],
        "num_frames": tracks.shape[1],
        "output_video_path": str(output_path)
    }

async def track_points(request: TrackingPointsRequest):
    """Track specific query points across the video."""
    
    if state.tracker is None:
        return {"error": "Tracker not active. Call /tracking/load_video first."}
    
    if state.tracking_video is None:
        return {"error": "No video loaded. Call /tracking/load_video first."}
    
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
        raise HTTPException(
            status_code=507,
            detail="CUDA out of memory during tracking. Try a smaller tracking workload and rerun.",
        ) from error
    except RuntimeError as error:
        if "out of memory" in str(error).lower():
            cleanup_cuda_memory()
            raise HTTPException(
                status_code=507,
                detail="CUDA out of memory during tracking. Try a smaller tracking workload and rerun.",
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
    
    fps = 30  # Default fps
    try:
        video_metadata = mediapy.read_video(state.tracking_video_path)
        if hasattr(video_metadata, 'metadata') and video_metadata.metadata and hasattr(video_metadata.metadata, 'fps'):
            fps = video_metadata.metadata.fps
    except:
        pass
    
    mediapy.write_video(str(output_path), painted_video, fps=fps)

    cleanup_cuda_memory()
    
    return {
        "message": "Point tracking completed",
        "tracks": tracks.tolist(),
        "visibility": visibility.tolist(),
        "num_points": tracks.shape[0],
        "num_frames": tracks.shape[1],
        "output_video_path": str(output_path)
    }

