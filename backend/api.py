from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import co_tracker as cot
import sam2_video_masker as svm
from utils import *
from pathlib import Path
from datetime import datetime
import mediapy
from fastapi.responses import FileResponse
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instances
video_masker: Optional[svm.SAM2VideoMasker] = None
tracker: Optional[cot.CoTracker] = None

# Video state
video_dir: Optional[str] = None
video_frame_files: list[str] = []
tracking_video: Optional[np.ndarray] = None
tracking_video_path: Optional[str] = None
PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
GENERATED_FRAMES_ROOT = PROJECT_ROOT / "backend/.data_engine_frames"


def _normalize_input_path(path_value: str) -> str:
    return path_value.strip().strip('"').strip("'")


def _resolve_input_path(path_value: str, expect_dir: Optional[bool] = None) -> Path:
    normalized = _normalize_input_path(path_value)
    if not normalized:
        raise HTTPException(status_code=400, detail="Path cannot be empty.")

    expanded = Path(os.path.expandvars(os.path.expanduser(normalized)))

    if expanded.is_absolute():
        candidate_paths = [expanded.resolve()]
    else:
        cwd_candidate = (Path.cwd() / expanded).resolve()
        project_candidate = (PROJECT_ROOT / expanded).resolve()
        candidate_paths = [cwd_candidate]
        if project_candidate != cwd_candidate:
            candidate_paths.append(project_candidate)

    resolved_path = next((path for path in candidate_paths if path.exists()), candidate_paths[0])

    if not resolved_path.exists():
        tried_paths = ", ".join(str(path) for path in candidate_paths)
        raise HTTPException(
            status_code=404,
            detail=f"Path not found: '{normalized}'. Tried: {tried_paths}"
        )

    if expect_dir is True and not resolved_path.is_dir():
        raise HTTPException(status_code=400, detail=f"Expected a directory path, got file: {resolved_path}")

    if expect_dir is False and not resolved_path.is_file():
        raise HTTPException(status_code=400, detail=f"Expected a file path, got directory: {resolved_path}")

    return resolved_path


class VideoInitStateRequest(BaseModel):
    video_frames_dir: str

class VideoAddPointsOrBoxRequest(BaseModel):
    frame_idx: int
    obj_id: int
    points: Optional[list[list[float]]] = None
    labels: Optional[list[int]] = None
    clear_old_points: bool = True
    box: Optional[list[float]] = None

class VideoPropagateRequest(BaseModel):
    start_frame_idx: Optional[int] = None
    max_frame_num_to_track: Optional[int] = None
    reverse: bool = False


class VideoAddMaskRequest(BaseModel):
    frame_idx: int
    obj_id: int
    mask: list[list[bool]]  # 2D boolean mask


class TrackingLoadVideoRequest(BaseModel):
    video_path: str


class TrackingGridRequest(BaseModel):
    grid_size: int = 15
    add_support_grid: bool = True


class TrackingPointsRequest(BaseModel):
    queries: list[list[float]]  # List of [t, x, y] coordinates
    add_support_grid: bool = True


@app.get("/")
async def root():
    return {"message": "Data Engine Backend"}


@app.get("/health")
async def health():
    """Simple health endpoint used by the C++ client to detect readiness."""
    return {"status": "ok"}


@app.get("/status")
async def status():
    global video_masker, tracker
    
    if video_masker is not None:
        return {"status": "video active", "device": video_masker.device.type}
    elif tracker is not None:
        return {"status": "tracking active", "device": tracker.device}
    else:
        return {"status": "inactive", "device": None}


@app.post("/video/init_state")
async def init_video_state(request: VideoInitStateRequest):
    global video_masker, video_dir, tracker, tracking_video, tracking_video_path, video_frame_files
    
    # Unload tracker if it's currently loaded
    if tracker is not None:
        del tracker
        tracker = None
        tracking_video = None
        tracking_video_path = None
    
    # Initialize video masker if not already created
    if video_masker is None:
        video_masker = svm.SAM2VideoMasker()
    
    resolved_input_path = _resolve_input_path(request.video_frames_dir)

    source_video_path = None
    if resolved_input_path.is_file():
        suffix = resolved_input_path.suffix.lower()
        if suffix in VIDEO_EXTENSIONS:
            try:
                resolved_video_dir = extract_video_to_frames(
                    resolved_input_path,
                    output_root=GENERATED_FRAMES_ROOT,
                    image_extensions=IMAGE_EXTENSIONS,
                )
            except ValueError as error:
                raise HTTPException(status_code=400, detail=str(error)) from error
            except Exception as error:
                raise HTTPException(status_code=500, detail=str(error)) from error
            source_video_path = str(resolved_input_path)
        else:
            if suffix in IMAGE_EXTENSIONS:
                detail = (
                    f"Expected a frames directory or video file, got a single image file: {resolved_input_path}. "
                    "Provide a directory containing image frames."
                )
            else:
                detail = (
                    f"Unsupported input file type: {resolved_input_path.suffix or '<none>'}. "
                    "Provide a directory of image frames or a video file (.mp4, .mov, .avi, .mkv, .webm, .m4v)."
                )
            raise HTTPException(status_code=400, detail=detail)
    else:
        resolved_video_dir = resolved_input_path

    video_dir = str(resolved_video_dir)
    video_masker.init_state(video_dir)
    
    # Scan for image files
    video_frame_files = sorted([
        frame_path.name
        for frame_path in resolved_video_dir.iterdir()
        if frame_path.is_file() and frame_path.suffix.lower() in IMAGE_EXTENSIONS
    ])

    if not video_frame_files:
        raise HTTPException(
            status_code=400,
            detail=f"No image frames found in directory: {resolved_video_dir}"
        )
    
    return {
        "message": "Video state initialized successfully",
        "num_frames": len(video_frame_files),
        "resolved_video_frames_dir": video_dir,
        "source_video_path": source_video_path
    }

@app.post("/video/reset_state")
async def reset_video_state():
    global video_masker
    if video_masker is None:
        return {"error": "Video masker not active."}
    video_masker.reset_state()
    return {"message": "Video state reset successfully"}

@app.post("/video/add_new_points_or_box")
async def add_new_points_or_box(request: VideoAddPointsOrBoxRequest):
    global video_masker
    if video_masker is None:
        return {"error": "Video masker not active."}
    out_obj_ids, out_mask_logits = video_masker.add_new_points_or_box(
        frame_idx=request.frame_idx,
        obj_id=request.obj_id,
        points=request.points,
        labels=request.labels,
        clear_old_points=request.clear_old_points,
        box=request.box
    )
    masks_list = [(out_mask_logits[i] > 0.0).squeeze(0).cpu().numpy().tolist() for i in range(len(out_obj_ids))]
    return {
        "out_obj_ids": out_obj_ids,
        "out_masks": masks_list
    }

@app.post("/video/add_new_mask")
async def add_new_mask(request: VideoAddMaskRequest):
    global video_masker
    if video_masker is None:
        return {"error": "Video masker not active."}
    
    # Convert mask from list to numpy array
    mask = np.array(request.mask, dtype=bool)
    
    frame_idx, out_obj_ids, out_mask_logits = video_masker.add_new_mask(
        frame_idx=request.frame_idx,
        obj_id=request.obj_id,
        mask=mask
    )
    
    masks_list = [(out_mask_logits[i] > 0.0).squeeze(0).cpu().numpy().tolist() for i in range(len(out_obj_ids))]
    return {
        "frame_idx": frame_idx,
        "out_obj_ids": out_obj_ids,
        "out_masks": masks_list
    }

@app.post("/video/propagate_in_video")
async def propagate_in_video(request: VideoPropagateRequest):
    global video_masker, video_dir
    if video_masker is None:
        return {"error": "Video masker not active."}
    if video_dir is None:
        return {"error": "Video directory not set. Call /video/init_state first."}
    
    video_segments = video_masker.propagate_in_video(
        start_frame_idx=request.start_frame_idx,
        max_frame_num_to_track=request.max_frame_num_to_track,
        reverse=request.reverse
    )
    
    # Save masks for each frame
    saved_mask_paths = save_video_masks(video_dir, video_segments)
    
    # Convert masks to lists for JSON serialization
    video_segments_serializable = {}
    for frame_idx, obj_dict in video_segments.items():
        video_segments_serializable[frame_idx] = {
            obj_id: mask.tolist() for obj_id, mask in obj_dict.items()
        }
    return {
        "video_segments": video_segments_serializable,
        "saved_mask_paths": saved_mask_paths
    }

@app.post("/video/clear_all_prompts_in_frame")
async def clear_all_prompts_in_frame(frame_idx: int, obj_id: int):
    global video_masker
    if video_masker is None:
        return {"error": "Video masker not active."}
    video_masker.clear_all_prompts_in_frame(frame_idx, obj_id)
    return {"message": "Cleared all prompts in frame successfully"}

@app.post("/video/remove_object")
async def remove_object(obj_id: int):
    global video_masker
    if video_masker is None:
        return {"error": "Video masker not active."}
    video_masker.remove_object(obj_id)
    return {"message": "Object removed successfully"}


@app.post("/tracking/load_video")
async def load_tracking_video(request: TrackingLoadVideoRequest):
    """Load a video file for tracking."""
    global tracker, tracking_video, tracking_video_path, video_masker, video_dir
    
    # Unload video masker if it's currently loaded
    if video_masker is not None:
        del video_masker
        video_masker = None
        video_dir = None
    
    # Initialize tracker if not already created
    if tracker is None:
        tracker = cot.CoTracker()
    
    resolved_video_path = _resolve_input_path(request.video_path, expect_dir=False)
    tracking_video_path = str(resolved_video_path)
    
    # Load video using mediapy
    tracking_video = mediapy.read_video(tracking_video_path)
    
    return {
        "message": "Video loaded successfully",
        "shape": tracking_video.shape,
        "num_frames": tracking_video.shape[0],
        "resolved_video_path": tracking_video_path
    }


@app.post("/tracking/track_grid")
async def track_grid(request: TrackingGridRequest):
    """Track a grid of points across the video."""
    global tracker, tracking_video, tracking_video_path
    
    if tracker is None:
        return {"error": "Tracker not active. Call /tracking/load_video first."}
    
    if tracking_video is None:
        return {"error": "No video loaded. Call /tracking/load_video first."}
    
    # Run tracking
    tracks, visibility = tracker.track(
        tracking_video, 
        queries=None, 
        grid_size=request.grid_size,
        add_support_grid=request.add_support_grid
    )
    
    # Save visualization
    painted_video = cot.paint_point_track(tracking_video, tracks, visibility)
    
    # Save output video
    video_name = Path(tracking_video_path).stem
    output_dir = Path(tracking_video_path).parent
    timestamp = int(datetime.now().timestamp())
    output_filename = f"{video_name}_tracked_grid_{timestamp}.mp4"
    output_path = output_dir / output_filename
    
    fps = 30  # Default fps
    try:
        video_metadata = mediapy.read_video(tracking_video_path)
        if hasattr(video_metadata, 'metadata') and video_metadata.metadata and hasattr(video_metadata.metadata, 'fps'):
            fps = video_metadata.metadata.fps
    except:
        pass
    
    mediapy.write_video(str(output_path), painted_video, fps=fps)
    
    return {
        "message": "Grid tracking completed",
        "tracks": tracks.tolist(),
        "visibility": visibility.tolist(),
        "num_points": tracks.shape[0],
        "num_frames": tracks.shape[1],
        "output_video_path": str(output_path)
    }


@app.post("/tracking/track_points")
async def track_points(request: TrackingPointsRequest):
    """Track specific query points across the video."""
    global tracker, tracking_video, tracking_video_path
    
    if tracker is None:
        return {"error": "Tracker not active. Call /tracking/load_video first."}
    
    if tracking_video is None:
        return {"error": "No video loaded. Call /tracking/load_video first."}
    
    # Convert queries to numpy array
    queries = np.array(request.queries)
    
    # Run tracking
    tracks, visibility = tracker.track(
        tracking_video,
        queries=queries,
        add_support_grid=request.add_support_grid
    )
    
    # Save visualization
    painted_video = cot.paint_point_track(tracking_video, tracks, visibility)
    
    # Save output video
    video_name = Path(tracking_video_path).stem
    output_dir = Path(tracking_video_path).parent
    timestamp = int(datetime.now().timestamp())
    output_filename = f"{video_name}_tracked_points_{timestamp}.mp4"
    output_path = output_dir / output_filename
    
    fps = 30  # Default fps
    try:
        video_metadata = mediapy.read_video(tracking_video_path)
        if hasattr(video_metadata, 'metadata') and video_metadata.metadata and hasattr(video_metadata.metadata, 'fps'):
            fps = video_metadata.metadata.fps
    except:
        pass
    
    mediapy.write_video(str(output_path), painted_video, fps=fps)
    
    return {
        "message": "Point tracking completed",
        "tracks": tracks.tolist(),
        "visibility": visibility.tolist(),
        "num_points": tracks.shape[0],
        "num_frames": tracks.shape[1],
        "output_video_path": str(output_path)
    }

@app.get("/video/info")
async def get_video_info():
    global video_frame_files
    return {
        "num_frames": len(video_frame_files),
        "frame_files": video_frame_files
    }

@app.get("/video/frame/{frame_idx}")
async def get_video_frame(frame_idx: int):
    global video_dir, video_frame_files
    if video_dir is None or not video_frame_files:
        return {"error": "Video not initialized"}
    
    if frame_idx < 0 or frame_idx >= len(video_frame_files):
        return {"error": "Frame index out of bounds"}
        
    file_path = Path(video_dir) / video_frame_files[frame_idx]
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Frame file not found: {file_path}")
    return FileResponse(str(file_path))
