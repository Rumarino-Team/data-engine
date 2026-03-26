from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import gc
import logging
import co_tracker as cot
import sam2_video_masker as svm
from utils import *
from pathlib import Path
from datetime import datetime
import mediapy
from fastapi.responses import FileResponse
import os
import torch
from urllib.parse import unquote, urlparse

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
DEFAULT_MAX_MASK_FRAMES_IN_RESPONSE = int(os.getenv("VIDEO_PROPAGATE_MAX_MASK_FRAMES", "0"))
DEFAULT_MAX_MASK_VALUES_IN_RESPONSE = int(os.getenv("VIDEO_PROPAGATE_MAX_MASK_VALUES", "0"))

logger = logging.getLogger(__name__)


def _cleanup_cuda_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _normalize_input_path(path_value: str) -> str:
    normalized = str(path_value).strip().strip('"').strip("'")
    if not normalized:
        return normalized

    if normalized.startswith("file://"):
        parsed = urlparse(normalized)
        if parsed.scheme == "file":
            normalized = parsed.path or ""
            if parsed.netloc and parsed.netloc != "localhost":
                normalized = f"//{parsed.netloc}{normalized}"

            if os.name == "nt" and normalized.startswith("/") and len(normalized) > 2 and normalized[2] == ":":
                normalized = normalized[1:]

    normalized = unquote(normalized)
    normalized = normalized.replace("\\ ", " ")
    return normalized


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


def _prepare_video_masker_for_video_init():
    global video_masker, tracker, tracking_video, tracking_video_path, video_dir

    if tracker is not None:
        del tracker
        tracker = None
        tracking_video = None
        tracking_video_path = None
        _cleanup_cuda_memory()

    if video_masker is None:
        video_masker = svm.SAM2VideoMasker()


def _initialize_video_state_from_resolved_input(
    resolved_input_path: Path,
    *,
    online_mode: bool,
    batch_size: Optional[int],
    offload_video_to_cpu: Optional[bool],
    offload_state_to_cpu: Optional[bool],
    async_loading_frames: bool,
):
    global video_masker, video_dir, video_frame_files

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
    try:
        video_masker.init_state(
            video_dir,
            online_mode=online_mode,
            batch_size=batch_size,
            offload_video_to_cpu=offload_video_to_cpu,
            offload_state_to_cpu=offload_state_to_cpu,
            async_loading_frames=async_loading_frames,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

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
        "source_video_path": source_video_path,
        "online_mode": video_masker.online_mode,
        "batch_size": video_masker.default_batch_size,
        "offload_video_to_cpu": video_masker.offload_video_to_cpu,
        "offload_state_to_cpu": video_masker.offload_state_to_cpu,
    }


def _serialize_video_segments_for_response(
    video_segments: dict,
    *,
    max_frames: int,
    max_mask_values: int,
) -> tuple[dict, bool, int, int]:
    """Serialize masks to JSON-safe payload with optional size limits."""
    serialized: dict[int, dict[int, list]] = {}
    total_mask_values = 0
    returned_frames = 0

    for frame_idx, obj_dict in sorted(video_segments.items(), key=lambda item: int(item[0])):
        if max_frames >= 0 and returned_frames >= max_frames:
            break

        frame_masks: dict[int, np.ndarray] = {}
        frame_mask_values = 0
        for obj_id, mask in obj_dict.items():
            mask_array = np.asarray(mask)
            frame_mask_values += int(mask_array.size)
            frame_masks[int(obj_id)] = mask_array

        if max_mask_values >= 0 and (total_mask_values + frame_mask_values) > max_mask_values:
            break

        serialized[int(frame_idx)] = {
            obj_id: mask_array.tolist()
            for obj_id, mask_array in frame_masks.items()
        }
        total_mask_values += frame_mask_values
        returned_frames += 1

    truncated = returned_frames < len(video_segments)
    return serialized, truncated, returned_frames, total_mask_values


class VideoInitStateRequest(BaseModel):
    video_frames_dir: str
    online_mode: bool = True
    batch_size: Optional[int] = None
    offload_video_to_cpu: Optional[bool] = None
    offload_state_to_cpu: Optional[bool] = None
    async_loading_frames: bool = False

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
    batch_size: Optional[int] = None
    online_mode: Optional[bool] = None
    include_masks_in_response: bool = False
    include_saved_mask_paths: bool = False
    max_frames_in_response: Optional[int] = None
    max_mask_values_in_response: Optional[int] = None


class VideoAddMaskRequest(BaseModel):
    frame_idx: int
    obj_id: int
    mask: list[list[bool]]  # 2D boolean mask


class TrackingLoadVideoRequest(BaseModel):
    video_path: str
    model_name: str = "cotracker3_offline"  # "cotracker3_offline" or "cotracker3_online"


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
    _prepare_video_masker_for_video_init()
    resolved_input_path = _resolve_input_path(request.video_frames_dir)
    return _initialize_video_state_from_resolved_input(
        resolved_input_path,
        online_mode=request.online_mode,
        batch_size=request.batch_size,
        offload_video_to_cpu=request.offload_video_to_cpu,
        offload_state_to_cpu=request.offload_state_to_cpu,
        async_loading_frames=request.async_loading_frames,
    )

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
    
    try:
        frame_files, masks_dir = prepare_video_masks_output(video_dir)
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Failed to prepare mask output directory: {error}") from error

    saved_mask_paths_serializable: dict[int, list[str]] = {}
    saved_mask_frame_count = 0
    save_failures = 0

    def _on_propagated_frame(out_frame_idx: int, frame_masks: dict[int, np.ndarray]):
        nonlocal saved_mask_frame_count, save_failures
        try:
            saved_path = save_single_video_mask_frame(frame_files, masks_dir, int(out_frame_idx), frame_masks)
            if saved_path is None:
                return
            saved_mask_frame_count += 1
            if request.include_saved_mask_paths:
                saved_mask_paths_serializable.setdefault(int(out_frame_idx), []).append(saved_path)
        except Exception:
            save_failures += 1
            logger.exception("Failed to save propagated mask frame %s", out_frame_idx)

    try:
        video_segments = video_masker.propagate_in_video(
            start_frame_idx=request.start_frame_idx,
            max_frame_num_to_track=request.max_frame_num_to_track,
            reverse=request.reverse,
            batch_size=request.batch_size,
            online_mode=request.online_mode,
            collect_segments=request.include_masks_in_response,
            frame_callback=_on_propagated_frame,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except torch.OutOfMemoryError as error:
        _cleanup_cuda_memory()
        raise HTTPException(
            status_code=507,
            detail="CUDA out of memory during propagation. Try lowering batch_size or enabling CPU offload.",
        ) from error
    except RuntimeError as error:
        if "out of memory" in str(error).lower():
            _cleanup_cuda_memory()
            raise HTTPException(
                status_code=507,
                detail="CUDA out of memory during propagation. Try lowering batch_size or enabling CPU offload.",
            ) from error
        raise HTTPException(status_code=500, detail=f"Failed to propagate video masks: {error}") from error
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Failed to propagate video masks: {error}") from error

    max_frames_in_response = request.max_frames_in_response
    if max_frames_in_response is None:
        max_frames_in_response = DEFAULT_MAX_MASK_FRAMES_IN_RESPONSE

    max_mask_values_in_response = request.max_mask_values_in_response
    if max_mask_values_in_response is None:
        max_mask_values_in_response = DEFAULT_MAX_MASK_VALUES_IN_RESPONSE

    video_segments_serializable: dict[int, dict[int, list]] = {}
    video_segments_truncated = False
    video_segments_returned_frames = 0
    video_segments_returned_mask_values = 0

    if request.include_masks_in_response:
        try:
            video_segments_serializable, video_segments_truncated, video_segments_returned_frames, video_segments_returned_mask_values = _serialize_video_segments_for_response(
                video_segments,
                max_frames=max_frames_in_response,
                max_mask_values=max_mask_values_in_response,
            )
        except MemoryError:
            logger.warning("Mask serialization skipped due to memory pressure.")
            video_segments_serializable = {}
            video_segments_truncated = len(video_segments) > 0
        except Exception:
            logger.exception("Mask serialization failed; returning saved mask paths only.")
            video_segments_serializable = {}
            video_segments_truncated = len(video_segments) > 0

    return {
        "video_segments": video_segments_serializable,
        "video_segments_total_frames": len(video_segments),
        "video_segments_returned_frames": video_segments_returned_frames,
        "video_segments_returned_mask_values": video_segments_returned_mask_values,
        "video_segments_truncated": video_segments_truncated,
        "saved_mask_frame_count": saved_mask_frame_count,
        "saved_mask_save_failures": save_failures,
        "saved_mask_paths": saved_mask_paths_serializable,
        "online_mode": video_masker.online_mode if request.online_mode is None else bool(request.online_mode),
        "batch_size": video_masker.default_batch_size if request.batch_size is None else int(request.batch_size),
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
        _cleanup_cuda_memory()
    
    # (Re-)initialise tracker with the requested model variant.
    # A new tracker is created if the model_name changed or no tracker exists.
    requested_model = getattr(request, "model_name", "cotracker3_offline")
    if tracker is None or getattr(tracker, "model_name", None) != requested_model:
        if tracker is not None:
            del tracker
            _cleanup_cuda_memory()
        tracker = cot.CoTracker(model_name=requested_model)
    
    resolved_video_path = _resolve_input_path(request.video_path, expect_dir=False)
    tracking_video_path = str(resolved_video_path)
    
    # Load video using mediapy
    tracking_video = mediapy.read_video(tracking_video_path)
    
    return {
        "message": "Video loaded successfully",
        "model_name": tracker.model_name,
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
    try:
        tracks, visibility = tracker.track(
            tracking_video,
            queries=None,
            grid_size=request.grid_size,
            add_support_grid=request.add_support_grid
        )
    except torch.OutOfMemoryError as error:
        _cleanup_cuda_memory()
        raise HTTPException(
            status_code=507,
            detail="CUDA out of memory during tracking. Try a smaller tracking workload and rerun.",
        ) from error
    except RuntimeError as error:
        if "out of memory" in str(error).lower():
            _cleanup_cuda_memory()
            raise HTTPException(
                status_code=507,
                detail="CUDA out of memory during tracking. Try a smaller tracking workload and rerun.",
            ) from error
        raise
    
    # Save visualization
    painted_video = cot.paint_point_track(tracking_video, tracks, visibility)
    
    # Save output video
    video_name = Path(tracking_video_path).stem
    output_dir = Path(tracking_video_path).parent
    timestamp = int(datetime.now().timestamp())
    mode_label = "online" if tracker.is_online else "offline"
    output_filename = f"{video_name}_tracked_grid_{mode_label}_{timestamp}.mp4"
    output_path = output_dir / output_filename
    
    fps = 30  # Default fps
    try:
        video_metadata = mediapy.read_video(tracking_video_path)
        if hasattr(video_metadata, 'metadata') and video_metadata.metadata and hasattr(video_metadata.metadata, 'fps'):
            fps = video_metadata.metadata.fps
    except:
        pass
    
    mediapy.write_video(str(output_path), painted_video, fps=fps)

    _cleanup_cuda_memory()
    
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
    try:
        tracks, visibility = tracker.track(
            tracking_video,
            queries=queries,
            add_support_grid=request.add_support_grid
        )
    except torch.OutOfMemoryError as error:
        _cleanup_cuda_memory()
        raise HTTPException(
            status_code=507,
            detail="CUDA out of memory during tracking. Try a smaller tracking workload and rerun.",
        ) from error
    except RuntimeError as error:
        if "out of memory" in str(error).lower():
            _cleanup_cuda_memory()
            raise HTTPException(
                status_code=507,
                detail="CUDA out of memory during tracking. Try a smaller tracking workload and rerun.",
            ) from error
        raise
    
    # Save visualization
    painted_video = cot.paint_point_track(tracking_video, tracks, visibility)
    
    # Save output video
    video_name = Path(tracking_video_path).stem
    output_dir = Path(tracking_video_path).parent
    timestamp = int(datetime.now().timestamp())
    mode_label = "online" if tracker.is_online else "offline"
    output_tag = "tracked_points_support" if request.add_support_grid else "tracked_points"
    output_filename = f"{video_name}_{output_tag}_{mode_label}_{timestamp}.mp4"
    output_path = output_dir / output_filename
    
    fps = 30  # Default fps
    try:
        video_metadata = mediapy.read_video(tracking_video_path)
        if hasattr(video_metadata, 'metadata') and video_metadata.metadata and hasattr(video_metadata.metadata, 'fps'):
            fps = video_metadata.metadata.fps
    except:
        pass
    
    mediapy.write_video(str(output_path), painted_video, fps=fps)

    _cleanup_cuda_memory()
    
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


@app.get("/video/mask_frame/{frame_idx}")
async def get_video_mask_frame(frame_idx: int):
    global video_dir
    if video_dir is None:
        return {"error": "Video not initialized"}

    if frame_idx < 0:
        return {"error": "Frame index out of bounds"}

    file_path = Path(video_dir) / "masks" / f"frame_{frame_idx:05d}_masks.png"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Mask frame not found: {file_path}")
    return FileResponse(str(file_path))
