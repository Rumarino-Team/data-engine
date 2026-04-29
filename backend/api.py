from typing import Optional, Any, Callable
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
import cv2
import shutil
import uuid
import threading
import re

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
video_source_path: Optional[str] = None
video_prompt_events: list[dict[str, Any]] = []
mask_manifest_path: Optional[str] = None
video_state_epoch: int = 0
active_session_dir: Optional[Path] = None
active_session_id: Optional[str] = None
active_session_saved_name: Optional[str] = None
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_ROOT = PROJECT_ROOT / "backend"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
CACHE_ROOT = BACKEND_ROOT / "cache"
GENERATED_FRAMES_ROOT = CACHE_ROOT / "frames"
WINDOW_FRAMES_ROOT = CACHE_ROOT / "windows"
SESSION_CACHE_ROOT = CACHE_ROOT / "sessions"
SAVED_ROOT = BACKEND_ROOT / "saved"
DEFAULT_MAX_MASK_FRAMES_IN_RESPONSE = int(os.getenv("VIDEO_PROPAGATE_MAX_MASK_FRAMES", "0"))
DEFAULT_MAX_MASK_VALUES_IN_RESPONSE = int(os.getenv("VIDEO_PROPAGATE_MAX_MASK_VALUES", "0"))
DEFAULT_PROMPT_TRACK_BATCH_SIZE = int(os.getenv("TRACK_PROMPT_BATCH_SIZE", "32"))

logger = logging.getLogger(__name__)
current_job: Optional[dict[str, Any]] = None
current_job_lock = threading.Lock()
_UNSET = object()


def _utc_now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _serialize_job(job: Optional[dict[str, Any]] = None) -> Optional[dict[str, Any]]:
    with current_job_lock:
        source = current_job if job is None else job
        return dict(source) if source is not None else None


def _active_job_exists() -> bool:
    return current_job is not None and current_job.get("status") in {"queued", "running"}


def _start_job(operation: str, *, stage: str, stage_label: str, message: str) -> dict[str, Any]:
    global current_job
    with current_job_lock:
        if _active_job_exists():
            raise HTTPException(status_code=409, detail="Another operation is already running.")

        now = _utc_now_iso()
        current_job = {
            "job_id": uuid.uuid4().hex,
            "operation": operation,
            "status": "queued",
            "stage": stage,
            "stage_label": stage_label,
            "progress": 0.0,
            "current": None,
            "total": None,
            "window_index": None,
            "window_count": None,
            "frame_idx": None,
            "stage_history": [],
            "message": message,
            "result": None,
            "error": None,
            "started_at": now,
            "updated_at": now,
            "completed_at": None,
        }
        return dict(current_job)


def _update_job(
    *,
    status: Optional[str] = None,
    stage: Optional[str] = None,
    stage_label: Optional[str] = None,
    progress: Any = _UNSET,
    current: Any = _UNSET,
    total: Any = _UNSET,
    window_index: Any = _UNSET,
    window_count: Any = _UNSET,
    frame_idx: Any = _UNSET,
    message: Optional[str] = None,
    append_history: bool = True,
) -> None:
    with current_job_lock:
        if current_job is None:
            return
        if status is not None:
            current_job["status"] = status
        if stage is not None:
            current_job["stage"] = stage
        if stage_label is not None:
            current_job["stage_label"] = stage_label
        if progress is not _UNSET:
            current_job["progress"] = None if progress is None else min(max(float(progress), 0.0), 1.0)
        if current is not _UNSET:
            current_job["current"] = None if current is None else int(current)
        if total is not _UNSET:
            current_job["total"] = None if total is None else int(total)
        if window_index is not _UNSET:
            current_job["window_index"] = None if window_index is None else int(window_index)
        if window_count is not _UNSET:
            current_job["window_count"] = None if window_count is None else int(window_count)
        if frame_idx is not _UNSET:
            current_job["frame_idx"] = None if frame_idx is None else int(frame_idx)
        if message is not None:
            current_job["message"] = message
        now = _utc_now_iso()
        current_job["updated_at"] = now
        if append_history and (stage is not None or stage_label is not None or message is not None):
            history = current_job.setdefault("stage_history", [])
            history.append(
                {
                    "stage": current_job.get("stage"),
                    "stage_label": current_job.get("stage_label"),
                    "message": current_job.get("message"),
                    "progress": current_job.get("progress"),
                    "updated_at": now,
                }
            )
            del history[:-8]


def _complete_job(result: dict[str, Any]) -> None:
    with current_job_lock:
        if current_job is None:
            return
        now = _utc_now_iso()
        current_job.update(
            {
                "status": "completed",
                "stage": "completed",
                "stage_label": "Completed",
                "progress": 1.0,
                "current": current_job.get("total") or current_job.get("current"),
                "window_index": None,
                "window_count": None,
                "frame_idx": None,
                "message": "Operation completed",
                "result": result,
                "error": None,
                "updated_at": now,
                "completed_at": now,
            }
        )


def _fail_job(error_code: str, message: str, detail: Optional[str] = None) -> None:
    with current_job_lock:
        if current_job is None:
            return
        now = _utc_now_iso()
        current_job.update(
            {
                "status": "failed",
                "error": {"code": error_code, "message": message, "detail": detail},
                "message": message,
                "updated_at": now,
                "completed_at": now,
            }
        )


def _job_error_from_exception(error: Exception) -> tuple[str, str, Optional[str]]:
    if isinstance(error, HTTPException):
        message = str(error.detail)
        if error.status_code == 507:
            return "cuda_out_of_memory", message, None
        if error.status_code in {400, 404, 409}:
            return "validation_error", message, None
        return "backend_error", message, None
    if isinstance(error, torch.OutOfMemoryError):
        return "cuda_out_of_memory", "CUDA out of memory during operation.", str(error)
    if isinstance(error, RuntimeError) and "out of memory" in str(error).lower():
        return "cuda_out_of_memory", "CUDA out of memory during operation.", str(error)
    return "backend_error", "Backend operation failed.", str(error)


def _run_job(job_id: str, worker: Callable[[], dict[str, Any]]) -> None:
    with current_job_lock:
        if current_job is None or current_job.get("job_id") != job_id:
            return
        current_job["status"] = "running"
        current_job["updated_at"] = _utc_now_iso()

    try:
        result = worker()
        _complete_job(result)
    except Exception as error:
        logger.exception("Background job failed")
        error_code, message, detail = _job_error_from_exception(error)
        _fail_job(error_code, message, detail)


def _queue_long_job(
    *,
    operation: str,
    stage: str,
    stage_label: str,
    message: str,
    worker: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    job = _start_job(operation, stage=stage, stage_label=stage_label, message=message)
    thread = threading.Thread(target=_run_job, args=(job["job_id"], worker), daemon=True)
    thread.start()
    return {
        "job_id": job["job_id"],
        "status": job["status"],
        "operation": job["operation"],
        "message": message,
    }


def _cleanup_cuda_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _path_is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _current_session_path() -> Optional[Path]:
    return active_session_dir.resolve() if active_session_dir is not None else None


def _current_frames_dir() -> Optional[Path]:
    session_path = _current_session_path()
    return session_path / "frames" if session_path is not None else None


def _current_masks_dir() -> Optional[Path]:
    session_path = _current_session_path()
    return session_path / "masks" if session_path is not None else None


def _write_session_metadata(extra: Optional[dict[str, Any]] = None) -> None:
    session_path = _current_session_path()
    if session_path is None:
        return
    metadata = {
        "session_id": active_session_id,
        "saved_name": active_session_saved_name,
        "source_video_path": video_source_path,
        "resolved_video_frames_dir": video_dir,
        "mask_manifest_path": mask_manifest_path,
        "num_frames": len(video_frame_files),
        "state_epoch": int(video_state_epoch),
        "updated_at": _utc_now_iso(),
    }
    if video_masker is not None:
        metadata.update(
            {
                "online_mode": video_masker.online_mode,
                "batch_size": video_masker.default_batch_size,
                "offload_video_to_cpu": video_masker.offload_video_to_cpu,
                "offload_state_to_cpu": video_masker.offload_state_to_cpu,
            }
        )
    if extra:
        metadata.update(extra)
    write_mask_manifest(session_path / "session.json", metadata)


def _clear_active_cache_session() -> None:
    session_path = _current_session_path()
    if session_path is None:
        return
    if _path_is_relative_to(session_path, SESSION_CACHE_ROOT):
        shutil.rmtree(session_path, ignore_errors=True)


def _clear_window_cache() -> None:
    shutil.rmtree(WINDOW_FRAMES_ROOT, ignore_errors=True)
    WINDOW_FRAMES_ROOT.mkdir(parents=True, exist_ok=True)


def _release_active_session(
    *,
    clear_video_masker: bool = True,
    clear_tracker: bool = True,
    clear_tracking_video: bool = True,
    clear_video_state: bool = True,
    clear_prompts: bool = True,
    clear_cache_session: bool = False,
) -> None:
    global video_masker, tracker, tracking_video, tracking_video_path
    global video_dir, video_frame_files, video_source_path, video_prompt_events
    global mask_manifest_path, active_session_dir, active_session_id, active_session_saved_name

    if clear_video_masker and video_masker is not None:
        del video_masker
        video_masker = None
    if clear_tracker and tracker is not None:
        del tracker
        tracker = None
    if clear_tracking_video:
        tracking_video = None
        tracking_video_path = None
    if clear_cache_session:
        _clear_active_cache_session()
    if clear_video_state:
        video_dir = None
        video_frame_files = []
        video_source_path = None
        mask_manifest_path = None
        active_session_dir = None
        active_session_id = None
        active_session_saved_name = None
    if clear_prompts:
        video_prompt_events = []
    _cleanup_cuda_memory()


@app.on_event("shutdown")
def _cleanup_backend_cache_on_shutdown() -> None:
    _release_active_session(clear_cache_session=False)
    shutil.rmtree(CACHE_ROOT, ignore_errors=True)
    _cleanup_cuda_memory()


def _bump_video_state_epoch() -> int:
    global video_state_epoch
    video_state_epoch += 1
    return video_state_epoch


def _reset_video_session_state():
    global video_prompt_events, mask_manifest_path, video_source_path
    video_prompt_events = []
    mask_manifest_path = None


def _record_prompt_event(request: "VideoAddPointsOrBoxRequest"):
    global video_prompt_events

    if request.clear_old_points:
        video_prompt_events = [
            event
            for event in video_prompt_events
            if not (event["frame_idx"] == request.frame_idx and event["obj_id"] == request.obj_id)
        ]

    event = {
        "frame_idx": int(request.frame_idx),
        "obj_id": int(request.obj_id),
        "points": [list(map(float, point)) for point in (request.points or [])],
        "labels": [int(label) for label in (request.labels or [])],
        "box": [float(v) for v in request.box] if request.box is not None else None,
        "clear_old_points": bool(request.clear_old_points),
    }
    video_prompt_events.append(event)


def _load_video_frames_as_numpy(video_dir_path: Path, frame_file_names: list[str]) -> np.ndarray:
    frames_rgb: list[np.ndarray] = []
    for name in frame_file_names:
        frame_bgr = cv2.imread(str(video_dir_path / name))
        if frame_bgr is None:
            continue
        frames_rgb.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

    if not frames_rgb:
        raise ValueError(f"No readable frames found under {video_dir_path}")

    return np.stack(frames_rgb, axis=0)


def _build_window_dir(
    frame_paths: list[Path],
    run_root: Path,
    window_name: str,
    progress_callback: Optional[Callable[[int, int, Path], None]] = None,
) -> Path:
    window_dir = run_root / window_name
    window_dir.mkdir(parents=True, exist_ok=True)
    total = len(frame_paths)

    for local_idx, source_path in enumerate(frame_paths):
        target_name = f"{local_idx:05d}{source_path.suffix.lower()}"
        target_path = window_dir / target_name
        try:
            os.symlink(source_path, target_path)
        except OSError:
            try:
                os.link(source_path, target_path)
            except OSError:
                shutil.copy2(source_path, target_path)
        if progress_callback is not None:
            progress_callback(local_idx + 1, total, source_path)

    return window_dir


def _link_or_copy_file(source_path: Path, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(source_path, target_path)
    except OSError:
        try:
            os.link(source_path, target_path)
        except OSError:
            shutil.copy2(source_path, target_path)


def _create_active_session(source_path: Path) -> Path:
    global active_session_dir, active_session_id, active_session_saved_name

    SESSION_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    session_id = uuid.uuid4().hex
    session_dir = SESSION_CACHE_ROOT / session_id
    (session_dir / "frames").mkdir(parents=True, exist_ok=False)
    (session_dir / "masks").mkdir(parents=True, exist_ok=True)
    active_session_dir = session_dir
    active_session_id = session_id
    active_session_saved_name = None
    _write_session_metadata(
        {
            "created_at": _utc_now_iso(),
            "source_input_path": str(source_path),
        }
    )
    return session_dir


def _copy_frames_directory_to_session(
    source_dir: Path,
    frames_dir: Path,
    progress_callback: Optional[Callable[[int, int, Path], None]] = None,
) -> list[str]:
    source_frames = sorted(
        frame_path
        for frame_path in source_dir.iterdir()
        if frame_path.is_file() and frame_path.suffix.lower() in IMAGE_EXTENSIONS
    )
    total = len(source_frames)
    frame_names: list[str] = []
    for index, source_path in enumerate(source_frames, start=1):
        target_name = f"{index - 1:05d}{source_path.suffix.lower()}"
        target_path = frames_dir / target_name
        _link_or_copy_file(source_path, target_path)
        frame_names.append(target_name)
        if progress_callback is not None:
            progress_callback(index, total, source_path)
    return frame_names


def _extract_video_to_session_frames(
    video_path: Path,
    frames_dir: Path,
    progress_callback: Optional[Callable[[int, Optional[int]], None]] = None,
) -> list[str]:
    frames_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Unable to open video file: {video_path}")

    raw_total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    total_frames: Optional[int] = raw_total_frames if raw_total_frames > 0 else None
    frame_idx = 0
    frame_names: list[str] = []
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break
            frame_name = f"{frame_idx:05d}.jpg"
            output_path = frames_dir / frame_name
            if not cv2.imwrite(str(output_path), frame):
                raise RuntimeError(f"Failed to write extracted frame: {output_path}")
            frame_names.append(frame_name)
            frame_idx += 1
            if progress_callback is not None:
                progress_callback(frame_idx, total_frames)
    finally:
        capture.release()

    if frame_idx == 0:
        raise ValueError(f"No frames could be extracted from video: {video_path}")
    return frame_names


def _sanitize_save_name(name: str) -> str:
    sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", name.strip())
    sanitized = re.sub(r"_+", "_", sanitized).strip(" ._")
    if not sanitized or sanitized in {".", ".."}:
        raise HTTPException(status_code=400, detail="Save name cannot be empty.")
    return sanitized


def _manifest_frame_payload(frame_masks: dict[int, np.ndarray]) -> dict[str, Any]:
    objects: dict[str, Any] = {}
    for obj_id, mask in frame_masks.items():
        mask_array = np.asarray(mask).astype(bool)
        if mask_array.ndim != 2:
            mask_array = np.squeeze(mask_array)
        if mask_array.ndim != 2:
            continue
        objects[str(int(obj_id))] = {
            "size": [int(mask_array.shape[0]), int(mask_array.shape[1])],
            "rle": encode_mask_to_rle(mask_array),
            "bbox": mask_bbox_xywh(mask_array),
        }
    return {"objects": objects}


def _mask_logits_to_2d_bool(mask_logits: Any) -> np.ndarray:
    mask_array = (mask_logits > 0.0).detach().cpu().numpy()
    mask_array = np.squeeze(mask_array).astype(bool)
    if mask_array.ndim == 3 and mask_array.shape[0] == 1:
        mask_array = np.squeeze(mask_array, axis=0)
    if mask_array.ndim != 2:
        raise ValueError(f"Expected 2D mask after squeeze, got shape {mask_array.shape}")
    return mask_array


def _ensure_tracker_model(model_name: str):
    global tracker
    if tracker is None or getattr(tracker, "model_name", None) != model_name:
        if tracker is not None:
            del tracker
            _cleanup_cuda_memory()
        tracker = cot.CoTracker(model_name=model_name)


def _restore_video_masker_from_prompt_events(
    *,
    online_mode: bool,
    batch_size: Optional[int],
    offload_video_to_cpu: Optional[bool],
    offload_state_to_cpu: Optional[bool],
    increment_epoch: bool = True,
) -> None:
    global video_masker, video_dir, video_prompt_events

    if video_dir is None:
        return

    def _on_progress(stage: str, label: str, progress: Optional[float], message: str) -> None:
        _update_job(stage=stage, stage_label=label, progress=progress, message=message)

    if video_masker is None:
        video_masker = svm.SAM2VideoMasker(progress_callback=_on_progress)

    video_masker.init_state(
        video_dir,
        online_mode=online_mode,
        batch_size=batch_size,
        offload_video_to_cpu=offload_video_to_cpu,
        offload_state_to_cpu=offload_state_to_cpu,
        async_loading_frames=False,
        progress_callback=_on_progress,
    )

    for event in video_prompt_events:
        points = event["points"] if event["points"] else None
        labels = event["labels"] if event["labels"] else None
        video_masker.add_new_points_or_box(
            frame_idx=int(event["frame_idx"]),
            obj_id=int(event["obj_id"]),
            points=points,
            labels=labels,
            clear_old_points=bool(event.get("clear_old_points", True)),
            box=event.get("box"),
        )

    if increment_epoch:
        _bump_video_state_epoch()
        _write_session_metadata()


def _load_tracking_video_from_current_video_state() -> tuple[np.ndarray, str]:
    global video_dir, video_frame_files
    if video_dir is None or not video_frame_files:
        raise ValueError("Video is not initialized for tracking.")

    # Keep tracking frame indices aligned with the exact frame sequence used by masking/frontend.
    # Decoding directly from source_video_path can introduce index drift across different decoders.
    frame_video = _load_video_frames_as_numpy(Path(video_dir), video_frame_files)
    return frame_video, str(video_dir)


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


def _validate_video_input_path(resolved_input_path: Path) -> None:
    if not resolved_input_path.is_file():
        return
    suffix = resolved_input_path.suffix.lower()
    if suffix in VIDEO_EXTENSIONS:
        return
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


def _prepare_video_masker_for_video_init():
    _release_active_session(clear_cache_session=True)


def _initialize_video_state_from_resolved_input(
    resolved_input_path: Path,
    *,
    online_mode: bool,
    batch_size: Optional[int],
    offload_video_to_cpu: Optional[bool],
    offload_state_to_cpu: Optional[bool],
    async_loading_frames: bool,
):
    global video_masker, video_dir, video_frame_files, video_source_path

    source_video_path = None
    session_dir = _create_active_session(resolved_input_path)
    frames_dir = session_dir / "frames"

    if resolved_input_path.is_file():
        suffix = resolved_input_path.suffix.lower()
        if suffix in VIDEO_EXTENSIONS:
            def _on_extract_progress(current: int, total: Optional[int]) -> None:
                if total:
                    progress = 0.35 + (0.3 * (current / total))
                    message = f"Extracted {current} of {total} frames"
                else:
                    progress = None
                    message = f"Extracted {current} frames"
                _update_job(
                    stage="extracting_frames",
                    stage_label="Extracting video frames",
                    progress=progress,
                    current=current,
                    total=total,
                    frame_idx=max(0, current - 1),
                    message=message,
                    append_history=current == 1 or (bool(total) and current == total),
                )

            try:
                indexed_frame_files = _extract_video_to_session_frames(
                    resolved_input_path,
                    frames_dir,
                    progress_callback=_on_extract_progress,
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
        candidate_count = len([
            path
            for path in resolved_input_path.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ])

        def _on_link_progress(current: int, total: int, source_path: Path) -> None:
            progress = 0.35 + (0.3 * (current / total)) if total else 0.65
            _update_job(
                stage="linking_frames",
                stage_label="Linking frame cache",
                progress=progress,
                current=current,
                total=total,
                frame_idx=max(0, current - 1),
                message=f"Linked {current} of {total} frames",
                append_history=current == 1 or current == total,
            )

        _update_job(
            stage="linking_frames",
            stage_label="Linking frame cache",
            progress=0.35,
            current=0,
            total=candidate_count,
            frame_idx=None,
            message="Preparing session-local frame cache",
        )
        indexed_frame_files = _copy_frames_directory_to_session(
            resolved_input_path,
            frames_dir,
            progress_callback=_on_link_progress,
        )

    video_dir = str(frames_dir)
    video_frame_files = indexed_frame_files

    if not video_frame_files:
        raise HTTPException(
            status_code=400,
            detail=f"No image frames found in directory: {resolved_input_path}"
        )

    def _on_sam2_progress(stage: str, label: str, progress: Optional[float], message: str) -> None:
        _update_job(
            stage=stage,
            stage_label=label,
            progress=progress,
            current=None,
            total=None,
            frame_idx=None,
            message=message,
        )

    if video_masker is None:
        video_masker = svm.SAM2VideoMasker(progress_callback=_on_sam2_progress)

    try:
        video_masker.init_state(
            video_dir,
            online_mode=online_mode,
            batch_size=batch_size,
            offload_video_to_cpu=offload_video_to_cpu,
            offload_state_to_cpu=offload_state_to_cpu,
            async_loading_frames=async_loading_frames,
            progress_callback=_on_sam2_progress,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    _update_job(
        stage="indexing_frames",
        stage_label="Indexing video frames",
        progress=0.85,
        message="SAM2 state initialized; indexing frame files",
    )

    candidate_paths = sorted([
        frame_path
        for frame_path in frames_dir.iterdir()
        if frame_path.is_file()
    ])
    total_candidates = len(candidate_paths)
    report_every = max(1, total_candidates // 100) if total_candidates else 1
    indexed_frame_files: list[str] = []
    for candidate_idx, frame_path in enumerate(candidate_paths, start=1):
        if frame_path.suffix.lower() in IMAGE_EXTENSIONS:
            indexed_frame_files.append(frame_path.name)
        should_report = (
            candidate_idx == 1
            or candidate_idx == total_candidates
            or candidate_idx % report_every == 0
        )
        if should_report:
            _update_job(
                stage="indexing_frames",
                stage_label="Indexing video frames",
                progress=0.85 + (0.1 * (candidate_idx / total_candidates)) if total_candidates else 0.95,
                current=len(indexed_frame_files),
                total=total_candidates,
                frame_idx=len(indexed_frame_files) - 1 if indexed_frame_files else None,
                message=f"Indexed {len(indexed_frame_files)} of {total_candidates} frame files",
                append_history=candidate_idx == 1 or candidate_idx == total_candidates,
            )
    video_source_path = source_video_path
    state_epoch = _bump_video_state_epoch()
    _write_session_metadata(
        {
            "created_at": _utc_now_iso(),
            "source_input_path": str(resolved_input_path),
        }
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
        "state_epoch": state_epoch,
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


class VideoSaveRequest(BaseModel):
    name: str


class TrackingLoadVideoRequest(BaseModel):
    video_path: str
    model_name: str = "cotracker3_offline"  # "cotracker3_offline" or "cotracker3_online"


class TrackingGridRequest(BaseModel):
    grid_size: int = 15
    add_support_grid: bool = True


class TrackingPointsRequest(BaseModel):
    queries: list[list[float]]  # List of [t, x, y] coordinates
    add_support_grid: bool = True


class TrackingPromptPointsRequest(BaseModel):
    model_name: str = "cotracker3_online"
    add_support_grid: bool = True


@app.get("/")
async def root():
    return {"message": "Data Engine Backend"}


@app.get("/health")
async def health():
    """Simple health endpoint used by the C++ client to detect readiness."""
    return {"status": "ok"}


@app.get("/jobs/current")
async def get_current_job():
    return {"job": _serialize_job()}


@app.get("/jobs/{job_id}")
async def get_job(job_id: str):
    job = _serialize_job()
    if job is None or job.get("job_id") != job_id:
        raise HTTPException(status_code=404, detail="Job not found.")
    return {"job": job}


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
    return _queue_long_job(
        operation="video_init",
        stage="resolving_input",
        stage_label="Resolving input",
        message="Video initialization queued",
        worker=lambda: _run_video_init_job(request),
    )


def _run_video_init_job(request: VideoInitStateRequest) -> dict[str, Any]:
    global video_masker
    _update_job(
        status="running",
        stage="resolving_input",
        stage_label="Resolving input",
        progress=0.05,
        message="Resolving video path",
    )
    _prepare_video_masker_for_video_init()
    resolved_input_path = _resolve_input_path(request.video_frames_dir)
    _validate_video_input_path(resolved_input_path)
    _update_job(
        stage="preparing_session_cache",
        stage_label="Preparing session cache",
        progress=0.10,
        message="Creating active session cache",
    )

    def _on_sam2_progress(stage: str, label: str, progress: Optional[float], message: str) -> None:
        _update_job(stage=stage, stage_label=label, progress=progress, message=message)

    video_masker = svm.SAM2VideoMasker(progress_callback=_on_sam2_progress)
    _update_job(
        stage="model_ready",
        stage_label="SAM2 model ready",
        progress=0.35,
        message="SAM2 model loaded",
    )
    if resolved_input_path.is_file() and resolved_input_path.suffix.lower() in VIDEO_EXTENSIONS:
        _update_job(
            stage="extracting_frames",
            stage_label="Extracting video frames",
            progress=0.35,
            message="Extracting video frames",
        )
    result = _initialize_video_state_from_resolved_input(
        resolved_input_path,
        online_mode=request.online_mode,
        batch_size=request.batch_size,
        offload_video_to_cpu=request.offload_video_to_cpu,
        offload_state_to_cpu=request.offload_state_to_cpu,
        async_loading_frames=request.async_loading_frames,
    )
    _update_job(
        stage="indexing_frames",
        stage_label="Indexing video frames",
        progress=0.95,
        current=int(result.get("num_frames", 0)),
        total=int(result.get("num_frames", 0)),
        message=f"Indexed {int(result.get('num_frames', 0))} frames",
    )
    return result

@app.post("/video/reset_state")
async def reset_video_state():
    global video_masker, mask_manifest_path
    if video_masker is None:
        return {"error": "Video masker not active."}
    video_masker.reset_state()
    _reset_video_session_state()
    masks_dir = _current_masks_dir()
    if masks_dir is not None:
        shutil.rmtree(masks_dir, ignore_errors=True)
        masks_dir.mkdir(parents=True, exist_ok=True)
    mask_manifest_path = None
    state_epoch = _bump_video_state_epoch()
    _write_session_metadata()
    return {
        "message": "Video state reset successfully",
        "state_epoch": state_epoch,
    }

@app.post("/video/add_new_points_or_box")
async def add_new_points_or_box(request: VideoAddPointsOrBoxRequest):
    global video_masker, video_frame_files, video_state_epoch
    if video_masker is None:
        return {"error": "Video masker not active."}

    if not video_frame_files:
        raise HTTPException(status_code=400, detail="No video frames available. Call /video/init_state first.")
    if request.frame_idx < 0 or request.frame_idx >= len(video_frame_files):
        raise HTTPException(
            status_code=400,
            detail=f"Frame index out of bounds: {request.frame_idx}. Expected 0..{len(video_frame_files) - 1}.",
        )

    out_frame_idx, out_obj_ids, out_mask_logits = video_masker.add_new_points_or_box(
        frame_idx=request.frame_idx,
        obj_id=request.obj_id,
        points=request.points,
        labels=request.labels,
        clear_old_points=request.clear_old_points,
        box=request.box
    )
    returned_frame_idx = int(out_frame_idx)
    if returned_frame_idx != int(request.frame_idx):
        raise HTTPException(
            status_code=409,
            detail=(
                "Frame mismatch in SAM2 response: "
                f"request_frame_idx={int(request.frame_idx)} response_frame_idx={returned_frame_idx}"
            ),
        )

    normalized_obj_ids = [int(obj_id) for obj_id in out_obj_ids]
    if len(normalized_obj_ids) != len(out_mask_logits):
        raise HTTPException(
            status_code=500,
            detail=(
                "Invalid SAM2 response: "
                f"{len(normalized_obj_ids)} object IDs but {len(out_mask_logits)} mask tensors."
            ),
        )

    masks_list: list[list[list[bool]]] = []
    mask_pixel_counts: dict[int, int] = {}
    mask_shapes: dict[int, list[int]] = {}
    for index, obj_id in enumerate(normalized_obj_ids):
        mask_2d = _mask_logits_to_2d_bool(out_mask_logits[index])
        masks_list.append(mask_2d.tolist())
        mask_pixel_counts[int(obj_id)] = int(np.count_nonzero(mask_2d))
        mask_shapes[int(obj_id)] = [int(mask_2d.shape[0]), int(mask_2d.shape[1])]

    selected_obj_id = int(request.obj_id)
    selected_obj_index = normalized_obj_ids.index(selected_obj_id) if selected_obj_id in normalized_obj_ids else None
    selected_obj_pixels = int(mask_pixel_counts.get(selected_obj_id, 0))
    has_positive_prompt = any(int(label) == 1 for label in (request.labels or []))
    used_single_frame_fallback = False

    # Some interactive clicks return an empty mask before memory preflight/consolidation.
    # If the selected object mask is empty, run a 1-frame propagate pass as a bounded fallback.
    if selected_obj_index is not None and selected_obj_pixels == 0 and has_positive_prompt:
        try:
            fallback_segments = video_masker.propagate_in_video(
                start_frame_idx=int(request.frame_idx),
                max_frame_num_to_track=1,
                reverse=False,
                batch_size=1,
                online_mode=video_masker.online_mode,
                collect_segments=True,
            )
            fallback_frame_masks = fallback_segments.get(int(request.frame_idx), {})
            fallback_mask = fallback_frame_masks.get(selected_obj_id)
            if fallback_mask is not None:
                fallback_mask_2d = np.asarray(fallback_mask).astype(bool)
                fallback_mask_2d = np.squeeze(fallback_mask_2d)
                if fallback_mask_2d.ndim == 2:
                    fallback_pixels = int(np.count_nonzero(fallback_mask_2d))
                    if fallback_pixels > 0:
                        masks_list[selected_obj_index] = fallback_mask_2d.tolist()
                        mask_pixel_counts[selected_obj_id] = fallback_pixels
                        mask_shapes[selected_obj_id] = [int(fallback_mask_2d.shape[0]), int(fallback_mask_2d.shape[1])]
                        used_single_frame_fallback = True
        except Exception:
            logger.exception(
                "Single-frame interactive fallback failed for frame=%s obj=%s",
                int(request.frame_idx),
                selected_obj_id,
            )

    _record_prompt_event(request)
    logger.info(
        "Interactive mask response frame=%s obj=%s pixels=%s fallback=%s",
        int(request.frame_idx),
        selected_obj_id,
        int(mask_pixel_counts.get(selected_obj_id, 0)),
        used_single_frame_fallback,
    )

    return {
        "request_frame_idx": int(request.frame_idx),
        "frame_idx": returned_frame_idx,
        "frame_file": video_frame_files[returned_frame_idx],
        "out_obj_ids": normalized_obj_ids,
        "out_masks": masks_list,
        "mask_pixel_counts": mask_pixel_counts,
        "mask_shapes": mask_shapes,
        "single_frame_fallback_used": used_single_frame_fallback,
        "state_epoch": int(video_state_epoch),
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
    
    masks_list = [_mask_logits_to_2d_bool(out_mask_logits[i]).tolist() for i in range(len(out_obj_ids))]
    return {
        "frame_idx": frame_idx,
        "out_obj_ids": out_obj_ids,
        "out_masks": masks_list
    }


@app.post("/video/save")
async def save_video_session(request: VideoSaveRequest):
    global active_session_dir, active_session_saved_name, video_dir, mask_manifest_path

    session_path = _current_session_path()
    if session_path is None or video_dir is None or not video_frame_files:
        raise HTTPException(status_code=400, detail="Video session is not initialized.")

    save_name = _sanitize_save_name(request.name)
    SAVED_ROOT.mkdir(parents=True, exist_ok=True)
    saved_path = (SAVED_ROOT / save_name).resolve()
    if saved_path.exists():
        raise HTTPException(status_code=409, detail=f"Saved session already exists: {save_name}")

    if _path_is_relative_to(session_path, SAVED_ROOT):
        raise HTTPException(status_code=409, detail="Current session is already saved.")

    session_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(session_path), str(saved_path))

    active_session_dir = saved_path
    active_session_saved_name = save_name
    video_dir = str(saved_path / "frames")
    manifest_path = saved_path / "masks" / "manifest.json"
    mask_manifest_path = str(manifest_path) if manifest_path.exists() else None
    _write_session_metadata(
        {
            "saved_name": save_name,
            "saved_path": str(saved_path),
            "saved_at": _utc_now_iso(),
        }
    )

    return {
        "message": "Session saved successfully",
        "name": save_name,
        "saved_path": str(saved_path),
        "state_epoch": int(video_state_epoch),
    }


@app.post("/video/propagate_in_video")
async def propagate_in_video(request: VideoPropagateRequest):
    return _queue_long_job(
        operation="mask_propagation",
        stage="validating_prompts",
        stage_label="Validating prompts",
        message="Mask propagation queued",
        worker=lambda: _run_propagation_job(request),
    )


def _run_propagation_job(request: VideoPropagateRequest) -> dict[str, Any]:
    global video_masker, video_dir, video_frame_files, mask_manifest_path, video_state_epoch
    def _propagation_progress(processed_frames_count: int, expected_frames_count: int) -> float:
        return (processed_frames_count / expected_frames_count) if expected_frames_count else 0.0

    _update_job(
        status="running",
        stage="validating_prompts",
        stage_label="Validating prompts",
        progress=0.0,
        message="Validating mask propagation inputs",
    )
    if video_masker is None:
        raise HTTPException(status_code=400, detail="Video masker not active.")
    if video_dir is None:
        raise HTTPException(status_code=400, detail="Video directory not set. Call /video/init_state first.")

    if request.reverse:
        raise HTTPException(
            status_code=400,
            detail="Reverse propagation is not supported in half-window mode.",
        )

    if not video_frame_files:
        raise HTTPException(status_code=400, detail="No video frames available. Call /video/init_state first.")

    if not video_prompt_events:
        raise HTTPException(status_code=400, detail="No prompts available for propagation.")

    effective_online_mode = video_masker.online_mode if request.online_mode is None else bool(request.online_mode)
    effective_batch_size = video_masker.default_batch_size if request.batch_size is None else int(request.batch_size)
    effective_offload_video_to_cpu = video_masker.offload_video_to_cpu
    effective_offload_state_to_cpu = video_masker.offload_state_to_cpu
    if effective_batch_size <= 0:
        raise HTTPException(status_code=400, detail="batch_size must be a positive integer.")

    num_frames = len(video_frame_files)
    if request.start_frame_idx is not None:
        start_frame_idx = int(request.start_frame_idx)
    else:
        start_frame_idx = min(int(event["frame_idx"]) for event in video_prompt_events)
    start_frame_idx = min(max(start_frame_idx, 0), num_frames - 1)

    if request.max_frame_num_to_track is None:
        end_frame_idx = num_frames - 1
    else:
        requested = int(request.max_frame_num_to_track)
        if requested <= 0:
            return {
                "video_segments": {},
                "video_segments_total_frames": 0,
                "video_segments_returned_frames": 0,
                "video_segments_returned_mask_values": 0,
                "video_segments_truncated": False,
                "saved_mask_frame_count": 0,
                "saved_mask_save_failures": 0,
                "saved_mask_paths": {},
                "online_mode": effective_online_mode,
                "batch_size": effective_batch_size,
                "mask_manifest_path": mask_manifest_path,
                "state_epoch": int(video_state_epoch),
            }
        end_frame_idx = min(num_frames - 1, start_frame_idx + requested - 1)

    if end_frame_idx < start_frame_idx:
        raise HTTPException(status_code=400, detail="Invalid propagation frame range.")

    expected_total_frames = end_frame_idx - start_frame_idx + 1
    masks_root = _current_masks_dir()
    if masks_root is None:
        raise HTTPException(status_code=400, detail="Video session cache is not initialized.")
    _clear_window_cache()
    _update_job(
        stage="clearing_previous_masks",
        stage_label="Clearing previous masks",
        progress=0.02,
        current=0,
        total=expected_total_frames,
        message="Preparing mask output directory",
    )

    frame_files, masks_dir = prepare_video_masks_output(video_dir, masks_root)
    manifest_file_path = masks_dir / "manifest.json"

    first_frame = cv2.imread(str(Path(video_dir) / video_frame_files[start_frame_idx]))
    if first_frame is None:
        raise HTTPException(status_code=500, detail="Unable to read first frame for manifest metadata.")

    manifest = build_empty_mask_manifest(
        source_video_path=video_source_path,
        resolved_video_frames_dir=str(video_dir),
        num_frames=num_frames,
        frame_height=int(first_frame.shape[0]),
        frame_width=int(first_frame.shape[1]),
    )
    manifest_frames: dict[str, Any] = manifest["frames"]

    split_frame_idx = (start_frame_idx + end_frame_idx) // 2
    windows: list[tuple[int, int]] = [(start_frame_idx, split_frame_idx)]
    if split_frame_idx < end_frame_idx:
        windows.append((split_frame_idx, end_frame_idx))

    run_root = WINDOW_FRAMES_ROOT / f"run_{uuid.uuid4().hex[:12]}"
    run_root.mkdir(parents=True, exist_ok=True)

    saved_mask_paths_serializable: dict[int, list[str]] = {}
    saved_mask_frame_count = 0
    save_failures = 0
    processed_frames: set[int] = set()
    boundary_masks: dict[int, np.ndarray] = {}
    boundary_frame_idx: Optional[int] = None

    video_segments_serializable: dict[int, dict[int, list]] = {}

    try:
        for window_index, (window_start, window_end) in enumerate(windows):
            window_number = window_index + 1
            window_count = len(windows)
            _update_job(
                stage="building_window",
                stage_label="Loading propagation window",
                current=0,
                total=window_end - window_start + 1,
                window_index=window_number,
                window_count=window_count,
                frame_idx=window_start,
                progress=_propagation_progress(len(processed_frames), expected_total_frames),
                message=f"Preparing window {window_number} of {window_count}: frames {window_start}-{window_end}",
            )
            window_frame_paths = [Path(video_dir) / video_frame_files[idx] for idx in range(window_start, window_end + 1)]
            window_name = f"window_{window_index}_{window_start}_{window_end}"

            def _on_window_build_progress(current: int, total: int, source_path: Path) -> None:
                source_frame_idx = window_start + current - 1
                _update_job(
                    stage="building_window",
                    stage_label="Loading propagation window",
                    current=current,
                    total=total,
                    window_index=window_number,
                    window_count=window_count,
                    frame_idx=source_frame_idx,
                    progress=_propagation_progress(len(processed_frames), expected_total_frames),
                    message=f"Linked {current} of {total} frames for window {window_number} of {window_count}",
                    append_history=current == 1 or current == total,
                )

            window_dir = _build_window_dir(
                window_frame_paths,
                run_root,
                window_name,
                progress_callback=_on_window_build_progress,
            )

            _update_job(
                stage="initializing_state",
                stage_label="Initializing SAM2 window state",
                current=len(processed_frames),
                total=expected_total_frames,
                window_index=window_number,
                window_count=window_count,
                frame_idx=window_start,
                progress=_propagation_progress(len(processed_frames), expected_total_frames),
                message=f"Initializing SAM2 state for window {window_number} of {window_count}",
            )

            video_masker.init_state(
                str(window_dir),
                online_mode=effective_online_mode,
                batch_size=effective_batch_size,
                offload_video_to_cpu=effective_offload_video_to_cpu,
                offload_state_to_cpu=effective_offload_state_to_cpu,
                async_loading_frames=False,
            )

            _update_job(
                stage="seeding_window",
                stage_label="Seeding prompts",
                current=len(processed_frames),
                total=expected_total_frames,
                window_index=window_number,
                window_count=window_count,
                frame_idx=window_start,
                progress=_propagation_progress(len(processed_frames), expected_total_frames),
                message=f"Seeding prompts for window {window_number} of {window_count}",
            )

            if window_index > 0 and boundary_masks and boundary_frame_idx is not None:
                _update_job(
                    stage="seeding_window",
                    stage_label="Seeding boundary masks",
                    current=len(processed_frames),
                    total=expected_total_frames,
                    window_index=window_number,
                    window_count=window_count,
                    frame_idx=boundary_frame_idx,
                    progress=_propagation_progress(len(processed_frames), expected_total_frames),
                    message=f"Seeding boundary masks for window {window_number} of {window_count}",
                )
                local_boundary_idx = int(boundary_frame_idx - window_start)
                for obj_id, obj_mask in boundary_masks.items():
                    video_masker.add_new_mask(
                        frame_idx=local_boundary_idx,
                        obj_id=int(obj_id),
                        mask=np.asarray(obj_mask).astype(bool),
                    )

            for event in video_prompt_events:
                event_frame_idx = int(event["frame_idx"])
                if event_frame_idx < window_start or event_frame_idx > window_end:
                    continue
                local_event_frame_idx = event_frame_idx - window_start
                points = event["points"] if event["points"] else None
                labels = event["labels"] if event["labels"] else None
                video_masker.add_new_points_or_box(
                    frame_idx=local_event_frame_idx,
                    obj_id=int(event["obj_id"]),
                    points=points,
                    labels=labels,
                    clear_old_points=bool(event.get("clear_old_points", True)),
                    box=event.get("box"),
                )

            local_start_frame = max(start_frame_idx, window_start) - window_start
            local_max_frames = (window_end - window_start + 1) - local_start_frame

            def _on_window_frame(local_frame_idx: int, frame_masks: dict[int, np.ndarray]):
                nonlocal saved_mask_frame_count, save_failures, boundary_masks, boundary_frame_idx
                global_frame_idx = int(window_start + local_frame_idx)
                if global_frame_idx < start_frame_idx or global_frame_idx > end_frame_idx:
                    return

                is_overlap_duplicate = (
                    window_index > 0
                    and global_frame_idx == window_start
                    and global_frame_idx in processed_frames
                )
                if is_overlap_duplicate:
                    boundary_masks = {
                        int(obj_id): np.asarray(mask).astype(bool)
                        for obj_id, mask in frame_masks.items()
                    }
                    boundary_frame_idx = global_frame_idx
                    return

                processed_frames.add(global_frame_idx)
                _update_job(
                    stage="propagating_window",
                    stage_label="Propagating masks",
                    current=len(processed_frames),
                    total=expected_total_frames,
                    window_index=window_number,
                    window_count=window_count,
                    frame_idx=global_frame_idx,
                    progress=_propagation_progress(len(processed_frames), expected_total_frames),
                    message=f"Processed {len(processed_frames)} of {expected_total_frames} frames",
                    append_history=len(processed_frames) == 1 or len(processed_frames) == expected_total_frames,
                )
                manifest_frames[str(global_frame_idx)] = _manifest_frame_payload(frame_masks)
                if request.include_masks_in_response:
                    video_segments_serializable[global_frame_idx] = {
                        int(obj_id): np.asarray(mask).astype(bool).tolist()
                        for obj_id, mask in frame_masks.items()
                    }

                try:
                    saved_path = save_single_video_mask_frame(
                        frame_files,
                        masks_dir,
                        global_frame_idx,
                        frame_masks,
                    )
                    if saved_path is not None:
                        saved_mask_frame_count += 1
                        if request.include_saved_mask_paths:
                            saved_mask_paths_serializable.setdefault(global_frame_idx, []).append(saved_path)
                except Exception:
                    save_failures += 1
                    logger.exception("Failed to save propagated mask frame %s", global_frame_idx)

                if global_frame_idx == window_end:
                    boundary_masks = {
                        int(obj_id): np.asarray(mask).astype(bool)
                        for obj_id, mask in frame_masks.items()
                    }
                    boundary_frame_idx = global_frame_idx

            video_masker.propagate_in_video(
                start_frame_idx=local_start_frame,
                max_frame_num_to_track=local_max_frames,
                reverse=False,
                batch_size=effective_batch_size,
                online_mode=effective_online_mode,
                collect_segments=False,
                frame_callback=_on_window_frame,
            )
            _update_job(
                stage="propagating_window",
                stage_label="Propagating masks",
                current=len(processed_frames),
                total=expected_total_frames,
                window_index=window_number,
                window_count=window_count,
                frame_idx=window_end,
                progress=_propagation_progress(len(processed_frames), expected_total_frames),
                message=f"Finished window {window_number} of {window_count}",
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
    finally:
        shutil.rmtree(run_root, ignore_errors=True)

    write_mask_manifest(manifest_file_path, manifest)
    mask_manifest_path = str(manifest_file_path)
    _write_session_metadata()
    _update_job(
        stage="saving_manifest",
        stage_label="Saving manifest",
        progress=0.98,
        current=len(processed_frames),
        total=expected_total_frames,
        window_index=None,
        window_count=None,
        frame_idx=None,
        message="Saving mask manifest",
    )

    try:
        # Rebind interactive state to the original full video frame index space.
        _update_job(
            stage="restoring_interactive_state",
            stage_label="Restoring interactive state",
            progress=None,
            current=len(processed_frames),
            total=expected_total_frames,
            window_index=None,
            window_count=None,
            frame_idx=None,
            message="Restoring interactive masking state",
        )
        _restore_video_masker_from_prompt_events(
            online_mode=effective_online_mode,
            batch_size=effective_batch_size,
            offload_video_to_cpu=effective_offload_video_to_cpu,
            offload_state_to_cpu=effective_offload_state_to_cpu,
        )
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Propagation completed but failed to restore interactive masking state: {error}",
        ) from error

    return {
        "video_segments": video_segments_serializable if request.include_masks_in_response else {},
        "video_segments_total_frames": len(processed_frames),
        "video_segments_returned_frames": len(video_segments_serializable),
        "video_segments_returned_mask_values": 0,
        "video_segments_truncated": False,
        "saved_mask_frame_count": saved_mask_frame_count,
        "saved_mask_save_failures": save_failures,
        "saved_mask_paths": saved_mask_paths_serializable,
        "online_mode": effective_online_mode,
        "batch_size": effective_batch_size,
        "mask_manifest_path": mask_manifest_path,
        "state_epoch": int(video_state_epoch),
    }

@app.post("/video/clear_all_prompts_in_frame")
async def clear_all_prompts_in_frame(frame_idx: int, obj_id: int):
    global video_masker, video_prompt_events
    if video_masker is None:
        return {"error": "Video masker not active."}
    video_masker.clear_all_prompts_in_frame(frame_idx, obj_id)
    video_prompt_events = [
        event
        for event in video_prompt_events
        if not (int(event["frame_idx"]) == int(frame_idx) and int(event["obj_id"]) == int(obj_id))
    ]
    return {"message": "Cleared all prompts in frame successfully"}

@app.post("/video/remove_object")
async def remove_object(obj_id: int):
    global video_masker, video_prompt_events
    if video_masker is None:
        return {"error": "Video masker not active."}
    video_masker.remove_object(obj_id)
    video_prompt_events = [
        event
        for event in video_prompt_events
        if int(event["obj_id"]) != int(obj_id)
    ]
    return {"message": "Object removed successfully"}


@app.post("/tracking/load_video")
async def load_tracking_video(request: TrackingLoadVideoRequest):
    """Load a video file for tracking."""
    global tracker, tracking_video, tracking_video_path

    _release_active_session(clear_tracker=False, clear_cache_session=True)
    _bump_video_state_epoch()
    
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


@app.post("/tracking/track_prompt_points")
async def track_prompt_points(request: TrackingPromptPointsRequest):
    return _queue_long_job(
        operation="prompt_tracking",
        stage="collecting_prompts",
        stage_label="Collecting prompts",
        message="Prompt tracking queued",
        worker=lambda: _run_prompt_tracking_job(request),
    )


def _run_prompt_tracking_job(request: TrackingPromptPointsRequest) -> dict[str, Any]:
    global tracker, tracking_video, tracking_video_path, video_masker, video_prompt_events, video_dir, video_state_epoch
    _update_job(
        status="running",
        stage="collecting_prompts",
        stage_label="Collecting prompts",
        progress=0.05,
        message="Collecting positive prompt points",
    )

    if not video_prompt_events:
        raise HTTPException(status_code=400, detail="No annotation prompts available for tracking.")

    positive_queries: list[list[float]] = []
    point_metadata: list[dict[str, Any]] = []

    for event_idx, event in enumerate(video_prompt_events):
        points = event.get("points", []) or []
        labels = event.get("labels", []) or [1] * len(points)
        frame_idx = int(event.get("frame_idx", 0))
        obj_id = int(event.get("obj_id", 0))
        for point_idx, point in enumerate(points):
            if point_idx >= len(labels) or int(labels[point_idx]) != 1:
                continue
            if len(point) < 2:
                continue
            x_coord = float(point[0])
            y_coord = float(point[1])
            positive_queries.append([float(frame_idx), x_coord, y_coord])
            point_metadata.append(
                {
                    "point_id": f"p{event_idx}_{point_idx}",
                    "obj_id": obj_id,
                    "source_frame_idx": frame_idx,
                    "source_x": x_coord,
                    "source_y": y_coord,
                }
            )

    if not positive_queries:
        raise HTTPException(status_code=400, detail="No positive prompt points available for tracking.")
    total_queries = len(positive_queries)
    _update_job(
        stage="loading_tracker",
        stage_label="Loading tracker",
        progress=0.2,
        current=0,
        total=total_queries,
        message=f"Preparing to track {total_queries} prompt points",
    )

    should_restore_video_masker = video_masker is not None and video_dir is not None
    restore_online_mode = video_masker.online_mode if video_masker is not None else True
    restore_batch_size = video_masker.default_batch_size if video_masker is not None else None
    restore_offload_video_to_cpu = video_masker.offload_video_to_cpu if video_masker is not None else None
    restore_offload_state_to_cpu = video_masker.offload_state_to_cpu if video_masker is not None else None

    def _restore_masker_state(*, raise_on_error: bool) -> None:
        if not should_restore_video_masker:
            return
        try:
            _restore_video_masker_from_prompt_events(
                online_mode=restore_online_mode,
                batch_size=restore_batch_size,
                offload_video_to_cpu=restore_offload_video_to_cpu,
                offload_state_to_cpu=restore_offload_state_to_cpu,
            )
        except Exception as error:
            logger.exception("Failed to restore interactive video masker state after prompt tracking")
            if raise_on_error:
                raise HTTPException(
                    status_code=500,
                    detail=f"Prompt-point tracking finished but failed to restore interactive masking state: {error}",
                ) from error

    if video_masker is not None:
        del video_masker
        video_masker = None
        _bump_video_state_epoch()
        _cleanup_cuda_memory()

    _ensure_tracker_model(request.model_name)
    _update_job(
        stage="loading_frames",
        stage_label="Loading frames",
        progress=0.35,
        current=0,
        total=total_queries,
        message="Loading video frames for tracking",
    )

    def _is_oom_runtime_error(error: RuntimeError) -> bool:
        return "out of memory" in str(error).lower()

    def _track_queries_batched(
        video: np.ndarray,
        query_array: np.ndarray,
        *,
        add_support_grid: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        if query_array.size == 0:
            raise ValueError("No query points provided for tracking.")

        initial_batch_size = int(DEFAULT_PROMPT_TRACK_BATCH_SIZE)
        if initial_batch_size <= 0:
            initial_batch_size = query_array.shape[0]
        batch_size = max(1, min(initial_batch_size, int(query_array.shape[0])))

        tracks_batches: list[np.ndarray] = []
        visibility_batches: list[np.ndarray] = []
        index = 0

        while index < query_array.shape[0]:
            end_index = min(query_array.shape[0], index + batch_size)
            query_batch = query_array[index:end_index]

            try:
                batch_tracks, batch_visibility = tracker.track(
                    video,
                    queries=query_batch,
                    add_support_grid=add_support_grid,
                )
            except torch.OutOfMemoryError as error:
                _cleanup_cuda_memory()
                if batch_size <= 1:
                    raise error
                batch_size = max(1, batch_size // 2)
                continue
            except RuntimeError as error:
                if _is_oom_runtime_error(error):
                    _cleanup_cuda_memory()
                    if batch_size <= 1:
                        raise error
                    batch_size = max(1, batch_size // 2)
                    continue
                raise

            tracks_batches.append(batch_tracks)
            visibility_batches.append(batch_visibility)
            index = end_index
            _update_job(
                stage="tracking_points",
                stage_label="Tracking prompt points",
                progress=0.35 + (0.55 * (index / query_array.shape[0])),
                current=index,
                total=query_array.shape[0],
                message=f"Tracked {index} of {query_array.shape[0]} prompt points",
            )

        tracks = np.concatenate(tracks_batches, axis=0)
        visibility = np.concatenate(visibility_batches, axis=0)
        return tracks, visibility

    try:
        tracking_video, tracking_video_path = _load_tracking_video_from_current_video_state()
        query_array = np.asarray(positive_queries, dtype=np.float32)
        support_grid_used = bool(request.add_support_grid)

        try:
            tracks, visibility = _track_queries_batched(
                tracking_video,
                query_array,
                add_support_grid=support_grid_used,
            )
        except torch.OutOfMemoryError:
            if not support_grid_used:
                raise
            support_grid_used = False
            tracks, visibility = _track_queries_batched(
                tracking_video,
                query_array,
                add_support_grid=False,
            )
        except RuntimeError as error:
            if not _is_oom_runtime_error(error) or not support_grid_used:
                raise
            support_grid_used = False
            tracks, visibility = _track_queries_batched(
                tracking_video,
                query_array,
                add_support_grid=False,
            )
    except ValueError as error:
        _restore_masker_state(raise_on_error=False)
        raise HTTPException(status_code=400, detail=str(error)) from error
    except torch.OutOfMemoryError as error:
        _cleanup_cuda_memory()
        _restore_masker_state(raise_on_error=False)
        raise HTTPException(
            status_code=507,
            detail="CUDA out of memory during tracking. Try online mode or fewer prompt points.",
        ) from error
    except RuntimeError as error:
        if "out of memory" in str(error).lower():
            _cleanup_cuda_memory()
            _restore_masker_state(raise_on_error=False)
            raise HTTPException(
                status_code=507,
                detail="CUDA out of memory during tracking. Try online mode or fewer prompt points.",
            ) from error
        _restore_masker_state(raise_on_error=False)
        raise HTTPException(status_code=500, detail=f"Prompt-point tracking failed: {error}") from error
    except Exception as error:
        _restore_masker_state(raise_on_error=False)
        raise HTTPException(status_code=500, detail=f"Prompt-point tracking failed: {error}") from error

    _cleanup_cuda_memory()
    _update_job(
        stage="restoring_masker",
        stage_label="Restoring masking state",
        progress=0.95,
        current=int(tracks.shape[0]),
        total=total_queries,
        message="Restoring interactive masking state",
    )
    _restore_masker_state(raise_on_error=True)

    return {
        "message": "Prompt-point tracking completed",
        "model_name": tracker.model_name,
        "num_points": int(tracks.shape[0]),
        "num_frames": int(tracks.shape[1]),
        "add_support_grid_used": support_grid_used,
        "tracks": tracks.tolist(),
        "visibility": visibility.tolist(),
        "points": point_metadata,
        "state_epoch": int(video_state_epoch),
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


@app.get("/video/mask_manifest")
async def get_mask_manifest():
    global video_dir, mask_manifest_path
    if video_dir is None:
        return {"error": "Video not initialized"}

    masks_dir = _current_masks_dir()
    manifest_path = Path(mask_manifest_path) if mask_manifest_path else (masks_dir / "manifest.json" if masks_dir is not None else Path(video_dir) / "masks" / "manifest.json")
    if not manifest_path.exists():
        return {"error": "Mask manifest not found. Run /video/propagate_in_video first."}

    manifest = load_mask_manifest(manifest_path)
    return {
        "version": manifest.get("version"),
        "source_video_path": manifest.get("source_video_path"),
        "resolved_video_frames_dir": manifest.get("resolved_video_frames_dir"),
        "num_frames": manifest.get("num_frames", 0),
        "frame_height": manifest.get("frame_height"),
        "frame_width": manifest.get("frame_width"),
        "mask_manifest_path": str(manifest_path),
    }


@app.get("/video/mask_data/{frame_idx}")
async def get_mask_data(frame_idx: int):
    global video_dir, mask_manifest_path
    if video_dir is None:
        return {"error": "Video not initialized"}
    if frame_idx < 0:
        return {"error": "Frame index out of bounds"}

    masks_dir = _current_masks_dir()
    manifest_path = Path(mask_manifest_path) if mask_manifest_path else (masks_dir / "manifest.json" if masks_dir is not None else Path(video_dir) / "masks" / "manifest.json")
    if not manifest_path.exists():
        return {"frame_idx": frame_idx, "objects": {}}

    manifest = load_mask_manifest(manifest_path)
    num_frames = int(manifest.get("num_frames", 0))
    if frame_idx >= num_frames:
        return {"error": "Frame index out of bounds"}

    frame_payload = manifest.get("frames", {}).get(str(frame_idx), {"objects": {}})
    objects_payload = frame_payload.get("objects", {})
    return {
        "frame_idx": int(frame_idx),
        "objects": objects_payload,
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

    masks_dir = _current_masks_dir() or Path(video_dir) / "masks"
    file_path = masks_dir / f"frame_{frame_idx:05d}_masks.png"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Mask frame not found: {file_path}")
    return FileResponse(str(file_path))
