import copy, logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from fastapi import HTTPException
import numpy as np
import torch
import co_tracker as cot
from core.config import DEFAULT_PROMPT_TRACK_BATCH_SIZE, DEFAULT_STREAMING_TRACK_FRAME_THRESHOLD
from core.jobs import queue_long_job, update_job
from core.runtime import cleanup_cuda_memory
from core.state import state
from schemas.tracking import TrackingPromptPointsRequest
from sessions.cache import bump_video_state_epoch
from tracking.results import save_prompt_tracking_result
from tracking.service import ensure_tracker_model, should_stream_tracking
from video.io import load_video_frames_as_numpy
from video.prompts import restore_video_masker_from_prompt_events

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class PromptTrackingSnapshot:
    session_dir: Path
    video_dir: str
    frame_files: list[str]
    prompt_events: list[dict[str, Any]]
    state_epoch: int
    restore_online_mode: bool
    restore_batch_size: int | None
    restore_offload_video_to_cpu: bool | None
    restore_offload_state_to_cpu: bool | None

def _resolve_tracking_frame_range(request: TrackingPromptPointsRequest, num_frames: int) -> tuple[int, int]:
    if num_frames <= 0:
        raise HTTPException(status_code=400, detail="No video frames available for tracking.")
    start_value = getattr(request, "start_frame_idx", None)
    end_value = getattr(request, "end_frame_idx", None)
    start_frame_idx = 0 if start_value is None else int(start_value)
    end_frame_idx = num_frames - 1 if end_value is None else int(end_value)
    start_frame_idx = min(max(start_frame_idx, 0), num_frames - 1)
    end_frame_idx = min(max(end_frame_idx, 0), num_frames - 1)
    if end_frame_idx < start_frame_idx:
        raise HTTPException(status_code=400, detail="end_frame_idx must be >= start_frame_idx.")
    return start_frame_idx, end_frame_idx

def _expand_tracks_to_full_timeline(
    tracks: np.ndarray,
    visibility: np.ndarray,
    *,
    num_frames: int,
    start_frame_idx: int,
    end_frame_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    expected_range_frames = int(end_frame_idx) - int(start_frame_idx) + 1
    if tracks.shape[1] != expected_range_frames or visibility.shape[1] != expected_range_frames:
        raise ValueError(
            "CoTracker returned an unexpected frame count: "
            f"expected {expected_range_frames}, got tracks={tracks.shape[1]} visibility={visibility.shape[1]}."
        )
    expanded_tracks = np.zeros((tracks.shape[0], int(num_frames), 2), dtype=np.float32)
    expanded_visibility = np.zeros((visibility.shape[0], int(num_frames)), dtype=bool)
    expanded_tracks[:, start_frame_idx : end_frame_idx + 1, :] = tracks
    expanded_visibility[:, start_frame_idx : end_frame_idx + 1] = visibility
    return expanded_tracks, expanded_visibility

def _snapshot_and_release_masker_for_tracking() -> PromptTrackingSnapshot:
    with state.video_state_lock:
        if not state.video_prompt_events:
            raise HTTPException(status_code=400, detail="No annotation prompts available for tracking.")
        if state.video_dir is None or not state.video_frame_files:
            raise HTTPException(status_code=400, detail="Video is not initialized for tracking.")
        if state.active_session_dir is None:
            raise HTTPException(status_code=400, detail="Video session cache is not initialized.")
        if state.video_masker is None:
            if state.video_masker_status == "restore_failed":
                raise HTTPException(
                    status_code=503,
                    detail="Video masker restore failed. Reinitialize the video session.",
                )
            raise HTTPException(status_code=400, detail="Video masker not active.")

        snapshot = PromptTrackingSnapshot(
            session_dir=state.active_session_dir.resolve(),
            video_dir=state.video_dir,
            frame_files=list(state.video_frame_files),
            prompt_events=copy.deepcopy(state.video_prompt_events),
            state_epoch=int(state.video_state_epoch),
            restore_online_mode=state.video_masker.online_mode,
            restore_batch_size=state.video_masker.default_batch_size,
            restore_offload_video_to_cpu=state.video_masker.offload_video_to_cpu,
            restore_offload_state_to_cpu=state.video_masker.offload_state_to_cpu,
        )
        state.video_masker = None
        state.video_masker_status = "released_for_tracking"
        state.video_masker_error = None
        bump_video_state_epoch()
        return snapshot

def _mark_masker_restoring() -> None:
    with state.video_state_lock:
        state.video_masker_status = "restoring"
        state.video_masker_error = None

def _mark_masker_ready() -> None:
    with state.video_state_lock:
        state.video_masker_status = "ready"
        state.video_masker_error = None

def _mark_masker_restore_failed(error: Exception) -> None:
    with state.video_state_lock:
        state.video_masker = None
        state.video_masker_status = "restore_failed"
        state.video_masker_error = str(error)

async def start_prompt_tracking(request: TrackingPromptPointsRequest):
    return queue_long_job(
        operation="prompt_tracking",
        stage="collecting_prompts",
        stage_label="Collecting prompts",
        message="Prompt tracking queued",
        worker=lambda: run_prompt_tracking_job(request),
    )

def run_prompt_tracking_job(request: TrackingPromptPointsRequest) -> dict[str, Any]:
    update_job(
        status="running",
        stage="collecting_prompts",
        stage_label="Collecting prompts",
        progress=0.05,
        message="Collecting positive prompt points",
    )

    snapshot = _snapshot_and_release_masker_for_tracking()
    cleanup_cuda_memory()

    num_video_frames = len(snapshot.frame_files)
    tracking_start_frame_idx, tracking_end_frame_idx = _resolve_tracking_frame_range(
        request,
        num_video_frames,
    )

    positive_queries: list[list[float]] = []
    point_metadata: list[dict[str, Any]] = []

    for event_idx, event in enumerate(snapshot.prompt_events):
        points = event.get("points", []) or []
        labels = event.get("labels", []) or [1] * len(points)
        frame_idx = int(event.get("frame_idx", 0))
        if frame_idx < tracking_start_frame_idx or frame_idx > tracking_end_frame_idx:
            continue
        obj_id = int(event.get("obj_id", 0))
        for point_idx, point in enumerate(points):
            if point_idx >= len(labels) or int(labels[point_idx]) != 1:
                continue
            if len(point) < 2:
                continue
            x_coord = float(point[0])
            y_coord = float(point[1])
            positive_queries.append([float(frame_idx - tracking_start_frame_idx), x_coord, y_coord])
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
        raise HTTPException(
            status_code=400,
            detail="No positive prompt points available in the requested tracking frame range.",
        )
    total_queries = len(positive_queries)
    update_job(
        stage="loading_tracker",
        stage_label="Loading tracker",
        progress=0.2,
        current=0,
        total=total_queries,
        message=f"Preparing tracker for {total_queries} prompt points",
    )

    def _restore_masker_state(*, raise_on_error: bool) -> None:
        _mark_masker_restoring()
        try:
            restore_video_masker_from_prompt_events(
                online_mode=snapshot.restore_online_mode,
                batch_size=snapshot.restore_batch_size,
                offload_video_to_cpu=snapshot.restore_offload_video_to_cpu,
                offload_state_to_cpu=snapshot.restore_offload_state_to_cpu,
                progress_stage_override="restoring_masker",
                progress_label_override="Restoring interactive masking state",
                progress_message_prefix="Restoring masking state after prompt tracking",
                prompt_events=snapshot.prompt_events,
            )
            _mark_masker_ready()
        except Exception as error:
            logger.exception("Failed to restore interactive video masker state after prompt tracking")
            _mark_masker_restore_failed(error)
            if raise_on_error:
                raise HTTPException(
                    status_code=500,
                    detail=f"Prompt-point tracking finished but failed to restore interactive masking state: {error}",
                ) from error

    try:
        ensure_tracker_model(cot.DEFAULT_COTRACKER_MODEL)
    except ValueError as error:
        _restore_masker_state(raise_on_error=False)
        raise HTTPException(status_code=400, detail=str(error)) from error
    update_job(
        stage="loading_frames",
        stage_label="Preparing tracking frames",
        progress=0.35,
        current=0,
        total=total_queries,
        message="Preparing video frames for prompt tracking",
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
                batch_tracks, batch_visibility = state.tracker.track(
                    video,
                    queries=query_batch,
                    add_support_grid=add_support_grid,
                )
            except torch.OutOfMemoryError as error:
                cleanup_cuda_memory()
                if batch_size <= 1:
                    raise error
                batch_size = max(1, batch_size // 2)
                continue
            except RuntimeError as error:
                if _is_oom_runtime_error(error):
                    cleanup_cuda_memory()
                    if batch_size <= 1:
                        raise error
                    batch_size = max(1, batch_size // 2)
                    continue
                raise

            tracks_batches.append(batch_tracks)
            visibility_batches.append(batch_visibility)
            index = end_index
            update_job(
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

    def _track_queries_streaming(
        query_array: np.ndarray,
        *,
        add_support_grid: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        if query_array.size == 0:
            raise ValueError("No query points provided for streaming tracking.")

        frame_loader = cot.FrameChunkLoader(
            video_dir=Path(snapshot.video_dir),
            frame_files=snapshot.frame_files[tracking_start_frame_idx : tracking_end_frame_idx + 1],
            device=state.tracker.device,
        )

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
                batch_tracks, batch_visibility = state.tracker.track_streaming(
                    frame_loader,
                    queries=query_batch,
                    add_support_grid=add_support_grid,
                )
            except torch.OutOfMemoryError as error:
                cleanup_cuda_memory()
                if batch_size <= 1:
                    raise error
                batch_size = max(1, batch_size // 2)
                continue
            except RuntimeError as error:
                if _is_oom_runtime_error(error):
                    cleanup_cuda_memory()
                    if batch_size <= 1:
                        raise error
                    batch_size = max(1, batch_size // 2)
                    continue
                raise

            tracks_batches.append(batch_tracks)
            visibility_batches.append(batch_visibility)
            index = end_index
            update_job(
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
        query_array = np.asarray(positive_queries, dtype=np.float32)
        support_grid_used = bool(request.add_support_grid)
        tracking_frame_count = tracking_end_frame_idx - tracking_start_frame_idx + 1
        use_streaming_tracking = should_stream_tracking(tracking_frame_count)
        tracking_mode = "streaming" if use_streaming_tracking else "in_memory"

        try:
            if use_streaming_tracking:
                tracks, visibility = _track_queries_streaming(
                    query_array,
                    add_support_grid=support_grid_used,
                )
            else:
                tracking_video = load_video_frames_as_numpy(Path(snapshot.video_dir), snapshot.frame_files)
                tracking_video = tracking_video[
                    tracking_start_frame_idx : tracking_end_frame_idx + 1
                ]
                tracks, visibility = _track_queries_batched(
                    tracking_video,
                    query_array,
                    add_support_grid=support_grid_used,
                )
        except torch.OutOfMemoryError:
            if not support_grid_used:
                raise
            support_grid_used = False
            if use_streaming_tracking:
                tracks, visibility = _track_queries_streaming(
                    query_array,
                    add_support_grid=False,
                )
            else:
                tracks, visibility = _track_queries_batched(
                    tracking_video,
                    query_array,
                    add_support_grid=False,
                )
        except RuntimeError as error:
            if not _is_oom_runtime_error(error) or not support_grid_used:
                raise
            support_grid_used = False
            if use_streaming_tracking:
                tracks, visibility = _track_queries_streaming(
                    query_array,
                    add_support_grid=False,
                )
            else:
                tracks, visibility = _track_queries_batched(
                    tracking_video,
                    query_array,
                    add_support_grid=False,
                )
    except ValueError as error:
        _restore_masker_state(raise_on_error=False)
        raise HTTPException(status_code=400, detail=str(error)) from error
    except torch.OutOfMemoryError as error:
        cleanup_cuda_memory()
        _restore_masker_state(raise_on_error=False)
        raise HTTPException(
            status_code=507,
            detail="CUDA out of memory during tracking. Try fewer prompt points or disable support grid.",
        ) from error
    except RuntimeError as error:
        if "out of memory" in str(error).lower():
            cleanup_cuda_memory()
            _restore_masker_state(raise_on_error=False)
            raise HTTPException(
                status_code=507,
                detail="CUDA out of memory during tracking. Try fewer prompt points or disable support grid.",
            ) from error
        _restore_masker_state(raise_on_error=False)
        raise HTTPException(status_code=500, detail=f"Prompt-point tracking failed: {error}") from error
    except Exception as error:
        _restore_masker_state(raise_on_error=False)
        raise HTTPException(status_code=500, detail=f"Prompt-point tracking failed: {error}") from error

    cleanup_cuda_memory()
    update_job(
        stage="restoring_masker",
        stage_label="Restoring interactive masking state",
        progress=0.95,
        current=int(tracks.shape[0]),
        total=total_queries,
        message="Restoring masking state after prompt tracking",
    )
    _restore_masker_state(raise_on_error=True)
    tracks, visibility = _expand_tracks_to_full_timeline(
        tracks,
        visibility,
        num_frames=num_video_frames,
        start_frame_idx=tracking_start_frame_idx,
        end_frame_idx=tracking_end_frame_idx,
    )

    tracking_result = save_prompt_tracking_result(
        model_name=state.tracker.model_name,
        num_points=int(tracks.shape[0]),
        num_frames=int(tracks.shape[1]),
        add_support_grid_used=support_grid_used,
        tracking_mode=tracking_mode,
        streaming_frame_threshold=int(DEFAULT_STREAMING_TRACK_FRAME_THRESHOLD),
        tracking_start_frame_idx=tracking_start_frame_idx,
        tracking_end_frame_idx=tracking_end_frame_idx,
        points=point_metadata,
        tracks=tracks.tolist(),
        visibility=visibility.tolist(),
    )

    return {
        "message": "Prompt-point tracking completed",
        "model_name": state.tracker.model_name,
        "num_points": int(tracks.shape[0]),
        "num_frames": int(tracks.shape[1]),
        "add_support_grid_used": support_grid_used,
        "tracking_mode": tracking_mode,
        "streaming_frame_threshold": int(DEFAULT_STREAMING_TRACK_FRAME_THRESHOLD),
        "tracking_result_id": tracking_result["result_id"],
        "tracking_start_frame_idx": tracking_start_frame_idx,
        "tracking_end_frame_idx": tracking_end_frame_idx,
        "state_epoch": int(state.video_state_epoch),
    }
