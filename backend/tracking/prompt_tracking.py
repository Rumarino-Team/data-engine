import logging
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
from tracking.service import ensure_tracker_model, load_tracking_video_from_current_video_state, should_stream_tracking
from video.prompts import restore_video_masker_from_prompt_events

logger = logging.getLogger(__name__)

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

    if not state.video_prompt_events:
        raise HTTPException(status_code=400, detail="No annotation prompts available for tracking.")

    positive_queries: list[list[float]] = []
    point_metadata: list[dict[str, Any]] = []

    for event_idx, event in enumerate(state.video_prompt_events):
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
    update_job(
        stage="loading_tracker",
        stage_label="Loading state.tracker",
        progress=0.2,
        current=0,
        total=total_queries,
        message=f"Preparing to track {total_queries} prompt points",
    )

    should_restore_video_masker = state.video_masker is not None and state.video_dir is not None
    restore_online_mode = state.video_masker.online_mode if state.video_masker is not None else True
    restore_batch_size = state.video_masker.default_batch_size if state.video_masker is not None else None
    restore_offload_video_to_cpu = state.video_masker.offload_video_to_cpu if state.video_masker is not None else None
    restore_offload_state_to_cpu = state.video_masker.offload_state_to_cpu if state.video_masker is not None else None

    def _restore_masker_state(*, raise_on_error: bool) -> None:
        if not should_restore_video_masker:
            return
        try:
            restore_video_masker_from_prompt_events(
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

    if state.video_masker is not None:
        state.video_masker = None
        bump_video_state_epoch()
        cleanup_cuda_memory()

    try:
        ensure_tracker_model(cot.DEFAULT_COTRACKER_MODEL)
    except ValueError as error:
        _restore_masker_state(raise_on_error=False)
        raise HTTPException(status_code=400, detail=str(error)) from error
    update_job(
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
        if state.video_dir is None or not state.video_frame_files:
            raise ValueError("Video is not initialized for streaming tracking.")
        if query_array.size == 0:
            raise ValueError("No query points provided for streaming tracking.")

        frame_loader = cot.FrameChunkLoader(
            video_dir=Path(state.video_dir),
            frame_files=state.video_frame_files,
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
        use_streaming_tracking = should_stream_tracking(len(state.video_frame_files))
        tracking_mode = "streaming" if use_streaming_tracking else "in_memory"

        try:
            if use_streaming_tracking:
                tracks, visibility = _track_queries_streaming(
                    query_array,
                    add_support_grid=support_grid_used,
                )
                state.tracking_video_path = str(state.video_dir)
            else:
                state.tracking_video, state.tracking_video_path = load_tracking_video_from_current_video_state()
                tracks, visibility = _track_queries_batched(
                    state.tracking_video,
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
                state.tracking_video_path = str(state.video_dir)
            else:
                tracks, visibility = _track_queries_batched(
                    state.tracking_video,
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
                state.tracking_video_path = str(state.video_dir)
            else:
                tracks, visibility = _track_queries_batched(
                    state.tracking_video,
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
        stage_label="Restoring masking state",
        progress=0.95,
        current=int(tracks.shape[0]),
        total=total_queries,
        message="Restoring interactive masking state",
    )
    _restore_masker_state(raise_on_error=True)

    return {
        "message": "Prompt-point tracking completed",
        "model_name": state.tracker.model_name,
        "num_points": int(tracks.shape[0]),
        "num_frames": int(tracks.shape[1]),
        "add_support_grid_used": support_grid_used,
        "tracking_mode": tracking_mode,
        "streaming_frame_threshold": int(DEFAULT_STREAMING_TRACK_FRAME_THRESHOLD),
        "tracks": tracks.tolist(),
        "visibility": visibility.tolist(),
        "points": point_metadata,
        "state_epoch": int(state.video_state_epoch),
    }

