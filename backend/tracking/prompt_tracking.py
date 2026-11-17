from dataclasses import dataclass
from core.errors import OperationError
from pathlib import Path
from typing import Any
import numpy as np
from tqdm import tqdm
import co_tracker as cot
from core.config import DEFAULT_PROMPT_TRACK_BATCH_SIZE, DEFAULT_STREAMING_TRACK_FRAME_THRESHOLD
from core.context import AppContext
from core.jobs import JobHandle, capture_job_inputs, validate_job_inputs
from core.runtime import cleanup_cuda_memory, is_out_of_memory
from sessions.checkpoints import saved_state
from sessions.publication import publish_results
from schemas.responses import JobOperation, TrackingJobResult
from schemas.tracking import TrackingPromptPointsRequest
from tracking.results import save_prompt_tracking_result
from tracking.runtime import VideoTracker, VideoTrackerRuntime
from tracking.service import create_tracker_runtime, load_tracking_video_from_current_video_state, should_stream_tracking
from video.prompts import create_masker_runtime
from video.runtime import MaskerRestorationError, VideoMaskerRuntime


def track_in_batches(track_batch, queries: np.ndarray, *, on_progress) -> tuple[np.ndarray, np.ndarray]:
    """Track `queries` in batches, halving the batch size whenever the GPU runs out of memory."""
    total = len(queries)
    batch_size = int(DEFAULT_PROMPT_TRACK_BATCH_SIZE)
    if batch_size <= 0:
        batch_size = total
    batch_size = max(1, min(batch_size, total))
    tracks: list[np.ndarray] = []
    visibility: list[np.ndarray] = []
    index = 0
    while index < total:
        end = min(total, index + batch_size)
        try:
            batch_tracks, batch_visibility = track_batch(queries[index:end])
        except RuntimeError as error:
            if not is_out_of_memory(error):
                raise
            cleanup_cuda_memory()
            if batch_size <= 1:
                raise
            batch_size = max(1, batch_size // 2)
            continue
        tracks.append(batch_tracks)
        visibility.append(batch_visibility)
        index = end
        on_progress(index, total)
    return np.concatenate(tracks, axis=0), np.concatenate(visibility, axis=0)


def track_prompt_queries(
    session, tracker: VideoTracker, queries: np.ndarray, *, add_support_grid: bool, streaming: bool, on_progress,
    on_frames=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Track the query points with the session's tracker, in memory or streamed from the frame files.

    Returns the tracks and their visibility. When the GPU runs out of memory the batch is halved; the support grid is
    never dropped, because it adds almost no memory and without it a point that is covered by something else is
    still reported visible (COTRACKER_AND_SAM2_FINDINGS.md).

    `on_progress(points_done, points_total)` is called after each batch of points. A streamed batch also calls
    `on_frames(points_done, batch_points, frames_done, frames_total)` as it moves through the video, which is the
    only progress there is when every point fits in one batch.
    """
    points_done = 0

    def batch_done(done: int, total: int) -> None:
        nonlocal points_done
        points_done = done
        on_progress(done, total)

    if streaming:
        if session.video_dir is None or not session.video_frame_files:
            raise ValueError("Video is not initialized for streaming tracking.")
        loader = cot.FrameChunkLoader(
            video_dir=Path(session.video_dir), frame_files=session.video_frame_files,
            device=tracker.device, resolution=tracker.model_resolution,
        )

        def batch_tracker(grid: bool):
            def track_batch(batch):
                report = None if on_frames is None else (
                    lambda done, total: on_frames(points_done, len(batch), done, total))
                return tracker.track_streaming(loader, queries=batch, add_support_grid=grid, on_frames=report)
            return track_batch
    else:
        video, _ = load_tracking_video_from_current_video_state(session)
        # Resize once; every query batch reuses the same model-resolution frames.
        prepared = tracker.prepare_video(video)

        def batch_tracker(grid: bool):
            return lambda batch: tracker.track(prepared, queries=batch, add_support_grid=grid)
    return track_in_batches(batch_tracker(add_support_grid), queries, on_progress=batch_done)


def hide_before_query_frame(visibility: np.ndarray, queries) -> np.ndarray:
    """Mark a point not visible on the frames before its own frame.

    CoTracker's online model only tracks forward: before the frame a point was placed on, its track just stays near
    the placed position and is sometimes reported visible. Those frames are not tracking results.
    """
    visibility = np.array(visibility, dtype=bool, copy=True)
    for row, query in enumerate(queries):
        visibility[row, : int(query[0])] = False
    return visibility


def start_prompt_tracking(ctx: AppContext, request: TrackingPromptPointsRequest, precondition):
    return ctx.start_job(
        precondition, operation=JobOperation.PROMPT_TRACKING, stage="collecting_prompts", stage_label="Collecting prompts",
        message="Prompt tracking queued", parameters=request.model_dump(),
        run=lambda job, inputs: run_prompt_tracking_job(ctx, request, job, inputs),
    )

@dataclass
class TrackingOutput:
    tracker: VideoTracker
    tracks: np.ndarray
    visibility: np.ndarray
    support_grid_used: bool
    mode: str


def collect_prompt_points(session) -> tuple[list[list[float]], list[dict[str, Any]]]:
    """The queries to track (frame, x, y) and the metadata a tracking result keeps for each point."""
    if not session.annotations.prompts:
        raise OperationError(400, "No annotation prompts available for tracking.")
    positive_points = session.annotations.positive_points()
    if not positive_points:
        raise OperationError(400, "No positive prompt points available for tracking.")
    queries = [[float(point.frame_idx), point.x, point.y] for point in positive_points]
    metadata = [
        {
            "point_id": f"p{point.event_idx}_{point.point_idx}",
            "obj_id": point.obj_id,
            "source_frame_idx": point.frame_idx,
            "source_x": point.x,
            "source_y": point.y,
        }
        for point in positive_points
    ]
    return queries, metadata


def track_prompts(session, tracker_runtime: VideoTrackerRuntime, request, job: JobHandle, queries: list[list[float]]) -> TrackingOutput:
    """Track the prompt points with the session's tracker; failures become the job's HTTP-shaped errors."""
    try:
        tracker = tracker_runtime.ensure(cot.DEFAULT_COTRACKER_MODEL)
    except ValueError as error:
        raise OperationError(400, str(error)) from error
    total_queries = len(queries)
    job.enter_stage(
        "loading_frames", "Preparing tracking frames",
        progress=None,
        current=0,
        total=total_queries,
        message="Preparing video frames for prompt tracking",
    )

    # The terminal's counterpart of the job progress: frames of the video tracked. Shown only in a terminal.
    bar = tqdm(total=len(session.video_frame_files), desc="track prompt points", unit="frame")

    # The bar is the tracking itself, frame by frame. Before it the bar is indeterminate and after it full; the
    # stage text says what is happening (loading the tracker, moving SAM2 back).
    def report_progress(done: int, total: int) -> None:
        job.enter_stage(
            "tracking_points", "Tracking prompt points",
            progress=done / total,
            current=done,
            total=total,
            message=f"Tracked {done} of {total} prompt points",
        )

    def report_frames(points_done: int, batch_points: int, frames_done: int, frames_total: int) -> None:
        # The frame counter is what moves: all the points of a batch advance through the video together.
        fraction = (points_done + batch_points * frames_done / frames_total) / total_queries
        points = (f"{total_queries} prompt points" if batch_points == total_queries
                  else f"prompt points {points_done + 1} to {points_done + batch_points} of {total_queries}")
        bar.total = frames_total
        bar.n = frames_done  # set, not added: a batch that ran out of memory starts the video again
        bar.set_description(f"track {points}")
        job.enter_stage(
            "tracking_points", "Tracking prompt points",
            progress=fraction,
            current=frames_done,
            total=frames_total,
            message=f"Tracked frame {frames_done} of {frames_total} for {points}",
        )

    try:
        streaming = should_stream_tracking(len(session.video_frame_files))
        support_grid_used = bool(request.add_support_grid)
        tracks, visibility = track_prompt_queries(
            session, tracker, np.asarray(queries, dtype=np.float32), add_support_grid=support_grid_used,
            streaming=streaming, on_progress=report_progress, on_frames=report_frames,
        )
        expected_frames = len(session.video_frame_files)
        expected_tracks_shape = (total_queries, expected_frames, 2)
        expected_visibility_shape = (total_queries, expected_frames)
        if tracks.shape != expected_tracks_shape or visibility.shape != expected_visibility_shape:
            raise RuntimeError(
                f"Tracking returned tracks shape {tracks.shape} and visibility shape {visibility.shape}; "
                f"expected {expected_tracks_shape} and {expected_visibility_shape} for the session frame sequence."
            )
        visibility = hide_before_query_frame(visibility, queries)
    except ValueError as error:
        raise OperationError(400, str(error)) from error
    except RuntimeError as error:
        if is_out_of_memory(error):
            cleanup_cuda_memory()
            raise OperationError(
                507, "CUDA out of memory during tracking. Try fewer prompt points or a smaller video."
            ) from error
        raise OperationError(500, f"Prompt-point tracking failed: {error}") from error
    except Exception as error:
        raise OperationError(500, f"Prompt-point tracking failed: {error}") from error
    finally:
        bar.close()
    return TrackingOutput(tracker, tracks, visibility, support_grid_used, "streaming" if streaming else "in_memory")


def run_prompt_tracking_job(
    ctx: AppContext, request: TrackingPromptPointsRequest, job: JobHandle, inputs=None, *,
    masker_runtime: VideoMaskerRuntime | None = None, tracker_runtime: VideoTrackerRuntime | None = None,
) -> dict[str, Any]:
    session = ctx.session
    inputs = inputs or capture_job_inputs(session, JobOperation.PROMPT_TRACKING)
    validate_job_inputs(session, inputs, JobOperation.PROMPT_TRACKING)
    job.enter_stage(
        "collecting_prompts", "Collecting prompts",
        status="running",
        progress=None,
        message="Collecting positive prompt points",
    )
    queries, point_metadata = collect_prompt_points(session)
    total_queries = len(queries)
    job.enter_stage(
        "loading_tracker", "Loading tracker",
        progress=None,
        current=0,
        total=total_queries,
        message=f"Preparing tracker for {total_queries} prompt points",
    )

    runtime = masker_runtime if masker_runtime is not None else create_masker_runtime(session)
    tracker_runtime = tracker_runtime if tracker_runtime is not None else create_tracker_runtime(session)

    try:
        # SAM2's weights wait in CPU RAM meanwhile and are back on the GPU before the result is published.
        with runtime.offloaded():
            output = track_prompts(session, tracker_runtime, request, job, queries)
            cleanup_cuda_memory()
            job.enter_stage(
                "restoring_masker", "Restoring interactive masking state",
                progress=1.0,
                current=int(output.tracks.shape[0]),
                total=total_queries,
                message="Moving SAM2 back to the GPU after prompt tracking",
            )
            validate_job_inputs(session, inputs, JobOperation.PROMPT_TRACKING)
    except MaskerRestorationError as error:
        raise OperationError(
            503, f"Prompt-point tracking finished but failed to restore interactive masking state: {error}"
        ) from error

    num_points, num_frames = int(output.tracks.shape[0]), int(output.tracks.shape[1])
    with publish_results(ctx, inputs, JobOperation.PROMPT_TRACKING) as session:
        tracking_result = save_prompt_tracking_result(
            session,
            model_name=output.tracker.model_name,
            num_points=num_points,
            num_frames=num_frames,
            add_support_grid_used=output.support_grid_used,
            tracking_mode=output.mode,
            streaming_frame_threshold=int(DEFAULT_STREAMING_TRACK_FRAME_THRESHOLD),
            points=point_metadata,
            tracks=output.tracks.tolist(),
            visibility=output.visibility.tolist(),
        )

    result = {
        "session_id": inputs["session_id"],
        "input_revision": inputs["input_revision"],
        "message": "Prompt-point tracking completed",
        "model_name": output.tracker.model_name,
        "num_points": num_points,
        "num_frames": num_frames,
        "add_support_grid_used": output.support_grid_used,
        "tracking_mode": output.mode,
        "streaming_frame_threshold": int(DEFAULT_STREAMING_TRACK_FRAME_THRESHOLD),
        "tracking_result_id": tracking_result["result_id"],
        "saved_state": saved_state(session),
        "history": session.history.status(),
    }
    return TrackingJobResult(**result).model_dump()
