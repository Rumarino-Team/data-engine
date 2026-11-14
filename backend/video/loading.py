import logging
from pathlib import Path
from typing import Any, Optional
from pydantic import ValidationError
from core.annotations import SessionAnnotations
from core.config import IMAGE_EXTENSIONS
from core.context import AppContext
from core.errors import OperationError
from core.jobs import JobHandle, utc_now_iso
from core.runtime import cleanup_cuda_memory
from core.state import SessionState
from schemas.responses import InitResult
from schemas.video import InteractiveState, VideoInitStateRequest
from sessions.artifacts import sweep_unreferenced_artifacts
from sessions.checkpoints import content_fingerprint
from sessions.metadata import interactive_state_projection, load_session_metadata, persisted_session_fields
from sessions.paths import resolve_input_path, resolve_saved_session_layout, validate_video_input_path
from tracking.results import restored_tracking_result_payload
from video.io import copy_frames_directory_to_session, create_active_session, discard_working_session, extract_video_to_session_frames, open_checkpoint_as_working_copy
from video.masks import override_pixels, validate_interactive_mask_references
from video.prompts import create_masker_runtime
from video.runtime import MaskerSettings, VideoMasker

logger = logging.getLogger(__name__)

def open_saved_session(checkpoint_dir: Path, session: SessionState, job: JobHandle) -> dict[str, Any]:
    """Open a checkpoint as a new working copy and restore its accepted state into `session`.

    Returns the restored payload for the response.
    """
    session_metadata = load_session_metadata(checkpoint_dir)
    checkpoint_id, display_name = session_metadata.get("checkpoint_id"), session_metadata.get("display_name")
    if not all(isinstance(value, str) and value.strip() for value in (checkpoint_id, display_name)):
        raise OperationError(400, "Saved session is missing its checkpoint identity or display name.")
    # The checkpoint stays untouched; all edits and results belong to the working copy.
    session_dir = open_checkpoint_as_working_copy(checkpoint_dir, session=session)
    frames_dir, masks_dir = session_dir / "frames", session_dir / "masks"
    session.adopt_checkpoint(checkpoint_id, checkpoint_dir, display_name)
    manifest_name = session_metadata.get("mask_manifest_file")
    manifest_path = masks_dir / Path(manifest_name).name if manifest_name else None
    session.mask_manifest_path = str(manifest_path) if manifest_path and manifest_path.exists() else None
    session.video_source_path = session_metadata.get("source_video_path")
    frame_files = sorted(
        frame_path.name
        for frame_path in frames_dir.iterdir()
        if frame_path.is_file() and frame_path.suffix.lower() in IMAGE_EXTENSIONS
    )
    job.enter_stage(
        "linking_frames", "Loading saved session frames",
        progress=0.65,
        current=len(frame_files),
        total=len(frame_files),
        frame_idx=len(frame_files) - 1 if frame_files else None,
        message=f"Found {len(frame_files)} saved session frames",
    )
    try:
        annotations = SessionAnnotations.model_validate(session_metadata["annotations"])
        if any(event.frame_idx >= len(frame_files) for event in [*annotations.prompts, *annotations.mask_overrides]):
            raise ValueError("Prompt frame is outside the video.")
        interactive_state = InteractiveState.model_validate(session_metadata.get("interactive_state") or {}).model_dump()
        for mask in annotations.mask_overrides:
            override_pixels(mask, session.active_session_dir)
        interactive_mask_files = validate_interactive_mask_references(session_metadata.get("interactive_mask_files", {}))
    except (KeyError, ValidationError, ValueError) as error:
        raise OperationError(400, f"Invalid session annotations or editor state: {error}") from error
    session.annotations = annotations
    session.interactive_mask_files = interactive_mask_files
    session.session_metadata = persisted_session_fields(session_metadata)
    restored = {
        "annotations": session.annotations.snapshot(),
        "interactive_state": interactive_state_projection(interactive_state, session=session),
        "has_mask_manifest": bool(session.mask_manifest_path),
    }
    session.video_dir = str(frames_dir)
    session.video_frame_files = frame_files
    tracking_result = restored_tracking_result_payload(session, session_dir, session_metadata)
    if tracking_result is not None:
        restored["tracking_result"] = tracking_result
    if not frame_files:
        raise OperationError(400, f"No image frames found under saved session frames directory: {frames_dir}")
    return restored


def import_frames(resolved_input_path: Path, session: SessionState, job: JobHandle) -> None:
    """Extract a video file, or link a folder of frames, into a new working copy of `session`."""
    session_dir = create_active_session(resolved_input_path, session=session)
    frames_dir = session_dir / "frames"
    if resolved_input_path.is_file():
        frame_files = _extract_video_frames(resolved_input_path, frames_dir, job)
        session.video_source_path = str(resolved_input_path)
    else:
        frame_files = _link_frame_directory(resolved_input_path, frames_dir, job)
        session.video_source_path = None
    session.mask_manifest_path = None
    session.video_dir = str(frames_dir)
    session.video_frame_files = frame_files
    if not frame_files:
        raise OperationError(400, f"No image frames found in directory: {resolved_input_path}")


def _extract_video_frames(video_path: Path, frames_dir: Path, job: JobHandle) -> list[str]:
    def on_extract_progress(current: int, total: Optional[int]) -> None:
        if total:
            progress = 0.35 + (0.3 * (current / total))
            message = f"Extracted {current} of {total} frames"
        else:
            progress = None
            message = f"Extracted {current} frames"
        job.enter_stage(
            "extracting_frames", "Extracting video frames",
            progress=progress,
            current=current,
            total=total,
            frame_idx=max(0, current - 1),
            message=message,
            append_history=current == 1 or (bool(total) and current == total),
        )

    try:
        return extract_video_to_session_frames(video_path, frames_dir, progress_callback=on_extract_progress)
    except ValueError as error:
        raise OperationError(400, str(error)) from error


def _link_frame_directory(source_dir: Path, frames_dir: Path, job: JobHandle) -> list[str]:
    candidate_count = len([
        path
        for path in source_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ])

    def on_link_progress(current: int, total: int, source_path: Path) -> None:
        job.enter_stage(
            "linking_frames", "Linking frame cache",
            progress=0.35 + (0.3 * (current / total)) if total else 0.65,
            current=current,
            total=total,
            frame_idx=max(0, current - 1),
            message=f"Linked {current} of {total} frames",
            append_history=current == 1 or current == total,
        )

    job.enter_stage(
        "linking_frames", "Linking frame cache",
        progress=0.35,
        current=0,
        total=candidate_count,
        message="Preparing session-local frame cache",
    )
    return copy_frames_directory_to_session(source_dir, frames_dir, progress_callback=on_link_progress)


def _initialize_predictor(
    session: SessionState, job: JobHandle, settings: MaskerSettings, *,
    async_loading_frames: bool, reuse: Optional[VideoMasker],
) -> None:
    if reuse is not None:
        # A different video replaces the previous one: free its frames before the new ones load.
        reuse.release_state()
        cleanup_cuda_memory()

    def on_progress(stage: str, label: str, progress: Optional[float], message: str) -> None:
        job.enter_stage(stage, label, progress=progress, message=message)

    try:
        create_masker_runtime(session).initialize(
            settings, async_loading_frames=async_loading_frames, progress_callback=on_progress, reuse=reuse)
    except ValueError as error:
        raise OperationError(400, str(error)) from error


def initialize_video_state_from_resolved_input(
    resolved_input_path: Path,
    *,
    online_mode: bool,
    batch_size: Optional[int],
    offload_video_to_cpu: Optional[bool],
    offload_state_to_cpu: Optional[bool],
    async_loading_frames: bool,
    session: SessionState,
    job: JobHandle,
    reuse: Optional[VideoMasker] = None,
):
    # Preparation may target a detached session; helpers must use that same session.
    saved_session_layout = resolve_saved_session_layout(resolved_input_path)
    if saved_session_layout is not None:
        source_type = "saved_session"
        restored_session_payload = open_saved_session(saved_session_layout[0], session, job)
    else:
        validate_video_input_path(resolved_input_path)
        source_type = "video_file" if resolved_input_path.is_file() else "frames_dir"
        restored_session_payload = None
        import_frames(resolved_input_path, session, job)

    _initialize_predictor(
        session, job,
        MaskerSettings(online_mode=online_mode, batch_size=batch_size,
                       offload_video_to_cpu=offload_video_to_cpu, offload_state_to_cpu=offload_state_to_cpu),
        async_loading_frames=async_loading_frames, reuse=reuse,
    )

    session.session_metadata["source_input_path"] = str(resolved_input_path)
    if restored_session_payload is None:
        session.annotations.create_object()
        session.session_metadata["created_at"] = utc_now_iso()
    else:
        session.session_metadata["interactive_state"] = restored_session_payload["interactive_state"]
    # A cloned checkpoint can carry files its metadata does not name.
    sweep_unreferenced_artifacts(session)
    # A freshly opened working copy matches its checkpoint, so re-save can copy only differences.
    session.checkpoint_artifacts = set(session.committed_artifacts) if restored_session_payload is not None else set()
    # A freshly loaded session has nothing to lose until its first edit, saved or not.
    session.saved_fingerprint = content_fingerprint(session)

    response_payload: dict[str, Any] = {
        "message": "Video state initialized successfully",
        "num_frames": len(session.video_frame_files),
        "resolved_video_frames_dir": session.video_dir,
        "source_video_path": session.video_source_path,
        "online_mode": session.video_masker.online_mode,
        "batch_size": session.video_masker.default_batch_size,
        "offload_video_to_cpu": session.video_masker.offload_video_to_cpu,
        "offload_state_to_cpu": session.video_masker.offload_state_to_cpu,
        "source_type": source_type,
    }
    if restored_session_payload is not None:
        response_payload["restored_session"] = restored_session_payload
    return InitResult(**response_payload).model_dump(exclude_unset=True)


def start_video_init(ctx: AppContext, request: VideoInitStateRequest):
    return ctx.jobs.start(
        operation="video_init",
        stage="resolving_input",
        stage_label="Resolving input",
        message="Video initialization queued",
        worker=lambda job: run_video_init_job(ctx, request, job),
    )

def run_video_init_job(ctx: AppContext, request: VideoInitStateRequest, job: JobHandle) -> dict[str, Any]:
    job.enter_stage(
        "resolving_input", "Resolving input",
        status="running",
        progress=0.05,
        message="Resolving video path",
    )
    resolved_input_path = resolve_input_path(request.video_frames_dir)
    previous = ctx.session
    previous_session_id = previous.active_session_id
    # The new video takes over the previous one's loaded weights and the previous frames are freed before the
    # new ones load (two videos are never in RAM together). The old session's accepted annotations stay
    # untouched until the replacement is published.
    reuse = previous.video_masker
    candidate = SessionState()
    try:
        result = initialize_video_state_from_resolved_input(
            resolved_input_path,
            online_mode=request.online_mode,
            batch_size=request.batch_size,
            offload_video_to_cpu=request.offload_video_to_cpu,
            offload_state_to_cpu=request.offload_state_to_cpu,
            async_loading_frames=request.async_loading_frames,
            session=candidate,
            job=job,
            reuse=reuse,
        )
        with ctx.lock:
            if ctx.session.active_session_id != previous_session_id:
                raise OperationError(409, "The active session changed while preparing its replacement.")
            previous_dir = ctx.session.active_session_dir
            ctx.sessions.replace(candidate)
    except BaseException:
        # The rejected candidate never became active; its working directory is unreachable.
        discard_working_session(candidate.active_session_dir)
        if candidate.video_masker is not None:
            candidate.video_masker.release_state()
        restore_previous_state(previous)
        raise
    # The replaced working copy is unreachable once the candidate is published.
    discard_working_session(previous_dir)
    return result


def restore_previous_state(session: SessionState) -> None:
    """After a failed load of another video, load the previous video's state back (best effort).

    The predictor only holds derived data. If it cannot be rebuilt the session keeps its accepted
    annotations and loses its predictor, and the video must be reloaded.
    """
    masker = session.video_masker
    if masker is None or masker.state_loaded or session.video_dir is None:
        return
    settings = MaskerSettings(
        online_mode=masker.online_mode, batch_size=masker.default_batch_size,
        offload_video_to_cpu=masker.offload_video_to_cpu, offload_state_to_cpu=masker.offload_state_to_cpu,
    )
    try:
        create_masker_runtime(session).restore(settings)
    except Exception:
        session.video_masker = None
        logger.exception("Could not rebuild the previous video's predictor state; the video must be reloaded.")
