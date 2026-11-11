import logging
from pathlib import Path
from typing import Any, Optional
import cv2
from fastapi.responses import FileResponse
import numpy as np
from core.config import IMAGE_EXTENSIONS, SAVED_ROOT
from core.errors import OperationError
from core.context import AppContext
from core.jobs import JobHandle, utc_now_iso
from core.runtime import cleanup_cuda_memory
from core.state import BackendState
from core.annotations import SessionAnnotations
from pydantic import ValidationError
from schemas.video import (
    VideoAddMaskRequest,
    VideoAddPointsOrBoxRequest,
    VideoInitStateRequest,
    VideoObject,
    VideoObjectUpdate,
    AnnotationPrecondition,
    VideoSaveRequest,
)
from sessions.cache import reset_video_session_state
from schemas.responses import InitResult
from schemas.video import InteractiveState
from sessions.artifacts import referenced_artifacts, sweep_unreferenced_artifacts
from sessions.checkpoints import content_fingerprint, new_checkpoint_location, normalize_display_name, saved_state, write_checkpoint
from sessions.metadata import current_masks_dir, current_session_path, interactive_state_projection, load_session_metadata, persisted_session_fields, session_metadata_document
from sessions.paths import path_is_relative_to, resolve_input_path, resolve_saved_session_layout, validate_video_input_path
from video.io import copy_frames_directory_to_session, create_active_session, discard_working_session, extract_video_to_session_frames, open_checkpoint_as_working_copy
from video.masks import mask_logits_to_2d_bool, store_mask_override, override_pixels, retain_interactive_masks, effective_mask_data
from video.manifest_cache import load_cached_mask_manifest
from video.mask_encoding import encode_live_mask_payload
from video.prompts import record_prompt_event, annotation_edit, create_masker_runtime
from video.runtime import MaskerSettings
from tracking.results import restored_tracking_result_payload, tracking_validity

logger = logging.getLogger(__name__)

def open_saved_session(
    checkpoint_dir: Path, session: BackendState, job: JobHandle,
) -> tuple[Path, list[str], Optional[str], dict[str, Any]]:
    """Open a checkpoint as a new working copy and restore its accepted state into `session`.

    Returns the working frames directory, its frame files, the source video path and the restored payload.
    """
    session_metadata = load_session_metadata(checkpoint_dir)
    checkpoint_id, display_name = session_metadata.get("checkpoint_id"), session_metadata.get("display_name")
    if not all(isinstance(value, str) and value.strip() for value in (checkpoint_id, display_name)):
        raise OperationError(400, "Saved session is missing its checkpoint identity or display name.")
    # The checkpoint stays untouched; all edits and results belong to the working copy.
    session_dir = open_checkpoint_as_working_copy(checkpoint_dir, session=session)
    frames_dir, masks_dir = session_dir / "frames", session_dir / "masks"
    session.checkpoint_id, session.checkpoint_dir, session.display_name = checkpoint_id, checkpoint_dir, display_name
    manifest_name = session_metadata.get("mask_manifest_file")
    manifest_path = masks_dir / Path(manifest_name).name if manifest_name else None
    session.mask_manifest_path = str(manifest_path) if manifest_path and manifest_path.exists() else None
    source_video_path = session_metadata.get("source_video_path")
    session.video_source_path = source_video_path
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
    except (KeyError, ValidationError, ValueError) as error:
        raise OperationError(400, f"Invalid session annotations or editor state: {error}") from error
    session.annotations = annotations
    session.interactive_mask_files = session_metadata.get("interactive_mask_files", {})
    session.session_metadata = persisted_session_fields(session_metadata)
    restored = {
        "annotations": session.annotations.snapshot(),
        "interactive_state": interactive_state_projection(interactive_state, session=session),
        "has_mask_manifest": bool(session.mask_manifest_path),
    }
    session.video_frame_files = frame_files
    tracking_result = restored_tracking_result_payload(session, session_dir, session_metadata)
    if tracking_result is not None:
        restored["tracking_result"] = tracking_result
    return frames_dir, frame_files, source_video_path, restored


def import_frames(resolved_input_path: Path, session: BackendState, job: JobHandle) -> tuple[Path, list[str], Optional[str]]:
    """Extract a video file, or link a folder of frames, into a new working copy.

    Returns the working frames directory, its frame files and the source video path (None for a folder).
    """
    session_dir = create_active_session(resolved_input_path, session=session)
    frames_dir = session_dir / "frames"

    if resolved_input_path.is_file():
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
            frame_files = extract_video_to_session_frames(resolved_input_path, frames_dir, progress_callback=on_extract_progress)
        except ValueError as error:
            raise OperationError(400, str(error)) from error
        except Exception as error:
            raise OperationError(500, str(error)) from error
        return frames_dir, frame_files, str(resolved_input_path)

    candidate_count = len([
        path
        for path in resolved_input_path.iterdir()
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
    frame_files = copy_frames_directory_to_session(resolved_input_path, frames_dir, progress_callback=on_link_progress)
    return frames_dir, frame_files, None


def initialize_video_state_from_resolved_input(
    resolved_input_path: Path,
    *,
    online_mode: bool,
    batch_size: Optional[int],
    offload_video_to_cpu: Optional[bool],
    offload_state_to_cpu: Optional[bool],
    async_loading_frames: bool,
    session: BackendState,
    job: JobHandle,
    reuse=None,
):
    # Preparation may target a detached session; helpers must use that same session.
    restored_session_payload: Optional[dict[str, Any]] = None
    saved_session_layout = resolve_saved_session_layout(resolved_input_path)
    if saved_session_layout is not None:
        source_type = "saved_session"
        frames_dir, frame_files, source_video_path, restored_session_payload = open_saved_session(
            saved_session_layout[0], session, job)
    else:
        validate_video_input_path(resolved_input_path)
        source_type = "video_file" if resolved_input_path.is_file() else "frames_dir"
        frames_dir, frame_files, source_video_path = import_frames(resolved_input_path, session, job)

    session.video_dir = str(frames_dir)
    session.video_frame_files = frame_files

    if not session.video_frame_files:
        if source_type == "saved_session":
            raise OperationError(400, f"No image frames found under saved session frames directory: {frames_dir}")
        raise OperationError(400, f"No image frames found in directory: {resolved_input_path}")

    if source_type != "saved_session":
        session.mask_manifest_path = None
        session.video_source_path = source_video_path

    def _on_sam2_progress(stage: str, label: str, progress: Optional[float], message: str) -> None:
        job.enter_stage(
            stage, label,
            progress=progress,
            message=message,
        )

    if reuse is not None:
        # A different video replaces the previous one: free its frames before the new ones load.
        reuse.release_state()
        cleanup_cuda_memory()
    try:
        create_masker_runtime(session).initialize(
            MaskerSettings(
                online_mode=online_mode, batch_size=batch_size,
                offload_video_to_cpu=offload_video_to_cpu, offload_state_to_cpu=offload_state_to_cpu,
            ),
            async_loading_frames=async_loading_frames,
            progress_callback=_on_sam2_progress,
            reuse=reuse,
        )
    except ValueError as error:
        raise OperationError(400, str(error)) from error

    metadata_updates = {
        "source_input_path": str(resolved_input_path),
    }
    if source_type != "saved_session":
        session.annotations.create_object()
        metadata_updates["created_at"] = utc_now_iso()
    else:
        metadata_updates["interactive_state"] = restored_session_payload["interactive_state"]

    response_payload: dict[str, Any] = {
        "message": "Video state initialized successfully",
        "num_frames": len(session.video_frame_files),
        "resolved_video_frames_dir": session.video_dir,
        "source_video_path": source_video_path,
        "online_mode": session.video_masker.online_mode,
        "batch_size": session.video_masker.default_batch_size,
        "offload_video_to_cpu": session.video_masker.offload_video_to_cpu,
        "offload_state_to_cpu": session.video_masker.offload_state_to_cpu,
        "source_type": source_type,
    }
    if restored_session_payload is not None:
        response_payload["restored_session"] = restored_session_payload
    session.session_metadata.update(metadata_updates)
    # A cloned checkpoint can carry files its metadata does not name.
    sweep_unreferenced_artifacts(session)
    # A freshly opened working copy matches its checkpoint, so re-save can copy only differences.
    session.checkpoint_artifacts = set(session.committed_artifacts) if source_type == "saved_session" else set()
    # A freshly loaded session has nothing to lose until its first edit, saved or not.
    session.saved_fingerprint = content_fingerprint(session)
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
    validate_video_input_path(resolved_input_path)
    previous = ctx.session
    previous_session_id = previous.active_session_id
    # The new video takes over the previous one's loaded weights and the previous frames are freed before the
    # new ones load (two videos are never in RAM together). The old session's accepted annotations stay
    # untouched until the replacement is published.
    reuse = previous.video_masker
    candidate = BackendState()
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


def restore_previous_state(session: BackendState) -> None:
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


def reset_video_state(ctx: AppContext, precondition: AnnotationPrecondition):
    with ctx.edit(precondition) as session:
        if session.video_masker is None:
            raise OperationError(400, "Video masker not active.")
        with annotation_edit(session):
            session.video_masker.reset_state()
            reset_video_session_state(session)
            # Reset publishes empty references and invalidates input revisions.
            # Keep artifacts intact until acceptance; physical reclamation is separate.
            session.mask_manifest_path = None
            return {
                **annotation_update(session, precondition),
                "message": "Video state reset successfully",
            }

def add_new_points_or_box(ctx: AppContext, request: VideoAddPointsOrBoxRequest, precondition: AnnotationPrecondition):
    with ctx.edit(precondition) as session:
        if session.video_masker is None:
            raise OperationError(400, "Video masker not active.")

        if not session.video_frame_files:
            raise OperationError(400, "No video frames available. Call /video/init_state first.")
        if request.frame_idx < 0 or request.frame_idx >= len(session.video_frame_files):
            raise OperationError(
                400,
                f"Frame index out of bounds: {request.frame_idx}. Expected 0..{len(session.video_frame_files) - 1}.",
            )

        if request.obj_id not in session.annotations.objects:
            raise OperationError(404, "Prompt object does not exist.")

        with annotation_edit(session):
            if any(mask.frame_idx == request.frame_idx and mask.obj_id == request.obj_id
                   for mask in session.annotations.mask_overrides):
                session.video_masker.clear_all_prompts_in_frame(request.frame_idx, request.obj_id)
            out_frame_idx, out_obj_ids, out_mask_logits = session.video_masker.add_new_points_or_box(
                frame_idx=request.frame_idx,
                obj_id=request.obj_id,
                points=request.points,
                labels=request.labels,
                clear_old_points=request.clear_old_points,
                box=request.box
            )
            returned_frame_idx = int(out_frame_idx)
            if returned_frame_idx != int(request.frame_idx):
                raise OperationError(
                    409,
                    "Frame mismatch in SAM2 response: "
                    f"request_frame_idx={int(request.frame_idx)} response_frame_idx={returned_frame_idx}",
                )

            normalized_obj_ids = [int(obj_id) for obj_id in out_obj_ids]
            if (len(normalized_obj_ids) != len(out_mask_logits)
                    or len(set(normalized_obj_ids)) != len(normalized_obj_ids)
                    or request.obj_id not in normalized_obj_ids
                    or any(obj_id not in session.annotations.objects for obj_id in normalized_obj_ids)):
                raise OperationError(
                    500,
                    "Invalid SAM2 response: "
                    f"{len(normalized_obj_ids)} object IDs but {len(out_mask_logits)} mask tensors.",
                )

            masks_list: list[np.ndarray] = []
            mask_pixel_counts: dict[int, int] = {}
            mask_shapes: dict[int, list[int]] = {}
            for index, obj_id in enumerate(normalized_obj_ids):
                mask_2d = mask_logits_to_2d_bool(out_mask_logits[index])
                masks_list.append(mask_2d)
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
                    fallback_segments = session.video_masker.propagate_in_video(
                        start_frame_idx=int(request.frame_idx),
                        max_frame_num_to_track=1,
                        reverse=False,
                        batch_size=1,
                        online_mode=session.video_masker.online_mode,
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
                                masks_list[selected_obj_index] = fallback_mask_2d
                                mask_pixel_counts[selected_obj_id] = fallback_pixels
                                mask_shapes[selected_obj_id] = [int(fallback_mask_2d.shape[0]), int(fallback_mask_2d.shape[1])]
                                used_single_frame_fallback = True
                except Exception:
                    logger.exception(
                        "Single-frame interactive fallback failed for frame=%s obj=%s",
                        int(request.frame_idx),
                        selected_obj_id,
                    )
                    raise

            encoded_masks = [encode_live_mask_payload(mask) for mask in masks_list]
            record_prompt_event(session, request)
            retain_interactive_masks(session, request.frame_idx, normalized_obj_ids, encoded_masks)
            mask_data = effective_mask_data(session, request.frame_idx)
            logger.info(
                "Interactive mask response frame=%s obj=%s pixels=%s fallback=%s",
                int(request.frame_idx),
                selected_obj_id,
                int(mask_pixel_counts.get(selected_obj_id, 0)),
                used_single_frame_fallback,
            )

            return {
                **annotation_update(session, precondition),
                "mask_data": mask_data,
                "request_frame_idx": int(request.frame_idx),
                "frame_idx": returned_frame_idx,
                "frame_file": session.video_frame_files[returned_frame_idx],
                "mask_pixel_counts": mask_pixel_counts,
                "mask_shapes": mask_shapes,
                "single_frame_fallback_used": used_single_frame_fallback,
            }

def add_new_mask(ctx: AppContext, request: VideoAddMaskRequest, precondition: AnnotationPrecondition):
    with ctx.edit(precondition) as session:
        if session.video_masker is None:
            raise OperationError(400, "Video masker not active.")
        if request.obj_id not in session.annotations.objects or request.frame_idx >= len(session.video_frame_files):
            raise OperationError(404, "Mask object or frame does not exist.")
        mask = np.asarray(request.mask, dtype=bool)
        image = cv2.imread(str(Path(session.video_dir) / session.video_frame_files[request.frame_idx]))
        if image is None or mask.shape != image.shape[:2]:
            raise OperationError(400, "Mask dimensions must match the video frame.")
        with annotation_edit(session):
            session.video_masker.clear_all_prompts_in_frame(request.frame_idx, request.obj_id)
            session.video_masker.add_new_mask(frame_idx=request.frame_idx, obj_id=request.obj_id, mask=mask)
            session.annotations.set_mask(store_mask_override(session.active_session_dir, request.frame_idx, request.obj_id, mask))
            mask_data = effective_mask_data(session, request.frame_idx)
            return {**annotation_update(session, precondition), "mask_data": mask_data}

def save_video_session(ctx: AppContext, request: VideoSaveRequest, precondition: AnnotationPrecondition):
    # Save persists the backend's current state, so it must be the state this client last saw.
    with ctx.edit(precondition):
        session = ctx.session

        session_path = current_session_path(session)
        if session_path is None or session.video_dir is None or not session.video_frame_files:
            raise OperationError(400, "Video session is not initialized.")

        # Re-save replaces this session's own checkpoint; Save As (or a first save) mints a new identity.
        resave = session.checkpoint_id is not None and not request.save_as
        if request.name is None and not resave:
            raise OperationError(400, "A session name is required for the first save and for Save As.")
        display_name = normalize_display_name(request.name) if request.name is not None else session.display_name
        if resave:
            checkpoint_id, target = session.checkpoint_id, session.checkpoint_dir
        else:
            SAVED_ROOT.mkdir(parents=True, exist_ok=True)
            checkpoint_id, target = new_checkpoint_location(display_name, SAVED_ROOT)

        saved_at = utc_now_iso()
        overrides: dict[str, Any] = {
            "checkpoint_id": checkpoint_id, "display_name": display_name, "saved_at": saved_at,
            "saved_revision": session.annotations.revision,
            "resolved_video_frames_dir": str(target / "frames"),
        }
        if request.interactive_state is not None:
            overrides["interactive_state"] = request.interactive_state.model_dump()
        references = referenced_artifacts(session)
        if references is None:
            raise OperationError(500, "The session's mask results cannot be read, so it cannot be saved.")
        target.parent.mkdir(parents=True, exist_ok=True)
        write_checkpoint(session_path, target, session_metadata_document(session, overrides), references,
                         session.checkpoint_artifacts if resave else None)

        # The checkpoint is published; the working copy adopts its identity.
        session.checkpoint_id, session.checkpoint_dir, session.display_name = checkpoint_id, target, display_name
        session.checkpoint_artifacts = references
        session.saved_fingerprint = content_fingerprint(session)
        session.session_metadata["saved_at"] = saved_at
        if "interactive_state" in overrides:
            session.session_metadata["interactive_state"] = overrides["interactive_state"]
        return {
            "message": "Session saved successfully",
            "name": display_name,
            "checkpoint_id": checkpoint_id,
            "saved_path": str(target),
            "saved_state": saved_state(session),
        }

def clear_all_prompts_in_frame(ctx: AppContext, frame_idx: int, obj_id: int, precondition: AnnotationPrecondition):
    with ctx.edit(precondition) as session:
        if session.video_masker is None:
            raise OperationError(400, "Video masker not active.")
        if obj_id not in session.annotations.objects or not 0 <= frame_idx < len(session.video_frame_files):
            raise OperationError(404, "Prompt object or frame does not exist.")
        with annotation_edit(session):
            session.video_masker.clear_all_prompts_in_frame(frame_idx, obj_id)
            session.annotations.clear_prompts(frame_idx, obj_id)
            mask_data = effective_mask_data(session, frame_idx)
            return {**annotation_update(session, precondition), "mask_data": mask_data,
                    "message": "Cleared all prompts in frame successfully"}

def remove_object(ctx: AppContext, obj_id: int, precondition: AnnotationPrecondition):
    with ctx.edit(precondition) as session:
        if session.video_masker is None:
            raise OperationError(400, "Video masker not active.")
        if obj_id not in session.annotations.objects:
            raise OperationError(404, "Object does not exist.")
        with annotation_edit(session):
            session.video_masker.remove_object(obj_id)
            session.annotations.remove_object(obj_id)
        return annotation_update(session, precondition)

def create_object(ctx: AppContext, precondition: AnnotationPrecondition):
    with ctx.edit(precondition) as session:
        with annotation_edit(session, restore_runtime=False):
            obj = session.annotations.create_object()
        return {**annotation_update(session, precondition), "created_object_id": obj.id}


def update_object(ctx: AppContext, obj_id: int, request: VideoObjectUpdate, precondition: AnnotationPrecondition):
    with ctx.edit(precondition) as session:
        if obj_id not in session.annotations.objects:
            raise OperationError(404, "Object does not exist.")
        obj = VideoObject(id=obj_id, **request.model_dump()).model_dump()
        with annotation_edit(session, restore_runtime=False):
            session.annotations.upsert_object(obj)
        return annotation_update(session, precondition)


def annotation_update(session: BackendState, precondition: AnnotationPrecondition):
    """Mutation response: only accepted changes since the validated precondition revision."""
    return {
        "session_id": session.active_session_id,
        "annotation_delta": session.annotations.take_delta(precondition.expected_revision),
        "tracking_validity": tracking_validity(session),
        "saved_state": saved_state(session),
    }


def annotation_snapshot(session: BackendState):
    return {
        "session_id": session.active_session_id,
        "annotations": session.annotations.snapshot() if current_session_path(session) is not None else None,
        "tracking_validity": tracking_validity(session),
        "saved_state": saved_state(session) if current_session_path(session) is not None else None,
    }


def get_video_info(ctx: AppContext):
    with ctx.lock:
        session = ctx.session
        return {
            **annotation_snapshot(session),
            "predictor_ready": session.video_masker is not None and bool(session.video_masker.state_loaded),
            "num_frames": len(session.video_frame_files),
            "frame_files": list(session.video_frame_files),
        }

def get_mask_data(ctx: AppContext, frame_idx: int):
    with ctx.lock:
        session = ctx.session
        if session.video_dir is None:
            raise OperationError(400, "Video not initialized")
        if not 0 <= frame_idx < len(session.video_frame_files):
            raise OperationError(404, "Frame index out of bounds")
        return effective_mask_data(session, frame_idx)

def get_video_frame(ctx: AppContext, frame_idx: int):
    with ctx.lock:
        session = ctx.session
        if session.video_dir is None or not session.video_frame_files:
            raise OperationError(400, "Video not initialized")

        if frame_idx < 0 or frame_idx >= len(session.video_frame_files):
            raise OperationError(404, "Frame index out of bounds")

        file_path = Path(session.video_dir) / session.video_frame_files[frame_idx]
        if not file_path.exists():
            raise OperationError(404, f"Frame file not found: {file_path}")
        return FileResponse(str(file_path))

def get_video_mask_frame(ctx: AppContext, frame_idx: int):
    with ctx.lock:
        session = ctx.session
        if session.video_dir is None or not 0 <= frame_idx < len(session.video_frame_files):
            raise OperationError(404, "Frame index out of bounds")
        masks_dir = current_masks_dir(session)
        manifest_path = Path(session.mask_manifest_path) if session.mask_manifest_path else None
        if not manifest_path or not manifest_path.exists():
            raise OperationError(404, "Mask frame not available")
        frame = load_cached_mask_manifest(manifest_path).get("frames", {}).get(str(frame_idx), {})
        effective = effective_mask_data(session, frame_idx)
        if (not frame.get("overlay_path") or effective["objects"] != frame.get("objects") or
                any(source != "manifest" for source in effective["sources"].values())):
            raise OperationError(404, "Mask overlay is obsolete")
        file_path = (masks_dir / frame["overlay_path"]).resolve()
        if not path_is_relative_to(file_path, masks_dir.resolve()) or not file_path.is_file():
            raise OperationError(404, "Mask frame not available")
        return FileResponse(str(file_path))
