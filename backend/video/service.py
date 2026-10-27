import logging, shutil
from pathlib import Path
from typing import Any, Optional
import cv2
from fastapi.responses import FileResponse
import numpy as np
import sam2_video_masker as svm
from core.config import IMAGE_EXTENSIONS, SAVED_ROOT, VIDEO_EXTENSIONS
from core.errors import OperationError
from core.jobs import queue_long_job, foreground_operation, foreground_lock, update_job, utc_now_iso
from core.state import state
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
from sessions.cache import bump_video_state_epoch, prepare_video_masker_for_video_init, reset_video_session_state
from schemas.video import InteractiveState
from sessions.metadata import current_masks_dir, current_session_path, interactive_state_projection, load_session_metadata, persisted_session_fields, sanitize_save_name, write_session_metadata
from sessions.paths import path_is_relative_to, resolve_input_path, resolve_saved_session_layout, validate_video_input_path
from utils import load_mask_manifest
from video.io import copy_frames_directory_to_session, create_active_session, extract_video_to_session_frames
from video.masks import mask_logits_to_2d_bool, store_mask_override, override_pixels, reclaim_override_artifacts, retain_interactive_masks, effective_mask_data
from video.manifest_cache import load_cached_mask_manifest
from video.mask_encoding import encode_live_mask_payload
from video.prompts import record_prompt_event, replay_annotation_prompts, prompt_runtime_edit
from tracking.results import restored_tracking_result_payload, tracking_validity

logger = logging.getLogger(__name__)

def initialize_video_state_from_resolved_input(
    resolved_input_path: Path,
    *,
    online_mode: bool,
    batch_size: Optional[int],
    offload_video_to_cpu: Optional[bool],
    offload_state_to_cpu: Optional[bool],
    async_loading_frames: bool,
):

    source_video_path = None
    source_type = "frames_dir"
    restored_session_payload: Optional[dict[str, Any]] = None
    saved_session_layout = resolve_saved_session_layout(resolved_input_path)

    if saved_session_layout is not None:
        session_dir, frames_dir, masks_dir = saved_session_layout
        session_metadata = load_session_metadata(session_dir)
        source_type = "saved_session"
        state.active_session_dir = session_dir
        state.active_session_saved_name = (
            str(session_metadata.get("saved_name")).strip()
            if session_metadata.get("saved_name")
            else session_dir.name
        )
        manifest_name = session_metadata.get("mask_manifest_file")
        manifest_path = masks_dir / Path(manifest_name).name if manifest_name else None
        state.mask_manifest_path = str(manifest_path) if manifest_path and manifest_path.exists() else None
        source_video_path = session_metadata.get("source_video_path")
        state.video_source_path = source_video_path
        indexed_frame_files = sorted(
            frame_path.name
            for frame_path in frames_dir.iterdir()
            if frame_path.is_file() and frame_path.suffix.lower() in IMAGE_EXTENSIONS
        )
        update_job(
            stage="linking_frames",
            stage_label="Loading saved session frames",
            progress=0.65,
            current=len(indexed_frame_files),
            total=len(indexed_frame_files),
            frame_idx=len(indexed_frame_files) - 1 if indexed_frame_files else None,
            message=f"Found {len(indexed_frame_files)} saved session frames",
        )
        try:
            session_id = session_metadata["session_id"]
            if not isinstance(session_id, str) or not session_id.strip():
                raise ValueError("Session identity must be a nonempty string.")
            annotations = SessionAnnotations.model_validate(session_metadata["annotations"])
            if any(event.frame_idx >= len(indexed_frame_files) for event in [*annotations.prompts, *annotations.mask_overrides]):
                raise ValueError("Prompt frame is outside the video.")
            interactive_state = InteractiveState.model_validate(
                session_metadata.get("interactive_state") or {}
            ).model_dump()
            for mask in annotations.mask_overrides:
                override_pixels(mask)
        except (KeyError, ValidationError, ValueError) as error:
            raise OperationError(400, f"Invalid session annotations or editor state: {error}") from error
        state.active_session_id = session_id
        state.annotations = annotations
        state.interactive_mask_files = session_metadata.get("interactive_mask_files", {})
        state.session_metadata = persisted_session_fields(session_metadata)
        state.committed_override_artifacts = {mask.artifact for mask in annotations.mask_overrides}
        # Orphans are artifacts from edits interrupted before or after their metadata commit.
        reclaim_override_artifacts(state.committed_override_artifacts)
        restored_session_payload = {
            "annotations": state.annotations.snapshot(),
            "interactive_state": interactive_state_projection(interactive_state),
            "has_mask_manifest": bool(state.mask_manifest_path),
        }
        state.video_frame_files = indexed_frame_files
        tracking_result = restored_tracking_result_payload(session_dir, session_metadata)
        if tracking_result is not None:
            restored_session_payload["tracking_result"] = tracking_result
    else:
        session_dir = create_active_session(resolved_input_path)
        frames_dir = session_dir / "frames"

        if resolved_input_path.is_file():
            source_type = "video_file"
            suffix = resolved_input_path.suffix.lower()
            if suffix in VIDEO_EXTENSIONS:
                def _on_extract_progress(current: int, total: Optional[int]) -> None:
                    if total:
                        progress = 0.35 + (0.3 * (current / total))
                        message = f"Extracted {current} of {total} frames"
                    else:
                        progress = None
                        message = f"Extracted {current} frames"
                    update_job(
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
                    indexed_frame_files = extract_video_to_session_frames(
                        resolved_input_path,
                        frames_dir,
                        progress_callback=_on_extract_progress,
                    )
                except ValueError as error:
                    raise OperationError(400, str(error)) from error
                except Exception as error:
                    raise OperationError(500, str(error)) from error
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
                raise OperationError(400, detail)
        else:
            source_type = "frames_dir"
            candidate_count = len([
                path
                for path in resolved_input_path.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            ])

            def _on_link_progress(current: int, total: int, source_path: Path) -> None:
                progress = 0.35 + (0.3 * (current / total)) if total else 0.65
                update_job(
                    stage="linking_frames",
                    stage_label="Linking frame cache",
                    progress=progress,
                    current=current,
                    total=total,
                    frame_idx=max(0, current - 1),
                    message=f"Linked {current} of {total} frames",
                    append_history=current == 1 or current == total,
                )

            update_job(
                stage="linking_frames",
                stage_label="Linking frame cache",
                progress=0.35,
                current=0,
                total=candidate_count,
                frame_idx=None,
                message="Preparing session-local frame cache",
            )
            indexed_frame_files = copy_frames_directory_to_session(
                resolved_input_path,
                frames_dir,
                progress_callback=_on_link_progress,
            )

    state.video_dir = str(frames_dir)
    state.video_frame_files = indexed_frame_files

    if not state.video_frame_files:
        if source_type == "saved_session":
            raise OperationError(400, f"No image frames found under saved session frames directory: {frames_dir}")
        raise OperationError(400, f"No image frames found in directory: {resolved_input_path}")

    if source_type != "saved_session":
        state.mask_manifest_path = None
        state.video_source_path = source_video_path

    if resolved_input_path.is_file():
        suffix = resolved_input_path.suffix.lower()
        if source_type != "video_file" and suffix in VIDEO_EXTENSIONS:
            source_type = "video_file"

    def _on_sam2_progress(stage: str, label: str, progress: Optional[float], message: str) -> None:
        update_job(
            stage=stage,
            stage_label=label,
            progress=progress,
            current=None,
            total=None,
            frame_idx=None,
            message=message,
        )

    if state.video_masker is None:
        state.video_masker = svm.SAM2VideoMasker(progress_callback=_on_sam2_progress)

    try:
        state.video_masker.init_state(
            state.video_dir,
            online_mode=online_mode,
            batch_size=batch_size,
            offload_video_to_cpu=offload_video_to_cpu,
            offload_state_to_cpu=offload_state_to_cpu,
            async_loading_frames=async_loading_frames,
            progress_callback=_on_sam2_progress,
        )
        replay_annotation_prompts()
    except ValueError as error:
        raise OperationError(400, str(error)) from error

    update_job(
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
            update_job(
                stage="indexing_frames",
                stage_label="Indexing video frames",
                progress=0.85 + (0.1 * (candidate_idx / total_candidates)) if total_candidates else 0.95,
                current=len(indexed_frame_files),
                total=total_candidates,
                frame_idx=len(indexed_frame_files) - 1 if indexed_frame_files else None,
                message=f"Indexed {len(indexed_frame_files)} of {total_candidates} frame files",
                append_history=candidate_idx == 1 or candidate_idx == total_candidates,
            )
    state_epoch = bump_video_state_epoch()
    metadata_updates = {
        "source_input_path": str(resolved_input_path),
    }
    if source_type != "saved_session":
        state.annotations.create_object()
        metadata_updates["created_at"] = utc_now_iso()
    else:
        metadata_updates["interactive_state"] = restored_session_payload["interactive_state"]
    write_session_metadata(metadata_updates)

    response_payload: dict[str, Any] = {
        "message": "Video state initialized successfully",
        "num_frames": len(state.video_frame_files),
        "resolved_video_frames_dir": state.video_dir,
        "source_video_path": source_video_path,
        "online_mode": state.video_masker.online_mode,
        "batch_size": state.video_masker.default_batch_size,
        "offload_video_to_cpu": state.video_masker.offload_video_to_cpu,
        "offload_state_to_cpu": state.video_masker.offload_state_to_cpu,
        "state_epoch": state_epoch,
        "source_type": source_type,
    }
    if restored_session_payload is not None:
        response_payload["restored_session"] = restored_session_payload
    return response_payload

async def start_video_init(request: VideoInitStateRequest):
    return queue_long_job(
        operation="video_init",
        stage="resolving_input",
        stage_label="Resolving input",
        message="Video initialization queued",
        worker=lambda: run_video_init_job(request),
    )

def run_video_init_job(request: VideoInitStateRequest) -> dict[str, Any]:
    update_job(
        status="running",
        stage="resolving_input",
        stage_label="Resolving input",
        progress=0.05,
        message="Resolving video path",
    )
    prepare_video_masker_for_video_init()
    resolved_input_path = resolve_input_path(request.video_frames_dir)
    validate_video_input_path(resolved_input_path)
    update_job(
        stage="preparing_session_cache",
        stage_label="Preparing session cache",
        progress=0.10,
        message="Creating active session cache",
    )

    def _on_sam2_progress(stage: str, label: str, progress: Optional[float], message: str) -> None:
        update_job(stage=stage, stage_label=label, progress=progress, message=message)

    state.video_masker = svm.SAM2VideoMasker(progress_callback=_on_sam2_progress)
    update_job(
        stage="model_ready",
        stage_label="SAM2 model ready",
        progress=0.35,
        message="SAM2 model loaded",
    )
    if resolved_input_path.is_file() and resolved_input_path.suffix.lower() in VIDEO_EXTENSIONS:
        update_job(
            stage="extracting_frames",
            stage_label="Extracting video frames",
            progress=0.35,
            message="Extracting video frames",
        )
    result = initialize_video_state_from_resolved_input(
        resolved_input_path,
        online_mode=request.online_mode,
        batch_size=request.batch_size,
        offload_video_to_cpu=request.offload_video_to_cpu,
        offload_state_to_cpu=request.offload_state_to_cpu,
        async_loading_frames=request.async_loading_frames,
    )
    update_job(
        stage="indexing_frames",
        stage_label="Indexing video frames",
        progress=0.95,
        current=int(result.get("num_frames", 0)),
        total=int(result.get("num_frames", 0)),
        message=f"Indexed {int(result.get('num_frames', 0))} frames",
    )
    return result

async def reset_video_state(precondition: AnnotationPrecondition):
    with foreground_operation():
        validate_annotation_precondition(precondition)
        if state.video_masker is None:
            raise OperationError(400, "Video masker not active.")
        with prompt_runtime_edit():
            state.video_masker.reset_state()
            reset_video_session_state()
            # Reset publishes empty references and invalidates input revisions.
            # Keep artifacts intact until acceptance; physical reclamation is separate.
            state.mask_manifest_path = None
            state_epoch = bump_video_state_epoch()
            write_session_metadata()
            return {
                **annotation_update(precondition),
                "message": "Video state reset successfully",
                "state_epoch": state_epoch,
            }

async def add_new_points_or_box(request: VideoAddPointsOrBoxRequest, precondition: AnnotationPrecondition):
    with foreground_operation():
        validate_annotation_precondition(precondition)
        if state.video_masker is None:
            raise OperationError(400, "Video masker not active.")

        if not state.video_frame_files:
            raise OperationError(400, "No video frames available. Call /video/init_state first.")
        if request.frame_idx < 0 or request.frame_idx >= len(state.video_frame_files):
            raise OperationError(
                400,
                f"Frame index out of bounds: {request.frame_idx}. Expected 0..{len(state.video_frame_files) - 1}.",
            )

        if request.obj_id not in state.annotations.objects:
            raise OperationError(404, "Prompt object does not exist.")

        with prompt_runtime_edit():
            if any(mask.frame_idx == request.frame_idx and mask.obj_id == request.obj_id
                   for mask in state.annotations.mask_overrides):
                state.video_masker.clear_all_prompts_in_frame(request.frame_idx, request.obj_id)
            out_frame_idx, out_obj_ids, out_mask_logits = state.video_masker.add_new_points_or_box(
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
                    or any(obj_id not in state.annotations.objects for obj_id in normalized_obj_ids)):
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
                    fallback_segments = state.video_masker.propagate_in_video(
                        start_frame_idx=int(request.frame_idx),
                        max_frame_num_to_track=1,
                        reverse=False,
                        batch_size=1,
                        online_mode=state.video_masker.online_mode,
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
            record_prompt_event(request)
            retain_interactive_masks(request.frame_idx, normalized_obj_ids, encoded_masks)
            mask_data = effective_mask_data(request.frame_idx)
            write_session_metadata()
            logger.info(
                "Interactive mask response frame=%s obj=%s pixels=%s fallback=%s",
                int(request.frame_idx),
                selected_obj_id,
                int(mask_pixel_counts.get(selected_obj_id, 0)),
                used_single_frame_fallback,
            )

            return {
                **annotation_update(precondition),
                "mask_data": mask_data,
                "request_frame_idx": int(request.frame_idx),
                "frame_idx": returned_frame_idx,
                "frame_file": state.video_frame_files[returned_frame_idx],
                "mask_pixel_counts": mask_pixel_counts,
                "mask_shapes": mask_shapes,
                "single_frame_fallback_used": used_single_frame_fallback,
                "state_epoch": int(state.video_state_epoch),
            }

async def add_new_mask(request: VideoAddMaskRequest, precondition: AnnotationPrecondition):
    with foreground_operation():
        validate_annotation_precondition(precondition)
        if state.video_masker is None:
            raise OperationError(400, "Video masker not active.")
        if request.obj_id not in state.annotations.objects or request.frame_idx >= len(state.video_frame_files):
            raise OperationError(404, "Mask object or frame does not exist.")
        mask = np.asarray(request.mask, dtype=bool)
        image = cv2.imread(str(Path(state.video_dir) / state.video_frame_files[request.frame_idx]))
        if image is None or mask.shape != image.shape[:2]:
            raise OperationError(400, "Mask dimensions must match the video frame.")
        with prompt_runtime_edit():
            state.video_masker.clear_all_prompts_in_frame(request.frame_idx, request.obj_id)
            state.video_masker.add_new_mask(frame_idx=request.frame_idx, obj_id=request.obj_id, mask=mask)
            state.annotations.set_mask(store_mask_override(request.frame_idx, request.obj_id, mask))
            mask_data = effective_mask_data(request.frame_idx)
            write_session_metadata()
            return {**annotation_update(precondition), "mask_data": mask_data}

async def save_video_session(request: VideoSaveRequest):
    with foreground_operation():

        session_path = current_session_path()
        if session_path is None or state.video_dir is None or not state.video_frame_files:
            raise OperationError(400, "Video session is not initialized.")

        save_name = sanitize_save_name(request.name)
        SAVED_ROOT.mkdir(parents=True, exist_ok=True)
        saved_path = (SAVED_ROOT / save_name).resolve()
        if saved_path.exists():
            raise OperationError(409, f"Saved session already exists: {save_name}")

        if path_is_relative_to(session_path, SAVED_ROOT):
            raise OperationError(409, "Current session is already saved.")

        interactive_state_payload = request.interactive_state.model_dump() if request.interactive_state else None
        saved_at = utc_now_iso()
        metadata_extra = {"saved_name": save_name, "saved_path": str(saved_path), "saved_at": saved_at}
        if interactive_state_payload is not None:
            metadata_extra["interactive_state"] = interactive_state_payload
        write_session_metadata(metadata_extra)

        session_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(session_path), str(saved_path))

        state.active_session_dir = saved_path
        state.active_session_saved_name = save_name
        state.video_dir = str(saved_path / "frames")
        manifest_path = saved_path / "masks" / Path(state.mask_manifest_path).name if state.mask_manifest_path else None
        state.mask_manifest_path = str(manifest_path) if manifest_path and manifest_path.exists() else None
        write_session_metadata(metadata_extra)

        return {
            "message": "Session saved successfully",
            "name": save_name,
            "saved_path": str(saved_path),
            "state_epoch": int(state.video_state_epoch),
        }

async def clear_all_prompts_in_frame(frame_idx: int, obj_id: int, precondition: AnnotationPrecondition):
    with foreground_operation():
        validate_annotation_precondition(precondition)
        if state.video_masker is None:
            raise OperationError(400, "Video masker not active.")
        if obj_id not in state.annotations.objects or not 0 <= frame_idx < len(state.video_frame_files):
            raise OperationError(404, "Prompt object or frame does not exist.")
        with prompt_runtime_edit():
            state.video_masker.clear_all_prompts_in_frame(frame_idx, obj_id)
            state.annotations.clear_prompts(frame_idx, obj_id)
            mask_data = effective_mask_data(frame_idx)
            write_session_metadata()
            return {**annotation_update(precondition), "mask_data": mask_data,
                    "message": "Cleared all prompts in frame successfully"}

async def remove_object(obj_id: int, precondition: AnnotationPrecondition):
    with foreground_operation():
        validate_annotation_precondition(precondition)
        if state.video_masker is None:
            raise OperationError(400, "Video masker not active.")
        state.video_masker.remove_object(obj_id)
        state.annotations.remove_object(obj_id)
        write_session_metadata()
        return annotation_update(precondition)

async def create_object(precondition: AnnotationPrecondition):
    with foreground_operation():
        validate_annotation_precondition(precondition)
        obj = state.annotations.create_object()
        write_session_metadata()
        return {**annotation_update(precondition), "created_object_id": obj.id}


async def update_object(obj_id: int, request: VideoObjectUpdate, precondition: AnnotationPrecondition):
    with foreground_operation():
        validate_annotation_precondition(precondition)
        if obj_id not in state.annotations.objects:
            raise OperationError(404, "Object does not exist.")
        obj = VideoObject(id=obj_id, **request.model_dump()).model_dump()
        state.annotations.upsert_object(obj)
        write_session_metadata()
        return annotation_update(precondition)


def validate_annotation_precondition(precondition: AnnotationPrecondition):
    if current_session_path() is None or precondition.session_id != state.active_session_id:
        raise OperationError(409, "The target session is no longer active. Refresh annotations.")
    if precondition.expected_revision != state.annotations.revision:
        raise OperationError(409, "Annotations changed. Refresh before editing again.")


def annotation_update(precondition: AnnotationPrecondition):
    """Mutation response: only accepted changes since the validated precondition revision."""
    return {
        "session_id": state.active_session_id,
        "state_epoch": int(state.video_state_epoch),
        "annotation_delta": state.annotations.take_delta(precondition.expected_revision),
        "tracking_validity": tracking_validity(),
    }


def annotation_snapshot():
    return {
        "session_id": state.active_session_id,
        "state_epoch": int(state.video_state_epoch),
        "annotations": state.annotations.snapshot() if current_session_path() is not None else None,
        "tracking_validity": tracking_validity(),
    }


async def get_video_info():
    with foreground_lock:
        return {
            **annotation_snapshot(),
            "predictor_ready": state.video_masker is not None,
            "num_frames": len(state.video_frame_files),
            "frame_files": list(state.video_frame_files),
        }

async def get_mask_manifest():
    with foreground_lock:
        if state.video_dir is None:
            raise OperationError(400, "Video not initialized")

        manifest_path = Path(state.mask_manifest_path) if state.mask_manifest_path else None
        if not manifest_path or not manifest_path.exists():
            raise OperationError(404, "Mask manifest not found. Run /video/propagate_in_video first.")

        manifest = load_cached_mask_manifest(manifest_path)
        return {
            "version": manifest.get("version"),
            "source_video_path": manifest.get("source_video_path"),
            "resolved_video_frames_dir": manifest.get("resolved_video_frames_dir"),
            "num_frames": manifest.get("num_frames", 0),
            "frame_height": manifest.get("frame_height"),
            "frame_width": manifest.get("frame_width"),
            "state.mask_manifest_path": str(manifest_path),
        }

async def get_mask_data(frame_idx: int):
    with foreground_lock:
        if state.video_dir is None:
            raise OperationError(400, "Video not initialized")
        if not 0 <= frame_idx < len(state.video_frame_files):
            raise OperationError(404, "Frame index out of bounds")
        return effective_mask_data(frame_idx)

async def get_video_frame(frame_idx: int):
    if state.video_dir is None or not state.video_frame_files:
        raise OperationError(400, "Video not initialized")

    if frame_idx < 0 or frame_idx >= len(state.video_frame_files):
        raise OperationError(404, "Frame index out of bounds")

    file_path = Path(state.video_dir) / state.video_frame_files[frame_idx]
    if not file_path.exists():
        raise OperationError(404, f"Frame file not found: {file_path}")
    return FileResponse(str(file_path))

async def get_video_mask_frame(frame_idx: int):
    with foreground_lock:
        if state.video_dir is None or not 0 <= frame_idx < len(state.video_frame_files):
            raise OperationError(404, "Frame index out of bounds")
        masks_dir = current_masks_dir()
        manifest_path = Path(state.mask_manifest_path) if state.mask_manifest_path else None
        if not manifest_path or not manifest_path.exists():
            raise OperationError(404, "Mask frame not available")
        frame = load_cached_mask_manifest(manifest_path).get("frames", {}).get(str(frame_idx), {})
        effective = effective_mask_data(frame_idx)
        if (not frame.get("overlay_path") or effective["objects"] != frame.get("objects") or
                any(source != "manifest" for source in effective["sources"].values())):
            raise OperationError(404, "Mask overlay is obsolete")
        file_path = (masks_dir / frame["overlay_path"]).resolve()
        if not path_is_relative_to(file_path, masks_dir.resolve()) or not file_path.is_file():
            raise OperationError(404, "Mask frame not available")
        return FileResponse(str(file_path))
