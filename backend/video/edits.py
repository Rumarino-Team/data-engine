import logging
from pathlib import Path
from typing import Any
import cv2
import numpy as np
from core.context import AppContext
from core.errors import OperationError
from core.state import SessionState
from schemas.video import (
    AnnotationPrecondition,
    VideoAddMaskRequest,
    VideoAddPointsOrBoxRequest,
    VideoObject,
    VideoObjectUpdate,
)
from sessions.cache import reset_video_session_state
from video.mask_encoding import encode_live_mask_payload
from video.masks import as_2d_bool, effective_mask_data, mask_logits_to_2d_bool, retain_interactive_masks, store_mask_override
from video.prompts import annotation_edit, record_prompt_event
from video.responses import annotation_update

logger = logging.getLogger(__name__)

def reset_video_state(ctx: AppContext, precondition: AnnotationPrecondition):
    with ctx.edit(precondition) as session:
        masker = session.require_masker()
        with annotation_edit(session):
            masker.reset_state()
            reset_video_session_state(session)
            # Reset publishes empty references and invalidates input revisions.
            # Keep artifacts intact until acceptance; physical reclamation is separate.
            session.mask_manifest_path = None
            return {
                **annotation_update(session, precondition),
                "message": "Video state reset successfully",
            }


def _checked_prediction(session: SessionState, request: VideoAddPointsOrBoxRequest,
                        frame_idx: Any, obj_ids: Any, mask_logits: Any) -> list[int]:
    """The object ids SAM2 answered for, after checking that the answer matches the request."""
    if int(frame_idx) != request.frame_idx:
        raise OperationError(
            409,
            f"Frame mismatch in SAM2 response: request_frame_idx={request.frame_idx} response_frame_idx={int(frame_idx)}",
        )
    answered = [int(obj_id) for obj_id in obj_ids]
    if (len(answered) != len(mask_logits)
            or len(set(answered)) != len(answered)
            or request.obj_id not in answered
            or any(obj_id not in session.annotations.objects for obj_id in answered)):
        raise OperationError(500, f"Invalid SAM2 response: {len(answered)} object IDs but {len(mask_logits)} mask tensors.")
    return answered


def _single_frame_mask(masker: Any, request: VideoAddPointsOrBoxRequest) -> np.ndarray | None:
    """Some clicks return an empty mask before memory preflight/consolidation; a 1-frame propagate pass can fill it."""
    segments = masker.propagate_in_video(
        start_frame_idx=request.frame_idx, max_frame_num_to_track=1, reverse=False,
        batch_size=1, online_mode=masker.online_mode, collect_segments=True,
    )
    mask = segments.get(request.frame_idx, {}).get(request.obj_id)
    mask_2d = as_2d_bool(mask) if mask is not None else None
    return mask_2d if mask_2d is not None and np.count_nonzero(mask_2d) > 0 else None


def add_new_points_or_box(ctx: AppContext, request: VideoAddPointsOrBoxRequest, precondition: AnnotationPrecondition):
    with ctx.edit(precondition) as session:
        masker = session.require_masker()
        session.require_frame(request.frame_idx)
        if request.obj_id not in session.annotations.objects:
            raise OperationError(404, "Prompt object does not exist.")

        with annotation_edit(session):
            if any(mask.frame_idx == request.frame_idx and mask.obj_id == request.obj_id
                   for mask in session.annotations.mask_overrides):
                masker.clear_all_prompts_in_frame(request.frame_idx, request.obj_id)
            frame_idx, obj_ids, mask_logits = masker.add_new_points_or_box(
                frame_idx=request.frame_idx,
                obj_id=request.obj_id,
                points=request.points,
                labels=request.labels,
                clear_old_points=request.clear_old_points,
                box=request.box,
            )
            obj_ids = _checked_prediction(session, request, frame_idx, obj_ids, mask_logits)
            masks = [mask_logits_to_2d_bool(logits) for logits in mask_logits]

            selected = obj_ids.index(request.obj_id)
            used_fallback = False
            if np.count_nonzero(masks[selected]) == 0 and any(label == 1 for label in request.labels or []):
                fallback = _single_frame_mask(masker, request)
                if fallback is not None:
                    masks[selected], used_fallback = fallback, True

            record_prompt_event(session, request)
            retain_interactive_masks(session, request.frame_idx, obj_ids, [encode_live_mask_payload(mask) for mask in masks])
            pixel_counts = {obj_id: int(np.count_nonzero(mask)) for obj_id, mask in zip(obj_ids, masks)}
            logger.info("Interactive mask response frame=%s obj=%s pixels=%s fallback=%s",
                        request.frame_idx, request.obj_id, pixel_counts[request.obj_id], used_fallback)

            return {
                **annotation_update(session, precondition),
                "mask_data": effective_mask_data(session, request.frame_idx),
                "request_frame_idx": request.frame_idx,
                "frame_idx": request.frame_idx,
                "frame_file": session.video_frame_files[request.frame_idx],
                "mask_pixel_counts": pixel_counts,
                "mask_shapes": {obj_id: [int(mask.shape[0]), int(mask.shape[1])] for obj_id, mask in zip(obj_ids, masks)},
                "single_frame_fallback_used": used_fallback,
            }


def _frame_shape(session: SessionState, frame_idx: int) -> tuple[int, ...] | None:
    """Height and width of a frame, read from the file; None when it cannot be read."""
    image = cv2.imread(str(Path(session.video_dir) / session.video_frame_files[frame_idx]))
    return None if image is None else image.shape[:2]


def add_new_mask(ctx: AppContext, request: VideoAddMaskRequest, precondition: AnnotationPrecondition):
    with ctx.edit(precondition) as session:
        masker = session.require_masker()
        session.require_frame(request.frame_idx)
        if request.obj_id not in session.annotations.objects:
            raise OperationError(404, "Mask object does not exist.")
        mask = np.asarray(request.mask, dtype=bool)
        if mask.shape != _frame_shape(session, request.frame_idx):
            raise OperationError(400, "Mask dimensions must match the video frame.")
        with annotation_edit(session):
            masker.clear_all_prompts_in_frame(request.frame_idx, request.obj_id)
            masker.add_new_mask(frame_idx=request.frame_idx, obj_id=request.obj_id, mask=mask)
            session.annotations.set_mask(store_mask_override(session.active_session_dir, request.frame_idx, request.obj_id, mask))
            mask_data = effective_mask_data(session, request.frame_idx)
            return {**annotation_update(session, precondition), "mask_data": mask_data}


def clear_all_prompts_in_frame(ctx: AppContext, frame_idx: int, obj_id: int, precondition: AnnotationPrecondition):
    with ctx.edit(precondition) as session:
        masker = session.require_masker()
        session.require_frame(frame_idx)
        if obj_id not in session.annotations.objects:
            raise OperationError(404, "Prompt object does not exist.")
        with annotation_edit(session):
            masker.clear_all_prompts_in_frame(frame_idx, obj_id)
            session.annotations.clear_prompts(frame_idx, obj_id)
            mask_data = effective_mask_data(session, frame_idx)
            return {**annotation_update(session, precondition), "mask_data": mask_data,
                    "message": "Cleared all prompts in frame successfully"}

def remove_object(ctx: AppContext, obj_id: int, precondition: AnnotationPrecondition):
    with ctx.edit(precondition) as session:
        masker = session.require_masker()
        if obj_id not in session.annotations.objects:
            raise OperationError(404, "Object does not exist.")
        with annotation_edit(session):
            masker.remove_object(obj_id)
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
