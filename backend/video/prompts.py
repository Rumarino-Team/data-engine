from contextlib import contextmanager
from sessions.artifacts import reclaim_superseded_artifacts, sweep_unreferenced_artifacts
from video.masks import override_pixels
from core.errors import OperationError
import sam2_video_masker as svm
from core.runtime import cleanup_cuda_memory
from core.state import SessionState
from schemas.video import VideoPrompt
from video.runtime import MaskerRestorationError, VideoMaskerRuntime

def record_prompt_event(session: SessionState, request: "VideoAddPointsOrBoxRequest"):
    event = VideoPrompt.model_validate({
        "frame_idx": int(request.frame_idx),
        "obj_id": int(request.obj_id),
        "points": [list(map(float, point)) for point in (request.points or [])],
        "labels": [int(label) for label in (request.labels or [])],
        "box": [float(v) for v in request.box] if request.box is not None else None,
    })
    session.annotations.set_prompt(event, clear_old_points=request.clear_old_points)


def replay_annotation_prompts(session: SessionState) -> None:
    for event in session.annotations.prompt_events():
        session.video_masker.add_new_points_or_box(
            frame_idx=event["frame_idx"], obj_id=event["obj_id"],
            points=event["points"] or None, labels=event["labels"] or None,
            clear_old_points=True, box=event["box"],
        )
    for mask in session.annotations.mask_overrides:
        session.video_masker.add_new_mask(frame_idx=mask.frame_idx, obj_id=mask.obj_id,
                                       mask=override_pixels(mask, session.active_session_dir))


def create_masker_runtime(session: SessionState) -> VideoMaskerRuntime:
    return VideoMaskerRuntime(
        session=session,
        create_masker=svm.SAM2VideoMasker,
        replay_inputs=replay_annotation_prompts,
        release_memory=cleanup_cuda_memory,
    )


@contextmanager
def annotation_edit(session: SessionState, *, restore_runtime: bool = True):
    """Stage accepted inputs and restore them when an edit fails.

    Caller holds `ctx.edit()`. The edit is accepted when the block exits without error, and files its
    new state no longer names are deleted then.
    Set restore_runtime=False for changes that never touch the predictor.
    """
    # Frame entries are replaced, never mutated by foreground edits.
    previous_manifest = session.mask_manifest_path
    previous_masks = session.interactive_mask_files
    session.interactive_mask_files = dict(previous_masks)
    previous = session.annotations
    session.annotations = previous.model_copy(deep=True)
    try:
        yield
    except Exception:
        session.annotations = previous
        session.interactive_mask_files = previous_masks
        session.mask_manifest_path = previous_manifest
        # Artifacts published by the rejected edit were never committed.
        sweep_unreferenced_artifacts(session)
        if not restore_runtime:
            raise
        try:
            create_masker_runtime(session).rebuild_inputs()
        except MaskerRestorationError as recovery_error:
            raise OperationError(
                503, "The edit was rejected and the predictor could not be restored. Reload the video."
            ) from recovery_error
        raise
    else:
        # Files become garbage only after the accepted state stops naming them.
        reclaim_superseded_artifacts(session)
