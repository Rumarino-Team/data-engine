from typing import Optional
from contextlib import contextmanager
import logging
from video.masks import override_pixels, discard_override_artifacts
from core.errors import OperationError
import sam2_video_masker as svm
from core.jobs import update_job
from core.state import BackendState, state
from sessions.cache import bump_video_state_epoch
from sessions.metadata import write_session_metadata
from schemas.video import VideoPrompt

def record_prompt_event(request: "VideoAddPointsOrBoxRequest"):
    event = VideoPrompt.model_validate({
        "frame_idx": int(request.frame_idx),
        "obj_id": int(request.obj_id),
        "points": [list(map(float, point)) for point in (request.points or [])],
        "labels": [int(label) for label in (request.labels or [])],
        "box": [float(v) for v in request.box] if request.box is not None else None,
    })
    state.annotations.set_prompt(event, clear_old_points=request.clear_old_points)


def replay_annotation_prompts(session: BackendState = state) -> None:
    for event in session.annotations.prompt_events():
        session.video_masker.add_new_points_or_box(
            frame_idx=event["frame_idx"], obj_id=event["obj_id"],
            points=event["points"] or None, labels=event["labels"] or None,
            clear_old_points=True, box=event["box"],
        )
    for mask in session.annotations.mask_overrides:
        session.video_masker.add_new_mask(frame_idx=mask.frame_idx, obj_id=mask.obj_id,
                                       mask=override_pixels(mask, session_path=session.active_session_dir))


def restore_video_masker_from_prompt_events(
    *,
    online_mode: bool,
    batch_size: Optional[int],
    offload_video_to_cpu: Optional[bool],
    offload_state_to_cpu: Optional[bool],
    increment_epoch: bool = True,
    progress_stage_override: Optional[str] = None,
    progress_label_override: Optional[str] = None,
    progress_message_prefix: Optional[str] = None,
) -> None:

    if state.video_dir is None:
        return

    def _on_progress(stage: str, label: str, progress: Optional[float], message: str) -> None:
        update_job(
            stage=progress_stage_override or stage,
            stage_label=progress_label_override or label,
            progress=progress,
            message=f"{progress_message_prefix}: {message}" if progress_message_prefix else message,
        )

    if state.video_masker is None:
        state.video_masker = svm.SAM2VideoMasker(progress_callback=_on_progress)

    state.video_masker.init_state(
        state.video_dir,
        online_mode=online_mode,
        batch_size=batch_size,
        offload_video_to_cpu=offload_video_to_cpu,
        offload_state_to_cpu=offload_state_to_cpu,
        async_loading_frames=False,
        progress_callback=_on_progress,
    )

    replay_annotation_prompts()

    if increment_epoch:
        bump_video_state_epoch()
        write_session_metadata()


@contextmanager
def annotation_edit(*, restore_runtime: bool = True):
    """Stage accepted inputs and restore them when an edit fails.

    Caller holds foreground_operation. Encode results before persisting acceptance.
    Set restore_runtime=False for changes that never touch the predictor.
    """
    # Frame entries are replaced, never mutated by foreground edits.
    previous_manifest = state.mask_manifest_path
    previous_masks = state.interactive_mask_files
    state.interactive_mask_files = dict(previous_masks)
    previous = state.annotations
    state.annotations = previous.model_copy(deep=True)
    try:
        yield
    except Exception:
        # Override artifacts published by the rejected edit were never committed.
        discard_override_artifacts({mask.artifact for mask in state.annotations.mask_overrides}
                                   - {mask.artifact for mask in previous.mask_overrides}
                                   - state.committed_override_artifacts)
        state.annotations = previous
        state.interactive_mask_files = previous_masks
        state.mask_manifest_path = previous_manifest
        if not restore_runtime:
            raise
        try:
            state.video_masker.reset_state()
            replay_annotation_prompts()
        except Exception as recovery_error:
            state.video_masker = None
            bump_video_state_epoch()
            logging.getLogger(__name__).exception("Prompt runtime recovery failed")
            raise OperationError(
                503, "The edit was rejected and the predictor could not be restored. Reload the video."
            ) from recovery_error
        bump_video_state_epoch()
        raise
