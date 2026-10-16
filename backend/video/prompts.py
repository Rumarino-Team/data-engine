from typing import Optional
import sam2_video_masker as svm
from core.jobs import update_job
from core.state import state
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
        "clear_old_points": bool(request.clear_old_points),
    })
    state.annotations.set_prompt(event)
    write_session_metadata()


def replay_annotation_prompts() -> None:
    for event in state.annotations.prompt_events():
        state.video_masker.add_new_points_or_box(
            frame_idx=event["frame_idx"], obj_id=event["obj_id"],
            points=event["points"] or None, labels=event["labels"] or None,
            clear_old_points=event["clear_old_points"], box=event["box"],
        )

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
