from typing import Optional
import sam2_video_masker as svm
from core.jobs import update_job
from core.state import state
from sessions.cache import bump_video_state_epoch
from sessions.metadata import write_session_metadata

def record_prompt_event(request: "VideoAddPointsOrBoxRequest"):

    if request.clear_old_points:
        state.video_prompt_events = [
            event
            for event in state.video_prompt_events
            if not (event["frame_idx"] == request.frame_idx and event["obj_id"] == request.obj_id)
        ]

    event = {
        "frame_idx": int(request.frame_idx),
        "obj_id": int(request.obj_id),
        "points": [list(map(float, point)) for point in (request.points or [])],
        "labels": [int(label) for label in (request.labels or [])],
        "box": [float(v) for v in request.box] if request.box is not None else None,
        "clear_old_points": bool(request.clear_old_points),
    }
    state.video_prompt_events.append(event)

def restore_video_masker_from_prompt_events(
    *,
    online_mode: bool,
    batch_size: Optional[int],
    offload_video_to_cpu: Optional[bool],
    offload_state_to_cpu: Optional[bool],
    increment_epoch: bool = True,
) -> None:

    if state.video_dir is None:
        return

    def _on_progress(stage: str, label: str, progress: Optional[float], message: str) -> None:
        update_job(stage=stage, stage_label=label, progress=progress, message=message)

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

    for event in state.video_prompt_events:
        points = event["points"] if event["points"] else None
        labels = event["labels"] if event["labels"] else None
        state.video_masker.add_new_points_or_box(
            frame_idx=int(event["frame_idx"]),
            obj_id=int(event["obj_id"]),
            points=points,
            labels=labels,
            clear_old_points=bool(event.get("clear_old_points", True)),
            box=event.get("box"),
        )

    if increment_epoch:
        bump_video_state_epoch()
        write_session_metadata()

