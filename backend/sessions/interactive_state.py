from typing import Any, Optional
from pydantic import ValidationError
from core.config import DEFAULT_MAX_INTERACTIVE_LIVE_MASKS
from schemas.video import InteractiveMaskRLE, InteractiveObject, InteractivePoint, InteractiveState

def sanitize_interactive_state(interactive_state: Optional[InteractiveState]) -> Optional[dict[str, Any]]:
    if interactive_state is None:
        return None
    payload = interactive_state.model_dump()
    payload["version"] = int(payload.get("version", 1))
    return payload

def validate_interactive_state_for_restore(raw_state: Any) -> tuple[Optional[dict[str, Any]], list[str]]:
    if raw_state is None:
        return None, []
    if not isinstance(raw_state, dict):
        return None, ["interactive_state was not a JSON object and was ignored."]

    try:
        validated = InteractiveState.model_validate(raw_state)
        return sanitize_interactive_state(validated), []
    except ValidationError as error:
        first_error = error.errors()[0] if error.errors() else {}
        warnings: list[str] = [
            f"interactive_state had validation issues: {first_error.get('msg', 'invalid payload')}"
        ]

    objects: list[dict[str, Any]] = []
    for index, raw_obj in enumerate(raw_state.get("objects", [])):
        try:
            objects.append(InteractiveObject.model_validate(raw_obj).model_dump())
        except ValidationError:
            warnings.append(f"Dropped invalid interactive_state.objects[{index}].")

    points: list[dict[str, Any]] = []
    for index, raw_point in enumerate(raw_state.get("points", [])):
        try:
            points.append(InteractivePoint.model_validate(raw_point).model_dump())
        except ValidationError:
            warnings.append(f"Dropped invalid interactive_state.points[{index}].")

    live_masks: list[dict[str, Any]] = []
    for index, raw_mask in enumerate(raw_state.get("live_masks", [])):
        try:
            live_masks.append(InteractiveMaskRLE.model_validate(raw_mask).model_dump())
        except ValidationError:
            warnings.append(f"Dropped invalid interactive_state.live_masks[{index}].")
        if len(live_masks) >= DEFAULT_MAX_INTERACTIVE_LIVE_MASKS:
            warnings.append(
                f"Truncated interactive_state.live_masks to {DEFAULT_MAX_INTERACTIVE_LIVE_MASKS} entries."
            )
            break

    interaction_mode_value = raw_state.get("interaction_mode")
    interaction_mode = str(interaction_mode_value) if interaction_mode_value in {"positive", "negative"} else None
    if interaction_mode is None and interaction_mode_value is not None:
        warnings.append("Dropped invalid interactive_state.interaction_mode.")

    selected_object_id_value = raw_state.get("selected_object_id")
    selected_object_id = (
        int(selected_object_id_value)
        if isinstance(selected_object_id_value, int) and selected_object_id_value > 0
        else None
    )
    if selected_object_id is None and selected_object_id_value is not None:
        warnings.append("Dropped invalid interactive_state.selected_object_id.")

    current_frame_idx_value = raw_state.get("current_frame_idx")
    current_frame_idx = (
        int(current_frame_idx_value)
        if isinstance(current_frame_idx_value, int) and current_frame_idx_value >= 0
        else None
    )
    if current_frame_idx is None and current_frame_idx_value is not None:
        warnings.append("Dropped invalid interactive_state.current_frame_idx.")

    sanitized = {
        "version": 1,
        "objects": objects,
        "selected_object_id": selected_object_id,
        "interaction_mode": interaction_mode,
        "current_frame_idx": current_frame_idx,
        "points": points,
        "live_masks": live_masks,
    }
    try:
        validated = InteractiveState.model_validate(sanitized)
        return sanitize_interactive_state(validated), warnings
    except ValidationError:
        warnings.append("interactive_state could not be restored and was ignored.")
        return None, warnings

def prompt_events_from_interactive_state(interactive_state: Optional[dict[str, Any]]) -> list[dict[str, Any]]:
    if not interactive_state:
        return []

    grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for point in interactive_state.get("points", []):
        frame_idx = int(point["frame_idx"])
        obj_id = int(point["obj_id"])
        grouped.setdefault((frame_idx, obj_id), []).append(point)

    events: list[dict[str, Any]] = []
    for (frame_idx, obj_id), points in sorted(grouped.items()):
        events.append(
            {
                "frame_idx": frame_idx,
                "obj_id": obj_id,
                "points": [[float(point["x"]), float(point["y"])] for point in points],
                "labels": [int(point["label"]) for point in points],
                "box": None,
                "clear_old_points": True,
            }
        )
    return events

