import re
import uuid
from pathlib import Path
from typing import Any

from core.errors import OperationError
from core.state import state
from core.jobs import utc_now_iso
from sessions.metadata import current_session_path, write_session_metadata
from tracking.guidance import tracked_prompt_keys, positive_prompt_keys
from utils import load_mask_manifest, write_mask_manifest

_RESULT_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


def _validate_result_id(result_id: str) -> str:
    normalized = str(result_id).strip()
    if not normalized or not _RESULT_ID_PATTERN.fullmatch(normalized):
        raise OperationError(400, "Invalid tracking result id.")
    return normalized


def _tracking_dir(session_path: Path) -> Path:
    return session_path / "tracking"


def _tracking_result_path(session_path: Path, result_id: str) -> Path:
    return _tracking_dir(session_path) / f"{result_id}.json"


def save_prompt_tracking_result(
    *,
    model_name: str,
    num_points: int,
    num_frames: int,
    add_support_grid_used: bool,
    tracking_mode: str,
    streaming_frame_threshold: int,
    points: list[dict[str, Any]],
    tracks: list[Any],
    visibility: list[Any],
) -> dict[str, Any]:
    session_path = current_session_path()
    if session_path is None:
        raise OperationError(400, "Video session cache is not initialized.")

    result_id = uuid.uuid4().hex
    created_at = utc_now_iso()
    payload = {
        "version": 3,
        "session_id": state.active_session_id,
        "input_revision": state.annotations.revision,
        "result_id": result_id,
        "created_at": created_at,
        "state_epoch": int(state.video_state_epoch),
        "model_name": model_name,
        "num_points": int(num_points),
        "num_frames": int(num_frames),
        "add_support_grid_used": bool(add_support_grid_used),
        "tracking_mode": tracking_mode,
        "streaming_frame_threshold": int(streaming_frame_threshold),
        "tracked_prompt_keys": tracked_prompt_keys(points),
        "points": points,
        "tracks": tracks,
        "visibility": visibility,
    }

    result_path = _tracking_result_path(session_path, result_id)
    write_mask_manifest(result_path, payload)

    summary = {
        "points": points,
        "session_id": state.active_session_id,
        "model_name": model_name,
        "num_points": int(num_points),
        "num_frames": int(num_frames),
        "add_support_grid_used": bool(add_support_grid_used),
        "tracking_mode": tracking_mode,
        "streaming_frame_threshold": int(streaming_frame_threshold),
    }
    previous_metadata = dict(state.session_metadata)
    try:
        write_session_metadata({
            "latest_tracking_result_id": result_id,
            "latest_tracking_result_path": f"tracking/{result_id}.json",
            "latest_tracking_result_updated_at": created_at,
            "latest_tracking_result_summary": summary,
        })
    except Exception:
        state.session_metadata = previous_metadata
        raise
    return {"result_id": result_id, "path": str(result_path), "summary": summary}


def load_prompt_tracking_result(result_id: str) -> dict[str, Any]:
    normalized_result_id = _validate_result_id(result_id)
    session_path = current_session_path()
    if session_path is None:
        raise OperationError(404, "No active session.")

    if normalized_result_id != state.session_metadata.get("latest_tracking_result_id"):
        raise OperationError(404, "Tracking result is not published for this session.")
    result_path = _tracking_result_path(session_path, normalized_result_id)
    try:
        resolved_result_path = result_path.resolve()
        resolved_tracking_dir = _tracking_dir(session_path).resolve()
        resolved_result_path.relative_to(resolved_tracking_dir)
    except ValueError as error:
        raise OperationError(400, "Invalid tracking result id.") from error

    if not result_path.exists():
        raise OperationError(404, "Tracking result not found.")
    result = load_mask_manifest(result_path)
    if not isinstance(result, dict):
        raise OperationError(500, "Tracking result payload is invalid.")
    return current_tracking_result(result)


def current_tracking_result(result: dict[str, Any]) -> dict[str, Any]:
    """One validity projection for display, guidance and session restoration."""
    if (result.get("session_id") != state.active_session_id or
            result.get("num_frames") != len(state.video_frame_files)):
        raise OperationError(409, "Tracking result does not match the current video session.")
    keys = positive_prompt_keys(state.annotations.prompt_events())
    points, tracks, visibility = result.get("points", []), result.get("tracks", []), result.get("visibility", [])
    if len(points) != len(tracks) or len(points) != len(visibility):
        raise OperationError(500, "Tracking result payload is invalid.")
    indices = [i for i, point in enumerate(points)
               if point["obj_id"] in state.annotations.objects and tracked_prompt_keys([point])[0] in keys]
    return {**result, "annotation_revision": state.annotations.revision,
            "state_epoch": state.video_state_epoch, "num_points": len(indices),
            "points": [points[i] for i in indices], "tracks": [tracks[i] for i in indices],
            "visibility": [visibility[i] for i in indices]}


def tracking_validity() -> dict[str, Any] | None:
    result_id = state.session_metadata.get("latest_tracking_result_id")
    summary = state.session_metadata.get("latest_tracking_result_summary", {})
    if (not result_id or summary.get("session_id") != state.active_session_id or
            summary.get("num_frames") != len(state.video_frame_files)):
        return None
    keys = positive_prompt_keys(state.annotations.prompt_events())
    return {"result_id": result_id, "point_ids": [point["point_id"] for point in summary.get("points", [])
            if point["obj_id"] in state.annotations.objects and tracked_prompt_keys([point])[0] in keys]}


def restored_tracking_result_payload(session_path: Path, session_metadata: dict[str, Any]) -> dict[str, Any] | None:
    result_id = session_metadata.get("latest_tracking_result_id")
    relative_path = session_metadata.get("latest_tracking_result_path")
    if not result_id or not relative_path:
        return None
    try:
        normalized_result_id = _validate_result_id(str(result_id))
    except OperationError:
        return None

    result_path = (session_path / str(relative_path)).resolve()
    try:
        result_path.relative_to(session_path.resolve())
    except ValueError:
        return None
    if not result_path.exists():
        return None

    try:
        result = current_tracking_result(load_mask_manifest(result_path))
        if not result["points"]:
            return None
    except (OperationError, OSError, ValueError, KeyError):
        return None

    summary = session_metadata.get("latest_tracking_result_summary")
    return {
        "result_id": normalized_result_id,
        "summary": summary if isinstance(summary, dict) else {},
    }
