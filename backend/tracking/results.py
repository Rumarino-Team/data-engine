import re
import uuid
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from core.jobs import utc_now_iso
from sessions.metadata import current_session_path, write_session_metadata
from utils import load_mask_manifest, write_mask_manifest

_RESULT_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


def _validate_result_id(result_id: str) -> str:
    normalized = str(result_id).strip()
    if not normalized or not _RESULT_ID_PATTERN.fullmatch(normalized):
        raise HTTPException(status_code=400, detail="Invalid tracking result id.")
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
        raise HTTPException(status_code=400, detail="Video session cache is not initialized.")

    result_id = uuid.uuid4().hex
    created_at = utc_now_iso()
    payload = {
        "version": 1,
        "result_id": result_id,
        "created_at": created_at,
        "model_name": model_name,
        "num_points": int(num_points),
        "num_frames": int(num_frames),
        "add_support_grid_used": bool(add_support_grid_used),
        "tracking_mode": tracking_mode,
        "streaming_frame_threshold": int(streaming_frame_threshold),
        "points": points,
        "tracks": tracks,
        "visibility": visibility,
    }

    result_path = _tracking_result_path(session_path, result_id)
    write_mask_manifest(result_path, payload)

    summary = {
        "model_name": model_name,
        "num_points": int(num_points),
        "num_frames": int(num_frames),
        "add_support_grid_used": bool(add_support_grid_used),
        "tracking_mode": tracking_mode,
        "streaming_frame_threshold": int(streaming_frame_threshold),
    }
    write_session_metadata(
        {
            "latest_tracking_result_id": result_id,
            "latest_tracking_result_path": f"tracking/{result_id}.json",
            "latest_tracking_result_updated_at": created_at,
            "latest_tracking_result_summary": summary,
        }
    )
    return {"result_id": result_id, "path": str(result_path), "summary": summary}


def load_prompt_tracking_result(result_id: str) -> dict[str, Any]:
    normalized_result_id = _validate_result_id(result_id)
    session_path = current_session_path()
    if session_path is None:
        raise HTTPException(status_code=404, detail="No active session.")

    result_path = _tracking_result_path(session_path, normalized_result_id)
    try:
        resolved_result_path = result_path.resolve()
        resolved_tracking_dir = _tracking_dir(session_path).resolve()
        resolved_result_path.relative_to(resolved_tracking_dir)
    except ValueError as error:
        raise HTTPException(status_code=400, detail="Invalid tracking result id.") from error

    if not result_path.exists():
        raise HTTPException(status_code=404, detail="Tracking result not found.")
    result = load_mask_manifest(result_path)
    if not isinstance(result, dict):
        raise HTTPException(status_code=500, detail="Tracking result payload is invalid.")
    return result


def restored_tracking_result_payload(session_path: Path, session_metadata: dict[str, Any]) -> dict[str, Any] | None:
    result_id = session_metadata.get("latest_tracking_result_id")
    relative_path = session_metadata.get("latest_tracking_result_path")
    if not result_id or not relative_path:
        return None
    try:
        normalized_result_id = _validate_result_id(str(result_id))
    except HTTPException:
        return None

    result_path = (session_path / str(relative_path)).resolve()
    try:
        result_path.relative_to(session_path.resolve())
    except ValueError:
        return None
    if not result_path.exists():
        return None

    summary = session_metadata.get("latest_tracking_result_summary")
    return {
        "result_id": normalized_result_id,
        "summary": summary if isinstance(summary, dict) else {},
    }
