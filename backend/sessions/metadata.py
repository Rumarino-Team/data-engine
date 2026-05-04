import logging, re
from pathlib import Path
from typing import Any, Optional
from fastapi import HTTPException
from core.jobs import utc_now_iso
from core.state import state
from utils import load_mask_manifest, write_mask_manifest

logger = logging.getLogger(__name__)

def current_session_path() -> Optional[Path]:
    return state.active_session_dir.resolve() if state.active_session_dir is not None else None

def current_frames_dir() -> Optional[Path]:
    session_path = current_session_path()
    return session_path / "frames" if session_path is not None else None

def current_masks_dir() -> Optional[Path]:
    session_path = current_session_path()
    return session_path / "masks" if session_path is not None else None

def write_session_metadata(extra: Optional[dict[str, Any]] = None) -> None:
    session_path = current_session_path()
    if session_path is None:
        return
    existing_metadata: dict[str, Any] = {}
    session_metadata_path = session_path / "session.json"
    if session_metadata_path.exists():
        try:
            loaded = load_mask_manifest(session_metadata_path)
            if isinstance(loaded, dict):
                existing_metadata = loaded
        except Exception:
            logger.warning("Failed to read existing session metadata from %s", session_metadata_path)
    metadata = {
        "session_id": state.active_session_id,
        "saved_name": state.active_session_saved_name,
        "source_video_path": state.video_source_path,
        "resolved_video_frames_dir": state.video_dir,
        "state.mask_manifest_path": state.mask_manifest_path,
        "num_frames": len(state.video_frame_files),
        "state_epoch": int(state.video_state_epoch),
        "updated_at": utc_now_iso(),
    }
    if state.video_masker is not None:
        metadata.update(
            {
                "online_mode": state.video_masker.online_mode,
                "batch_size": state.video_masker.default_batch_size,
                "offload_video_to_cpu": state.video_masker.offload_video_to_cpu,
                "offload_state_to_cpu": state.video_masker.offload_state_to_cpu,
            }
        )
    for key in (
        "schema_version",
        "interactive_state",
        "created_at",
        "saved_at",
        "saved_path",
        "source_input_path",
        "latest_tracking_result_id",
        "latest_tracking_result_path",
        "latest_tracking_result_updated_at",
        "latest_tracking_result_summary",
    ):
        if key in existing_metadata:
            metadata[key] = existing_metadata[key]
    if extra:
        metadata.update(extra)
    write_mask_manifest(session_metadata_path, metadata)

def load_session_metadata(session_root: Path) -> dict[str, Any]:
    session_json_path = session_root / "session.json"
    try:
        metadata = load_mask_manifest(session_json_path)
    except FileNotFoundError as error:
        raise HTTPException(
            status_code=400,
            detail=f"Saved session directory is missing session.json: {session_json_path}",
        ) from error
    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail=f"Unable to parse saved session metadata: {session_json_path}",
        ) from error
    if not isinstance(metadata, dict):
        raise HTTPException(
            status_code=400,
            detail=f"Saved session metadata must be a JSON object: {session_json_path}",
        )
    return metadata

def sanitize_save_name(name: str) -> str:
    sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", name.strip())
    sanitized = re.sub(r"_+", "_", sanitized).strip(" ._")
    if not sanitized or sanitized in {".", ".."}:
        raise HTTPException(status_code=400, detail="Save name cannot be empty.")
    return sanitized

def merge_session_metadata(
    *,
    existing_meta: dict[str, Any],
    interactive_state: Optional[dict[str, Any]],
    save_name: str,
    saved_path: Path,
    saved_at: str,
) -> dict[str, Any]:
    merged = dict(existing_meta)
    merged["schema_version"] = 2
    merged["saved_name"] = save_name
    merged["saved_path"] = str(saved_path)
    merged["saved_at"] = saved_at
    if interactive_state is not None:
        merged["interactive_state"] = interactive_state
    return merged
