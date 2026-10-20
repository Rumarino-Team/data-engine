import json, logging, os, re, tempfile
from pathlib import Path
from typing import Any, Optional
from fastapi import HTTPException
from core.jobs import utc_now_iso
from core.state import state
from utils import load_mask_manifest

logger = logging.getLogger(__name__)

def current_session_path() -> Optional[Path]:
    return state.active_session_dir.resolve() if state.active_session_dir is not None else None

def current_frames_dir() -> Optional[Path]:
    session_path = current_session_path()
    return session_path / "frames" if session_path is not None else None

def current_masks_dir() -> Optional[Path]:
    session_path = current_session_path()
    return session_path / "masks" if session_path is not None else None

# Session fields owned outside annotations; kept in memory so writes never re-read session.json.
PERSISTED_METADATA_KEYS = (
    "interactive_state",
    "created_at",
    "saved_at",
    "saved_path",
    "source_input_path",
    "latest_tracking_result_id",
    "latest_tracking_result_path",
    "latest_tracking_result_updated_at",
    "latest_tracking_result_summary",
)

def persisted_session_fields(metadata: dict[str, Any]) -> dict[str, Any]:
    return {key: metadata[key] for key in PERSISTED_METADATA_KEYS if key in metadata}

def interactive_state_projection(raw_state: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """Editor preferences that still reference accepted objects."""
    interactive = dict(raw_state or {})
    interactive["version"] = 1
    if interactive.get("selected_object_id") not in state.annotations.objects:
        interactive["selected_object_id"] = None
    return interactive

def write_session_metadata(extra: Optional[dict[str, Any]] = None) -> None:
    from video.masks import discard_override_artifacts
    session_path = current_session_path()
    if session_path is None:
        return
    session_metadata_path = session_path / "session.json"
    if extra:
        state.session_metadata.update(extra)
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
    metadata.update(state.session_metadata)
    # Editor caches cannot replace accepted annotation inputs.
    metadata["interactive_mask_files"] = state.interactive_mask_files
    metadata["annotations"] = state.annotations.snapshot()
    metadata["interactive_state"] = interactive_state_projection(metadata.get("interactive_state"))
    metadata["schema_version"] = 4
    # A failed write must leave the last complete annotations readable.
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=session_path,
                                         prefix=".session-", suffix=".json", delete=False) as handle:
            temporary_path = Path(handle.name)
            json.dump(metadata, handle, ensure_ascii=True, separators=(",", ":"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, session_metadata_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    # Replaced override pixels become garbage only after metadata stops naming them.
    referenced = {mask.artifact for mask in state.annotations.mask_overrides}
    stale = state.committed_override_artifacts - referenced
    state.committed_override_artifacts = referenced
    discard_override_artifacts(stale)

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
