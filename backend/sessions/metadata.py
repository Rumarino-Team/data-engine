import json, logging, os, re, tempfile
from pathlib import Path
from typing import Any, Optional
from core.errors import OperationError
from core.jobs import utc_now_iso
from core.state import BackendState
from sessions.artifacts import reclaim_superseded_artifacts
from utils import load_mask_manifest

logger = logging.getLogger(__name__)

def current_session_path(session: BackendState) -> Optional[Path]:
    return session.active_session_dir.resolve() if session.active_session_dir is not None else None

def current_frames_dir(session: BackendState) -> Optional[Path]:
    session_path = current_session_path(session)
    return session_path / "frames" if session_path is not None else None

def current_masks_dir(session: BackendState) -> Optional[Path]:
    session_path = current_session_path(session)
    return session_path / "masks" if session_path is not None else None

# Session fields owned outside annotations; kept in memory so writes never re-read session.json.
PERSISTED_METADATA_KEYS = (
    "interactive_state",
    "created_at",
    "saved_at",
    "source_input_path",
    "latest_tracking_result_id",
    "latest_tracking_result_path",
    "latest_tracking_result_updated_at",
    "latest_tracking_result_summary",
)

def persisted_session_fields(metadata: dict[str, Any]) -> dict[str, Any]:
    return {key: metadata[key] for key in PERSISTED_METADATA_KEYS if key in metadata}

def interactive_state_projection(raw_state: Optional[dict[str, Any]] = None, *, session: BackendState) -> dict[str, Any]:
    """Editor preferences that still reference accepted objects."""
    interactive = dict(raw_state or {})
    interactive["version"] = 1
    if interactive.get("selected_object_id") not in session.annotations.objects:
        interactive["selected_object_id"] = None
    return interactive

def session_metadata_document(session: BackendState, overrides: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    metadata = {
        "session_id": session.active_session_id,
        "checkpoint_id": session.checkpoint_id,
        "display_name": session.display_name,
        "source_video_path": session.video_source_path,
        "resolved_video_frames_dir": session.video_dir,
        "state.mask_manifest_path": session.mask_manifest_path,
        "mask_manifest_file": Path(session.mask_manifest_path).name if session.mask_manifest_path else None,
        "num_frames": len(session.video_frame_files),
        "state_epoch": int(session.video_state_epoch),
        "updated_at": utc_now_iso(),
    }
    if session.video_masker is not None:
        metadata.update(
            {
                "online_mode": session.video_masker.online_mode,
                "batch_size": session.video_masker.default_batch_size,
                "offload_video_to_cpu": session.video_masker.offload_video_to_cpu,
                "offload_state_to_cpu": session.video_masker.offload_state_to_cpu,
            }
        )
    metadata.update(session.session_metadata)
    metadata.update(overrides or {})
    # Editor caches cannot replace accepted annotation inputs.
    metadata["interactive_mask_files"] = session.interactive_mask_files
    metadata["annotations"] = session.annotations.snapshot()
    metadata["interactive_state"] = interactive_state_projection(metadata.get("interactive_state"), session=session)
    metadata["schema_version"] = 5
    return metadata

def write_metadata_file(directory: Path, metadata: dict[str, Any]) -> None:
    # A failed write must leave the last complete annotations readable.
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=directory,
                                         prefix=".session-", suffix=".json", delete=False) as handle:
            temporary_path = Path(handle.name)
            json.dump(metadata, handle, ensure_ascii=True, separators=(",", ":"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, directory / "session.json")
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

def write_session_metadata(session: BackendState, extra: Optional[dict[str, Any]] = None) -> None:
    session_path = current_session_path(session)
    if session_path is None:
        return
    if extra:
        session.session_metadata.update(extra)
    write_metadata_file(session_path, session_metadata_document(session))
    # Files become garbage only after committed metadata stops naming them.
    reclaim_superseded_artifacts(session)

def load_session_metadata(session_root: Path) -> dict[str, Any]:
    session_json_path = session_root / "session.json"
    try:
        metadata = load_mask_manifest(session_json_path)
    except FileNotFoundError as error:
        raise OperationError(
            400,
            f"Saved session directory is missing session.json: {session_json_path}",
        ) from error
    except Exception as error:
        raise OperationError(
            400,
            f"Unable to parse saved session metadata: {session_json_path}",
        ) from error
    if not isinstance(metadata, dict):
        raise OperationError(
            400,
            f"Saved session metadata must be a JSON object: {session_json_path}",
        )
    return metadata
