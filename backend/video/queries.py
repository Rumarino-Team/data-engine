from pathlib import Path
from typing import Any, Optional
from core.context import AppContext
from core.errors import OperationError
from core.state import SessionState
from video.masks import effective_mask_data, mask_overlay_path
from video.responses import annotation_snapshot


def get_video_info(ctx: AppContext) -> dict[str, Any]:
    with ctx.lock:
        session = ctx.session
        return {
            **annotation_snapshot(session),
            "predictor_ready": session.video_masker is not None and bool(session.video_masker.state_loaded),
            "num_frames": len(session.video_frame_files),
            "frame_files": list(session.video_frame_files),
        }


def get_mask_data(ctx: AppContext, frame_idx: int) -> dict[str, Any]:
    with ctx.lock:
        session = ctx.session
        session.require_frame(frame_idx)
        return effective_mask_data(session, frame_idx)


def _require_requested_session(session: SessionState, session_id: Optional[str]) -> None:
    """A request that names its session must find that session still active (like an edit's precondition)."""
    if session_id is not None and session_id != session.active_session_id:
        raise OperationError(409, "The target session is no longer active. Refresh annotations.")


def get_video_frame_path(ctx: AppContext, frame_idx: int, session_id: Optional[str] = None) -> Path:
    with ctx.lock:
        session = ctx.session
        _require_requested_session(session, session_id)
        session.require_frame(frame_idx)
        file_path = Path(session.video_dir) / session.video_frame_files[frame_idx]
        if not file_path.exists():
            raise OperationError(404, f"Frame file not found: {file_path}")
        return file_path


def get_video_mask_frame_path(ctx: AppContext, frame_idx: int, session_id: Optional[str] = None) -> Path:
    with ctx.lock:
        session = ctx.session
        _require_requested_session(session, session_id)
        session.require_frame(frame_idx)
        return mask_overlay_path(session, frame_idx)
