from pathlib import Path
from typing import Any
from core.context import AppContext
from core.errors import OperationError
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


def get_video_frame_path(ctx: AppContext, frame_idx: int) -> Path:
    with ctx.lock:
        session = ctx.session
        session.require_frame(frame_idx)
        file_path = Path(session.video_dir) / session.video_frame_files[frame_idx]
        if not file_path.exists():
            raise OperationError(404, f"Frame file not found: {file_path}")
        return file_path


def get_video_mask_frame_path(ctx: AppContext, frame_idx: int) -> Path:
    with ctx.lock:
        session = ctx.session
        session.require_frame(frame_idx)
        return mask_overlay_path(session, frame_idx)
