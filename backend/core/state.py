from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
import numpy as np
from core.annotations import SessionAnnotations

@dataclass
class BackendState:
    video_masker: Optional[Any] = None
    tracker: Optional[Any] = None
    video_dir: Optional[str] = None
    video_frame_files: list[str] = field(default_factory=list)
    tracking_video: Optional[np.ndarray] = None
    tracking_video_path: Optional[str] = None
    video_source_path: Optional[str] = None
    annotations: SessionAnnotations = field(default_factory=SessionAnnotations)
    interactive_mask_files: dict[str, str] = field(default_factory=dict)
    # Non-annotation session.json fields kept in memory so writes never re-read the file.
    session_metadata: dict[str, Any] = field(default_factory=dict)
    # Session-relative artifact paths the last metadata commit named.
    committed_artifacts: set[str] = field(default_factory=set)
    mask_manifest_path: Optional[str] = None
    video_state_epoch: int = 0
    active_session_dir: Optional[Path] = None
    active_session_id: Optional[str] = None
    # Checkpoint this working copy was opened from or last saved to; the working directory is never one.
    checkpoint_id: Optional[str] = None
    checkpoint_dir: Optional[Path] = None
    display_name: Optional[str] = None
    # Session-relative artifact files the checkpoint holds, so re-save copies and deletes only differences.
    checkpoint_artifacts: set[str] = field(default_factory=set)
    # Content identity at the last open/save; None means the session was never saved.
    saved_fingerprint: Optional[tuple] = None

state = BackendState()
