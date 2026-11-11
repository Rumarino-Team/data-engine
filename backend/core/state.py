from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
from core.annotations import SessionAnnotations

@dataclass
class BackendState:
    video_masker: Optional[Any] = None
    tracker: Optional[Any] = None
    video_dir: Optional[str] = None
    video_frame_files: list[str] = field(default_factory=list)
    video_source_path: Optional[str] = None
    annotations: SessionAnnotations = field(default_factory=SessionAnnotations)
    interactive_mask_files: dict[str, str] = field(default_factory=dict)
    # Non-annotation session fields, held in memory; Save writes them into the checkpoint's session.json.
    session_metadata: dict[str, Any] = field(default_factory=dict)
    # Session-relative artifact paths the last metadata commit named.
    committed_artifacts: set[str] = field(default_factory=set)
    mask_manifest_path: Optional[str] = None
    active_session_dir: Optional[Path] = None
    active_session_id: Optional[str] = None
    # Checkpoint this working copy was opened from or last saved to; the working directory is never one.
    checkpoint_id: Optional[str] = None
    checkpoint_dir: Optional[Path] = None
    display_name: Optional[str] = None
    # Session-relative artifact files the checkpoint holds, so re-save copies and deletes only differences.
    checkpoint_artifacts: set[str] = field(default_factory=set)
    # Content identity when the session was loaded or last saved; None only before initialization completes.
    saved_fingerprint: Optional[tuple] = None

