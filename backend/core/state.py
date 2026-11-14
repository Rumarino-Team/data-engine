from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
from core.annotations import SessionAnnotations
from core.errors import OperationError

@dataclass
class SessionState:
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


    def adopt_checkpoint(self, checkpoint_id: str, checkpoint_dir: Path, display_name: str) -> None:
        """Take the identity of the checkpoint this working copy was opened from or just saved to."""
        self.checkpoint_id, self.checkpoint_dir, self.display_name = checkpoint_id, checkpoint_dir, display_name

    def require_masker(self) -> Any:
        if self.video_masker is None:
            raise OperationError(400, "Video masker not active.")
        return self.video_masker

    def require_frame(self, frame_idx: int) -> None:
        """A request about a frame needs a loaded video (400) and an index inside it (404)."""
        if self.video_dir is None or not self.video_frame_files:
            raise OperationError(400, "Video not initialized. Call /video/init_state first.")
        if not 0 <= frame_idx < len(self.video_frame_files):
            raise OperationError(
                404, f"Frame index out of bounds: {frame_idx}. Expected 0..{len(self.video_frame_files) - 1}.")
