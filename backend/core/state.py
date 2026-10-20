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
    mask_manifest_path: Optional[str] = None
    video_state_epoch: int = 0
    active_session_dir: Optional[Path] = None
    active_session_id: Optional[str] = None
    active_session_saved_name: Optional[str] = None

state = BackendState()
