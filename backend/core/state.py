from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
import threading
import numpy as np

@dataclass
class BackendState:
    video_masker: Optional[Any] = None
    tracker: Optional[Any] = None
    video_dir: Optional[str] = None
    video_frame_files: list[str] = field(default_factory=list)
    tracking_video: Optional[np.ndarray] = None
    tracking_video_path: Optional[str] = None
    video_source_path: Optional[str] = None
    video_prompt_events: list[dict[str, Any]] = field(default_factory=list)
    mask_manifest_path: Optional[str] = None
    video_state_epoch: int = 0
    video_masker_status: str = "inactive"
    video_masker_error: Optional[str] = None
    active_session_dir: Optional[Path] = None
    active_session_id: Optional[str] = None
    active_session_saved_name: Optional[str] = None
    video_state_lock: threading.RLock = field(default_factory=threading.RLock)

state = BackendState()
