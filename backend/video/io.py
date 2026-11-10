import logging, os, shutil, uuid
from pathlib import Path
from typing import Callable, Optional
import cv2
import numpy as np
from core.config import IMAGE_EXTENSIONS, SESSION_CACHE_ROOT
from core.state import BackendState
from sessions.checkpoints import clone_session_directory

def load_video_frames_as_numpy(video_dir_path: Path, frame_file_names: list[str]) -> np.ndarray:
    frames_rgb: list[np.ndarray] = []
    for frame_idx, name in enumerate(frame_file_names):
        frame_path = video_dir_path / name
        frame_bgr = cv2.imread(str(frame_path))
        if frame_bgr is None:
            raise ValueError(f"Unable to read frame {frame_idx}: {frame_path}")
        frames_rgb.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

    if not frames_rgb:
        raise ValueError(f"No readable frames found under {video_dir_path}")

    return np.stack(frames_rgb, axis=0)

def build_window_dir(
    frame_paths: list[Path],
    run_root: Path,
    window_name: str,
    progress_callback: Optional[Callable[[int, int, Path], None]] = None,
) -> Path:
    window_dir = run_root / window_name
    window_dir.mkdir(parents=True, exist_ok=True)
    total = len(frame_paths)

    for local_idx, source_path in enumerate(frame_paths):
        target_name = f"{local_idx:05d}{source_path.suffix.lower()}"
        target_path = window_dir / target_name
        link_or_copy_file(source_path, target_path)
        if progress_callback is not None:
            progress_callback(local_idx + 1, total, source_path)

    return window_dir

def link_or_copy_file(source_path: Path, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(source_path, target_path)
    except OSError:
        try:
            os.link(source_path, target_path)
        except OSError:
            shutil.copy2(source_path, target_path)

def create_active_session(source_path: Path, *, session: BackendState) -> Path:

    SESSION_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    session_id = uuid.uuid4().hex
    session_dir = SESSION_CACHE_ROOT / session_id
    (session_dir / "frames").mkdir(parents=True, exist_ok=False)
    (session_dir / "masks").mkdir(parents=True, exist_ok=True)
    session.active_session_dir = session_dir
    session.active_session_id = session_id
    return session_dir

def discard_working_session(session_dir: Optional[Path]) -> None:
    """Delete a working directory this backend created; checkpoints and any other path are never touched."""
    if session_dir is None:
        return
    try:
        resolved = Path(session_dir).resolve()
        if resolved.parent != SESSION_CACHE_ROOT.resolve():
            return
        shutil.rmtree(resolved)
    except OSError:
        # Best effort; anything left behind is removed by the startup sweep.
        logging.getLogger(__name__).warning("Unable to remove working session %s", session_dir, exc_info=True)

def open_checkpoint_as_working_copy(checkpoint_dir: Path, *, session: BackendState) -> Path:
    """Edits and generated results go to a fresh working directory; the checkpoint is never written."""
    SESSION_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    session_id = uuid.uuid4().hex
    session_dir = SESSION_CACHE_ROOT / session_id
    clone_session_directory(checkpoint_dir, session_dir)
    session.active_session_dir = session_dir
    session.active_session_id = session_id
    return session_dir

def copy_frames_directory_to_session(
    source_dir: Path,
    frames_dir: Path,
    progress_callback: Optional[Callable[[int, int, Path], None]] = None,
) -> list[str]:
    source_frames = sorted(
        frame_path
        for frame_path in source_dir.iterdir()
        if frame_path.is_file() and frame_path.suffix.lower() in IMAGE_EXTENSIONS
    )
    total = len(source_frames)
    frame_names: list[str] = []
    for index, source_path in enumerate(source_frames, start=1):
        target_name = f"{index - 1:05d}{source_path.suffix.lower()}"
        target_path = frames_dir / target_name
        link_or_copy_file(source_path, target_path)
        frame_names.append(target_name)
        if progress_callback is not None:
            progress_callback(index, total, source_path)
    return frame_names

def extract_video_to_session_frames(
    video_path: Path,
    frames_dir: Path,
    progress_callback: Optional[Callable[[int, Optional[int]], None]] = None,
) -> list[str]:
    frames_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Unable to open video file: {video_path}")

    raw_total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    total_frames: Optional[int] = raw_total_frames if raw_total_frames > 0 else None
    frame_idx = 0
    frame_names: list[str] = []
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break
            frame_name = f"{frame_idx:05d}.jpg"
            output_path = frames_dir / frame_name
            if not cv2.imwrite(str(output_path), frame):
                raise RuntimeError(f"Failed to write extracted frame: {output_path}")
            frame_names.append(frame_name)
            frame_idx += 1
            if progress_callback is not None:
                progress_callback(frame_idx, total_frames)
    finally:
        capture.release()

    if frame_idx == 0:
        raise ValueError(f"No frames could be extracted from video: {video_path}")
    return frame_names

