import logging, os, shutil, uuid
from pathlib import Path
from typing import Callable, Optional
import cv2
import numpy as np
from core.config import IMAGE_EXTENSIONS, JPEG_EXTENSIONS, JPEG_QUALITY, SESSION_CACHE_ROOT
from core.state import SessionState
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

def link_or_copy_file(source_path: Path, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(source_path, target_path)
    except OSError:
        try:
            os.link(source_path, target_path)
        except OSError:
            shutil.copy2(source_path, target_path)

def create_active_session(source_path: Path, *, session: SessionState) -> Path:

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

def open_checkpoint_as_working_copy(checkpoint_dir: Path, *, session: SessionState) -> Path:
    """Edits and generated results go to a fresh working directory; the checkpoint is never written."""
    SESSION_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    session_id = uuid.uuid4().hex
    session_dir = SESSION_CACHE_ROOT / session_id
    clone_session_directory(checkpoint_dir, session_dir)
    session.active_session_dir = session_dir
    session.active_session_id = session_id
    return session_dir

def needs_jpeg_conversion(path: Path) -> bool:
    return path.suffix.lower() not in JPEG_EXTENSIONS

def convert_image_to_jpeg(source_path: Path, target_path: Path) -> None:
    """Write `source_path` (PNG, BMP) as a JPEG, the only format SAM2 reads.

    Decoded to 8-bit color, so an alpha channel is dropped (not blended) and 16-bit or grayscale images become
    8-bit RGB. JPEG is lossy; the quality matches the frames extracted from a video file."""
    image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unable to read image: {source_path}")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(target_path), image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]):
        raise RuntimeError(f"Failed to write converted frame: {target_path}")

def copy_frames_directory_to_session(
    source_dir: Path,
    frames_dir: Path,
    progress_callback: Optional[Callable[[int, int, Path], None]] = None,
) -> list[str]:
    """Make the folder's images the working copy's frames, numbered in file-name order.

    JPEG files are linked (or copied) as they are; every other format is converted to JPEG."""
    source_frames = sorted(
        frame_path
        for frame_path in source_dir.iterdir()
        if frame_path.is_file() and frame_path.suffix.lower() in IMAGE_EXTENSIONS
    )
    total = len(source_frames)
    frame_names: list[str] = []
    for index, source_path in enumerate(source_frames, start=1):
        if needs_jpeg_conversion(source_path):
            target_name = f"{index - 1:05d}.jpg"
            convert_image_to_jpeg(source_path, frames_dir / target_name)
        else:
            target_name = f"{index - 1:05d}{source_path.suffix.lower()}"
            link_or_copy_file(source_path, frames_dir / target_name)
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
            if not cv2.imwrite(str(output_path), frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]):
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

