from pathlib import Path
import numpy as np
import co_tracker as cot
from core.config import DEFAULT_STREAMING_TRACK_FRAME_THRESHOLD
from core.runtime import cleanup_cuda_memory
from core.state import BackendState
from tracking.runtime import VideoTrackerRuntime
from video.io import load_video_frames_as_numpy

def create_tracker_runtime(session: BackendState) -> VideoTrackerRuntime:
    return VideoTrackerRuntime(session=session, create_tracker=cot.CoTracker, release_memory=cleanup_cuda_memory)

def load_tracking_video_from_current_video_state(session: BackendState) -> tuple[np.ndarray, str]:
    if session.video_dir is None or not session.video_frame_files:
        raise ValueError("Video is not initialized for tracking.")

    # Keep tracking frame indices aligned with the exact frame sequence used by masking/frontend.
    # Decoding directly from source_video_path can introduce index drift across different decoders.
    frame_video = load_video_frames_as_numpy(Path(session.video_dir), session.video_frame_files)
    return frame_video, str(session.video_dir)

def should_stream_tracking(num_frames: int) -> bool:
    threshold = int(DEFAULT_STREAMING_TRACK_FRAME_THRESHOLD)
    if threshold <= 0:
        return True
    return int(num_frames) >= threshold
