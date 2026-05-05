import shutil
from core.config import SESSION_CACHE_ROOT, WINDOW_FRAMES_ROOT
from core.state import state
from sessions.metadata import current_session_path
from sessions.paths import path_is_relative_to

def clear_active_cache_session() -> None:
    session_path = current_session_path()
    if session_path is None:
        return
    if path_is_relative_to(session_path, SESSION_CACHE_ROOT):
        shutil.rmtree(session_path, ignore_errors=True)

def clear_window_cache() -> None:
    shutil.rmtree(WINDOW_FRAMES_ROOT, ignore_errors=True)
    WINDOW_FRAMES_ROOT.mkdir(parents=True, exist_ok=True)

def release_active_session(*, clear_video_masker: bool = True, clear_tracker: bool = True, clear_tracking_video: bool = True, clear_video_state: bool = True, clear_prompts: bool = True, clear_cache_session: bool = False) -> None:
    from core.runtime import cleanup_cuda_memory
    if clear_video_masker and state.video_masker is not None:
        state.video_masker = None
    if clear_tracker and state.tracker is not None:
        state.tracker = None
    if clear_tracking_video:
        state.tracking_video = None
        state.tracking_video_path = None
    if clear_cache_session:
        clear_active_cache_session()
    if clear_video_state:
        state.video_dir = None
        state.video_frame_files = []
        state.video_source_path = None
        state.mask_manifest_path = None
        state.active_session_dir = None
        state.active_session_id = None
        state.active_session_saved_name = None
        state.video_masker_status = "inactive"
        state.video_masker_error = None
    if clear_prompts:
        state.video_prompt_events = []
    cleanup_cuda_memory()

def bump_video_state_epoch() -> int:
    state.video_state_epoch += 1
    return state.video_state_epoch

def reset_video_session_state():
    state.video_prompt_events = []
    state.mask_manifest_path = None

def prepare_video_masker_for_video_init():
    release_active_session(clear_cache_session=True)
