import shutil
from core.config import SESSION_CACHE_ROOT, WINDOW_FRAMES_ROOT
from core.state import BackendState
from core.annotations import SessionAnnotations
from sessions.metadata import current_session_path
from sessions.paths import path_is_relative_to

def clear_active_cache_session(session: BackendState) -> None:
    session_path = current_session_path(session)
    if session_path is None:
        return
    if path_is_relative_to(session_path, SESSION_CACHE_ROOT):
        shutil.rmtree(session_path, ignore_errors=True)

def clear_window_cache() -> None:
    shutil.rmtree(WINDOW_FRAMES_ROOT, ignore_errors=True)
    WINDOW_FRAMES_ROOT.mkdir(parents=True, exist_ok=True)

def release_active_session(session: BackendState, *, clear_video_masker: bool = True, clear_tracker: bool = True, clear_video_state: bool = True, clear_prompts: bool = True, clear_cache_session: bool = False) -> None:
    from core.runtime import cleanup_cuda_memory
    if clear_video_masker and session.video_masker is not None:
        session.video_masker = None
    if clear_tracker and session.tracker is not None:
        session.tracker = None
    if clear_cache_session:
        clear_active_cache_session(session)
    if clear_video_state:
        session.video_dir = None
        session.video_frame_files = []
        session.video_source_path = None
        session.mask_manifest_path = None
        session.active_session_dir = None
        session.active_session_id = None
        session.checkpoint_id = None
        session.checkpoint_dir = None
        session.display_name = None
        session.saved_fingerprint = None
        session.checkpoint_artifacts = set()
        session.annotations = SessionAnnotations()
        session.interactive_mask_files = {}
        session.session_metadata = {}
        session.committed_artifacts = set()
    if clear_prompts:
        session.annotations.clear_prompts()
    cleanup_cuda_memory()

def reset_video_session_state(session: BackendState):
    session.interactive_mask_files = {}
    session.annotations.clear_prompts()
    session.annotations.invalidate_inputs()
    session.mask_manifest_path = None
