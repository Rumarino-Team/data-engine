import gc, shutil
import torch
from core.config import CACHE_ROOT

def cleanup_cuda_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def sweep_stale_sessions_on_startup() -> None:
    # Working changes are discarded across restarts; only checkpoints are durable.
    from core.config import SAVED_ROOT, SESSION_CACHE_ROOT
    from sessions.checkpoints import sweep_stale_session_storage
    sweep_stale_session_storage(SESSION_CACHE_ROOT, SAVED_ROOT)

def cleanup_backend_cache_on_shutdown() -> None:
    from sessions.cache import release_active_session
    release_active_session(clear_cache_session=False)
    shutil.rmtree(CACHE_ROOT, ignore_errors=True)
    cleanup_cuda_memory()
