"""What the application does when it starts and stops: housekeeping that needs the session storage."""
import shutil

from core.config import CACHE_ROOT, SAVED_ROOT, SESSION_CACHE_ROOT
from core.context import AppContext
from core.runtime import cleanup_cuda_memory
from sessions.cache import release_active_session
from sessions.checkpoints import sweep_stale_session_storage


def sweep_stale_sessions_on_startup() -> None:
    # Working changes are discarded across restarts; only checkpoints are durable.
    sweep_stale_session_storage(SESSION_CACHE_ROOT, SAVED_ROOT)


def cleanup_backend_cache_on_shutdown(context: AppContext) -> None:
    release_active_session(context.session, clear_cache_session=False)
    shutil.rmtree(CACHE_ROOT, ignore_errors=True)
    cleanup_cuda_memory()
