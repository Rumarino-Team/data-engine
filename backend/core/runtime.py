import gc, shutil
import torch
from core.config import CACHE_ROOT

def cleanup_cuda_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def cleanup_backend_cache_on_shutdown() -> None:
    from sessions.cache import release_active_session
    release_active_session(clear_cache_session=False)
    shutil.rmtree(CACHE_ROOT, ignore_errors=True)
    cleanup_cuda_memory()
