import ctypes
import gc
import torch

def cleanup_cuda_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def release_heap_memory():
    """Hand the C heap's free pages back to the OS (glibc only). After a long job the memory it freed, mostly
    NumPy and torch buffers, otherwise stays in the process's resident size."""
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass

def is_out_of_memory(error: BaseException) -> bool:
    return isinstance(error, torch.OutOfMemoryError) or (
        isinstance(error, RuntimeError) and "out of memory" in str(error).lower()
    )
