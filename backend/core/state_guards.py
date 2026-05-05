from contextlib import contextmanager
from typing import Iterator

from fastapi import HTTPException

from core.jobs import is_job_active
from core.state import state


ACTIVE_JOB_DETAIL = "Another operation is already running."


def require_no_active_job_locked(operation_label: str = "operation") -> None:
    if is_job_active():
        raise HTTPException(status_code=409, detail=ACTIVE_JOB_DETAIL)


def require_video_read_allowed() -> None:
    if is_job_active():
        raise HTTPException(status_code=409, detail=ACTIVE_JOB_DETAIL)


@contextmanager
def guarded_video_write(operation_label: str = "operation") -> Iterator[None]:
    with state.video_state_lock:
        require_no_active_job_locked(operation_label)
        yield


@contextmanager
def guarded_video_read() -> Iterator[None]:
    with state.video_state_lock:
        require_video_read_allowed()
        yield
