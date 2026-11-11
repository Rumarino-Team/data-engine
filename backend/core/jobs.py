"""The one background job and the lock that orders foreground edits against it."""
from typing import Any, Callable, Optional
from datetime import datetime
import logging, threading, uuid
from typing import TYPE_CHECKING
import torch

from core.errors import OperationError

if TYPE_CHECKING:
    from core.state import BackendState

logger = logging.getLogger(__name__)
_UNSET = object()
_POSITION_FIELDS = (
    "current", "total", "frame_idx",
    "batch_current", "batch_total", "batch_index", "batch_count",
)
# A finished job keeps its position only as a count; the rest is meaningless once it is done.
_CLEARED_ON_COMPLETION = tuple(name for name in _POSITION_FIELDS if name not in ("current", "total"))
_RESULT_KEYS_KEPT_AFTER_CLEAR = (
    "message", "model_name", "num_points", "num_frames", "tracking_result_id",
)


def utc_now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _job_error_from_exception(error: Exception) -> tuple[str, str, Optional[str]]:
    if isinstance(error, OperationError):
        return error.code, error.message, error.detail
    if isinstance(error, torch.OutOfMemoryError):
        return "cuda_out_of_memory", "CUDA out of memory during operation.", str(error)
    if isinstance(error, RuntimeError) and "out of memory" in str(error).lower():
        return "cuda_out_of_memory", "CUDA out of memory during operation.", str(error)
    return "backend_error", "Backend operation failed.", str(error)


class JobHandle:
    """What a running worker may do to its own job; a worker never touches a newer job."""

    def __init__(self, manager: "JobManager", job_id: str):
        self._manager = manager
        self.job_id = job_id

    def update(
        self, *, status: Optional[str] = None, stage: Optional[str] = None,
        stage_label: Optional[str] = None, progress: Any = _UNSET, message: Optional[str] = None,
        append_history: bool = True, **position: Any,
    ) -> None:
        """Refine the current stage: only the fields given change."""
        unknown = set(position) - set(_POSITION_FIELDS)
        if unknown:
            raise TypeError(f"Unknown job position fields: {sorted(unknown)}")
        self._manager._update(
            self.job_id, status=status, stage=stage, stage_label=stage_label, progress=progress,
            message=message, append_history=append_history, position=position,
        )

    def enter_stage(
        self, stage: str, stage_label: str, *, message: str, progress: Any = _UNSET,
        status: Optional[str] = None, append_history: bool = True, **position: Any,
    ) -> None:
        """Start a stage: position fields (frame and batch counters) not given are cleared.

        `update` only touches what it is given, so a stage that forgot to reset a counter
        would show the previous stage's position.
        """
        unknown = set(position) - set(_POSITION_FIELDS)
        if unknown:
            raise TypeError(f"Unknown job position fields: {sorted(unknown)}")
        self.update(
            status=status, stage=stage, stage_label=stage_label, message=message, progress=progress,
            append_history=append_history, **{name: position.get(name) for name in _POSITION_FIELDS},
        )


class JobManager:
    """Owns the single job slot. Job admission takes the session's `foreground_lock`, shared with
    foreground edits: once admitted, an active job excludes edits until its worker has finished."""

    def __init__(self, foreground_lock: "threading.RLock") -> None:
        self._job: Optional[dict[str, Any]] = None
        self._lock = threading.Lock()
        self._foreground_lock = foreground_lock

    def _active(self) -> bool:
        return self._job is not None and self._job.get("status") in {"queued", "running"}

    def is_active(self) -> bool:
        with self._lock:
            return self._active()

    def require_idle(self) -> None:
        if self.is_active():
            raise OperationError(409, "Another operation is already running.")

    def snapshot(self) -> Optional[dict[str, Any]]:
        with self._lock:
            return dict(self._job) if self._job is not None else None

    def clear_result(self, job_id: str) -> bool:
        with self._lock:
            job = self._current(job_id)
            if job is None or job.get("status") not in {"completed", "failed"}:
                return False
            result = job.get("result")
            job["result"] = (
                {key: result[key] for key in _RESULT_KEYS_KEPT_AFTER_CLEAR if key in result}
                if isinstance(result, dict) else None
            )
            job["updated_at"] = utc_now_iso()
            return True

    def start(
        self, *, operation: str, stage: str, stage_label: str, message: str,
        worker: Callable[[JobHandle], dict[str, Any]],
    ) -> dict[str, Any]:
        """Admit a job and run `worker(handle)` on a thread; raises 409 while another job is active."""
        job_id = self._admit(operation, stage=stage, stage_label=stage_label, message=message)
        threading.Thread(target=self._run, args=(job_id, worker), daemon=True).start()
        return {"job_id": job_id, "status": "queued", "operation": operation, "message": message}

    def _admit(self, operation: str, *, stage: str, stage_label: str, message: str) -> str:
        with self._foreground_lock, self._lock:
            if self._active():
                raise OperationError(409, "Another operation is already running.")
            now = utc_now_iso()
            self._job = {
                "job_id": uuid.uuid4().hex,
                "operation": operation,
                "status": "queued",
                "stage": stage,
                "stage_label": stage_label,
                "progress": 0.0,
                **{name: None for name in _POSITION_FIELDS},
                "stage_history": [],
                "message": message,
                "result": None,
                "error": None,
                "started_at": now,
                "updated_at": now,
                "completed_at": None,
            }
            return self._job["job_id"]

    def _current(self, job_id: str) -> Optional[dict[str, Any]]:
        """The job record while `job_id` is still the registered job; caller holds the lock."""
        return self._job if self._job is not None and self._job.get("job_id") == job_id else None

    def _update(
        self, job_id: str, *, status, stage, stage_label, progress, message, append_history,
        position: dict[str, Any],
    ) -> None:
        with self._lock:
            job = self._current(job_id)
            if job is None:
                return
            if status is not None:
                job["status"] = status
            if stage is not None:
                job["stage"] = stage
            if stage_label is not None:
                job["stage_label"] = stage_label
            if progress is not _UNSET:
                job["progress"] = None if progress is None else min(max(float(progress), 0.0), 1.0)
            for name, value in position.items():
                if value is not _UNSET:
                    job[name] = None if value is None else int(value)
            if message is not None:
                job["message"] = message
            now = utc_now_iso()
            job["updated_at"] = now
            if append_history and (stage is not None or stage_label is not None or message is not None):
                history = job.setdefault("stage_history", [])
                history.append({
                    "stage": job.get("stage"), "stage_label": job.get("stage_label"),
                    "message": job.get("message"), "progress": job.get("progress"), "updated_at": now,
                })
                del history[:-8]

    def _run(self, job_id: str, worker: Callable[[JobHandle], dict[str, Any]]) -> None:
        with self._lock:
            job = self._current(job_id)
            if job is None:
                return
            job["status"] = "running"
            job["updated_at"] = utc_now_iso()
        try:
            result = worker(JobHandle(self, job_id))
        except Exception as error:
            logger.exception("Background job failed")
            self._finish(job_id, failure=_job_error_from_exception(error))
        else:
            self._finish(job_id, result=result)

    def _finish(self, job_id: str, *, result: Optional[dict[str, Any]] = None, failure=None) -> None:
        with self._lock:
            job = self._current(job_id)
            if job is None:
                return
            now = utc_now_iso()
            if failure is not None:
                code, message, detail = failure
                job.update(status="failed", error={"code": code, "message": message, "detail": detail},
                           message=message, updated_at=now, completed_at=now)
                return
            job.update(
                status="completed", stage="completed", stage_label="Completed", progress=1.0,
                current=job.get("total") or job.get("current"),
                **{name: None for name in _CLEARED_ON_COMPLETION},
                message="Operation completed", result=result, error=None, updated_at=now, completed_at=now,
            )



def capture_job_inputs(session: "BackendState", operation: str) -> dict[str, Any]:
    return {
        "session_id": session.active_session_id,
        "input_revision": session.annotations.revision,
        "dependencies": (sorted(session.annotations.positive_prompt_keys())
                         if operation == "prompt_tracking" else dict(session.annotations.input_revisions)),
    }


def validate_job_inputs(session: "BackendState", inputs: dict[str, Any], operation: str) -> None:
    current = capture_job_inputs(session, operation)
    if inputs["session_id"] != current["session_id"] or inputs["dependencies"] != current["dependencies"]:
        raise OperationError(409, "Job inputs are no longer current.")
