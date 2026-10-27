from typing import Any, Callable, Optional
from contextlib import contextmanager
from datetime import datetime
import logging, threading, uuid
import torch

from core.errors import OperationError

logger = logging.getLogger(__name__)
current_job: Optional[dict[str, Any]] = None
current_job_lock = threading.Lock()
# Foreground edits and job admission share this boundary. Once admitted, an active
# job excludes foreground edits until its worker has finished.
foreground_lock = threading.RLock()


@contextmanager
def foreground_operation():
    """Serialize synchronous mutation bodies; never suspend/await inside this block."""
    with foreground_lock:
        require_no_active_job()
        yield

_UNSET = object()

def utc_now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"

def serialize_job(job: Optional[dict[str, Any]] = None) -> Optional[dict[str, Any]]:
    with current_job_lock:
        source = current_job if job is None else job
        return dict(source) if source is not None else None

def _active_job_exists() -> bool:
    return current_job is not None and current_job.get("status") in {"queued", "running"}

def is_job_active() -> bool:
    with current_job_lock:
        return _active_job_exists()

def require_no_active_job(operation_label: str = "operation") -> None:
    if is_job_active():
        raise OperationError(409, "Another operation is already running.")

def clear_current_job_result(job_id: str) -> bool:
    with current_job_lock:
        if current_job is None or current_job.get("job_id") != job_id:
            return False
        if current_job.get("status") not in {"completed", "failed"}:
            return False
        result = current_job.get("result")
        if isinstance(result, dict):
            current_job["result"] = {
                key: result.get(key)
                for key in (
                    "message",
                    "model_name",
                    "num_points",
                    "num_frames",
                    "tracking_result_id",
                    "state_epoch",
                )
                if key in result
            }
        else:
            current_job["result"] = None
        current_job["updated_at"] = utc_now_iso()
        return True

def capture_job_inputs(operation: str) -> dict[str, Any]:
    from core.state import state
    from tracking.guidance import positive_prompt_keys
    return {
        "session_id": state.active_session_id,
        "input_revision": state.annotations.revision,
        "dependencies": (sorted(positive_prompt_keys(state.annotations.prompt_events()))
                         if operation == "prompt_tracking" else dict(state.annotations.input_revisions)),
    }


def validate_job_inputs(inputs: dict[str, Any], operation: str) -> None:
    current = capture_job_inputs(operation)
    if inputs["session_id"] != current["session_id"] or inputs["dependencies"] != current["dependencies"]:
        raise OperationError(409, "Job inputs are no longer current.")


def _start_job(operation: str, *, stage: str, stage_label: str, message: str, inputs=None) -> dict[str, Any]:
    global current_job
    with foreground_lock, current_job_lock:
        if _active_job_exists():
            raise OperationError(409, "Another operation is already running.")

        now = utc_now_iso()
        current_job = {
            "job_id": uuid.uuid4().hex,
            "inputs": inputs,
            "operation": operation,
            "status": "queued",
            "stage": stage,
            "stage_label": stage_label,
            "progress": 0.0,
            "current": None,
            "total": None,
            "window_index": None,
            "window_count": None,
            "frame_idx": None,
            "batch_current": None,
            "batch_total": None,
            "batch_index": None,
            "batch_count": None,
            "stage_history": [],
            "message": message,
            "result": None,
            "error": None,
            "started_at": now,
            "updated_at": now,
            "completed_at": None,
        }
        return dict(current_job)

def update_job(
    *,
    status: Optional[str] = None,
    stage: Optional[str] = None,
    stage_label: Optional[str] = None,
    progress: Any = _UNSET,
    current: Any = _UNSET,
    total: Any = _UNSET,
    window_index: Any = _UNSET,
    window_count: Any = _UNSET,
    frame_idx: Any = _UNSET,
    batch_current: Any = _UNSET,
    batch_total: Any = _UNSET,
    batch_index: Any = _UNSET,
    batch_count: Any = _UNSET,
    message: Optional[str] = None,
    append_history: bool = True,
) -> None:
    with current_job_lock:
        if current_job is None:
            return
        if status is not None:
            current_job["status"] = status
        if stage is not None:
            current_job["stage"] = stage
        if stage_label is not None:
            current_job["stage_label"] = stage_label
        if progress is not _UNSET:
            current_job["progress"] = None if progress is None else min(max(float(progress), 0.0), 1.0)
        if current is not _UNSET:
            current_job["current"] = None if current is None else int(current)
        if total is not _UNSET:
            current_job["total"] = None if total is None else int(total)
        if window_index is not _UNSET:
            current_job["window_index"] = None if window_index is None else int(window_index)
        if window_count is not _UNSET:
            current_job["window_count"] = None if window_count is None else int(window_count)
        if frame_idx is not _UNSET:
            current_job["frame_idx"] = None if frame_idx is None else int(frame_idx)
        if batch_current is not _UNSET:
            current_job["batch_current"] = None if batch_current is None else int(batch_current)
        if batch_total is not _UNSET:
            current_job["batch_total"] = None if batch_total is None else int(batch_total)
        if batch_index is not _UNSET:
            current_job["batch_index"] = None if batch_index is None else int(batch_index)
        if batch_count is not _UNSET:
            current_job["batch_count"] = None if batch_count is None else int(batch_count)
        if message is not None:
            current_job["message"] = message
        now = utc_now_iso()
        current_job["updated_at"] = now
        if append_history and (stage is not None or stage_label is not None or message is not None):
            history = current_job.setdefault("stage_history", [])
            history.append(
                {
                    "stage": current_job.get("stage"),
                    "stage_label": current_job.get("stage_label"),
                    "message": current_job.get("message"),
                    "progress": current_job.get("progress"),
                    "updated_at": now,
                }
            )
            del history[:-8]

def _complete_job(job_id: str, result: dict[str, Any]) -> None:
    with current_job_lock:
        if current_job is None or current_job.get("job_id") != job_id:
            return
        now = utc_now_iso()
        current_job.update(
            {
                "status": "completed",
                "stage": "completed",
                "stage_label": "Completed",
                "progress": 1.0,
                "current": current_job.get("total") or current_job.get("current"),
                "window_index": None,
                "window_count": None,
                "frame_idx": None,
                "batch_current": None,
                "batch_total": None,
                "batch_index": None,
                "batch_count": None,
                "message": "Operation completed",
                "result": result,
                "error": None,
                "updated_at": now,
                "completed_at": now,
            }
        )

def _fail_job(job_id: str, error_code: str, message: str, detail: Optional[str] = None) -> None:
    with current_job_lock:
        if current_job is None or current_job.get("job_id") != job_id:
            return
        now = utc_now_iso()
        current_job.update(
            {
                "status": "failed",
                "error": {"code": error_code, "message": message, "detail": detail},
                "message": message,
                "updated_at": now,
                "completed_at": now,
            }
        )

def _job_error_from_exception(error: Exception) -> tuple[str, str, Optional[str]]:
    if isinstance(error, OperationError):
        return error.code, error.message, error.detail
    if isinstance(error, torch.OutOfMemoryError):
        return "cuda_out_of_memory", "CUDA out of memory during operation.", str(error)
    if isinstance(error, RuntimeError) and "out of memory" in str(error).lower():
        return "cuda_out_of_memory", "CUDA out of memory during operation.", str(error)
    return "backend_error", "Backend operation failed.", str(error)

def _run_job(job_id: str, worker: Callable[[], dict[str, Any]]) -> None:
    with current_job_lock:
        if current_job is None or current_job.get("job_id") != job_id:
            return
        current_job["status"] = "running"
        current_job["updated_at"] = utc_now_iso()

    try:
        result = worker()
        _complete_job(job_id, result)
    except Exception as error:
        logger.exception("Background job failed")
        error_code, message, detail = _job_error_from_exception(error)
        _fail_job(job_id, error_code, message, detail)

def queue_long_job(
    *,
    operation: str,
    stage: str,
    stage_label: str,
    message: str,
    worker: Callable[[], dict[str, Any]],
    inputs: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    job = _start_job(operation, stage=stage, stage_label=stage_label, message=message, inputs=inputs)
    thread = threading.Thread(target=_run_job, args=(job["job_id"], worker), daemon=True)
    thread.start()
    return {
        "job_id": job["job_id"],
        "status": job["status"],
        "operation": job["operation"],
        "message": message,
    }
