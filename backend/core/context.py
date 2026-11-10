"""What one running backend shares: the active session, the lock ordering edits, and the job slot."""
from contextlib import contextmanager
import threading
from typing import Any, Callable, Iterator

from core.errors import OperationError
from core.jobs import JobHandle, JobManager, capture_job_inputs
from core.state import BackendState
from schemas.video import AnnotationPrecondition


def validate_annotation_precondition(session: BackendState, precondition: AnnotationPrecondition) -> None:
    """The client's edit must target the session and revision it last saw."""
    if session.active_session_dir is None or precondition.session_id != session.active_session_id:
        raise OperationError(409, "The target session is no longer active. Refresh annotations.")
    if precondition.expected_revision != session.annotations.revision:
        raise OperationError(409, "Annotations changed. Refresh before editing again.")


class SessionManager:
    """Owns the active session. Replacing it is one reference swap; whoever holds the old one keeps a
    consistent (now unreachable) session instead of a half-overwritten one."""

    def __init__(self, session: BackendState | None = None) -> None:
        self.current = session if session is not None else BackendState()
        # Foreground edits, snapshots, session replacement and job admission all take this lock.
        self.lock = threading.RLock()

    def replace(self, candidate: BackendState) -> None:
        """Make `candidate` the active session; the caller holds `lock`."""
        self.current = candidate


class AppContext:
    """Created once per application (`create_app`) and handed to every service."""

    def __init__(self, sessions: SessionManager | None = None, jobs: JobManager | None = None) -> None:
        self.sessions = sessions if sessions is not None else SessionManager()
        self.jobs = jobs if jobs is not None else JobManager(self.sessions.lock)

    @property
    def session(self) -> BackendState:
        return self.sessions.current

    @property
    def lock(self) -> threading.RLock:
        return self.sessions.lock

    @contextmanager
    def edit(self, precondition: AnnotationPrecondition | None = None) -> Iterator[BackendState]:
        """Serialize a synchronous mutation of the session and yield it; refused while a job runs.
        With a precondition, the edit must also target the session and revision the client last saw.
        Never suspend/await inside this block."""
        with self.lock:
            self.jobs.require_idle()
            session = self.session
            if precondition is not None:
                validate_annotation_precondition(session, precondition)
            yield session

    def start_job(
        self, precondition: AnnotationPrecondition, *, operation: str, stage: str, stage_label: str,
        message: str, parameters: dict[str, Any], run: Callable[[JobHandle, dict[str, Any]], dict[str, Any]],
    ) -> dict[str, Any]:
        """Admit a background job against the state the client last saw; `run(job, inputs)` is its worker."""
        with self.edit(precondition) as session:
            inputs = capture_job_inputs(session, operation)
            inputs["parameters"] = parameters
            return self.jobs.start(
                operation=operation, stage=stage, stage_label=stage_label, message=message,
                worker=lambda job: run(job, inputs), inputs=inputs,
            )
