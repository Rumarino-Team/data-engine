"""What one running backend shares: the active session, the lock ordering edits, and the job slot."""
from contextlib import contextmanager
import threading

from core.jobs import JobManager
from core.state import BackendState


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
    def edit(self):
        """Serialize a synchronous mutation of the session; refused while a job runs.
        Never suspend/await inside this block."""
        with self.lock:
            self.jobs.require_idle()
            yield
