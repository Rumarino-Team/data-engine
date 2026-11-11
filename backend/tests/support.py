"""Shared fixtures for tests that need a backend: every test builds its own AppContext.

`state` is a read/write view of the running test's active session, so assertions can say `state.annotations`
without threading the context everywhere; the context itself is `self.ctx`.
"""
import sys
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any, Optional
from unittest.mock import Mock

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.factory import create_app
from core.context import AppContext, SessionManager
from core.jobs import JobHandle, JobManager
from core.state import BackendState
from sessions.artifacts import reclaim_superseded_artifacts

_active: Optional[AppContext] = None


class _ActiveSession:
    """The running test's session; never holds state of its own."""

    @property
    def __dict__(self):
        return _active.session.__dict__

    def __getattr__(self, name):
        return getattr(_active.session, name)

    def __setattr__(self, name, value):
        setattr(_active.session, name, value)


state = _ActiveSession()


class BackendStateFixtures:
    """Mixin for TestCase and IsolatedAsyncioTestCase: isolated `state`, idle jobs, temp dirs, API client."""

    ctx: Optional[AppContext] = None

    def reset_backend_state(self, **fields: Any) -> BackendState:
        """Start from a fresh session and idle jobs. The first call creates this test's context; later calls
        (a table test starting over) reset it in place so clients already bound to it stay valid."""
        global _active
        if getattr(self, "ctx", None) is None:
            self.ctx = _active = AppContext(SessionManager(BackendState(**fields)))
            self.addCleanup(_deactivate)
        else:
            self.ctx.sessions.current = BackendState(**fields)
            self.ctx.jobs = JobManager(self.ctx.sessions.lock)
        return self.ctx.session

    def temp_dir(self) -> Path:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        return Path(directory.name)

    def make_masker(self, **overrides: Any) -> Mock:
        settings = dict(online_mode=True, default_batch_size=1, offload_video_to_cpu=True, offload_state_to_cpu=True)
        return Mock(**{**settings, **overrides})

    def working_session(self, path: Optional[Path] = None, *, session_id: str = "session",
                        frames: tuple[str, ...] = ("00000.jpg",), frame_bytes: bytes = b"",
                        **fields: Any) -> Path:
        """Create a working directory with frames/ and masks/ and make it the active session."""
        path = path or self.temp_dir() / "working"
        (path / "frames").mkdir(parents=True)
        (path / "masks").mkdir()
        for name in frames:
            (path / "frames" / name).write_bytes(frame_bytes)
        state.active_session_dir = path
        state.active_session_id = session_id
        state.video_dir = str(path / "frames")
        state.video_frame_files = list(frames)
        for name, value in fields.items():
            setattr(state, name, value)
        return path

    def make_client(self, *, preconditions: bool = True) -> TestClient:
        """An API client; by default every request carries the active session and revision headers."""
        client = TestClient(create_app(self.ctx))
        if preconditions:
            def add_preconditions(request):
                request.headers.setdefault("X-Session-ID", state.active_session_id or "absent")
                request.headers.setdefault("X-Annotation-Revision", str(state.annotations.revision))
            client.event_hooks["request"].append(add_preconditions)
        return client

    def activate_job(self, status: str = "running") -> str:
        """Make the job manager report an active job, as a long operation would; returns its id."""
        jobs = self.ctx.jobs
        job_id = jobs._admit("video_init", stage="working", stage_label="Working", message="working")
        jobs._job["status"] = status
        return job_id

    def finished_job(self, result: dict[str, Any]) -> str:
        """Run a job to completion with this result; returns its id."""
        jobs = self.ctx.jobs
        jobs.start(operation="video_init", stage="init", stage_label="Init", message="init",
                   worker=lambda handle: result)
        deadline = time.time() + 5
        while jobs.is_active() and time.time() < deadline:
            time.sleep(0.001)
        return jobs.snapshot()["job_id"]

    def job_handle(self) -> Mock:
        """A stand-in for the handle a job worker receives; progress calls are recorded, not stored."""
        return Mock(spec=JobHandle)


def commit_session(session, extra=None) -> None:
    """What an accepted change does to a working copy: record in-memory metadata, reclaim superseded files."""
    session.session_metadata.update(extra or {})
    reclaim_superseded_artifacts(session)


def _deactivate() -> None:
    global _active
    _active = None


class BackendStateTestCase(BackendStateFixtures, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.reset_backend_state()
