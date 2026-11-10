"""Shared fixtures for tests that touch the process-global backend state.

`state` and the job registry are globals, so every test that uses them must start from a clean copy and put
the original back. Keeping that in one place also means a future move to explicitly constructed application
state only has to change this module, not every test file.
"""
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Optional
from unittest.mock import Mock

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.factory import create_app
from core import jobs
from core.state import BackendState, state


class BackendStateFixtures:
    """Mixin for TestCase and IsolatedAsyncioTestCase: isolated `state`, idle jobs, temp dirs, API client."""

    _state_snapshot: Optional[dict[str, Any]] = None

    def reset_backend_state(self, **fields: Any) -> BackendState:
        """Replace `state` with a fresh BackendState and clear the job registry; undone when the test ends."""
        if self._state_snapshot is None:
            self._state_snapshot = state.__dict__.copy()
            with jobs.current_job_lock:
                self._job_snapshot = jobs.current_job
            self.addCleanup(self._restore_backend_state)
        state.__dict__.clear()
        state.__dict__.update(BackendState(**fields).__dict__)
        with jobs.current_job_lock:
            jobs.current_job = None
        return state

    def _restore_backend_state(self) -> None:
        state.__dict__.clear()
        state.__dict__.update(self._state_snapshot)
        with jobs.current_job_lock:
            jobs.current_job = self._job_snapshot
        self._state_snapshot = None

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
        client = TestClient(create_app())
        if preconditions:
            def add_preconditions(request):
                request.headers.setdefault("X-Session-ID", state.active_session_id or "absent")
                request.headers.setdefault("X-Annotation-Revision", str(state.annotations.revision))
            client.event_hooks["request"].append(add_preconditions)
        return client

    def activate_job(self, status: str = "running") -> None:
        """Make the job registry report an active job, as a long operation would."""
        with jobs.current_job_lock:
            jobs.current_job = {
                "job_id": "active", "operation": "video_init", "status": status, "stage": "working",
                "stage_label": "Working", "progress": 0.5, "current": None, "total": None,
                "window_index": None, "window_count": None, "frame_idx": None, "stage_history": [],
                "message": "working", "result": None, "error": None, "started_at": "now",
                "updated_at": "now", "completed_at": None,
            }


class BackendStateTestCase(BackendStateFixtures, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.reset_backend_state()
