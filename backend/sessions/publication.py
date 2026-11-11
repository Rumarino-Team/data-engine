"""Accepting a finished background job's results into the active session."""
from contextlib import contextmanager
from typing import Any, Iterator

from core.context import AppContext
from core.jobs import validate_job_inputs
from core.state import SessionState
from sessions.artifacts import reclaim_superseded_artifacts, sweep_unreferenced_artifacts


@contextmanager
def publish_results(ctx: AppContext, inputs: dict[str, Any], operation: str) -> Iterator[SessionState]:
    """Hold the session lock and accept a job's results only if the inputs it ran on are still current.

    If publishing fails (or the inputs went stale), the references it may have changed are restored and
    the files the job wrote are swept. When it succeeds, files the new references no longer name are deleted.
    """
    with ctx.lock:
        session = ctx.session
        previous = (session.mask_manifest_path, session.interactive_mask_files, dict(session.session_metadata))
        try:
            validate_job_inputs(session, inputs, operation)
            yield session
        except BaseException:
            session.mask_manifest_path, session.interactive_mask_files, session.session_metadata = previous
            sweep_unreferenced_artifacts(session)
            raise
        reclaim_superseded_artifacts(session)
