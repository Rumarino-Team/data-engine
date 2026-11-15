"""Accepting a finished background job's results into the active session."""
from contextlib import contextmanager
from typing import Any, Iterator

from core.context import AppContext
from core.history import HistoryState, tracking_metadata
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
        before = tracking_metadata(previous[2])
        if (operation == "prompt_tracking"
                and session.session_metadata.get("latest_tracking_result_id") != before.get("latest_tracking_result_id")):
            # A tracking run is an undo step (it brings back the previous result); a propagation is not: its
            # masks are derived output and large, so undoing an input never takes one away.
            session.history.record(HistoryState(
                "Track points", session.annotations.model_copy(deep=True), dict(previous[1]), before))
        reclaim_superseded_artifacts(session)
