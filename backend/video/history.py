"""Undo and redo: the accepted inputs go back to what they were before a recent change, or forward again."""
import copy

from core.context import AppContext
from core.errors import OperationError
from core.history import TRACKING_METADATA_KEYS, HistoryState, tracking_metadata
from core.state import SessionState
from schemas.video import AnnotationPrecondition
from video.prompts import annotation_edit, create_masker_runtime
from video.responses import annotation_update


def capture_state(session: SessionState, label: str) -> HistoryState:
    return HistoryState(label, session.annotations.model_copy(deep=True), dict(session.interactive_mask_files),
                        tracking_metadata(session.session_metadata))


def restore_state(session: SessionState, state: HistoryState) -> None:
    session.annotations.restore_to(state.annotations)
    session.interactive_mask_files = dict(state.live_masks)
    for key in TRACKING_METADATA_KEYS:
        if key in state.tracking:
            session.session_metadata[key] = copy.deepcopy(state.tracking[key])
        else:
            session.session_metadata.pop(key, None)


def _step(ctx: AppContext, precondition: AnnotationPrecondition, *, undoing: bool) -> dict:
    verb = "Undo" if undoing else "Redo"
    with ctx.edit(precondition) as session:
        session.require_masker()
        history = session.history
        source, other = (history.undo, history.redo) if undoing else (history.redo, history.undo)
        if not source:
            raise OperationError(400, f"Nothing to {verb.lower()}.")
        target = source[-1]
        now = capture_state(session, target.label)
        metadata_before = dict(session.session_metadata)
        try:
            with annotation_edit(session):
                restore_state(session, target)
                create_masker_runtime(session).rebuild_inputs()
                # Moved inside the block: the files `now` names must be kept before the commit reclaims anything.
                source.pop()
                other.append(now)
        except Exception:
            session.session_metadata = metadata_before
            raise
        return {**annotation_update(session, precondition), "message": f"{verb}: {target.label}"}


def undo_last_change(ctx: AppContext, precondition: AnnotationPrecondition) -> dict:
    return _step(ctx, precondition, undoing=True)


def redo_last_change(ctx: AppContext, precondition: AnnotationPrecondition) -> dict:
    return _step(ctx, precondition, undoing=False)
