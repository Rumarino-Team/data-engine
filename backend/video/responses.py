from core.state import SessionState
from schemas.video import AnnotationPrecondition
from sessions.checkpoints import saved_state
from sessions.metadata import current_session_path
from tracking.results import tracking_validity

def annotation_update(session: SessionState, precondition: AnnotationPrecondition):
    """Mutation response: only accepted changes since the validated precondition revision."""
    return {
        "session_id": session.active_session_id,
        "annotation_delta": session.annotations.take_delta(precondition.expected_revision),
        "tracking_validity": tracking_validity(session),
        "saved_state": saved_state(session),
    }


def annotation_snapshot(session: SessionState):
    return {
        "session_id": session.active_session_id,
        "annotations": session.annotations.snapshot() if current_session_path(session) is not None else None,
        "tracking_validity": tracking_validity(session),
        "saved_state": saved_state(session) if current_session_path(session) is not None else None,
    }
