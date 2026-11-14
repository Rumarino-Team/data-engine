from typing import Any
from core.config import SAVED_ROOT
from core.context import AppContext
from core.errors import OperationError
from core.jobs import utc_now_iso
from schemas.video import AnnotationPrecondition, VideoSaveRequest
from sessions.artifacts import referenced_artifacts
from sessions.checkpoints import content_fingerprint, new_checkpoint_location, normalize_display_name, saved_state, write_checkpoint
from sessions.metadata import current_session_path, session_metadata_document

def save_video_session(ctx: AppContext, request: VideoSaveRequest, precondition: AnnotationPrecondition):
    # Save persists the backend's current state, so it must be the state this client last saw.
    with ctx.edit(precondition):
        session = ctx.session

        session_path = current_session_path(session)
        if session_path is None or session.video_dir is None or not session.video_frame_files:
            raise OperationError(400, "Video session is not initialized.")

        # Re-save replaces this session's own checkpoint; Save As (or a first save) mints a new identity.
        resave = session.checkpoint_id is not None and not request.save_as
        if request.name is None and not resave:
            raise OperationError(400, "A session name is required for the first save and for Save As.")
        display_name = normalize_display_name(request.name) if request.name is not None else session.display_name
        if resave:
            checkpoint_id, target = session.checkpoint_id, session.checkpoint_dir
        else:
            SAVED_ROOT.mkdir(parents=True, exist_ok=True)
            checkpoint_id, target = new_checkpoint_location(display_name, SAVED_ROOT)

        saved_at = utc_now_iso()
        overrides: dict[str, Any] = {
            "checkpoint_id": checkpoint_id, "display_name": display_name, "saved_at": saved_at,
            "saved_revision": session.annotations.revision,
            "resolved_video_frames_dir": str(target / "frames"),
        }
        if request.interactive_state is not None:
            overrides["interactive_state"] = request.interactive_state.model_dump()
        references = referenced_artifacts(session)
        if references is None:
            raise OperationError(500, "The session's mask results cannot be read, so it cannot be saved.")
        target.parent.mkdir(parents=True, exist_ok=True)
        write_checkpoint(session_path, target, session_metadata_document(session, overrides), references,
                         session.checkpoint_artifacts if resave else None)

        # The checkpoint is published; the working copy adopts its identity.
        session.adopt_checkpoint(checkpoint_id, target, display_name)
        session.checkpoint_artifacts = references
        session.saved_fingerprint = content_fingerprint(session)
        session.session_metadata["saved_at"] = saved_at
        if "interactive_state" in overrides:
            session.session_metadata["interactive_state"] = overrides["interactive_state"]
        return {
            "message": "Session saved successfully",
            "name": display_name,
            "checkpoint_id": checkpoint_id,
            "saved_path": str(target),
            "saved_state": saved_state(session),
        }
