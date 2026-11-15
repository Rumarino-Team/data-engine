"""Saved checkpoints: isolated working copies, dirty tracking and atomic checkpoint publication."""
import os, re, shutil, uuid
from pathlib import Path
from typing import Any
from core.errors import OperationError
from core.session_layout import FRAMES_DIR, MASKS_DIR, OVERLAY_DIR_PREFIX, SESSION_FILE, TRACKING_RESULT_ID
from core.state import SessionState

MAX_DISPLAY_NAME_LENGTH = 120

def content_fingerprint(session: SessionState) -> tuple:
    """Identity of everything a checkpoint persists apart from editor preferences."""
    manifest = Path(session.mask_manifest_path).name if session.mask_manifest_path else None
    return (session.annotations.revision, manifest, session.session_metadata.get(TRACKING_RESULT_ID))

def saved_state(session: SessionState) -> dict[str, Any]:
    dirty = session.saved_fingerprint is None or session.saved_fingerprint != content_fingerprint(session)
    return {
        "checkpoint_id": session.checkpoint_id,
        "display_name": session.display_name,
        "saved_at": session.session_metadata.get("saved_at") if session.checkpoint_id else None,
        "dirty": dirty,
    }

def normalize_display_name(name: str) -> str:
    normalized = re.sub(r"\s+", " ", re.sub(r"[\x00-\x1f\x7f]", " ", name)).strip()
    if not normalized:
        raise OperationError(400, "Session name cannot be empty.")
    if len(normalized) > MAX_DISPLAY_NAME_LENGTH:
        raise OperationError(400, f"Session name cannot exceed {MAX_DISPLAY_NAME_LENGTH} characters.")
    return normalized

def new_checkpoint_location(display_name: str, root: Path) -> tuple[str, Path]:
    """A fresh identity and a readable directory; the display name is never the identity."""
    slug = re.sub(r"[^A-Za-z0-9]+", "-", display_name).strip("-").lower()[:40] or "session"
    while True:
        checkpoint_id = uuid.uuid4().hex
        directory = root / f"{slug}-{checkpoint_id[:8]}"
        if not directory.exists():
            return checkpoint_id, directory

def _link_or_copy(source: str, target: str) -> None:
    # Artifacts are written once and replaced atomically, so sharing storage cannot leak edits across copies.
    try:
        # Resolved first: on Linux a plain link() of a symlink links the symlink itself, not the file it names.
        os.link(os.path.realpath(source), target)
    except OSError:
        shutil.copy2(source, target)

def clone_session_directory(source: Path, target: Path) -> None:
    """Copy a session directory without session.json, which each copy writes for itself."""
    shutil.copytree(
        source, target, symlinks=True, copy_function=_link_or_copy,
        ignore=shutil.ignore_patterns(SESSION_FILE, ".session-*"),
    )

def link_files(source_root: Path, target_root: Path, relatives: set[str]) -> list[Path]:
    """Link or copy session-relative files that the target lacks; names are immutable, so existing ones match.

    Removes what it added if any file fails. Overlays are decorative, so a missing one is skipped."""
    added: list[Path] = []
    try:
        for relative in sorted(relatives):
            source, target = source_root / relative, target_root / relative
            if target.exists():
                continue
            if not source.is_file():
                if relative.startswith(f"{MASKS_DIR}/{OVERLAY_DIR_PREFIX}"):
                    continue
                raise OperationError(500, f"Session artifact is missing: {relative}")
            target.parent.mkdir(parents=True, exist_ok=True)
            _link_or_copy(str(source), str(target))
            added.append(target)
    except BaseException:
        for path in added:
            path.unlink(missing_ok=True)
        raise
    return added

def _write_frames(source: Path, target: Path) -> None:
    """Give a checkpoint its own files for every frame, so it does not depend on where the video came from.

    A working copy of a frames folder links to the user's original files. Copying those links as links would
    leave the checkpoint broken as soon as that folder is moved, deleted or opened on another machine, so each
    link is followed: the checkpoint gets a hard link to the file itself (a copy across filesystems). Moving or
    deleting the original no longer affects it; only rewriting the original's bytes in place would still show."""
    try:
        shutil.copytree(source, target, symlinks=False, copy_function=_link_or_copy)
    except shutil.Error as error:
        unreadable = error.args[0][0][0] if error.args and error.args[0] else source
        raise OperationError(500, f"A frame of this session can no longer be read, so it cannot be saved: {unreadable}") from error

def write_checkpoint(working: Path, target: Path, document: dict[str, Any], references: set[str],
                     known: set[str] | None) -> None:
    """Publish the working state to `target` without touching the previous checkpoint on failure.

    With `known` (the files an existing checkpoint holds) only the difference is copied: new files are added,
    the replaced `session.json` is the commit point, and superseded files are deleted afterward. Without it, a
    complete checkpoint (frames plus referenced files) is staged and moved into place."""
    from sessions.artifacts import remove_artifacts
    from sessions.metadata import write_metadata_file
    if known is not None and (target / SESSION_FILE).is_file():
        added = link_files(working, target, references - known)
        try:
            write_metadata_file(target, document)
        except BaseException:
            for path in added:
                path.unlink(missing_ok=True)
            raise
        remove_artifacts(target, known - references)
        return
    if target.exists():
        raise OperationError(409, "The checkpoint directory already exists or is damaged. Use Save As.")
    staging = target.parent / f".staging-{uuid.uuid4().hex}"
    try:
        (staging / MASKS_DIR).mkdir(parents=True)
        _write_frames(working / FRAMES_DIR, staging / FRAMES_DIR)
        link_files(working, staging, references)
        write_metadata_file(staging, document)
        os.replace(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

def sweep_stale_session_storage(working_root: Path, checkpoint_root: Path) -> None:
    """Working changes do not survive a restart; abandoned working and staging directories are removed."""
    if working_root.is_dir():
        for child in working_root.iterdir():
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
    if checkpoint_root.is_dir():
        for child in checkpoint_root.glob(".staging-*"):
            shutil.rmtree(child, ignore_errors=True)
