"""Saved checkpoints: isolated working copies, dirty tracking and atomic checkpoint publication."""
import os, re, shutil, uuid
from pathlib import Path
from typing import Any
from core.errors import OperationError
from core.state import BackendState, state

MAX_DISPLAY_NAME_LENGTH = 120

def content_fingerprint(session: BackendState = state) -> tuple:
    """Identity of everything a checkpoint persists apart from editor preferences."""
    manifest = Path(session.mask_manifest_path).name if session.mask_manifest_path else None
    return (session.annotations.revision, manifest, session.session_metadata.get("latest_tracking_result_id"))

def saved_state(session: BackendState = state) -> dict[str, Any]:
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
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)

def clone_session_directory(source: Path, target: Path) -> None:
    """Copy a session directory without session.json, which each copy writes for itself."""
    shutil.copytree(
        source, target, symlinks=True, copy_function=_link_or_copy,
        ignore=shutil.ignore_patterns("session.json", ".session-*"),
    )

def publish_checkpoint(staging: Path, target: Path) -> None:
    """Replace the checkpoint with a complete staged one; the previous checkpoint survives any failure."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        os.replace(staging, target)
        return
    retired = target.with_name(f".retired-{uuid.uuid4().hex}")
    os.replace(target, retired)
    try:
        os.replace(staging, target)
    except BaseException:
        os.replace(retired, target)
        raise
    shutil.rmtree(retired, ignore_errors=True)

def sweep_stale_session_storage(working_root: Path, checkpoint_root: Path) -> None:
    """Working changes do not survive a restart; abandoned working and staging directories are removed."""
    if working_root.is_dir():
        for child in working_root.iterdir():
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
    if checkpoint_root.is_dir():
        for child in checkpoint_root.glob(".staging-*"):
            shutil.rmtree(child, ignore_errors=True)
