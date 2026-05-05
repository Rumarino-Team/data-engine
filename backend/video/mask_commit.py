import shutil
import uuid
from pathlib import Path

from fastapi import HTTPException


def cleanup_stale_mask_commit_dirs(session_dir: Path) -> None:
    session_dir = Path(session_dir)
    masks_dir = session_dir / "masks"
    pending_dirs = list(session_dir.glob(".masks_pending_*"))
    backup_dirs = list(session_dir.glob(".masks_backup_*"))

    for pending_dir in pending_dirs:
        shutil.rmtree(pending_dir, ignore_errors=True)

    if masks_dir.exists() and (masks_dir / "manifest.json").exists():
        for backup_dir in backup_dirs:
            shutil.rmtree(backup_dir, ignore_errors=True)
        return

    if not masks_dir.exists() and len(backup_dirs) == 1:
        backup_dirs[0].rename(masks_dir)
        return

    for backup_dir in backup_dirs:
        shutil.rmtree(backup_dir, ignore_errors=True)


def prepare_pending_masks_dir(session_dir: Path) -> Path:
    cleanup_stale_mask_commit_dirs(session_dir)
    pending_dir = Path(session_dir) / f".masks_pending_{uuid.uuid4().hex}"
    pending_dir.mkdir(parents=True, exist_ok=False)
    return pending_dir


def commit_pending_masks_dir(session_dir: Path, pending_dir: Path) -> Path:
    session_dir = Path(session_dir)
    pending_dir = Path(pending_dir)
    committed_dir = session_dir / "masks"
    backup_dir = session_dir / f".masks_backup_{uuid.uuid4().hex}"
    moved_old = False

    try:
        if committed_dir.exists():
            committed_dir.rename(backup_dir)
            moved_old = True
        pending_dir.rename(committed_dir)
    except Exception as error:
        if committed_dir.exists() and not moved_old:
            pass
        elif not committed_dir.exists() and moved_old and backup_dir.exists():
            backup_dir.rename(committed_dir)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to commit propagated masks: {error}",
        ) from error

    shutil.rmtree(backup_dir, ignore_errors=True)
    return committed_dir
