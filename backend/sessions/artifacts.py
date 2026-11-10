"""Physical artifact lifecycle: a working session deletes files its committed metadata no longer names.

Deletion only follows a commit (or the restoration of committed references after a rejected edit), so a
file is removed once no accepted state can reach it. It is best effort and never fails an accepted edit.
Only directories and name patterns this backend writes are touched.

Each commit deletes only what the previous commit named and this one does not, so edit cost follows the
number of changed files rather than the number of existing ones. A full directory scan (`sweep_...`) runs
when a session opens and after rejected edits or publications, which can leave unreferenced files behind.
"""
import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional
from core.state import BackendState

logger = logging.getLogger(__name__)

@lru_cache(maxsize=1)
def _read_manifest_overlays(manifest_path: str, mtime_ns: int, size: int) -> frozenset[str]:
    from utils import load_mask_manifest
    frames = load_mask_manifest(Path(manifest_path)).get("frames", {})
    return frozenset(f"masks/{Path(frame['overlay_path']).as_posix()}"
                     for frame in frames.values() if frame.get("overlay_path"))

def _manifest_overlays(manifest_path: str) -> frozenset[str]:
    # Published manifests are immutable, so the overlay set is cached per file version.
    stat = Path(manifest_path).stat()
    return _read_manifest_overlays(manifest_path, stat.st_mtime_ns, stat.st_size)

def referenced_artifacts(session: BackendState) -> Optional[set[str]]:
    """Session-relative paths the session's accepted state names; None when that cannot be determined."""
    refs = {f"masks/overrides/{mask.artifact}" for mask in session.annotations.mask_overrides}
    refs.update(f"masks/interactive/{name}" for name in session.interactive_mask_files.values() if isinstance(name, str))
    tracking_path = session.session_metadata.get("latest_tracking_result_path")
    if tracking_path:
        refs.add(f"tracking/{Path(tracking_path).name}")
    if session.mask_manifest_path:
        manifest = Path(session.mask_manifest_path)
        try:
            overlays = _manifest_overlays(str(manifest))
        except (OSError, ValueError):
            # Without a readable manifest its overlays are unknown, so keep every result file.
            return None
        refs.add(f"masks/{manifest.name}")
        refs.update(overlays)
    return refs

def _unlink(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        logger.warning("Unable to remove obsolete artifact %s", path, exc_info=True)

def _remove_empty_overlay_dir(directory: Path) -> None:
    if directory.name.startswith("overlays_"):
        try:
            directory.rmdir()  # Only succeeds once empty; a directory still holding kept overlays stays.
        except OSError:
            pass

def remove_artifacts(root: Path, relatives: set[str]) -> None:
    """Best-effort deletion of session-relative files and of overlay directories they leave empty."""
    for relative in relatives:
        path = root / relative
        _unlink(path)
        _remove_empty_overlay_dir(path.parent)

def reclaim_superseded_artifacts(session: BackendState) -> None:
    """After a commit: delete what the previous commit named and this one does not."""
    if session.active_session_dir is None:
        return
    try:
        refs = referenced_artifacts(session)
        if refs is None:
            return
        remove_artifacts(Path(session.active_session_dir), session.committed_artifacts - refs)
        session.committed_artifacts = refs
    except Exception:
        logger.warning("Unable to reclaim superseded session artifacts", exc_info=True)

def sweep_unreferenced_artifacts(session: BackendState) -> None:
    """Full scan for files no accepted state names: leftovers of rejected edits and of opened checkpoints."""
    if session.active_session_dir is None:
        return
    try:
        refs = referenced_artifacts(session)
        if refs is None:
            return
        root = Path(session.active_session_dir)
        owned = [(root / "masks" / "overrides", "*.json"), (root / "masks" / "interactive", "*.json"),
                 (root / "tracking", "*.json"), (root / "masks", "manifest-*.json")]
        for directory, pattern in owned:
            if directory.is_dir():
                for path in directory.glob(pattern):
                    if path.is_file() and path.relative_to(root).as_posix() not in refs:
                        _unlink(path)
        for overlay_dir in (root / "masks").glob("overlays_*"):
            if overlay_dir.is_dir():
                for path in overlay_dir.rglob("*"):
                    if path.is_file() and path.relative_to(root).as_posix() not in refs:
                        _unlink(path)
                _remove_empty_overlay_dir(overlay_dir)
        session.committed_artifacts = refs
    except Exception:
        logger.warning("Unable to sweep unreferenced session artifacts", exc_info=True)
