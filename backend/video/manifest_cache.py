"""Keep only the most recently requested mask manifest, invalidating on file changes."""

from functools import lru_cache
import json
from pathlib import Path
from typing import Any


@lru_cache(maxsize=1)
def _read_manifest(path: str, version: tuple[int, int, int, int]) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_cached_mask_manifest(path: Path) -> dict[str, Any]:
    """Return shared read-only data; callers must not mutate the returned manifest."""
    resolved = path.resolve()
    stat = resolved.stat()
    version = (stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size, stat.st_ino)
    return _read_manifest(str(resolved), version)
