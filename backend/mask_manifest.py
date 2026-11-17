"""Propagation manifests (masks/manifest-<id>.json) read and written without parsing the whole file at once.

A manifest holds one entry per frame with every mask run of every object, so a long video makes it large:
parsed into Python objects it takes about ten times its size on disk. Only one frame is ever needed at a
time, or a few numbers per frame, so this module keeps a small index (where each frame sits in the file, which
input revision each of its masks was made under, its overlay) and reads a frame's JSON on demand. The file
format is unchanged. Manifests are published by atomic replacement, so a change shows up as a new file
identity or modification time; an in-place rewrite that forges both is not detected, and nothing writes one.
"""
import json
from pathlib import Path
from typing import Any, Iterator, NamedTuple, Optional
from utils import write_atomically

_CHUNK = 1 << 20


class FrameRecord(NamedTuple):
    """One frame while a job collects it: its JSON as it will be written, and what the index keeps of it."""
    raw: bytes
    stamps: dict[str, int]  # object -> input revision its mask in this frame was made under
    overlay: Optional[str]


class FrameEntry(NamedTuple):
    """One frame in a manifest file."""
    offset: int
    length: int
    stamps: dict[str, int]
    overlay: Optional[str]


def _summary(frame: dict[str, Any]) -> tuple[dict[str, int], Optional[str]]:
    revisions = frame.get("input_revisions", {})
    return {key: int(revisions.get(key, 0)) for key in frame.get("objects", {})}, frame.get("overlay_path") or None


def frame_record(frame: dict[str, Any]) -> FrameRecord:
    return FrameRecord(json.dumps(frame, ensure_ascii=True, separators=(",", ":")).encode("ascii"), *_summary(frame))


def _scan(path: Path) -> Iterator[tuple[str, int, int, dict[str, Any]]]:
    """Yield (key, start, end, frame) for each entry of the manifest's "frames", parsing one at a time.

    The file is read as latin-1 so a position in the text is a position in the file; manifests are written
    ASCII-only. Separators between values are skipped, not checked.
    """
    decoder = json.JSONDecoder()
    with path.open("r", encoding="latin-1", newline="") as handle:
        buf, i, base = "", 0, 0  # base is the file position of buf[0]

        def more() -> bool:
            nonlocal buf, i, base
            chunk = handle.read(max(_CHUNK, len(buf) - i))
            if chunk:
                base, buf, i = base + i, buf[i:] + chunk, 0
            return bool(chunk)

        def peek() -> str:
            nonlocal i
            while True:
                while i < len(buf) and buf[i] in " \t\r\n,:":
                    i += 1
                if i < len(buf):
                    return buf[i]
                if not more():
                    return ""

        def value() -> Any:
            nonlocal i
            peek()
            while True:
                try:
                    result, end = decoder.raw_decode(buf, i)
                    if end < len(buf) or not more():  # a value that ends with the buffer may be cut short
                        i = end
                        return result
                except json.JSONDecodeError:
                    if not more():
                        raise

        if peek() != "{":
            raise json.JSONDecodeError("Expecting a manifest object", buf, i)
        i += 1
        while peek() != "}":
            key = value()
            if key != "frames":
                value()
                continue
            if peek() != "{":
                raise json.JSONDecodeError("Expecting the frames object", buf, i)
            i += 1
            while peek() != "}":
                frame_key = value()
                peek()
                start = base + i
                frame = value()
                yield frame_key, start, base + i, frame
            i += 1


# The index of the manifest most recently read or written, with the file identity it describes.
_latest: Optional[tuple[tuple, dict[str, FrameEntry]]] = None


def _version(path: Path) -> tuple:
    stat = path.stat()
    return str(path), stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size, stat.st_ino


def manifest_index(path: Path) -> dict[str, FrameEntry]:
    """The frame key -> entry index of a manifest file; shared read-only data, callers must not mutate it."""
    global _latest
    path = path.resolve()
    version = _version(path)
    latest = _latest
    if latest is None or latest[0] != version:
        latest = _latest = (version, {key: FrameEntry(start, end - start, *_summary(frame))
                                      for key, start, end, frame in _scan(path)})
    return latest[1]


def read_frame(path: Path, key: str) -> dict[str, Any]:
    """One frame of a manifest file, parsed on request; empty when the manifest has no such frame."""
    path = path.resolve()
    entry = manifest_index(path).get(key)
    if entry is None:
        return {}
    with path.open("rb") as handle:
        handle.seek(entry.offset)
        return json.loads(handle.read(entry.length))


def read_records(path: Path) -> dict[str, FrameRecord]:
    """Every frame of a manifest file as written, in file order, without parsing it."""
    index = manifest_index(path)
    with path.resolve().open("rb") as handle:
        records = {}
        for key, entry in index.items():
            handle.seek(entry.offset)
            records[key] = FrameRecord(handle.read(entry.length), entry.stamps, entry.overlay)
        return records


def write_manifest(path: Path, header: dict[str, Any], records: dict[str, FrameRecord]) -> None:
    """Write `header` (a manifest whose "frames" is empty) with the frames put in place, exactly as
    json.dump of the whole manifest would, and remember the index of the new file."""
    global _latest
    head, tail = json.dumps(header, ensure_ascii=True, separators=(",", ":")).split('"frames":{}')
    index: dict[str, FrameEntry] = {}

    def write(handle) -> None:
        offset = handle.write(f'{head}"frames":{{'.encode("ascii"))
        for n, (key, record) in enumerate(records.items()):
            offset += handle.write(f'{"," if n else ""}{json.dumps(key)}:'.encode("ascii"))
            handle.write(record.raw)
            index[key] = FrameEntry(offset, len(record.raw), record.stamps, record.overlay)
            offset += len(record.raw)
        handle.write(f"}}{tail}".encode("ascii"))

    write_atomically(path, write, "wb")
    path = path.resolve()
    _latest = (_version(path), index)
