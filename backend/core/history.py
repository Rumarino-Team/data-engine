"""The undo history of one session: what its accepted inputs were before each recent accepted change.

A step holds references and small copies, never mask files: masks are immutable files named by uuid, so
"the live masks as they were" is a dict of file names. Propagation results are deliberately not part of a
step (they are derived output and large); undo restores the inputs and the results of clicks.
"""
import copy
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional

from core.annotations import SessionAnnotations
from core.config import TRACKING_METADATA_KEYS, UNDO_LIMIT


def tracking_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    return {key: copy.deepcopy(metadata[key]) for key in TRACKING_METADATA_KEYS if key in metadata}


@dataclass(frozen=True)
class HistoryState:
    """The accepted inputs as they were before (or, on the redo side, after) one change."""
    label: str
    annotations: SessionAnnotations
    # Frame -> interactive result artifact: the mask each click showed.
    live_masks: dict[str, str]
    tracking: dict[str, Any]


@dataclass
class UndoHistory:
    limit: int = UNDO_LIMIT
    undo: list[HistoryState] = field(default_factory=list)
    redo: list[HistoryState] = field(default_factory=list)
    # The state before an edit that is still being applied; responses built inside the edit count it.
    pending: Optional[HistoryState] = None

    def record(self, before: HistoryState) -> None:
        """A new accepted change: it can be undone, and whatever could have been redone no longer can."""
        self.redo.clear()
        self.push_undo(before)

    def push_undo(self, state: HistoryState) -> None:
        if self.limit <= 0:
            return
        self.undo.append(state)
        if len(self.undo) > self.limit:
            del self.undo[: len(self.undo) - self.limit]

    def states(self) -> Iterator[HistoryState]:
        yield from self.undo
        yield from self.redo

    def status(self, pending: Optional[HistoryState] = None) -> dict[str, Any]:
        """How many steps can be undone and redone, counting `pending` as the newest undo step."""
        undo = [*self.undo, pending] if pending is not None and self.limit > 0 else list(self.undo)
        undo = undo[-self.limit:] if self.limit > 0 else []
        redo = [] if pending is not None else self.redo
        return {"undo": len(undo), "redo": len(redo),
                "next_undo": undo[-1].label if undo else None,
                "next_redo": redo[-1].label if redo else None}
