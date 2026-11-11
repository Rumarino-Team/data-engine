"""Session-bound CoTracker lifecycle, independent of jobs and concrete model imports."""

from dataclasses import dataclass
from typing import Any, Callable, Protocol

import numpy as np

from core.state import SessionState


class VideoTracker(Protocol):
    model_name: str
    device: str
    model_resolution: tuple[int, int]

    def prepare_video(self, video: np.ndarray) -> Any: ...

    def track(self, video: Any, *, queries: np.ndarray, add_support_grid: bool) -> tuple[np.ndarray, np.ndarray]: ...

    def track_streaming(
        self, frame_loader: Any, *, queries: np.ndarray, add_support_grid: bool,
    ) -> tuple[np.ndarray, np.ndarray]: ...


class TrackerFactory(Protocol):
    def __call__(self, *, model_name: str) -> VideoTracker: ...


@dataclass
class VideoTrackerRuntime:
    """Caller serializes mutations. The tracker holds model weights only, so unlike the masker
    there are no accepted inputs to replay and nothing to restore after it is released."""

    session: SessionState
    create_tracker: TrackerFactory
    release_memory: Callable[[], None]

    def ensure(self, model_name: str) -> VideoTracker:
        """The session's tracker for `model_name`, built if absent; a different model replaces the old one."""
        tracker = self.session.tracker
        if tracker is not None and tracker.model_name == model_name:
            return tracker
        self.release()
        self.session.tracker = self.create_tracker(model_name=model_name)
        return self.session.tracker

    def release(self) -> None:
        if self.session.tracker is not None:
            self.session.tracker = None
            self.release_memory()
