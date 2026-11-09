"""Session-bound SAM2 lifecycle, independent of jobs and concrete model imports."""

from contextlib import contextmanager
from dataclasses import dataclass
import logging
from typing import Callable, Iterator, Protocol

from core.state import BackendState


ProgressCallback = Callable[[str, str, float | None, str], None]


class VideoMasker(Protocol):
    online_mode: bool
    default_batch_size: int
    offload_video_to_cpu: bool
    offload_state_to_cpu: bool

    def init_state(
        self, video_dir: str, *, online_mode: bool, batch_size: int | None,
        offload_video_to_cpu: bool | None, offload_state_to_cpu: bool | None,
        async_loading_frames: bool, progress_callback: ProgressCallback | None,
    ) -> None: ...


class MaskerFactory(Protocol):
    def __call__(self, *, progress_callback: ProgressCallback | None) -> VideoMasker: ...


@dataclass(frozen=True)
class MaskerSettings:
    online_mode: bool
    batch_size: int | None
    offload_video_to_cpu: bool | None
    offload_state_to_cpu: bool | None


class MaskerRestorationError(RuntimeError):
    """Interactive segmentation is unavailable after attempted restoration."""


@dataclass
class VideoMaskerRuntime:
    """Caller serializes mutations; accepted annotations and publication stay external."""

    session: BackendState
    create_masker: MaskerFactory
    replay_inputs: Callable[[BackendState], None]
    persist_session: Callable[[BackendState], None]
    release_memory: Callable[[], None]

    def restore(
        self, settings: MaskerSettings, *, progress_callback: ProgressCallback | None = None,
        increment_epoch: bool = True,
    ) -> None:
        if self.session.video_dir is None:
            return
        if self.session.video_masker is None:
            self.session.video_masker = self.create_masker(progress_callback=progress_callback)
        self.session.video_masker.init_state(
            self.session.video_dir,
            online_mode=settings.online_mode,
            batch_size=settings.batch_size,
            offload_video_to_cpu=settings.offload_video_to_cpu,
            offload_state_to_cpu=settings.offload_state_to_cpu,
            async_loading_frames=False,
            progress_callback=progress_callback,
        )
        self.replay_inputs(self.session)
        if increment_epoch:
            self.session.video_state_epoch += 1
            self.persist_session(self.session)

    def _settings_for_restoration(self) -> MaskerSettings | None:
        masker = self.session.video_masker
        if masker is None or self.session.video_dir is None:
            return None
        return MaskerSettings(
            online_mode=masker.online_mode,
            batch_size=masker.default_batch_size,
            offload_video_to_cpu=masker.offload_video_to_cpu,
            offload_state_to_cpu=masker.offload_state_to_cpu,
        )

    def _restore_after_suspension(
        self, settings: MaskerSettings | None, progress_callback: ProgressCallback | None,
        *, raise_on_error: bool,
    ) -> None:
        if settings is None:
            return
        try:
            self.restore(settings, progress_callback=progress_callback)
        except Exception as error:
            self.session.video_masker = None
            self.session.video_state_epoch += 1
            logging.getLogger(__name__).exception("Failed to restore interactive video masker state")
            if raise_on_error:
                raise MaskerRestorationError(str(error)) from error

    @contextmanager
    def suspended(self, *, progress_callback: ProgressCallback | None = None) -> Iterator[None]:
        """Release SAM2 for another computation and restore it before returning.

        If computation fails, restoration is best effort and the original error wins.
        If computation succeeds, failed restoration prevents subsequent publication.
        """
        # Retain settings only: holding the previous masker would retain its weights.
        settings = self._settings_for_restoration()
        if self.session.video_masker is not None:
            self.session.video_masker = None
            self.session.video_state_epoch += 1
            self.release_memory()
        try:
            yield
        except Exception:
            self._restore_after_suspension(settings, progress_callback, raise_on_error=False)
            raise
        else:
            self._restore_after_suspension(settings, progress_callback, raise_on_error=True)
