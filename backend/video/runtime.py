"""Session-bound SAM2 lifecycle, independent of jobs and concrete model imports."""

import contextlib
from contextlib import contextmanager
from dataclasses import dataclass
import logging
from typing import Callable, Iterator, Protocol

from core.state import SessionState


ProgressCallback = Callable[[str, str, float | None, str], None]


class VideoMasker(Protocol):
    online_mode: bool
    default_batch_size: int
    offload_video_to_cpu: bool
    offload_state_to_cpu: bool
    state_loaded: bool

    def reset_state(self) -> None: ...

    def offload_weights(self) -> None: ...

    def reload_weights(self) -> None: ...

    def release_state(self) -> None: ...

    def init_state(
        self, video_dir: str, *, online_mode: bool, batch_size: int | None,
        offload_video_to_cpu: bool | None, offload_state_to_cpu: bool | None,
        async_loading_frames: bool, progress_callback: ProgressCallback | None,
    ) -> None: ...


class MaskerFactory(Protocol):
    def __call__(self, *, progress_callback: ProgressCallback | None, reuse: VideoMasker | None = None) -> VideoMasker: ...


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

    session: SessionState
    create_masker: MaskerFactory
    replay_inputs: Callable[[SessionState], None]
    release_memory: Callable[[], None]

    def initialize(
        self, settings: MaskerSettings, *, async_loading_frames: bool = False,
        progress_callback: ProgressCallback | None = None, reuse: VideoMasker | None = None,
    ) -> None:
        """Build the model if absent, load the session's frames and replay its accepted inputs.

        With `reuse`, the new masker takes the weights another masker already loaded.
        """
        if self.session.video_masker is None:
            options = {"progress_callback": progress_callback}
            if reuse is not None:
                options["reuse"] = reuse
            self.session.video_masker = self.create_masker(**options)
        self.session.video_masker.init_state(
            self.session.video_dir,
            online_mode=settings.online_mode,
            batch_size=settings.batch_size,
            offload_video_to_cpu=settings.offload_video_to_cpu,
            offload_state_to_cpu=settings.offload_state_to_cpu,
            async_loading_frames=async_loading_frames,
            progress_callback=progress_callback,
        )
        self.replay_inputs(self.session)

    def restore(
        self, settings: MaskerSettings, *, progress_callback: ProgressCallback | None = None,
    ) -> None:
        if self.session.video_dir is None:
            return
        self.initialize(settings, progress_callback=progress_callback)

    def rebuild_inputs(self) -> None:
        """Reset the predictor and replay accepted inputs, e.g. after a rejected edit.

        On failure the predictor is disabled, because it no longer matches accepted state.
        """
        try:
            self.session.video_masker.reset_state()
            self.replay_inputs(self.session)
        except Exception as error:
            self.session.video_masker = None
            logging.getLogger(__name__).exception("Prompt runtime recovery failed")
            raise MaskerRestorationError(str(error)) from error

    @contextmanager
    def offloaded(self) -> Iterator[None]:
        """Move SAM2's weights to CPU RAM while another model needs the GPU, then move them back.

        The video state stays loaded, so nothing is rebuilt. If the weights cannot come back after the work
        succeeded, MaskerRestorationError is raised so the caller does not publish results beside a predictor
        that cannot run; if the work failed, moving them back is best effort and the original error wins.
        """
        masker = self.session.video_masker
        if masker is None:
            yield
            return
        try:
            masker.offload_weights()
        except Exception:
            with contextlib.suppress(MaskerRestorationError):
                self._reload_weights(masker)
            raise
        self.release_memory()
        try:
            yield
        except Exception:
            with contextlib.suppress(MaskerRestorationError):
                self._reload_weights(masker)
            raise
        else:
            self._reload_weights(masker)

    def _reload_weights(self, masker: VideoMasker) -> None:
        try:
            masker.reload_weights()
        except Exception as error:
            self.session.video_masker = None
            logging.getLogger(__name__).exception("Failed to move the SAM2 weights back to the GPU")
            raise MaskerRestorationError(str(error)) from error

    @contextmanager
    def propagating(self) -> Iterator[VideoMasker]:
        """Run a long computation on the session's own video state, which stays loaded.

        The state is reset to the accepted inputs before the computation and again afterwards, so what the
        computation added never leaks into interactive editing, and no frames are loaded again. If the
        computation succeeds but the state cannot be put back, MaskerRestorationError is raised so the
        caller does not publish results next to a predictor that no longer matches them; if the computation
        fails, putting the state back is best effort and the original error wins.
        """
        if self.session.video_masker is None:
            raise ValueError("Video masker not active.")
        self.rebuild_inputs()
        try:
            yield self.session.video_masker
        except Exception:
            with contextlib.suppress(MaskerRestorationError):
                self.rebuild_inputs()
            raise
        else:
            self.rebuild_inputs()
