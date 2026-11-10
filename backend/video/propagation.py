from dataclasses import dataclass, field
import logging, shutil, uuid
from pathlib import Path
from typing import Any, Optional
import cv2
import numpy as np
from core.config import WINDOW_FRAMES_ROOT
from core.context import AppContext
from core.errors import OperationError
from core.jobs import JobHandle, capture_job_inputs, validate_job_inputs
from core.runtime import cleanup_cuda_memory, is_out_of_memory
from core.state import BackendState
from schemas.video import VideoPropagateRequest
from sessions.cache import clear_window_cache
from sessions.checkpoints import saved_state
from sessions.metadata import current_masks_dir
from sessions.publication import publish_results
from utils import build_empty_mask_manifest, load_mask_manifest, prepare_video_masks_output, save_single_video_mask_frame, write_mask_manifest
from video.io import build_window_dir
from video.masks import manifest_frame_payload, override_pixels
from video.prompts import create_masker_runtime
from video.runtime import MaskerRestorationError, MaskerSettings
from tracking.guidance import TrackingGuidance, load_latest_tracking_guidance

logger = logging.getLogger(__name__)

def propagation_batch_progress(
    *,
    processed_before_frame: int,
    window_total_frames: int,
    batch_size: int,
) -> dict[str, int]:
    effective_batch_size = max(1, int(batch_size))
    total_frames = max(1, int(window_total_frames))
    processed_before = min(max(0, int(processed_before_frame)), total_frames - 1)
    batch_index = (processed_before // effective_batch_size) + 1
    batch_count = (total_frames + effective_batch_size - 1) // effective_batch_size
    batch_start = (batch_index - 1) * effective_batch_size
    batch_total = min(effective_batch_size, total_frames - batch_start)
    batch_current = (processed_before - batch_start) + 1
    return {
        "batch_current": batch_current,
        "batch_total": batch_total,
        "batch_index": batch_index,
        "batch_count": batch_count,
    }

def start_propagation(ctx: AppContext, request: VideoPropagateRequest, precondition):
    return ctx.start_job(
        precondition, operation="mask_propagation", stage="validating_prompts", stage_label="Validating prompts",
        message="Mask propagation queued", parameters=request.model_dump(),
        run=lambda job, inputs: run_propagation_job(ctx, request, job, inputs),
    )


@dataclass
class Window:
    """A part of the requested range that SAM2 is initialized on, so only its frames are held in memory."""
    index: int
    count: int
    start: int
    end: int
    expected_frames: int = 0  # frames this window adds to the output
    processed: int = 0

    @property
    def number(self) -> int:
        return self.index + 1


@dataclass
class PropagationRun:
    """Everything one propagation job shares between its steps."""
    session: BackendState
    request: VideoPropagateRequest
    job: JobHandle
    settings: MaskerSettings
    start_frame_idx: int
    end_frame_idx: int
    frame_files: list
    masks_dir: Path
    manifest_path: Path
    manifest: dict[str, Any]
    frame_width: int
    frame_height: int
    guidance: Optional[TrackingGuidance]
    windows: list[Window]
    run_root: Path
    overlay_dir: Path
    # Filled in while the windows run.
    processed_frames: set[int] = field(default_factory=set)
    boundary_masks: dict[int, np.ndarray] = field(default_factory=dict)
    boundary_frame_idx: Optional[int] = None
    saved_mask_frame_count: int = 0
    save_failures: int = 0
    saved_mask_paths: dict[int, list[str]] = field(default_factory=dict)
    video_segments: dict[int, dict[int, list]] = field(default_factory=dict)
    tracked_points_seeded: int = 0
    tracked_points_seeded_frames: set[int] = field(default_factory=set)

    @property
    def expected_total_frames(self) -> int:
        return self.end_frame_idx - self.start_frame_idx + 1

    @property
    def manifest_frames(self) -> dict[str, Any]:
        return self.manifest["frames"]

    def report(self, window: Window, stage: str, label: str, message: str, *, frame_idx: int,
               current: Optional[int] = None, total: Optional[int] = None, **extra: Any) -> None:
        processed = len(self.processed_frames)
        self.job.enter_stage(
            stage, label,
            current=processed if current is None else current,
            total=self.expected_total_frames if total is None else total,
            window_index=window.number,
            window_count=window.count,
            frame_idx=frame_idx,
            progress=(processed / self.expected_total_frames) if self.expected_total_frames else 0.0,
            message=message,
            **extra,
        )


def plan_propagation(session: BackendState, request: VideoPropagateRequest) -> tuple[MaskerSettings, int, int]:
    """Check the session can propagate; return the settings to use and the frame range to cover."""
    if session.video_masker is None:
        raise OperationError(400, "Video masker not active.")
    if session.video_dir is None:
        raise OperationError(400, "Video directory not set. Call /video/init_state first.")
    if not session.video_frame_files:
        raise OperationError(400, "No video frames available. Call /video/init_state first.")
    if not session.annotations.prompts and not session.annotations.mask_overrides:
        raise OperationError(400, "No prompts available for propagation.")

    masker = session.video_masker
    batch_size = masker.default_batch_size if request.batch_size is None else int(request.batch_size)
    if batch_size <= 0:
        raise OperationError(400, "batch_size must be a positive integer.")
    settings = MaskerSettings(
        online_mode=masker.online_mode if request.online_mode is None else bool(request.online_mode),
        batch_size=batch_size,
        offload_video_to_cpu=masker.offload_video_to_cpu,
        offload_state_to_cpu=masker.offload_state_to_cpu,
    )

    num_frames = len(session.video_frame_files)
    if request.start_frame_idx is not None:
        start_frame_idx = int(request.start_frame_idx)
    else:
        start_frame_idx = min(event.frame_idx for event in [*session.annotations.prompts, *session.annotations.mask_overrides])
    start_frame_idx = min(max(start_frame_idx, 0), num_frames - 1)
    if request.max_frame_num_to_track is None:
        end_frame_idx = num_frames - 1
    else:
        end_frame_idx = min(num_frames - 1, start_frame_idx + int(request.max_frame_num_to_track) - 1)
    if end_frame_idx < start_frame_idx:
        raise OperationError(400, "Invalid propagation frame range.")
    return settings, start_frame_idx, end_frame_idx


def open_run(session: BackendState, request: VideoPropagateRequest, job: JobHandle, inputs: dict[str, Any]) -> PropagationRun:
    """Plan the job and prepare its outputs: the manifest to fill in, the tracked-point guidance and the windows."""
    settings, start_frame_idx, end_frame_idx = plan_propagation(session, request)
    expected_total_frames = end_frame_idx - start_frame_idx + 1
    masks_root = current_masks_dir(session)
    if masks_root is None:
        raise OperationError(400, "Video session cache is not initialized.")
    clear_window_cache()
    job.enter_stage(
        "preparing_masks", "Preparing mask output",
        progress=0.02,
        current=0,
        total=expected_total_frames,
        message="Preparing mask output directory",
    )

    frame_files, masks_dir = prepare_video_masks_output(session.video_dir, masks_root, clear_existing=False)
    manifest_path = masks_dir / f"manifest-{uuid.uuid4().hex}.json"
    first_frame = cv2.imread(str(Path(session.video_dir) / session.video_frame_files[start_frame_idx]))
    if first_frame is None:
        raise OperationError(500, "Unable to read first frame for manifest metadata.")
    frame_height, frame_width = int(first_frame.shape[0]), int(first_frame.shape[1])
    manifest = build_empty_mask_manifest(
        source_video_path=session.video_source_path,
        resolved_video_frames_dir=str(session.video_dir),
        num_frames=len(session.video_frame_files),
        frame_height=frame_height,
        frame_width=frame_width,
    )
    manifest["session_id"] = inputs["session_id"]
    manifest["input_revision"] = inputs["input_revision"]
    previous_manifest_path = Path(session.mask_manifest_path) if session.mask_manifest_path else None
    if previous_manifest_path and previous_manifest_path.exists():
        # Each processed frame replaces its previous result; other ranges survive.
        manifest["frames"].update(load_mask_manifest(previous_manifest_path).get("frames", {}))

    guidance = None
    if request.use_tracked_points:
        guidance = load_latest_tracking_guidance(session, num_frames=len(session.video_frame_files))

    split_frame_idx = (start_frame_idx + end_frame_idx) // 2
    ranges = [(start_frame_idx, split_frame_idx)]
    if split_frame_idx < end_frame_idx:
        ranges.append((split_frame_idx, end_frame_idx))
    windows = [Window(index, len(ranges), start, end) for index, (start, end) in enumerate(ranges)]

    run_root = WINDOW_FRAMES_ROOT / f"run_{uuid.uuid4().hex[:12]}"
    run_root.mkdir(parents=True, exist_ok=True)
    overlay_dir = masks_dir / f"overlays_{uuid.uuid4().hex}"
    if request.export_overlay_frames:
        overlay_dir.mkdir()
    return PropagationRun(
        session=session, request=request, job=job, settings=settings,
        start_frame_idx=start_frame_idx, end_frame_idx=end_frame_idx, frame_files=frame_files,
        masks_dir=masks_dir, manifest_path=manifest_path, manifest=manifest,
        frame_width=frame_width, frame_height=frame_height, guidance=guidance, windows=windows,
        run_root=run_root, overlay_dir=overlay_dir,
    )


def build_window(run: PropagationRun, window: Window) -> Path:
    """Link the window's frames into their own directory for SAM2 to load."""
    session = run.session
    run.report(
        window, "building_window", "Loading propagation window",
        f"Preparing window {window.number} of {window.count}: frames {window.start}-{window.end}",
        frame_idx=window.start, current=0, total=window.end - window.start + 1,
    )
    frame_paths = [Path(session.video_dir) / session.video_frame_files[idx] for idx in range(window.start, window.end + 1)]

    def on_progress(current: int, total: int, source_path: Path) -> None:
        run.report(
            window, "building_window", "Loading propagation window",
            f"Linked {current} of {total} frames for window {window.number} of {window.count}",
            frame_idx=window.start + current - 1, current=current, total=total,
            append_history=current == 1 or current == total,
        )

    return build_window_dir(frame_paths, run.run_root, f"window_{window.index}_{window.start}_{window.end}",
                            progress_callback=on_progress)


def seed_window(run: PropagationRun, masker, window: Window) -> None:
    """Give SAM2 everything it propagates from: the previous window's last masks, prompts, explicit masks and tracked points."""
    session, request = run.session, run.request
    run.report(window, "seeding_window", "Seeding prompts",
               f"Seeding prompts for window {window.number} of {window.count}", frame_idx=window.start)

    seeds_boundary = window.index > 0 and bool(run.boundary_masks) and run.boundary_frame_idx is not None
    if seeds_boundary:
        run.report(window, "seeding_window", "Seeding boundary masks",
                   f"Seeding boundary masks for window {window.number} of {window.count}",
                   frame_idx=run.boundary_frame_idx)
        local_boundary_idx = int(run.boundary_frame_idx - window.start)
        for obj_id, obj_mask in run.boundary_masks.items():
            masker.add_new_mask(frame_idx=local_boundary_idx, obj_id=int(obj_id), mask=np.asarray(obj_mask).astype(bool))

    for event in session.annotations.prompt_events():
        event_frame_idx = int(event["frame_idx"])
        if not window.start <= event_frame_idx <= window.end:
            continue
        masker.add_new_points_or_box(
            frame_idx=event_frame_idx - window.start,
            obj_id=int(event["obj_id"]),
            points=event["points"] if event["points"] else None,
            labels=event["labels"] if event["labels"] else None,
            clear_old_points=True,
            box=event.get("box"),
        )

    for mask in session.annotations.mask_overrides:
        if window.start <= mask.frame_idx <= window.end:
            masker.add_new_mask(frame_idx=mask.frame_idx - window.start, obj_id=mask.obj_id,
                                mask=override_pixels(mask, session.active_session_dir))

    if run.guidance is not None and run.guidance.points:
        excluded = {(mask.frame_idx, mask.obj_id) for mask in session.annotations.mask_overrides}
        if seeds_boundary:
            excluded.update((int(run.boundary_frame_idx), int(obj_id)) for obj_id in run.boundary_masks)
        batches, seeded_count, seeded_frames = run.guidance.build_window_batches(
            window_start=window.start,
            window_end=window.end,
            propagation_start_frame_idx=run.start_frame_idx,
            frame_width=run.frame_width,
            frame_height=run.frame_height,
            keyframe_interval=request.tracked_point_keyframe_interval,
            max_points_per_object_per_frame=request.max_tracked_points_per_object_per_frame,
            excluded_frame_objects=excluded,
        )
        for local_frame_idx, obj_batches in sorted(batches.items()):
            for obj_id, points in sorted(obj_batches.items()):
                if points:
                    masker.add_new_points_or_box(
                        frame_idx=int(local_frame_idx), obj_id=int(obj_id), points=points,
                        labels=[1] * len(points), clear_old_points=False, box=None,
                    )
        run.tracked_points_seeded += seeded_count
        run.tracked_points_seeded_frames.update(seeded_frames)


def remember_boundary(run: PropagationRun, frame_idx: int, frame_masks: dict[int, np.ndarray]) -> None:
    run.boundary_masks = {int(obj_id): np.asarray(mask).astype(bool) for obj_id, mask in frame_masks.items()}
    run.boundary_frame_idx = frame_idx


def collect_frame(run: PropagationRun, window: Window, local_frame_idx: int, frame_masks: dict[int, np.ndarray]) -> None:
    """Take one propagated frame: report it, add it to the manifest and export its overlay."""
    session, request = run.session, run.request
    global_frame_idx = int(window.start + local_frame_idx)
    if global_frame_idx < run.start_frame_idx or global_frame_idx > run.end_frame_idx:
        return
    if window.index > 0 and global_frame_idx == window.start and global_frame_idx in run.processed_frames:
        # The previous window already produced this overlap frame; it only supplies the boundary masks.
        remember_boundary(run, global_frame_idx, frame_masks)
        return

    batch_progress = propagation_batch_progress(
        processed_before_frame=window.processed,
        window_total_frames=max(1, window.expected_frames),
        batch_size=run.settings.batch_size,
    )
    window.processed += 1
    run.processed_frames.add(global_frame_idx)
    processed = len(run.processed_frames)
    run.report(
        window, "propagating_window", "Propagating masks", f"Processed {processed} of {run.expected_total_frames} frames",
        frame_idx=global_frame_idx, append_history=processed == 1 or processed == run.expected_total_frames,
        **batch_progress,
    )
    # Explicit masks remain exact inputs even if inference alters their pixels.
    for mask in session.annotations.mask_overrides:
        if mask.frame_idx == global_frame_idx:
            frame_masks[mask.obj_id] = override_pixels(mask, session.active_session_dir)
    run.manifest_frames[str(global_frame_idx)] = {
        **manifest_frame_payload(frame_masks),
        "input_revisions": {str(obj_id): session.annotations.input_revisions.get(int(obj_id), 0)
                            for obj_id in frame_masks},
    }
    if request.include_masks_in_response:
        run.video_segments[global_frame_idx] = {
            int(obj_id): np.asarray(mask).astype(bool).tolist() for obj_id, mask in frame_masks.items()
        }
    if request.export_overlay_frames:
        try:
            saved_path = save_single_video_mask_frame(run.frame_files, run.overlay_dir, global_frame_idx, frame_masks)
            if saved_path is not None:
                run.manifest_frames[str(global_frame_idx)]["overlay_path"] = str(Path(saved_path).relative_to(run.masks_dir))
                run.saved_mask_frame_count += 1
                if request.include_saved_mask_paths:
                    run.saved_mask_paths.setdefault(global_frame_idx, []).append(saved_path)
        except Exception:
            run.save_failures += 1
            logger.exception("Failed to save propagated mask frame %s", global_frame_idx)
    if global_frame_idx == window.end:
        remember_boundary(run, global_frame_idx, frame_masks)


def propagate_window(run: PropagationRun, masker, window: Window) -> None:
    settings = run.settings
    output_start = max(run.start_frame_idx, window.start)
    local_start_frame = output_start - window.start
    window.expected_frames = window.end - output_start + 1
    if window.index > 0 and window.start in run.processed_frames:
        window.expected_frames = max(0, window.expected_frames - 1)
    window.processed = 0
    masker.propagate_in_video(
        start_frame_idx=local_start_frame,
        max_frame_num_to_track=(window.end - window.start + 1) - local_start_frame,
        reverse=False,
        batch_size=settings.batch_size,
        online_mode=settings.online_mode,
        collect_segments=False,
        frame_callback=lambda local_frame_idx, frame_masks: collect_frame(run, window, local_frame_idx, frame_masks),
    )
    run.report(window, "propagating_window", "Propagating masks",
               f"Finished window {window.number} of {window.count}", frame_idx=window.end)


def run_window(run: PropagationRun, masker, window: Window) -> None:
    settings = run.settings
    window_dir = build_window(run, window)
    run.report(window, "initializing_state", "Initializing SAM2 window state",
               f"Initializing SAM2 state for window {window.number} of {window.count}", frame_idx=window.start)
    masker.init_state(
        str(window_dir),
        online_mode=settings.online_mode,
        batch_size=settings.batch_size,
        offload_video_to_cpu=settings.offload_video_to_cpu,
        offload_state_to_cpu=settings.offload_state_to_cpu,
        async_loading_frames=False,
    )
    seed_window(run, masker, window)
    propagate_window(run, masker, window)


def publish_run(ctx: AppContext, run: PropagationRun, inputs: dict[str, Any]) -> BackendState:
    """Accept the manifest as the session's current result, replacing the frames it re-processed."""
    run.job.enter_stage(
        "saving_manifest", "Saving manifest",
        progress=0.98,
        current=len(run.processed_frames),
        total=run.expected_total_frames,
        message="Saving mask manifest",
    )
    with publish_results(ctx, inputs, "mask_propagation") as session:
        write_mask_manifest(run.manifest_path, run.manifest)
        session.interactive_mask_files = {key: value for key, value in session.interactive_mask_files.items()
                                          if int(key) not in run.processed_frames}
        session.mask_manifest_path = str(run.manifest_path)
    return session


def propagation_result(run: PropagationRun, inputs: dict[str, Any], session: BackendState) -> dict[str, Any]:
    request, settings = run.request, run.settings
    seeded = run.tracked_points_seeded
    guidance = run.guidance
    skipped_reason = guidance.skipped_reason if guidance is not None else None
    should_warn = guidance.should_warn if guidance is not None else False
    return {
        "session_id": inputs["session_id"],
        "input_revision": inputs["input_revision"],
        "video_segments": run.video_segments if request.include_masks_in_response else {},
        "video_segments_total_frames": len(run.processed_frames),
        "video_segments_returned_frames": len(run.video_segments),
        "saved_mask_frame_count": run.saved_mask_frame_count,
        "saved_mask_save_failures": run.save_failures,
        "saved_mask_paths": run.saved_mask_paths,
        "online_mode": settings.online_mode,
        "batch_size": settings.batch_size,
        "tracked_points_used": seeded > 0,
        "tracked_points_skipped_reason": (
            None
            if seeded > 0 or not request.use_tracked_points or not should_warn
            else skipped_reason or "No CoTracker points were seeded during propagation."
        ),
        "tracked_points_seeded_count": seeded,
        "tracked_points_seeded_frames": len(run.tracked_points_seeded_frames),
        "mask_manifest_path": session.mask_manifest_path,
        "saved_state": saved_state(session),
    }


def run_propagation_job(ctx: AppContext, request: VideoPropagateRequest, job: JobHandle, inputs=None) -> dict[str, Any]:
    session = ctx.session
    inputs = inputs or capture_job_inputs(session, "mask_propagation")
    validate_job_inputs(session, inputs, "mask_propagation")
    job.enter_stage(
        "validating_prompts", "Validating prompts",
        status="running",
        progress=0.0,
        message="Validating mask propagation inputs",
    )
    run = open_run(session, request, job, inputs)

    def report_restore_progress(_stage: str, _label: str, progress: Optional[float], message: str) -> None:
        job.enter_stage("restoring_interactive_state", "Restoring interactive state", progress=progress, message=message)

    try:
        # Leaving this scope reloads the session's own frames and inputs before publication.
        with create_masker_runtime(session).borrowed(run.settings, progress_callback=report_restore_progress) as masker:
            for window in run.windows:
                run_window(run, masker, window)
    except ValueError as error:
        raise OperationError(400, str(error)) from error
    except MaskerRestorationError as error:
        raise OperationError(503, "Predictor restoration failed. Reload the video; no results were published.") from error
    except RuntimeError as error:
        if is_out_of_memory(error):
            cleanup_cuda_memory()
            raise OperationError(
                507, "CUDA out of memory during propagation. Try lowering batch_size or enabling CPU offload."
            ) from error
        raise OperationError(500, f"Failed to propagate video masks: {error}") from error
    except Exception as error:
        raise OperationError(500, f"Failed to propagate video masks: {error}") from error
    finally:
        shutil.rmtree(run.run_root, ignore_errors=True)

    published = publish_run(ctx, run, inputs)
    return propagation_result(run, inputs, published)
