from core.errors import OperationError
import logging, shutil, uuid
from pathlib import Path
from typing import Any, Optional
import cv2
import numpy as np
import torch
from core.config import WINDOW_FRAMES_ROOT
from core.context import AppContext
from core.jobs import JobHandle, capture_job_inputs, validate_job_inputs
from core.runtime import cleanup_cuda_memory
from schemas.video import VideoPropagateRequest
from sessions.artifacts import reclaim_superseded_artifacts, sweep_unreferenced_artifacts
from sessions.cache import clear_window_cache
from sessions.checkpoints import saved_state
from sessions.metadata import current_masks_dir
from utils import build_empty_mask_manifest, load_mask_manifest, prepare_video_masks_output, save_single_video_mask_frame, write_mask_manifest
from video.io import build_window_dir
from video.masks import manifest_frame_payload, override_pixels
from video.prompts import create_masker_runtime
from video.runtime import MaskerRestorationError, MaskerSettings
from tracking.guidance import load_latest_tracking_guidance

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
    from video.service import validate_annotation_precondition
    with ctx.edit():
        session = ctx.session
        validate_annotation_precondition(session, precondition)
        inputs = capture_job_inputs(session, "mask_propagation")
        inputs["parameters"] = request.model_dump()
        return ctx.jobs.start(
            operation="mask_propagation",
            stage="validating_prompts",
            stage_label="Validating prompts",
            message="Mask propagation queued",
            worker=lambda job: run_propagation_job(ctx, request, job, inputs),
            inputs=inputs,
        )

def run_propagation_job(ctx: AppContext, request: VideoPropagateRequest, job: JobHandle, inputs=None) -> dict[str, Any]:
    session = ctx.session
    inputs = inputs or capture_job_inputs(session, "mask_propagation")
    validate_job_inputs(session, inputs, "mask_propagation")
    def _propagation_progress(processed_frames_count: int, expected_frames_count: int) -> float:
        return (processed_frames_count / expected_frames_count) if expected_frames_count else 0.0

    job.enter_stage(
        "validating_prompts", "Validating prompts",
        status="running",
        progress=0.0,
        message="Validating mask propagation inputs",
    )
    if session.video_masker is None:
        raise OperationError(400, "Video masker not active.")
    if session.video_dir is None:
        raise OperationError(400, "Video directory not set. Call /video/init_state first.")

    if not session.video_frame_files:
        raise OperationError(400, "No video frames available. Call /video/init_state first.")

    if not session.annotations.prompts and not session.annotations.mask_overrides:
        raise OperationError(400, "No prompts available for propagation.")

    effective_online_mode = session.video_masker.online_mode if request.online_mode is None else bool(request.online_mode)
    effective_batch_size = session.video_masker.default_batch_size if request.batch_size is None else int(request.batch_size)
    effective_offload_video_to_cpu = session.video_masker.offload_video_to_cpu
    effective_offload_state_to_cpu = session.video_masker.offload_state_to_cpu
    if effective_batch_size <= 0:
        raise OperationError(400, "batch_size must be a positive integer.")

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
    previous_manifest_path = Path(session.mask_manifest_path) if session.mask_manifest_path else None
    manifest_file_path = masks_dir / f"manifest-{uuid.uuid4().hex}.json"

    first_frame = cv2.imread(str(Path(session.video_dir) / session.video_frame_files[start_frame_idx]))
    if first_frame is None:
        raise OperationError(500, "Unable to read first frame for manifest metadata.")

    manifest = build_empty_mask_manifest(
        source_video_path=session.video_source_path,
        resolved_video_frames_dir=str(session.video_dir),
        num_frames=num_frames,
        frame_height=int(first_frame.shape[0]),
        frame_width=int(first_frame.shape[1]),
    )
    manifest["session_id"] = inputs["session_id"]
    manifest["input_revision"] = inputs["input_revision"]
    manifest_frames: dict[str, Any] = manifest["frames"]
    if previous_manifest_path and previous_manifest_path.exists():
        # Each processed frame replaces its previous result; other ranges survive.
        manifest_frames.update(load_mask_manifest(previous_manifest_path).get("frames", {}))
    frame_height = int(first_frame.shape[0])
    frame_width = int(first_frame.shape[1])

    tracking_guidance = None
    tracked_points_skipped_reason: Optional[str] = None
    tracked_points_skip_should_warn = False
    if request.use_tracked_points:
        tracking_guidance = load_latest_tracking_guidance(session, num_frames=num_frames)
        tracked_points_skipped_reason = tracking_guidance.skipped_reason
        tracked_points_skip_should_warn = tracking_guidance.should_warn
    total_tracked_points_seeded = 0
    tracked_points_seeded_frames: set[int] = set()

    split_frame_idx = (start_frame_idx + end_frame_idx) // 2
    windows: list[tuple[int, int]] = [(start_frame_idx, split_frame_idx)]
    if split_frame_idx < end_frame_idx:
        windows.append((split_frame_idx, end_frame_idx))

    run_root = WINDOW_FRAMES_ROOT / f"run_{uuid.uuid4().hex[:12]}"
    run_root.mkdir(parents=True, exist_ok=True)
    overlay_dir = masks_dir / f"overlays_{uuid.uuid4().hex}"
    if request.export_overlay_frames:
        overlay_dir.mkdir()

    saved_mask_paths_serializable: dict[int, list[str]] = {}
    saved_mask_frame_count = 0
    save_failures = 0
    processed_frames: set[int] = set()
    boundary_masks: dict[int, np.ndarray] = {}
    boundary_frame_idx: Optional[int] = None

    video_segments_serializable: dict[int, dict[int, list]] = {}

    runtime = create_masker_runtime(session)
    restore_settings = MaskerSettings(
        online_mode=effective_online_mode, batch_size=effective_batch_size,
        offload_video_to_cpu=effective_offload_video_to_cpu,
        offload_state_to_cpu=effective_offload_state_to_cpu,
    )

    def _report_restore_progress(_stage: str, _label: str, progress: Optional[float], message: str) -> None:
        job.enter_stage(
            "restoring_interactive_state", "Restoring interactive state",
            progress=progress, message=message,
        )

    try:
        # Leaving this scope reloads the session's own frames and inputs before publication.
        with runtime.borrowed(restore_settings, progress_callback=_report_restore_progress) as masker:
            for window_index, (window_start, window_end) in enumerate(windows):
                window_number = window_index + 1
                window_count = len(windows)
                job.enter_stage(
                    "building_window", "Loading propagation window",
                    current=0,
                    total=window_end - window_start + 1,
                    window_index=window_number,
                    window_count=window_count,
                    frame_idx=window_start,
                    progress=_propagation_progress(len(processed_frames), expected_total_frames),
                    message=f"Preparing window {window_number} of {window_count}: frames {window_start}-{window_end}",
                )
                window_frame_paths = [Path(session.video_dir) / session.video_frame_files[idx] for idx in range(window_start, window_end + 1)]
                window_name = f"window_{window_index}_{window_start}_{window_end}"

                def _on_window_build_progress(current: int, total: int, source_path: Path) -> None:
                    source_frame_idx = window_start + current - 1
                    job.enter_stage(
                        "building_window", "Loading propagation window",
                        current=current,
                        total=total,
                        window_index=window_number,
                        window_count=window_count,
                        frame_idx=source_frame_idx,
                        progress=_propagation_progress(len(processed_frames), expected_total_frames),
                        message=f"Linked {current} of {total} frames for window {window_number} of {window_count}",
                        append_history=current == 1 or current == total,
                    )

                window_dir = build_window_dir(
                    window_frame_paths,
                    run_root,
                    window_name,
                    progress_callback=_on_window_build_progress,
                )

                job.enter_stage(
                    "initializing_state", "Initializing SAM2 window state",
                    current=len(processed_frames),
                    total=expected_total_frames,
                    window_index=window_number,
                    window_count=window_count,
                    frame_idx=window_start,
                    progress=_propagation_progress(len(processed_frames), expected_total_frames),
                    message=f"Initializing SAM2 state for window {window_number} of {window_count}",
                )

                masker.init_state(
                    str(window_dir),
                    online_mode=effective_online_mode,
                    batch_size=effective_batch_size,
                    offload_video_to_cpu=effective_offload_video_to_cpu,
                    offload_state_to_cpu=effective_offload_state_to_cpu,
                    async_loading_frames=False,
                )

                job.enter_stage(
                    "seeding_window", "Seeding prompts",
                    current=len(processed_frames),
                    total=expected_total_frames,
                    window_index=window_number,
                    window_count=window_count,
                    frame_idx=window_start,
                    progress=_propagation_progress(len(processed_frames), expected_total_frames),
                    message=f"Seeding prompts for window {window_number} of {window_count}",
                )

                if window_index > 0 and boundary_masks and boundary_frame_idx is not None:
                    job.enter_stage(
                        "seeding_window", "Seeding boundary masks",
                        current=len(processed_frames),
                        total=expected_total_frames,
                        window_index=window_number,
                        window_count=window_count,
                        frame_idx=boundary_frame_idx,
                        progress=_propagation_progress(len(processed_frames), expected_total_frames),
                        message=f"Seeding boundary masks for window {window_number} of {window_count}",
                    )
                    local_boundary_idx = int(boundary_frame_idx - window_start)
                    for obj_id, obj_mask in boundary_masks.items():
                        masker.add_new_mask(
                            frame_idx=local_boundary_idx,
                            obj_id=int(obj_id),
                            mask=np.asarray(obj_mask).astype(bool),
                        )

                for event in session.annotations.prompt_events():
                    event_frame_idx = int(event["frame_idx"])
                    if event_frame_idx < window_start or event_frame_idx > window_end:
                        continue
                    local_event_frame_idx = event_frame_idx - window_start
                    points = event["points"] if event["points"] else None
                    labels = event["labels"] if event["labels"] else None
                    masker.add_new_points_or_box(
                        frame_idx=local_event_frame_idx,
                        obj_id=int(event["obj_id"]),
                        points=points,
                        labels=labels,
                        clear_old_points=True,
                        box=event.get("box"),
                    )

                for mask in session.annotations.mask_overrides:
                    if window_start <= mask.frame_idx <= window_end:
                        masker.add_new_mask(frame_idx=mask.frame_idx - window_start,
                                                       obj_id=mask.obj_id, mask=override_pixels(mask, session.active_session_dir))

                if tracking_guidance is not None and tracking_guidance.points:
                    excluded_tracked_guidance = {(mask.frame_idx, mask.obj_id) for mask in session.annotations.mask_overrides}
                    if window_index > 0 and boundary_masks and boundary_frame_idx is not None:
                        excluded_tracked_guidance.update(
                            (int(boundary_frame_idx), int(obj_id))
                            for obj_id in boundary_masks.keys()
                        )
                    tracked_batches, seeded_count, seeded_frames = tracking_guidance.build_window_batches(
                        window_start=window_start,
                        window_end=window_end,
                        propagation_start_frame_idx=start_frame_idx,
                        frame_width=frame_width,
                        frame_height=frame_height,
                        keyframe_interval=request.tracked_point_keyframe_interval,
                        max_points_per_object_per_frame=request.max_tracked_points_per_object_per_frame,
                        excluded_frame_objects=excluded_tracked_guidance,
                    )
                    for local_frame_idx, obj_batches in sorted(tracked_batches.items()):
                        for obj_id, points in sorted(obj_batches.items()):
                            if not points:
                                continue
                            labels = [1] * len(points)
                            masker.add_new_points_or_box(
                                frame_idx=int(local_frame_idx),
                                obj_id=int(obj_id),
                                points=points,
                                labels=labels,
                                clear_old_points=False,
                                box=None,
                            )
                    total_tracked_points_seeded += seeded_count
                    tracked_points_seeded_frames.update(seeded_frames)

                local_start_frame = max(start_frame_idx, window_start) - window_start
                local_max_frames = (window_end - window_start + 1) - local_start_frame
                window_output_start = max(start_frame_idx, window_start)
                window_expected_frames = window_end - window_output_start + 1
                if window_index > 0 and window_start in processed_frames:
                    window_expected_frames = max(0, window_expected_frames - 1)
                processed_window_frames_count = 0

                def _on_window_frame(local_frame_idx: int, frame_masks: dict[int, np.ndarray]):
                    nonlocal saved_mask_frame_count, save_failures, boundary_masks, boundary_frame_idx, processed_window_frames_count
                    global_frame_idx = int(window_start + local_frame_idx)
                    if global_frame_idx < start_frame_idx or global_frame_idx > end_frame_idx:
                        return

                    is_overlap_duplicate = (
                        window_index > 0
                        and global_frame_idx == window_start
                        and global_frame_idx in processed_frames
                    )
                    if is_overlap_duplicate:
                        boundary_masks = {
                            int(obj_id): np.asarray(mask).astype(bool)
                            for obj_id, mask in frame_masks.items()
                        }
                        boundary_frame_idx = global_frame_idx
                        return

                    batch_progress = propagation_batch_progress(
                        processed_before_frame=processed_window_frames_count,
                        window_total_frames=max(1, window_expected_frames),
                        batch_size=effective_batch_size,
                    )
                    processed_window_frames_count += 1
                    processed_frames.add(global_frame_idx)
                    job.enter_stage(
                        "propagating_window", "Propagating masks",
                        current=len(processed_frames),
                        total=expected_total_frames,
                        window_index=window_number,
                        window_count=window_count,
                        frame_idx=global_frame_idx,
                        progress=_propagation_progress(len(processed_frames), expected_total_frames),
                        batch_current=batch_progress["batch_current"],
                        batch_total=batch_progress["batch_total"],
                        batch_index=batch_progress["batch_index"],
                        batch_count=batch_progress["batch_count"],
                        message=f"Processed {len(processed_frames)} of {expected_total_frames} frames",
                        append_history=len(processed_frames) == 1 or len(processed_frames) == expected_total_frames,
                    )
                    # Explicit masks remain exact inputs even if inference alters their pixels.
                    for mask in session.annotations.mask_overrides:
                        if mask.frame_idx == global_frame_idx:
                            frame_masks[mask.obj_id] = override_pixels(mask, session.active_session_dir)
                    manifest_frames[str(global_frame_idx)] = {
                        **manifest_frame_payload(frame_masks),
                        "input_revisions": {str(obj_id): session.annotations.input_revisions.get(int(obj_id), 0)
                                            for obj_id in frame_masks},
                    }
                    if request.include_masks_in_response:
                        video_segments_serializable[global_frame_idx] = {
                            int(obj_id): np.asarray(mask).astype(bool).tolist()
                            for obj_id, mask in frame_masks.items()
                        }

                    if request.export_overlay_frames:
                        try:
                            saved_path = save_single_video_mask_frame(
                                frame_files,
                                overlay_dir,
                                global_frame_idx,
                                frame_masks,
                            )
                            if saved_path is not None:
                                manifest_frames[str(global_frame_idx)]["overlay_path"] = str(Path(saved_path).relative_to(masks_dir))
                                saved_mask_frame_count += 1
                                if request.include_saved_mask_paths:
                                    saved_mask_paths_serializable.setdefault(global_frame_idx, []).append(saved_path)
                        except Exception:
                            save_failures += 1
                            logger.exception("Failed to save propagated mask frame %s", global_frame_idx)

                    if global_frame_idx == window_end:
                        boundary_masks = {
                            int(obj_id): np.asarray(mask).astype(bool)
                            for obj_id, mask in frame_masks.items()
                        }
                        boundary_frame_idx = global_frame_idx

                masker.propagate_in_video(
                    start_frame_idx=local_start_frame,
                    max_frame_num_to_track=local_max_frames,
                    reverse=False,
                    batch_size=effective_batch_size,
                    online_mode=effective_online_mode,
                    collect_segments=False,
                    frame_callback=_on_window_frame,
                )
                job.enter_stage(
                    "propagating_window", "Propagating masks",
                    current=len(processed_frames),
                    total=expected_total_frames,
                    window_index=window_number,
                    window_count=window_count,
                    frame_idx=window_end,
                    progress=_propagation_progress(len(processed_frames), expected_total_frames),
                    message=f"Finished window {window_number} of {window_count}",
                )
    except ValueError as error:
        raise OperationError(400, str(error)) from error
    except MaskerRestorationError as error:
        raise OperationError(503, "Predictor restoration failed. Reload the video; no results were published.") from error
    except torch.OutOfMemoryError as error:
        cleanup_cuda_memory()
        raise OperationError(
            507, "CUDA out of memory during propagation. Try lowering batch_size or enabling CPU offload."
        ) from error
    except RuntimeError as error:
        if "out of memory" in str(error).lower():
            cleanup_cuda_memory()
            raise OperationError(
                507, "CUDA out of memory during propagation. Try lowering batch_size or enabling CPU offload."
            ) from error
        raise OperationError(500, f"Failed to propagate video masks: {error}") from error
    except Exception as error:
        raise OperationError(500, f"Failed to propagate video masks: {error}") from error
    finally:
        shutil.rmtree(run_root, ignore_errors=True)

    job.enter_stage(
        "saving_manifest", "Saving manifest",
        progress=0.98,
        current=len(processed_frames),
        total=expected_total_frames,
        message="Saving mask manifest",
    )

    with ctx.lock:
        previous_masks = session.interactive_mask_files
        previous_path = session.mask_manifest_path
        try:
            validate_job_inputs(session, inputs, "mask_propagation")
            write_mask_manifest(manifest_file_path, manifest)
            session.interactive_mask_files = {key: value for key, value in previous_masks.items()
                                            if int(key) not in processed_frames}
            session.mask_manifest_path = str(manifest_file_path)
        except BaseException:
            session.interactive_mask_files = previous_masks
            session.mask_manifest_path = previous_path
            # The unpublished manifest and overlays are unreachable; the previous run stays intact.
            sweep_unreferenced_artifacts(session)
            raise
        # The previous run's files are deleted once the new manifest is accepted.
        reclaim_superseded_artifacts(session)

    return {
        "session_id": inputs["session_id"],
        "input_revision": inputs["input_revision"],
        "video_segments": video_segments_serializable if request.include_masks_in_response else {},
        "video_segments_total_frames": len(processed_frames),
        "video_segments_returned_frames": len(video_segments_serializable),
        "saved_mask_frame_count": saved_mask_frame_count,
        "saved_mask_save_failures": save_failures,
        "saved_mask_paths": saved_mask_paths_serializable,
        "online_mode": effective_online_mode,
        "batch_size": effective_batch_size,
        "tracked_points_used": total_tracked_points_seeded > 0,
        "tracked_points_skipped_reason": (
            None
            if total_tracked_points_seeded > 0 or not request.use_tracked_points or not tracked_points_skip_should_warn
            else tracked_points_skipped_reason or "No CoTracker points were seeded during propagation."
        ),
        "tracked_points_seeded_count": total_tracked_points_seeded,
        "tracked_points_seeded_frames": len(tracked_points_seeded_frames),
        "mask_manifest_path": session.mask_manifest_path,
        "saved_state": saved_state(session),
    }
