import logging, shutil, uuid
from pathlib import Path
from typing import Any, Optional
import cv2
from fastapi import HTTPException
import numpy as np
import torch
from core.config import WINDOW_FRAMES_ROOT
from core.jobs import queue_long_job, update_job
from core.runtime import cleanup_cuda_memory
from core.state import state
from schemas.video import VideoPropagateRequest
from sessions.cache import clear_window_cache
from sessions.metadata import current_masks_dir, write_session_metadata
from utils import build_empty_mask_manifest, prepare_video_masks_output, save_single_video_mask_frame, write_mask_manifest
from video.io import build_window_dir
from video.masks import manifest_frame_payload
from video.prompts import restore_video_masker_from_prompt_events
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

async def start_propagation(request: VideoPropagateRequest):
    return queue_long_job(
        operation="mask_propagation",
        stage="validating_prompts",
        stage_label="Validating prompts",
        message="Mask propagation queued",
        worker=lambda: run_propagation_job(request),
    )

def run_propagation_job(request: VideoPropagateRequest) -> dict[str, Any]:
    def _propagation_progress(processed_frames_count: int, expected_frames_count: int) -> float:
        return (processed_frames_count / expected_frames_count) if expected_frames_count else 0.0

    update_job(
        status="running",
        stage="validating_prompts",
        stage_label="Validating prompts",
        progress=0.0,
        batch_current=None,
        batch_total=None,
        batch_index=None,
        batch_count=None,
        message="Validating mask propagation inputs",
    )
    if state.video_masker is None:
        raise HTTPException(status_code=400, detail="Video masker not active.")
    if state.video_dir is None:
        raise HTTPException(status_code=400, detail="Video directory not set. Call /video/init_state first.")

    if request.reverse:
        raise HTTPException(
            status_code=400,
            detail="Reverse propagation is not supported in half-window mode.",
        )

    if not state.video_frame_files:
        raise HTTPException(status_code=400, detail="No video frames available. Call /video/init_state first.")

    if not state.video_prompt_events:
        raise HTTPException(status_code=400, detail="No prompts available for propagation.")

    effective_online_mode = state.video_masker.online_mode if request.online_mode is None else bool(request.online_mode)
    effective_batch_size = state.video_masker.default_batch_size if request.batch_size is None else int(request.batch_size)
    effective_offload_video_to_cpu = state.video_masker.offload_video_to_cpu
    effective_offload_state_to_cpu = state.video_masker.offload_state_to_cpu
    if effective_batch_size <= 0:
        raise HTTPException(status_code=400, detail="batch_size must be a positive integer.")

    num_frames = len(state.video_frame_files)
    if request.start_frame_idx is not None:
        start_frame_idx = int(request.start_frame_idx)
    else:
        start_frame_idx = min(int(event["frame_idx"]) for event in state.video_prompt_events)
    start_frame_idx = min(max(start_frame_idx, 0), num_frames - 1)

    if request.max_frame_num_to_track is None:
        end_frame_idx = num_frames - 1
    else:
        requested = int(request.max_frame_num_to_track)
        if requested <= 0:
            return {
                "video_segments": {},
                "video_segments_total_frames": 0,
                "video_segments_returned_frames": 0,
                "video_segments_returned_mask_values": 0,
                "video_segments_truncated": False,
                "saved_mask_frame_count": 0,
                "saved_mask_save_failures": 0,
                "saved_mask_paths": {},
                "online_mode": effective_online_mode,
                "batch_size": effective_batch_size,
                "tracked_points_used": False,
                "tracked_points_skipped_reason": None,
                "tracked_points_seeded_count": 0,
                "tracked_points_seeded_frames": 0,
                "state.mask_manifest_path": state.mask_manifest_path,
                "state_epoch": int(state.video_state_epoch),
            }
        end_frame_idx = min(num_frames - 1, start_frame_idx + requested - 1)

    if end_frame_idx < start_frame_idx:
        raise HTTPException(status_code=400, detail="Invalid propagation frame range.")

    expected_total_frames = end_frame_idx - start_frame_idx + 1
    masks_root = current_masks_dir()
    if masks_root is None:
        raise HTTPException(status_code=400, detail="Video session cache is not initialized.")
    clear_window_cache()
    update_job(
        stage="clearing_previous_masks",
        stage_label="Clearing previous masks",
        progress=0.02,
        current=0,
        total=expected_total_frames,
        batch_current=None,
        batch_total=None,
        batch_index=None,
        batch_count=None,
        message="Preparing mask output directory",
    )

    frame_files, masks_dir = prepare_video_masks_output(state.video_dir, masks_root)
    manifest_file_path = masks_dir / "manifest.json"

    first_frame = cv2.imread(str(Path(state.video_dir) / state.video_frame_files[start_frame_idx]))
    if first_frame is None:
        raise HTTPException(status_code=500, detail="Unable to read first frame for manifest metadata.")

    manifest = build_empty_mask_manifest(
        source_video_path=state.video_source_path,
        resolved_video_frames_dir=str(state.video_dir),
        num_frames=num_frames,
        frame_height=int(first_frame.shape[0]),
        frame_width=int(first_frame.shape[1]),
    )
    manifest_frames: dict[str, Any] = manifest["frames"]
    frame_height = int(first_frame.shape[0])
    frame_width = int(first_frame.shape[1])

    tracking_guidance = None
    tracked_points_skipped_reason: Optional[str] = None
    tracked_points_skip_should_warn = False
    if request.use_tracked_points:
        tracking_guidance = load_latest_tracking_guidance(num_frames=num_frames)
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

    saved_mask_paths_serializable: dict[int, list[str]] = {}
    saved_mask_frame_count = 0
    save_failures = 0
    processed_frames: set[int] = set()
    boundary_masks: dict[int, np.ndarray] = {}
    boundary_frame_idx: Optional[int] = None

    video_segments_serializable: dict[int, dict[int, list]] = {}

    try:
        for window_index, (window_start, window_end) in enumerate(windows):
            window_number = window_index + 1
            window_count = len(windows)
            update_job(
                stage="building_window",
                stage_label="Loading propagation window",
                current=0,
                total=window_end - window_start + 1,
                window_index=window_number,
                window_count=window_count,
                frame_idx=window_start,
                progress=_propagation_progress(len(processed_frames), expected_total_frames),
                batch_current=None,
                batch_total=None,
                batch_index=None,
                batch_count=None,
                message=f"Preparing window {window_number} of {window_count}: frames {window_start}-{window_end}",
            )
            window_frame_paths = [Path(state.video_dir) / state.video_frame_files[idx] for idx in range(window_start, window_end + 1)]
            window_name = f"window_{window_index}_{window_start}_{window_end}"

            def _on_window_build_progress(current: int, total: int, source_path: Path) -> None:
                source_frame_idx = window_start + current - 1
                update_job(
                    stage="building_window",
                    stage_label="Loading propagation window",
                    current=current,
                    total=total,
                    window_index=window_number,
                    window_count=window_count,
                    frame_idx=source_frame_idx,
                    progress=_propagation_progress(len(processed_frames), expected_total_frames),
                    batch_current=None,
                    batch_total=None,
                    batch_index=None,
                    batch_count=None,
                    message=f"Linked {current} of {total} frames for window {window_number} of {window_count}",
                    append_history=current == 1 or current == total,
                )

            window_dir = build_window_dir(
                window_frame_paths,
                run_root,
                window_name,
                progress_callback=_on_window_build_progress,
            )

            update_job(
                stage="initializing_state",
                stage_label="Initializing SAM2 window state",
                current=len(processed_frames),
                total=expected_total_frames,
                window_index=window_number,
                window_count=window_count,
                frame_idx=window_start,
                progress=_propagation_progress(len(processed_frames), expected_total_frames),
                batch_current=None,
                batch_total=None,
                batch_index=None,
                batch_count=None,
                message=f"Initializing SAM2 state for window {window_number} of {window_count}",
            )

            state.video_masker.init_state(
                str(window_dir),
                online_mode=effective_online_mode,
                batch_size=effective_batch_size,
                offload_video_to_cpu=effective_offload_video_to_cpu,
                offload_state_to_cpu=effective_offload_state_to_cpu,
                async_loading_frames=False,
            )

            update_job(
                stage="seeding_window",
                stage_label="Seeding prompts",
                current=len(processed_frames),
                total=expected_total_frames,
                window_index=window_number,
                window_count=window_count,
                frame_idx=window_start,
                progress=_propagation_progress(len(processed_frames), expected_total_frames),
                batch_current=None,
                batch_total=None,
                batch_index=None,
                batch_count=None,
                message=f"Seeding prompts for window {window_number} of {window_count}",
            )

            if window_index > 0 and boundary_masks and boundary_frame_idx is not None:
                update_job(
                    stage="seeding_window",
                    stage_label="Seeding boundary masks",
                    current=len(processed_frames),
                    total=expected_total_frames,
                    window_index=window_number,
                    window_count=window_count,
                    frame_idx=boundary_frame_idx,
                    progress=_propagation_progress(len(processed_frames), expected_total_frames),
                    batch_current=None,
                    batch_total=None,
                    batch_index=None,
                    batch_count=None,
                    message=f"Seeding boundary masks for window {window_number} of {window_count}",
                )
                local_boundary_idx = int(boundary_frame_idx - window_start)
                for obj_id, obj_mask in boundary_masks.items():
                    state.video_masker.add_new_mask(
                        frame_idx=local_boundary_idx,
                        obj_id=int(obj_id),
                        mask=np.asarray(obj_mask).astype(bool),
                    )

            for event in state.video_prompt_events:
                event_frame_idx = int(event["frame_idx"])
                if event_frame_idx < window_start or event_frame_idx > window_end:
                    continue
                local_event_frame_idx = event_frame_idx - window_start
                points = event["points"] if event["points"] else None
                labels = event["labels"] if event["labels"] else None
                state.video_masker.add_new_points_or_box(
                    frame_idx=local_event_frame_idx,
                    obj_id=int(event["obj_id"]),
                    points=points,
                    labels=labels,
                    clear_old_points=bool(event.get("clear_old_points", True)),
                    box=event.get("box"),
                )

            if tracking_guidance is not None and tracking_guidance.points:
                excluded_tracked_guidance: set[tuple[int, int]] = set()
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
                        state.video_masker.add_new_points_or_box(
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
                update_job(
                    stage="propagating_window",
                    stage_label="Propagating masks",
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
                manifest_frames[str(global_frame_idx)] = manifest_frame_payload(frame_masks)
                if request.include_masks_in_response:
                    video_segments_serializable[global_frame_idx] = {
                        int(obj_id): np.asarray(mask).astype(bool).tolist()
                        for obj_id, mask in frame_masks.items()
                    }

                try:
                    saved_path = save_single_video_mask_frame(
                        frame_files,
                        masks_dir,
                        global_frame_idx,
                        frame_masks,
                    )
                    if saved_path is not None:
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

            state.video_masker.propagate_in_video(
                start_frame_idx=local_start_frame,
                max_frame_num_to_track=local_max_frames,
                reverse=False,
                batch_size=effective_batch_size,
                online_mode=effective_online_mode,
                collect_segments=False,
                frame_callback=_on_window_frame,
            )
            update_job(
                stage="propagating_window",
                stage_label="Propagating masks",
                current=len(processed_frames),
                total=expected_total_frames,
                window_index=window_number,
                window_count=window_count,
                frame_idx=window_end,
                progress=_propagation_progress(len(processed_frames), expected_total_frames),
                batch_current=None,
                batch_total=None,
                batch_index=None,
                batch_count=None,
                message=f"Finished window {window_number} of {window_count}",
            )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except torch.OutOfMemoryError as error:
        cleanup_cuda_memory()
        raise HTTPException(
            status_code=507,
            detail="CUDA out of memory during propagation. Try lowering batch_size or enabling CPU offload.",
        ) from error
    except RuntimeError as error:
        if "out of memory" in str(error).lower():
            cleanup_cuda_memory()
            raise HTTPException(
                status_code=507,
                detail="CUDA out of memory during propagation. Try lowering batch_size or enabling CPU offload.",
            ) from error
        raise HTTPException(status_code=500, detail=f"Failed to propagate video masks: {error}") from error
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Failed to propagate video masks: {error}") from error
    finally:
        shutil.rmtree(run_root, ignore_errors=True)

    write_mask_manifest(manifest_file_path, manifest)
    state.mask_manifest_path = str(manifest_file_path)
    write_session_metadata()
    update_job(
        stage="saving_manifest",
        stage_label="Saving manifest",
        progress=0.98,
        current=len(processed_frames),
        total=expected_total_frames,
        window_index=None,
        window_count=None,
        frame_idx=None,
        message="Saving mask manifest",
        batch_current=None,
        batch_total=None,
        batch_index=None,
        batch_count=None,
    )

    try:
        # Rebind interactive state to the original full video frame index space.
        update_job(
            stage="restoring_interactive_state",
            stage_label="Restoring interactive state",
            progress=None,
            current=len(processed_frames),
            total=expected_total_frames,
            window_index=None,
            window_count=None,
            frame_idx=None,
            message="Restoring interactive masking state",
            batch_current=None,
            batch_total=None,
            batch_index=None,
            batch_count=None,
        )
        restore_video_masker_from_prompt_events(
            online_mode=effective_online_mode,
            batch_size=effective_batch_size,
            offload_video_to_cpu=effective_offload_video_to_cpu,
            offload_state_to_cpu=effective_offload_state_to_cpu,
        )
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Propagation completed but failed to restore interactive masking state: {error}",
        ) from error

    return {
        "video_segments": video_segments_serializable if request.include_masks_in_response else {},
        "video_segments_total_frames": len(processed_frames),
        "video_segments_returned_frames": len(video_segments_serializable),
        "video_segments_returned_mask_values": 0,
        "video_segments_truncated": False,
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
        "mask_manifest_path": state.mask_manifest_path,
        "state.mask_manifest_path": state.mask_manifest_path,
        "state_epoch": int(state.video_state_epoch),
    }
