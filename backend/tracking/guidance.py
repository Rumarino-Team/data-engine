from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from core.annotations import SessionAnnotations
from core.session_layout import TRACKING_RESULT_PATH
from core.state import SessionState
from sessions.metadata import current_session_path
from tracking.results import current_tracking_result
from utils import load_mask_manifest


# CoTracker's online model works in windows of 16 frames. The frames of the final window have no later frames to
# refine them (usually within 7 px of a longer run, once 35 px), so the video's last frames are never seeded.
TRACKER_TAIL_FRAMES = 16


@dataclass(frozen=True)
class TrackedGuidancePoint:
    obj_id: int
    source_frame_idx: int
    tracks: list[Any]
    visibility: list[Any]


@dataclass
class TrackingGuidance:
    points: list[TrackedGuidancePoint]
    skipped_reason: Optional[str] = None
    should_warn: bool = False

    def build_seed_batches(
        self,
        *,
        range_start: int,
        range_end: int,
        num_video_frames: int,
        frame_width: int,
        frame_height: int,
        keyframe_interval: int,
        max_keyframes: int,
        max_points_per_object_per_frame: int,
        excluded_frame_objects: Optional[set[tuple[int, int]]] = None,
        tail_frames: int = TRACKER_TAIL_FRAMES,
    ) -> tuple[dict[int, dict[int, list[list[float]]]], int, set[int]]:
        """Where each object may be seeded: {frame offset in the range: {object id: points}}, the number of
        points, and the global frames used.

        A seeded frame becomes a conditioning frame: SAM2 segments it from these points alone, with no memory,
        emits that mask as it is, and every other frame then attends to it as memory. So a frame is seeded only
        when the points stand for the whole of what the user defined:
        - `excluded_frame_objects` are the (frame, object) pairs the user prompted or gave a mask for. Those
          frames are never touched, and a point stops speaking for its object at the object's next such frame.
        - every tracked point of the object that is in effect on a frame must be visible there, or the object is
          left alone on that frame (a subset of the clicks would define a smaller object).
        - the last `tail_frames` of the video are never seeded, and the range's last frame is not forced.
        """
        keyframe_interval = max(1, int(keyframe_interval))
        max_points_per_object_per_frame = max(1, int(max_points_per_object_per_frame))
        excluded_frame_objects = excluded_frame_objects or set()
        # `tail_frames` are measured on the tracked video, not the propagated range: a range that stops early
        # ends where the tracker is fine.
        seed_end = min(int(range_end), int(num_video_frames) - 1 - max(0, int(tail_frames)))
        if seed_end < range_start:
            return {}, 0, set()
        # The first frame of the range and every `keyframe_interval` frames from it.
        selected_frames = set(range(int(range_start), seed_end + 1, keyframe_interval))
        max_keyframes = max(1, int(max_keyframes))
        if len(selected_frames) > max_keyframes:
            # Every seeded keyframe slows every frame of the propagation, so a long range gets at most
            # `max_keyframes` evenly spaced ones (both ends of the seedable span included).
            span = seed_end - int(range_start)
            selected_frames = {int(range_start)} if max_keyframes == 1 else {
                int(range_start) + round(index * span / (max_keyframes - 1)) for index in range(max_keyframes)}

        prompted_frames: dict[int, list[int]] = {}
        for frame_idx, obj_id in excluded_frame_objects:
            prompted_frames.setdefault(int(obj_id), []).append(int(frame_idx))

        def in_effect(point: TrackedGuidancePoint, frame_idx: int) -> bool:
            """A point speaks for its object from where it was placed until the user's next prompt for it."""
            next_prompt = min((frame for frame in prompted_frames.get(point.obj_id, ())
                               if frame > point.source_frame_idx), default=None)
            return point.source_frame_idx <= frame_idx and (next_prompt is None or frame_idx < next_prompt)

        def visible_position(point: TrackedGuidancePoint, frame_idx: int) -> Optional[list[float]]:
            if frame_idx >= len(point.tracks) or frame_idx >= len(point.visibility) or not point.visibility[frame_idx]:
                return None
            track = point.tracks[frame_idx]
            if not isinstance(track, (list, tuple)) or len(track) < 2:
                return None
            try:
                x_coord, y_coord = float(track[0]), float(track[1])
            except (TypeError, ValueError):
                return None
            if not np.isfinite(x_coord) or not np.isfinite(y_coord):
                return None
            return [min(max(x_coord, 0.0), float(frame_width - 1)), min(max(y_coord, 0.0), float(frame_height - 1))]

        points_by_object: dict[int, list[TrackedGuidancePoint]] = {}
        for tracked_point in self.points:
            points_by_object.setdefault(int(tracked_point.obj_id), []).append(tracked_point)

        batches: dict[int, dict[int, list[list[float]]]] = {}
        seeded_count = 0
        seeded_frames: set[int] = set()
        for obj_id, object_points in points_by_object.items():
            for global_frame_idx in sorted(selected_frames):
                if (global_frame_idx, obj_id) in excluded_frame_objects:
                    continue
                positions = [visible_position(point, global_frame_idx)
                             for point in object_points if in_effect(point, global_frame_idx)]
                if not positions or any(position is None for position in positions):
                    continue  # nothing tracked here, or only part of what defines the object
                points = positions[:max_points_per_object_per_frame]
                batches.setdefault(int(global_frame_idx - range_start), {})[obj_id] = points
                seeded_count += len(points)
                seeded_frames.add(global_frame_idx)

        return batches, seeded_count, seeded_frames


def seedable_object_ids(annotations: SessionAnnotations) -> set[int]:
    """The objects a tracked positive point can stand in for: those defined by positive points alone.

    Tracking carries only positive points, so a negative point, a box or an explicit mask (on any frame) is a
    refinement that a seeded frame would drop."""
    refined = {prompt.obj_id for prompt in annotations.prompts if prompt.box is not None or 0 in prompt.labels}
    refined.update(mask.obj_id for mask in annotations.mask_overrides)
    return set(annotations.objects) - refined


def _latest_tracking_result_path(session: SessionState, session_path: Path) -> Optional[Path]:
    relative_path = session.session_metadata.get(TRACKING_RESULT_PATH)
    if not relative_path:
        return None
    result_path = (session_path / str(relative_path)).resolve()
    try:
        result_path.relative_to(session_path.resolve())
    except ValueError:
        return None
    return result_path


def load_latest_tracking_guidance(session: SessionState, *, num_frames: int) -> TrackingGuidance:
    session_path = current_session_path(session)
    if session_path is None:
        return TrackingGuidance(points=[], skipped_reason="No active session for tracking guidance.")

    result_path = _latest_tracking_result_path(session, session_path)
    if result_path is None or not result_path.exists():
        return TrackingGuidance(points=[], skipped_reason="No CoTracker result available for propagation.", should_warn=False)

    try:
        result = load_mask_manifest(result_path)
    except Exception:
        return TrackingGuidance(points=[], skipped_reason="Latest CoTracker result is invalid.", should_warn=True)
    if not isinstance(result, dict):
        return TrackingGuidance(points=[], skipped_reason="Latest CoTracker result is invalid.", should_warn=True)
    try:
        result_num_frames = int(result.get("num_frames", -1))
    except (TypeError, ValueError):
        return TrackingGuidance(points=[], skipped_reason="Latest CoTracker result is invalid.", should_warn=True)
    if result_num_frames != int(num_frames):
        return TrackingGuidance(points=[], skipped_reason="Latest CoTracker result does not match the current video.", should_warn=True)

    result_points = result.get("points")
    result_tracks = result.get("tracks")
    result_visibility = result.get("visibility")
    if not isinstance(result_points, list) or not isinstance(result_tracks, list) or not isinstance(result_visibility, list):
        return TrackingGuidance(points=[], skipped_reason="Latest CoTracker result is missing tracked point data.", should_warn=True)
    if len(result_points) != len(result_tracks) or len(result_points) != len(result_visibility):
        return TrackingGuidance(points=[], skipped_reason="Latest CoTracker result has inconsistent tracked point data.", should_warn=True)

    try:
        result = current_tracking_result(session, result)
    except Exception:
        return TrackingGuidance(points=[], skipped_reason="Latest CoTracker result does not match the current session.", should_warn=True)
    matched_points = [TrackedGuidancePoint(
        obj_id=int(point["obj_id"]), source_frame_idx=int(point["source_frame_idx"]),
        tracks=tracks, visibility=visibility,
    ) for point, tracks, visibility in zip(result["points"], result["tracks"], result["visibility"])
        if isinstance(tracks, list) and isinstance(visibility, list)
        and len(tracks) >= num_frames and len(visibility) >= num_frames]

    if not matched_points:
        return TrackingGuidance(
            points=[],
            skipped_reason="No CoTracker points match the current positive prompts.",
            should_warn=True,
        )
    seedable = seedable_object_ids(session.annotations)
    skipped_objects = sorted({point.obj_id for point in matched_points} - seedable)
    matched_points = [point for point in matched_points if point.obj_id in seedable]
    if not matched_points:
        return TrackingGuidance(
            points=[],
            skipped_reason="Objects with negative points, boxes or masks are not seeded from tracked points.",
            should_warn=True,
        )
    if result.get("add_support_grid_used") is not True:
        # Measured: without the support grid a tracked point can sit on another object while flagged visible.
        return TrackingGuidance(
            points=[],
            skipped_reason="The latest CoTracker result was tracked without the support grid, so its points are not used.",
            should_warn=True,
        )
    # Not an error: kept so that the propagation result can say why an object was left out if nothing was seeded.
    skipped_reason = (f"Objects {', '.join(map(str, skipped_objects))} have negative points, boxes or masks and "
                      "are not seeded from tracked points.") if skipped_objects else None
    return TrackingGuidance(points=matched_points, skipped_reason=skipped_reason)
