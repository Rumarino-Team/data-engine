from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from core.state import BackendState
from sessions.metadata import current_session_path
from tracking.results import current_tracking_result
from utils import load_mask_manifest


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
        frame_width: int,
        frame_height: int,
        keyframe_interval: int,
        max_keyframes: int,
        max_points_per_object_per_frame: int,
        excluded_frame_objects: Optional[set[tuple[int, int]]] = None,
    ) -> tuple[dict[int, dict[int, list[list[float]]]], int, set[int]]:
        keyframe_interval = max(1, int(keyframe_interval))
        max_points_per_object_per_frame = max(1, int(max_points_per_object_per_frame))
        excluded_frame_objects = excluded_frame_objects or set()
        # The first and last frame of the range, and every `keyframe_interval` frames from the first.
        selected_frames = {int(range_start), int(range_end), *range(int(range_start), int(range_end) + 1, keyframe_interval)}
        max_keyframes = max(1, int(max_keyframes))
        if len(selected_frames) > max_keyframes:
            # Every seeded keyframe slows every frame of the propagation, so a long range gets at most
            # `max_keyframes` evenly spaced ones (both ends included) instead of one per interval.
            span = int(range_end) - int(range_start)
            selected_frames = {int(range_start)} if max_keyframes == 1 else {
                int(range_start) + round(index * span / (max_keyframes - 1)) for index in range(max_keyframes)}

        batches: dict[int, dict[int, list[list[float]]]] = {}
        seeded_count = 0
        seeded_frames: set[int] = set()

        for tracked_point in self.points:
            candidate_frames = set(selected_frames)
            if range_start <= tracked_point.source_frame_idx <= range_end:
                candidate_frames.add(tracked_point.source_frame_idx)
            for global_frame_idx in sorted(candidate_frames):
                if (int(global_frame_idx), int(tracked_point.obj_id)) in excluded_frame_objects:
                    continue
                if global_frame_idx < 0 or global_frame_idx >= len(tracked_point.tracks):
                    continue
                if global_frame_idx < len(tracked_point.visibility) and tracked_point.visibility[global_frame_idx] is False:
                    continue

                track = tracked_point.tracks[global_frame_idx]
                if not isinstance(track, (list, tuple)) or len(track) < 2:
                    continue
                try:
                    x_coord = float(track[0])
                    y_coord = float(track[1])
                except (TypeError, ValueError):
                    continue
                if not np.isfinite(x_coord) or not np.isfinite(y_coord):
                    continue
                x_coord = min(max(x_coord, 0.0), float(frame_width - 1))
                y_coord = min(max(y_coord, 0.0), float(frame_height - 1))

                local_frame_idx = int(global_frame_idx - range_start)
                obj_batches = batches.setdefault(local_frame_idx, {})
                points = obj_batches.setdefault(int(tracked_point.obj_id), [])
                if len(points) >= max_points_per_object_per_frame:
                    continue
                points.append([x_coord, y_coord])
                seeded_count += 1
                seeded_frames.add(global_frame_idx)

        return batches, seeded_count, seeded_frames


def _latest_tracking_result_path(session: BackendState, session_path: Path) -> Optional[Path]:
    relative_path = session.session_metadata.get("latest_tracking_result_path")
    if not relative_path:
        return None
    result_path = (session_path / str(relative_path)).resolve()
    try:
        result_path.relative_to(session_path.resolve())
    except ValueError:
        return None
    return result_path


def load_latest_tracking_guidance(session: BackendState, *, num_frames: int) -> TrackingGuidance:
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
    return TrackingGuidance(points=matched_points)
