from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from core.state import state
from sessions.metadata import current_session_path
from utils import load_mask_manifest


def prompt_point_key(*, obj_id: int, frame_idx: int, x: float, y: float) -> str:
    return f"obj:{int(obj_id)}|frame:{int(frame_idx)}|x:{float(x):.3f}|y:{float(y):.3f}"


def positive_prompt_keys(prompt_events: list[dict[str, Any]]) -> set[str]:
    keys: set[str] = set()
    for event in prompt_events:
        points = event.get("points", []) or []
        labels = event.get("labels", []) or [1] * len(points)
        frame_idx = int(event.get("frame_idx", 0))
        obj_id = int(event.get("obj_id", 0))
        for point_idx, point in enumerate(points):
            if point_idx >= len(labels) or int(labels[point_idx]) != 1:
                continue
            if len(point) < 2:
                continue
            keys.add(
                prompt_point_key(
                    obj_id=obj_id,
                    frame_idx=frame_idx,
                    x=float(point[0]),
                    y=float(point[1]),
                )
            )
    return keys


def tracked_prompt_keys(points: list[dict[str, Any]]) -> list[str]:
    return [
        prompt_point_key(
            obj_id=int(point["obj_id"]),
            frame_idx=int(point["source_frame_idx"]),
            x=float(point["source_x"]),
            y=float(point["source_y"]),
        )
        for point in points
    ]


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

    def build_window_batches(
        self,
        *,
        window_start: int,
        window_end: int,
        propagation_start_frame_idx: int,
        frame_width: int,
        frame_height: int,
        keyframe_interval: int,
        max_points_per_object_per_frame: int,
        excluded_frame_objects: Optional[set[tuple[int, int]]] = None,
    ) -> tuple[dict[int, dict[int, list[list[float]]]], int, set[int]]:
        keyframe_interval = max(1, int(keyframe_interval))
        max_points_per_object_per_frame = max(1, int(max_points_per_object_per_frame))
        excluded_frame_objects = excluded_frame_objects or set()
        selected_frames = {int(window_start), int(window_end)}
        first_interval_frame = int(propagation_start_frame_idx)
        if first_interval_frame < window_start:
            offset = window_start - first_interval_frame
            steps = (offset + keyframe_interval - 1) // keyframe_interval
            first_interval_frame += steps * keyframe_interval
        for frame_idx in range(first_interval_frame, window_end + 1, keyframe_interval):
            if frame_idx >= window_start:
                selected_frames.add(frame_idx)

        batches: dict[int, dict[int, list[list[float]]]] = {}
        seeded_count = 0
        seeded_frames: set[int] = set()

        for tracked_point in self.points:
            candidate_frames = set(selected_frames)
            if window_start <= tracked_point.source_frame_idx <= window_end:
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

                local_frame_idx = int(global_frame_idx - window_start)
                obj_batches = batches.setdefault(local_frame_idx, {})
                points = obj_batches.setdefault(int(tracked_point.obj_id), [])
                if len(points) >= max_points_per_object_per_frame:
                    continue
                points.append([x_coord, y_coord])
                seeded_count += 1
                seeded_frames.add(global_frame_idx)

        return batches, seeded_count, seeded_frames


def _latest_tracking_result_path(session_path: Path) -> Optional[Path]:
    session_metadata_path = session_path / "session.json"
    if not session_metadata_path.exists():
        return None
    try:
        metadata = load_mask_manifest(session_metadata_path)
    except Exception:
        return None
    if not isinstance(metadata, dict):
        return None
    relative_path = metadata.get("latest_tracking_result_path")
    if not relative_path:
        return None
    result_path = (session_path / str(relative_path)).resolve()
    try:
        result_path.relative_to(session_path.resolve())
    except ValueError:
        return None
    return result_path


def load_latest_tracking_guidance(*, num_frames: int) -> TrackingGuidance:
    session_path = current_session_path()
    if session_path is None:
        return TrackingGuidance(points=[], skipped_reason="No active session for tracking guidance.")

    result_path = _latest_tracking_result_path(session_path)
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

    current_keys = positive_prompt_keys(state.video_prompt_events)
    matched_points: list[TrackedGuidancePoint] = []

    for index, point in enumerate(result_points):
        if not isinstance(point, dict):
            continue
        try:
            source_key = prompt_point_key(
                obj_id=int(point["obj_id"]),
                frame_idx=int(point["source_frame_idx"]),
                x=float(point["source_x"]),
                y=float(point["source_y"]),
            )
        except (KeyError, TypeError, ValueError):
            continue
        if source_key not in current_keys:
            continue

        tracks = result_tracks[index]
        visibility = result_visibility[index]
        if not isinstance(tracks, list) or not isinstance(visibility, list):
            continue
        if len(tracks) < num_frames or len(visibility) < num_frames:
            continue
        matched_points.append(
            TrackedGuidancePoint(
                obj_id=int(point["obj_id"]),
                source_frame_idx=int(point["source_frame_idx"]),
                tracks=tracks,
                visibility=visibility,
            )
        )

    if not matched_points:
        return TrackingGuidance(
            points=[],
            skipped_reason="No CoTracker points match the current positive prompts.",
            should_warn=True,
        )
    return TrackingGuidance(points=matched_points)
