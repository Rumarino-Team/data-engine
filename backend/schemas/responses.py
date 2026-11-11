"""Everything the API returns, as models.

Models forbid undeclared fields, so a service that adds or renames a field fails its tests instead of
quietly changing what the frontend receives. `frontend-ng/.../backend-api.types.ts` declares the subset
the frontend reads; `tests/test_api_contract.py` checks that every field it names exists here.
"""
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict

from schemas.video import InteractiveState, MaskOverride, VideoObject, VideoPrompt


class Response(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---- accepted state ---------------------------------------------------------------------------------------------------

class SavedState(Response):
    checkpoint_id: Optional[str]
    display_name: Optional[str]
    saved_at: Optional[str]
    dirty: bool


class TrackingValidity(Response):
    result_id: str
    point_ids: list[str]


class AnnotationsSnapshot(Response):
    version: Literal[1]
    revision: int
    next_object_id: int
    objects: dict[str, VideoObject]
    prompts: list[VideoPrompt]
    mask_overrides: list[MaskOverride]
    input_revisions: dict[str, int]


class AnnotationDelta(Response):
    """Current values for the keys a mutation changed, applicable only at `base_revision`."""
    base_revision: int
    revision: int
    next_object_id: int
    objects: dict[str, VideoObject]
    removed_objects: list[int]
    prompts: list[VideoPrompt]
    removed_prompts: list[tuple[int, int]]
    mask_overrides: list[MaskOverride]
    removed_mask_overrides: list[tuple[int, int]]
    input_revisions: dict[str, int]
    removed_input_revisions: list[int]


# ---- masks ------------------------------------------------------------------------------------------------------------

class RleMask(Response):
    size: tuple[int, int]
    rle: list[list[int]]
    bbox: tuple[int, int, int, int]


class PackedMask(Response):
    size: tuple[int, int]
    encoding: Literal["packed-bits"]
    data: str


class MaskData(Response):
    """The effective masks for one frame: explicit override, then interactive result, then propagation."""
    session_id: str
    annotation_revision: int
    frame_idx: int
    objects: dict[str, RleMask | PackedMask]
    sources: dict[str, Literal["live", "manifest", "override"]]


# ---- video routes -----------------------------------------------------------------------------------------------------

class AnnotationUpdate(Response):
    """Every mutation answers with what changed since the revision the client sent."""
    session_id: Optional[str]
    annotation_delta: AnnotationDelta
    tracking_validity: Optional[TrackingValidity]
    saved_state: Optional[SavedState]


class ResetResponse(AnnotationUpdate):
    message: str


class MaskEditResponse(AnnotationUpdate):
    mask_data: MaskData


class ClearPromptsResponse(MaskEditResponse):
    message: str


class PointsResponse(MaskEditResponse):
    request_frame_idx: int
    frame_idx: int
    frame_file: str
    mask_pixel_counts: dict[int, int]
    mask_shapes: dict[int, tuple[int, int]]
    single_frame_fallback_used: bool


class ObjectCreationResponse(AnnotationUpdate):
    created_object_id: int


class VideoInfoResponse(Response):
    session_id: Optional[str]
    annotations: Optional[AnnotationsSnapshot]
    tracking_validity: Optional[TrackingValidity]
    saved_state: Optional[SavedState]
    predictor_ready: bool
    num_frames: int
    frame_files: list[str]


class SaveResponse(Response):
    message: str
    name: str
    checkpoint_id: str
    saved_path: str
    saved_state: SavedState


# ---- jobs -------------------------------------------------------------------------------------------------------------

JobStatus = Literal["queued", "running", "completed", "failed"]
JobOperation = Literal["video_init", "mask_propagation", "prompt_tracking"]
TrackingMode = Literal["streaming", "in_memory"]


class JobStartResponse(Response):
    job_id: str
    status: JobStatus
    operation: JobOperation
    message: str


class JobError(Response):
    code: str
    message: str
    detail: Optional[str]


class JobStageEntry(Response):
    stage: str
    stage_label: str
    message: str
    progress: Optional[float]
    updated_at: str


class JobSnapshot(Response):
    """A job's progress. `result` is the operation's own result model once it completes."""
    job_id: str
    operation: JobOperation
    status: JobStatus
    stage: str
    stage_label: str
    progress: Optional[float]
    current: Optional[int]
    total: Optional[int]
    window_index: Optional[int]
    window_count: Optional[int]
    frame_idx: Optional[int]
    batch_current: Optional[int]
    batch_total: Optional[int]
    batch_index: Optional[int]
    batch_count: Optional[int]
    stage_history: list[JobStageEntry]
    message: str
    result: Optional[dict[str, Any]]
    error: Optional[JobError]
    started_at: str
    updated_at: str
    completed_at: Optional[str]


class JobResponse(Response):
    job: JobSnapshot


class CurrentJobResponse(Response):
    job: Optional[JobSnapshot]


class ClearResultResponse(Response):
    cleared: bool


# ---- job results ------------------------------------------------------------------------------------------------------
# Each job builds its result through one of these, so what lands in `JobSnapshot.result` has a declared shape.

class RestoredTrackingResult(Response):
    result_id: str
    summary: dict[str, Any]


class RestoredSession(Response):
    annotations: AnnotationsSnapshot
    interactive_state: InteractiveState
    has_mask_manifest: bool
    tracking_result: Optional[RestoredTrackingResult] = None


class InitResult(Response):
    message: str
    num_frames: int
    resolved_video_frames_dir: str
    source_video_path: Optional[str]
    online_mode: bool
    batch_size: int
    offload_video_to_cpu: bool
    offload_state_to_cpu: bool
    source_type: Literal["frames_dir", "video_file", "saved_session"]
    restored_session: Optional[RestoredSession] = None


class PropagationResult(Response):
    session_id: str
    input_revision: int
    video_segments: dict[int, dict[int, list]]
    video_segments_total_frames: int
    video_segments_returned_frames: int
    saved_mask_frame_count: int
    saved_mask_save_failures: int
    saved_mask_paths: dict[int, list[str]]
    online_mode: bool
    batch_size: int
    tracked_points_used: bool
    tracked_points_skipped_reason: Optional[str]
    tracked_points_seeded_count: int
    tracked_points_seeded_frames: int
    mask_manifest_path: Optional[str]
    saved_state: SavedState


class TrackingJobResult(Response):
    session_id: str
    input_revision: int
    message: str
    model_name: str
    num_points: int
    num_frames: int
    add_support_grid_used: bool
    tracking_mode: TrackingMode
    streaming_frame_threshold: int
    tracking_result_id: str
    saved_state: SavedState


# ---- tracking routes --------------------------------------------------------------------------------------------------

class TrackedPoint(Response):
    point_id: str
    obj_id: int
    source_frame_idx: int
    source_x: float
    source_y: float


class TrackingResult(Response):
    version: int
    session_id: str
    input_revision: int
    annotation_revision: int
    result_id: str
    created_at: str
    model_name: str
    num_points: int
    num_frames: int
    add_support_grid_used: bool
    tracking_mode: TrackingMode
    streaming_frame_threshold: int
    tracked_prompt_keys: list[str]
    points: list[TrackedPoint]
    tracks: list[list[list[float]]]
    visibility: list[list[bool]]


class TrackingResultResponse(Response):
    result: TrackingResult


# ---- root -------------------------------------------------------------------------------------------------------------

class RootResponse(Response):
    message: str


class HealthResponse(Response):
    status: str


class StatusResponse(Response):
    status: str
    device: Optional[str]
