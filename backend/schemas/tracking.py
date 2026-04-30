from pydantic import BaseModel
import co_tracker as cot

class TrackingLoadVideoRequest(BaseModel):
    video_path: str
    model_name: str = cot.DEFAULT_COTRACKER_MODEL

class TrackingGridRequest(BaseModel):
    grid_size: int = 15
    add_support_grid: bool = True

class TrackingPointsRequest(BaseModel):
    queries: list[list[float]]  # List of [t, x, y] coordinates
    add_support_grid: bool = True

class TrackingPromptPointsRequest(BaseModel):
    add_support_grid: bool = True

class TrackPromptPointMetadata(BaseModel):
    point_id: str
    obj_id: int
    source_frame_idx: int
    source_x: float
    source_y: float

class TrackPromptPointsJobResponse(BaseModel):
    message: str
    model_name: str
    num_points: int
    num_frames: int
    add_support_grid_used: bool
    tracking_mode: str
    streaming_frame_threshold: int
    tracking_result_id: str
    state_epoch: int

class TrackPromptPointsResult(BaseModel):
    version: int
    result_id: str
    model_name: str
    num_points: int
    num_frames: int
    add_support_grid_used: bool
    tracking_mode: str
    streaming_frame_threshold: int
    points: list[TrackPromptPointMetadata]
    tracks: list[list[list[float]]]
    visibility: list[list[bool]]
