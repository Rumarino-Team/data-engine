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
