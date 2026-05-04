from typing import Optional
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from core.config import DEFAULT_MAX_INTERACTIVE_LIVE_MASKS

class VideoInitStateRequest(BaseModel):
    video_frames_dir: str
    online_mode: bool = True
    batch_size: Optional[int] = None
    offload_video_to_cpu: Optional[bool] = None
    offload_state_to_cpu: Optional[bool] = None
    async_loading_frames: bool = False

class VideoAddPointsOrBoxRequest(BaseModel):
    frame_idx: int
    obj_id: int
    points: Optional[list[list[float]]] = None
    labels: Optional[list[int]] = None
    clear_old_points: bool = True
    box: Optional[list[float]] = None

class VideoPropagateRequest(BaseModel):
    start_frame_idx: Optional[int] = None
    max_frame_num_to_track: Optional[int] = None
    reverse: bool = False
    batch_size: Optional[int] = None
    online_mode: Optional[bool] = None
    use_tracked_points: bool = True
    tracked_point_keyframe_interval: int = Field(default=8, ge=1)
    max_tracked_points_per_object_per_frame: int = Field(default=16, ge=1)
    include_masks_in_response: bool = False
    include_saved_mask_paths: bool = False
    max_frames_in_response: Optional[int] = None
    max_mask_values_in_response: Optional[int] = None

class VideoAddMaskRequest(BaseModel):
    frame_idx: int
    obj_id: int
    mask: list[list[bool]]

class InteractiveObject(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int = Field(ge=1)
    name: str = Field(min_length=1, max_length=200)
    color: str = Field(pattern=r"^#[0-9A-Fa-f]{6}$")

class InteractivePoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    frame_idx: int = Field(ge=0)
    obj_id: int = Field(ge=1)
    x: float
    y: float
    label: int

    @field_validator("x", "y")
    @classmethod
    def _validate_finite_coordinate(cls, value: float) -> float:
        if not np.isfinite(value):
            raise ValueError("Point coordinates must be finite numbers.")
        return float(value)

    @field_validator("label")
    @classmethod
    def _validate_label(cls, value: int) -> int:
        if int(value) not in (0, 1):
            raise ValueError("Point label must be 0 or 1.")
        return int(value)

class InteractiveMaskRLE(BaseModel):
    model_config = ConfigDict(extra="forbid")

    frame_idx: int = Field(ge=0)
    obj_id: int = Field(ge=1)
    height: int = Field(ge=1)
    width: int = Field(ge=1)
    counts: list[int]

    @field_validator("counts")
    @classmethod
    def _validate_counts_members(cls, counts: list[int]) -> list[int]:
        if not counts:
            raise ValueError("Live mask counts cannot be empty.")
        normalized = [int(value) for value in counts]
        if any(value < 0 for value in normalized):
            raise ValueError("Live mask counts cannot contain negative values.")
        return normalized

    @model_validator(mode="after")
    def _validate_counts_shape(self) -> "InteractiveMaskRLE":
        total = sum(self.counts)
        expected = int(self.height) * int(self.width)
        if total != expected:
            raise ValueError(
                f"Live mask counts sum ({total}) does not match width*height ({expected})."
            )
        return self

class InteractiveState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int = Field(default=1, ge=1)
    objects: list[InteractiveObject] = Field(default_factory=list)
    selected_object_id: Optional[int] = Field(default=None, ge=1)
    interaction_mode: Optional[str] = None
    current_frame_idx: Optional[int] = Field(default=None, ge=0)
    points: list[InteractivePoint] = Field(default_factory=list)
    live_masks: list[InteractiveMaskRLE] = Field(default_factory=list)

    @field_validator("interaction_mode")
    @classmethod
    def _validate_interaction_mode(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        if value not in {"positive", "negative"}:
            raise ValueError("interaction_mode must be positive or negative.")
        return value

    @model_validator(mode="after")
    def _validate_limits(self) -> "InteractiveState":
        if len(self.live_masks) > DEFAULT_MAX_INTERACTIVE_LIVE_MASKS:
            raise ValueError(
                f"Too many live masks in interactive_state ({len(self.live_masks)}). "
                f"Maximum allowed is {DEFAULT_MAX_INTERACTIVE_LIVE_MASKS}."
            )
        return self

class VideoSaveRequest(BaseModel):
    name: str
    interactive_state: Optional[InteractiveState] = None
