from pydantic import BaseModel


class TrackingPromptPointsRequest(BaseModel):
    add_support_grid: bool = True
