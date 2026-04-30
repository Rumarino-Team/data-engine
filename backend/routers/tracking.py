from fastapi import APIRouter

from schemas.tracking import (
    TrackingGridRequest,
    TrackingLoadVideoRequest,
    TrackingPointsRequest,
    TrackingPromptPointsRequest,
)
from tracking.prompt_tracking import start_prompt_tracking
from tracking.results import load_prompt_tracking_result
from tracking.service import load_tracking_video, track_grid, track_points


router = APIRouter()


@router.post("/tracking/load_video")
async def load_tracking_video_route(request: TrackingLoadVideoRequest):
    return await load_tracking_video(request)


@router.post("/tracking/track_prompt_points")
async def track_prompt_points(request: TrackingPromptPointsRequest):
    return await start_prompt_tracking(request)


@router.post("/tracking/track_grid")
async def track_grid_route(request: TrackingGridRequest):
    return await track_grid(request)


@router.post("/tracking/track_points")
async def track_points_route(request: TrackingPointsRequest):
    return await track_points(request)


@router.get("/tracking/results/{result_id}")
async def get_tracking_result(result_id: str):
    return {"result": load_prompt_tracking_result(result_id)}
