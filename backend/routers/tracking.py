from fastapi import APIRouter, Depends
from routers.video import annotation_precondition
from schemas.video import AnnotationPrecondition
from core.jobs import foreground_lock

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
def load_tracking_video_route(request: TrackingLoadVideoRequest):
    return load_tracking_video(request)


@router.post("/tracking/track_prompt_points")
def track_prompt_points(request: TrackingPromptPointsRequest,
                              precondition: AnnotationPrecondition = Depends(annotation_precondition)):
    return start_prompt_tracking(request, precondition)


@router.post("/tracking/track_grid")
def track_grid_route(request: TrackingGridRequest):
    return track_grid(request)


@router.post("/tracking/track_points")
def track_points_route(request: TrackingPointsRequest):
    return track_points(request)


@router.get("/tracking/results/{result_id}")
def get_tracking_result(result_id: str):
    with foreground_lock:
        return {"result": load_prompt_tracking_result(result_id)}
