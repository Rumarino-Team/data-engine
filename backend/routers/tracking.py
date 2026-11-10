from fastapi import APIRouter, Depends

from app.dependencies import get_context
from core.context import AppContext
from routers.video import annotation_precondition
from schemas.video import AnnotationPrecondition

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
def load_tracking_video_route(request: TrackingLoadVideoRequest, ctx: AppContext = Depends(get_context)):
    return load_tracking_video(ctx, request)


@router.post("/tracking/track_prompt_points")
def track_prompt_points(
    request: TrackingPromptPointsRequest,
    precondition: AnnotationPrecondition = Depends(annotation_precondition),
    ctx: AppContext = Depends(get_context),
):
    return start_prompt_tracking(ctx, request, precondition)


@router.post("/tracking/track_grid")
def track_grid_route(request: TrackingGridRequest, ctx: AppContext = Depends(get_context)):
    return track_grid(ctx, request)


@router.post("/tracking/track_points")
def track_points_route(request: TrackingPointsRequest, ctx: AppContext = Depends(get_context)):
    return track_points(ctx, request)


@router.get("/tracking/results/{result_id}")
def get_tracking_result(result_id: str, ctx: AppContext = Depends(get_context)):
    with ctx.lock:
        return {"result": load_prompt_tracking_result(ctx.session, result_id)}
