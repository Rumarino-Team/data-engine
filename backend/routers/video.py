from fastapi import APIRouter, Depends, Header, Path

from app.dependencies import get_context
from core.context import AppContext
from schemas.responses import (
    AnnotationUpdate,
    ClearPromptsResponse,
    JobStartResponse,
    MaskData,
    MaskEditResponse,
    ObjectCreationResponse,
    PointsResponse,
    ResetResponse,
    SaveResponse,
    VideoInfoResponse,
)
from schemas.video import (
    VideoAddMaskRequest,
    VideoAddPointsOrBoxRequest,
    VideoInitStateRequest,
    VideoObjectUpdate,
    AnnotationPrecondition,
    VideoPropagateRequest,
    VideoSaveRequest,
)
from video.propagation import start_propagation
from video.service import (
    add_new_mask,
    add_new_points_or_box,
    clear_all_prompts_in_frame,
    get_mask_data,
    get_video_frame,
    get_video_info,
    get_video_mask_frame,
    remove_object,
    reset_video_state,
    save_video_session,
    start_video_init,
    create_object,
    update_object,
)


# Blocking services (including lock waits) run as one unit in FastAPI's worker
# pool. Calling them from an async route would block the event loop instead.
router = APIRouter()


def annotation_precondition(
    session_id: str = Header(alias="X-Session-ID", min_length=1),
    expected_revision: int = Header(alias="X-Annotation-Revision", ge=0),
) -> AnnotationPrecondition:
    return AnnotationPrecondition(session_id=session_id, expected_revision=expected_revision)


@router.post("/video/init_state", response_model=JobStartResponse)
def init_video_state(request: VideoInitStateRequest, ctx: AppContext = Depends(get_context)):
    return start_video_init(ctx, request)


@router.post("/video/reset_state", response_model=ResetResponse)
def reset_video_state_route(
    precondition: AnnotationPrecondition = Depends(annotation_precondition),
    ctx: AppContext = Depends(get_context),
):
    return reset_video_state(ctx, precondition)


@router.post("/video/add_new_points_or_box", response_model=PointsResponse)
def add_new_points_or_box_route(
    request: VideoAddPointsOrBoxRequest,
    precondition: AnnotationPrecondition = Depends(annotation_precondition),
    ctx: AppContext = Depends(get_context),
):
    return add_new_points_or_box(ctx, request, precondition)


@router.post("/video/add_new_mask", response_model=MaskEditResponse)
def add_new_mask_route(
    request: VideoAddMaskRequest,
    precondition: AnnotationPrecondition = Depends(annotation_precondition),
    ctx: AppContext = Depends(get_context),
):
    return add_new_mask(ctx, request, precondition)


@router.post("/video/save", response_model=SaveResponse)
def save_video_session_route(
    request: VideoSaveRequest,
    precondition: AnnotationPrecondition = Depends(annotation_precondition),
    ctx: AppContext = Depends(get_context),
):
    return save_video_session(ctx, request, precondition)


@router.post("/video/propagate_in_video", response_model=JobStartResponse)
def propagate_in_video(
    request: VideoPropagateRequest,
    precondition: AnnotationPrecondition = Depends(annotation_precondition),
    ctx: AppContext = Depends(get_context),
):
    return start_propagation(ctx, request, precondition)


@router.post("/video/clear_all_prompts_in_frame", response_model=ClearPromptsResponse)
def clear_all_prompts_in_frame_route(
    frame_idx: int,
    obj_id: int,
    precondition: AnnotationPrecondition = Depends(annotation_precondition),
    ctx: AppContext = Depends(get_context),
):
    return clear_all_prompts_in_frame(ctx, frame_idx, obj_id, precondition)


@router.post("/video/objects", response_model=ObjectCreationResponse)
def create_object_route(
    precondition: AnnotationPrecondition = Depends(annotation_precondition),
    ctx: AppContext = Depends(get_context),
):
    return create_object(ctx, precondition)


@router.put("/video/objects/{obj_id}", response_model=AnnotationUpdate)
def update_object_route(
    request: VideoObjectUpdate,
    obj_id: int = Path(ge=1),
    precondition: AnnotationPrecondition = Depends(annotation_precondition),
    ctx: AppContext = Depends(get_context),
):
    return update_object(ctx, obj_id, request, precondition)


@router.post("/video/remove_object", response_model=AnnotationUpdate)
def remove_object_route(
    obj_id: int,
    precondition: AnnotationPrecondition = Depends(annotation_precondition),
    ctx: AppContext = Depends(get_context),
):
    return remove_object(ctx, obj_id, precondition)


@router.get("/video/info", response_model=VideoInfoResponse)
def get_video_info_route(ctx: AppContext = Depends(get_context)):
    return get_video_info(ctx)


@router.get("/video/mask_data/{frame_idx}", response_model=MaskData)
def get_mask_data_route(frame_idx: int, ctx: AppContext = Depends(get_context)):
    return get_mask_data(ctx, frame_idx)


@router.get("/video/frame/{frame_idx}")
def get_video_frame_route(frame_idx: int, ctx: AppContext = Depends(get_context)):
    return get_video_frame(ctx, frame_idx)


@router.get("/video/mask_frame/{frame_idx}")
def get_video_mask_frame_route(frame_idx: int, ctx: AppContext = Depends(get_context)):
    return get_video_mask_frame(ctx, frame_idx)
