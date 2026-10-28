from fastapi import APIRouter, Depends, Header, Path

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
    get_mask_manifest,
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


@router.post("/video/init_state")
def init_video_state(request: VideoInitStateRequest):
    return start_video_init(request)


@router.post("/video/reset_state")
def reset_video_state_route(precondition: AnnotationPrecondition = Depends(annotation_precondition)):
    return reset_video_state(precondition)


@router.post("/video/add_new_points_or_box")
def add_new_points_or_box_route(request: VideoAddPointsOrBoxRequest,
                                      precondition: AnnotationPrecondition = Depends(annotation_precondition)):
    return add_new_points_or_box(request, precondition)


@router.post("/video/add_new_mask")
def add_new_mask_route(request: VideoAddMaskRequest,
                             precondition: AnnotationPrecondition = Depends(annotation_precondition)):
    return add_new_mask(request, precondition)


@router.post("/video/save")
def save_video_session_route(request: VideoSaveRequest):
    return save_video_session(request)


@router.post("/video/propagate_in_video")
def propagate_in_video(request: VideoPropagateRequest,
                             precondition: AnnotationPrecondition = Depends(annotation_precondition)):
    return start_propagation(request, precondition)


@router.post("/video/clear_all_prompts_in_frame")
def clear_all_prompts_in_frame_route(frame_idx: int, obj_id: int,
                                         precondition: AnnotationPrecondition = Depends(annotation_precondition)):
    return clear_all_prompts_in_frame(frame_idx, obj_id, precondition)


@router.post("/video/objects")
def create_object_route(precondition: AnnotationPrecondition = Depends(annotation_precondition)):
    return create_object(precondition)


@router.put("/video/objects/{obj_id}")
def update_object_route(
    request: VideoObjectUpdate, obj_id: int = Path(ge=1),
    precondition: AnnotationPrecondition = Depends(annotation_precondition),
):
    return update_object(obj_id, request, precondition)


@router.post("/video/remove_object")
def remove_object_route(obj_id: int, precondition: AnnotationPrecondition = Depends(annotation_precondition)):
    return remove_object(obj_id, precondition)


@router.get("/video/info")
def get_video_info_route():
    return get_video_info()


@router.get("/video/mask_manifest")
def get_mask_manifest_route():
    return get_mask_manifest()


@router.get("/video/mask_data/{frame_idx}")
def get_mask_data_route(frame_idx: int):
    return get_mask_data(frame_idx)


@router.get("/video/frame/{frame_idx}")
def get_video_frame_route(frame_idx: int):
    return get_video_frame(frame_idx)


@router.get("/video/mask_frame/{frame_idx}")
def get_video_mask_frame_route(frame_idx: int):
    return get_video_mask_frame(frame_idx)
