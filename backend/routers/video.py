from fastapi import APIRouter

from schemas.video import (
    VideoAddMaskRequest,
    VideoAddPointsOrBoxRequest,
    VideoInitStateRequest,
    VideoPropagateRequest,
    VideoSaveRequest,
)
from video.propagation import start_propagation
from video.service import (
    add_new_mask,
    add_new_points_or_box,
    clear_all_prompts_in_frame,
    get_mask_data,
    get_mask_data_window,
    get_mask_manifest,
    get_video_frame,
    get_video_info,
    get_video_mask_frame,
    remove_object,
    reset_video_state,
    save_video_session,
    start_video_init,
)


router = APIRouter()


@router.post("/video/init_state")
async def init_video_state(request: VideoInitStateRequest):
    return await start_video_init(request)


@router.post("/video/reset_state")
async def reset_video_state_route():
    return await reset_video_state()


@router.post("/video/add_new_points_or_box")
async def add_new_points_or_box_route(request: VideoAddPointsOrBoxRequest):
    return await add_new_points_or_box(request)


@router.post("/video/add_new_mask")
async def add_new_mask_route(request: VideoAddMaskRequest):
    return await add_new_mask(request)


@router.post("/video/save")
async def save_video_session_route(request: VideoSaveRequest):
    return await save_video_session(request)


@router.post("/video/propagate_in_video")
async def propagate_in_video(request: VideoPropagateRequest):
    return await start_propagation(request)


@router.post("/video/clear_all_prompts_in_frame")
async def clear_all_prompts_in_frame_route(frame_idx: int, obj_id: int):
    return await clear_all_prompts_in_frame(frame_idx, obj_id)


@router.post("/video/remove_object")
async def remove_object_route(obj_id: int):
    return await remove_object(obj_id)


@router.get("/video/info")
async def get_video_info_route():
    return await get_video_info()


@router.get("/video/mask_manifest")
async def get_mask_manifest_route():
    return await get_mask_manifest()


@router.get("/video/mask_data/{frame_idx}")
async def get_mask_data_route(frame_idx: int):
    return await get_mask_data(frame_idx)


@router.get("/video/mask_data_window")
async def get_mask_data_window_route(
    start_frame_idx: int,
    end_frame_idx: int,
    object_ids: str | None = None,
    include_empty: bool = False,
):
    return await get_mask_data_window(
        start_frame_idx=start_frame_idx,
        end_frame_idx=end_frame_idx,
        object_ids=object_ids,
        include_empty=include_empty,
    )


@router.get("/video/frame/{frame_idx}")
async def get_video_frame_route(frame_idx: int):
    return await get_video_frame(frame_idx)


@router.get("/video/mask_frame/{frame_idx}")
async def get_video_mask_frame_route(frame_idx: int):
    return await get_video_mask_frame(frame_idx)
