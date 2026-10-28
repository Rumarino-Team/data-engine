from fastapi import APIRouter

from core.state import state


router = APIRouter()


@router.get("/")
async def root():
    return {"message": "Data Engine Backend"}


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.get("/status")
async def status():
    # A worker can replace/release a runtime while this lightweight probe runs.
    video_masker, tracker = state.video_masker, state.tracker
    if video_masker is not None:
        return {"status": "video active", "device": video_masker.device.type}
    if tracker is not None:
        return {"status": "tracking active", "device": tracker.device}
    return {"status": "inactive", "device": None}
