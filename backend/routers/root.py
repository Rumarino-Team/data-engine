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
    if state.video_masker is not None:
        return {"status": "video active", "device": state.video_masker.device.type}
    if state.tracker is not None:
        return {"status": "tracking active", "device": state.tracker.device}
    return {"status": "inactive", "device": None}
