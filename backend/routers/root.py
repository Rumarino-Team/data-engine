from fastapi import APIRouter, Depends

from app.dependencies import get_context
from core.context import AppContext
from schemas.responses import HealthResponse, RootResponse, StatusResponse


router = APIRouter()


@router.get("/", response_model=RootResponse)
async def root():
    return {"message": "Data Engine Backend"}


@router.get("/health", response_model=HealthResponse)
async def health():
    return {"status": "ok"}


@router.get("/status", response_model=StatusResponse)
async def status(ctx: AppContext = Depends(get_context)):
    # A worker can replace/release a runtime while this lightweight probe runs.
    session = ctx.session
    video_masker, tracker = session.video_masker, session.tracker
    if video_masker is not None:
        return {"status": "video active", "device": video_masker.device.type}
    if tracker is not None:
        return {"status": "tracking active", "device": tracker.device}
    return {"status": "inactive", "device": None}
