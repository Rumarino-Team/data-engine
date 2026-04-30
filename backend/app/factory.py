from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.runtime import cleanup_backend_cache_on_shutdown
from routers import jobs, root, tracking, video


def create_app() -> FastAPI:
    app = FastAPI()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(root.router)
    app.include_router(jobs.router)
    app.include_router(video.router)
    app.include_router(tracking.router)
    app.add_event_handler("shutdown", cleanup_backend_cache_on_shutdown)

    return app
