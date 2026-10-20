from fastapi import FastAPI, HTTPException, Request
from fastapi.exception_handlers import http_exception_handler as _default_http_exception_handler
from fastapi.middleware.cors import CORSMiddleware

from core.errors import OperationError
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

    @app.exception_handler(OperationError)
    async def operation_error_handler(request: Request, exc: OperationError):
        return await _default_http_exception_handler(
            request, HTTPException(status_code=exc.status_code, detail=exc.message)
        )

    app.include_router(root.router)
    app.include_router(jobs.router)
    app.include_router(video.router)
    app.include_router(tracking.router)
    app.add_event_handler("shutdown", cleanup_backend_cache_on_shutdown)

    return app
