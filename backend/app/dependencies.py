from fastapi import Request

from core.context import AppContext


# Async on purpose: a sync dependency runs in the worker pool, which long lock waits can saturate,
# and the async control routes (/health, /jobs/...) must keep answering then.
async def get_context(request: Request) -> AppContext:
    return request.app.state.context
