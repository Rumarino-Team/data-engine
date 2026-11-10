from fastapi import APIRouter, Depends

from core.errors import OperationError
from app.dependencies import get_context
from core.context import AppContext


router = APIRouter()


@router.get("/jobs/current")
async def get_current_job(ctx: AppContext = Depends(get_context)):
    return {"job": ctx.jobs.snapshot()}


@router.get("/jobs/{job_id}")
async def get_job(job_id: str, ctx: AppContext = Depends(get_context)):
    job = ctx.jobs.snapshot()
    if job is None or job.get("job_id") != job_id:
        raise OperationError(404, "Job not found.")
    return {"job": job}


@router.post("/jobs/{job_id}/clear_result")
async def clear_job_result(job_id: str, ctx: AppContext = Depends(get_context)):
    return {"cleared": ctx.jobs.clear_result(job_id)}
