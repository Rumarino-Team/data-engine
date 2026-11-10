from fastapi import APIRouter

from core.errors import OperationError
from core.jobs import job_manager


router = APIRouter()


@router.get("/jobs/current")
async def get_current_job():
    return {"job": job_manager.snapshot()}


@router.get("/jobs/{job_id}")
async def get_job(job_id: str):
    job = job_manager.snapshot()
    if job is None or job.get("job_id") != job_id:
        raise OperationError(404, "Job not found.")
    return {"job": job}


@router.post("/jobs/{job_id}/clear_result")
async def clear_job_result(job_id: str):
    return {"cleared": job_manager.clear_result(job_id)}
