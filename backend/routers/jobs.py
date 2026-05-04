from fastapi import APIRouter, HTTPException

from core.jobs import clear_current_job_result, serialize_job


router = APIRouter()


@router.get("/jobs/current")
async def get_current_job():
    return {"job": serialize_job()}


@router.get("/jobs/{job_id}")
async def get_job(job_id: str):
    job = serialize_job()
    if job is None or job.get("job_id") != job_id:
        raise HTTPException(status_code=404, detail="Job not found.")
    return {"job": job}


@router.post("/jobs/{job_id}/clear_result")
async def clear_job_result(job_id: str):
    return {"cleared": clear_current_job_result(job_id)}
