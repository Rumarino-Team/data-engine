from fastapi import APIRouter, HTTPException

from core.jobs import serialize_job


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
