"""Jobs router – GET /jobs/{job_id}.

Provides a status-polling endpoint so clients can track async analysis jobs.
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models import Job

router = APIRouter(tags=["jobs"])


class JobResponse(BaseModel):
    job_id: int
    status: str
    error_message: str | None = None


@router.get("/jobs/{job_id}", response_model=JobResponse)
def get_job(job_id: int, db: Session = Depends(get_db)) -> JobResponse:
    """Return the current status of an analysis job.

    Raises HTTP 404 if the job does not exist.
    """
    job: Job | None = db.get(Job, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")
    return JobResponse(
        job_id=job.id,
        status=job.status.value,
        error_message=job.error_message,
    )
