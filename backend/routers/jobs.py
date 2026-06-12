"""Jobs router – GET /jobs/{job_id} and POST /jobs/upload.

Provides endpoints for creating analysis jobs and tracking their status.
"""

import uuid
from datetime import datetime

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models import Job, JobStatus
from backend.worker import run_analysis

router = APIRouter(tags=["jobs"])


class JobResponse(BaseModel):
    job_id: int
    status: str
    error_message: str | None = None


class JobUploadResponse(BaseModel):
    job_id: str
    status: str


@router.post("/jobs/upload", response_model=JobUploadResponse, status_code=202)
def create_job(
    request: Request,
    background_tasks: BackgroundTasks,
    technique: str = "roundhouse_kick",
    db: Session = Depends(get_db),
) -> JobUploadResponse:
    """Create an analysis job for the given technique without a video upload.

    The job is enqueued for background processing immediately.  Coaching
    feedback is generated using the Anthropic client stored in ``app.state``
    (set up by the application lifespan).

    Returns HTTP 202 with ``{job_id, status}``; the client should poll
    ``GET /jobs/{job_id}/results`` for the final result.
    """
    job_uuid = str(uuid.uuid4())
    now = datetime.utcnow()
    job_row = Job(
        job_id=job_uuid,
        technique=technique,
        status=JobStatus.pending,
        created_at=now,
        updated_at=now,
    )
    db.add(job_row)
    db.commit()
    db.refresh(job_row)

    anthropic_client = getattr(request.app.state, "anthropic_client", None)
    background_tasks.add_task(run_analysis, job_row.id, anthropic_client)

    return JobUploadResponse(job_id=job_uuid, status="pending")


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
