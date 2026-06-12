"""Upload router – POST /upload.

Accepts a multipart video file, persists it to the configured upload
directory, creates an Upload record and a Job record (status=pending),
enqueues the background analysis task, and returns HTTP 202 immediately.
"""

import uuid
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, Form, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.config import UPLOAD_DIR
from backend.database import get_db
from backend.models import Job, JobStatus, TechniqueType, Upload
from backend.worker import run_analysis

router = APIRouter(tags=["upload"])


class UploadResponse(BaseModel):
    job_id: int
    status: str


@router.post("/upload", response_model=UploadResponse, status_code=202)
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile,
    technique: TechniqueType = Form(default=TechniqueType.front_kick),
    db: Session = Depends(get_db),
) -> UploadResponse:
    """Accept a video upload, persist it, and kick off async analysis.

    Returns HTTP 202 with ``{job_id, status}`` immediately — the client
    should poll ``GET /jobs/{job_id}`` for the final result.

    Raises HTTP 422 if the uploaded file's MIME type is not a video type.
    """
    # ── MIME validation ───────────────────────────────────────────────────────
    content_type: str = file.content_type or ""
    if not content_type.startswith("video/"):
        raise HTTPException(
            status_code=422,
            detail=(
                f"Unsupported media type '{content_type}'. "
                "Only video/* files are accepted."
            ),
        )

    # ── persist file ─────────────────────────────────────────────────────────
    upload_dir: Path = UPLOAD_DIR
    upload_dir.mkdir(parents=True, exist_ok=True)

    # Use a UUID prefix to avoid filename collisions.
    safe_filename = f"{uuid.uuid4().hex}_{file.filename or 'upload'}"
    dest: Path = upload_dir / safe_filename

    contents = await file.read()
    dest.write_bytes(contents)

    # ── database rows ─────────────────────────────────────────────────────────
    now = datetime.utcnow()

    upload_row = Upload(
        filename=file.filename or safe_filename,
        storage_path=str(dest),
        technique=technique,
        created_at=now,
    )
    db.add(upload_row)
    db.flush()  # populate upload_row.id without committing yet

    job_row = Job(
        upload_id=upload_row.id,
        status=JobStatus.pending,
        created_at=now,
        updated_at=now,
    )
    db.add(job_row)
    db.commit()
    db.refresh(job_row)

    # ── enqueue background analysis ───────────────────────────────────────────
    background_tasks.add_task(run_analysis, job_row.id)

    return UploadResponse(job_id=job_row.id, status=job_row.status.value)
