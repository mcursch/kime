"""FastAPI router for async pose analysis jobs.

Endpoints
---------
POST /api/analyze
    Accept a multipart video upload.  Save to a temp file, kick off landmark
    extraction in a background thread, and return HTTP 202 with a job_id.

GET /api/analyze/{job_id}
    Return job status (pending | complete | error) and, when complete,
    the landmarks array produced by the extractor.
"""

from __future__ import annotations

import asyncio
import logging
import tempfile
import uuid
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, UploadFile, status
from pydantic import BaseModel

from backend.extractors.landmarks import extract_landmarks

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/analyze", tags=["analyze"])

# ---------------------------------------------------------------------------
# In-memory job store
# ---------------------------------------------------------------------------

JobStatus = Literal["pending", "complete", "error"]


class Job(BaseModel):
    job_id: str
    status: JobStatus
    landmarks: list[dict[str, Any]] | None = None
    error: str | None = None


# Maps job_id → Job.  Shared across requests; fine for single-process use.
_jobs: dict[str, Job] = {}


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------


def _run_extraction(job_id: str, video_path: Path) -> None:
    """Synchronous worker executed in a thread pool."""
    try:
        landmarks = extract_landmarks(video_path)
        _jobs[job_id] = Job(
            job_id=job_id,
            status="complete",
            landmarks=landmarks,
        )
        logger.info("Job %s complete – %d frames", job_id, len(landmarks))
    except Exception as exc:  # noqa: BLE001
        logger.exception("Job %s failed", job_id)
        _jobs[job_id] = Job(job_id=job_id, status="error", error=str(exc))
    finally:
        # Clean up the temp file regardless of outcome.
        try:
            video_path.unlink(missing_ok=True)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


class SubmitResponse(BaseModel):
    job_id: str


@router.post(
    "",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=SubmitResponse,
    summary="Submit a video for pose analysis",
)
async def submit_video(file: UploadFile) -> SubmitResponse:
    """Accept a multipart video upload and start a background analysis job.

    Returns HTTP 202 immediately with a *job_id* that can be polled via
    ``GET /api/analyze/{job_id}``.
    """
    job_id = str(uuid.uuid4())

    # Persist upload to a named temp file that the background thread can open.
    suffix = Path(file.filename or "upload").suffix or ".mp4"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        contents = await file.read()
        tmp.write(contents)
        tmp.flush()
        video_path = Path(tmp.name)
    finally:
        tmp.close()

    # Register job as pending before the thread starts so the poll endpoint
    # never sees an unknown ID between submission and first status write.
    _jobs[job_id] = Job(job_id=job_id, status="pending")

    loop = asyncio.get_running_loop()
    future = loop.run_in_executor(
        None,  # default ThreadPoolExecutor
        _run_extraction,
        job_id,
        video_path,
    )

    def _on_done(fut: asyncio.Future) -> None:  # type: ignore[type-arg]
        """Surface any exception that escaped _run_extraction's try/except."""
        exc = fut.exception()
        if exc is not None:
            logger.error(
                "Unhandled exception in extraction future for job %s",
                job_id,
                exc_info=exc,
            )
            _jobs[job_id] = Job(job_id=job_id, status="error", error=str(exc))

    future.add_done_callback(_on_done)

    return SubmitResponse(job_id=job_id)


class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    landmarks: list[dict[str, Any]] | None = None
    error: str | None = None


@router.get(
    "/{job_id}",
    response_model=JobResponse,
    summary="Poll a pose analysis job",
)
async def get_job(job_id: str) -> JobResponse:
    """Return the current status of a job.

    - **pending** – extraction still running.
    - **complete** – extraction finished; ``landmarks`` is populated.
    - **error** – extraction failed; ``error`` contains the message.

    Returns HTTP 404 when *job_id* is unknown.
    """
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    return JobResponse(**job.model_dump())
