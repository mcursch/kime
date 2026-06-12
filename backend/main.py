"""
FastAPI application for Kime.

Endpoints
---------
POST /jobs/upload
    Accept a video file (or a technique name for stub testing) and enqueue
    an analysis job.  Returns immediately with a job ID.

GET /jobs/{job_id}/results
    Poll for job status.  When the job is ``complete`` the response includes
    per-criterion scores and the Claude-generated coaching feedback.

Startup validation
------------------
The app refuses to start when ``ANTHROPIC_API_KEY`` is not set in the
environment.  This provides a clear, immediate error rather than a
confusing ``AuthenticationError`` on the first request.
"""

import json
import os
import uuid
from contextlib import asynccontextmanager
from typing import Optional

import anthropic
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.database import DB_PATH, get_connection, init_db
from backend.worker import process_job

# Module-level client; populated during lifespan startup.
_anthropic_client: Optional[anthropic.Anthropic] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: validate env, initialise DB, create client."""
    global _anthropic_client  # noqa: PLW0603

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY environment variable is not set. "
            "Export it before starting the server:\n"
            "  export ANTHROPIC_API_KEY=sk-ant-..."
        )

    _anthropic_client = anthropic.Anthropic(api_key=api_key)

    # Create tables on first run (idempotent).
    init_db()

    yield

    # Cleanup (nothing to tear down for a sync client).
    _anthropic_client = None


app = FastAPI(title="Kime", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------


class JobResponse(BaseModel):
    job_id: str
    status: str


class ResultsResponse(BaseModel):
    job_id: str
    status: str
    technique: Optional[str] = None
    overall_score: Optional[float] = None
    criteria: Optional[dict] = None
    feedback: Optional[str] = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post("/jobs/upload", response_model=JobResponse, status_code=202)
async def upload_video(
    background_tasks: BackgroundTasks,
    technique: str = "roundhouse_kick",
    file: Optional[UploadFile] = File(default=None),
):
    """Upload a video and start an analysis job.

    The video is not yet processed synchronously (pipeline Phase 3–4);
    the ``technique`` query parameter drives the stub scoring logic.
    Returns immediately with ``status="pending"`` and a ``job_id`` to poll.
    """
    job_id = str(uuid.uuid4())

    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO jobs (id, status, technique) VALUES (?, 'pending', ?)",
            (job_id, technique),
        )
        conn.commit()
    finally:
        conn.close()

    background_tasks.add_task(_dispatch_worker, job_id, technique)

    return JobResponse(job_id=job_id, status="pending")


def _dispatch_worker(job_id: str, technique: str) -> None:
    """Thin wrapper so BackgroundTasks can call :func:`process_job`."""
    process_job(
        job_id=job_id,
        technique=technique,
        db_path=DB_PATH,
        anthropic_client=_anthropic_client,
    )


@app.get("/jobs/{job_id}/results", response_model=ResultsResponse)
def get_results(job_id: str):
    """Return the current status (and, when complete, scores + feedback)."""
    conn = get_connection()
    try:
        job = conn.execute(
            "SELECT id, status, technique FROM jobs WHERE id = ?",
            (job_id,),
        ).fetchone()

        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")

        scores_row = conn.execute(
            """
            SELECT overall_score, criteria, feedback
            FROM   scores
            WHERE  job_id = ?
            ORDER  BY id DESC
            LIMIT  1
            """,
            (job_id,),
        ).fetchone()
    finally:
        conn.close()

    result = ResultsResponse(
        job_id=job_id,
        status=job["status"],
        technique=job["technique"],
    )

    if scores_row is not None:
        result.overall_score = scores_row["overall_score"]
        result.feedback = scores_row["feedback"]
        try:
            result.criteria = (
                json.loads(scores_row["criteria"])
                if scores_row["criteria"]
                else None
            )
        except json.JSONDecodeError:
            result.criteria = None

    return result
