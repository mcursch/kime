"""Background analysis worker.

``run_analysis`` is called by FastAPI's ``BackgroundTasks`` after a successful
upload.  It transitions the job through ``processing → completed | failed`` and
writes a stub ``Score`` row on completion so that downstream consumers always
have a row to work with once a job is finished.
"""

import logging
from datetime import datetime

from sqlalchemy.orm import Session

from backend.database import SessionLocal
from backend.models import Job, JobStatus, Score

logger = logging.getLogger(__name__)

# Stub criteria emitted by the worker until the real judging engine is wired in.
_STUB_CRITERIA: list[tuple[str, float, float | None, str]] = [
    ("chamber_height", 0.0, None, "Chamber height (stub)"),
    ("hip_rotation", 0.0, None, "Hip rotation at impact (stub)"),
    ("extension_angle", 0.0, None, "Extension angle (stub)"),
]


def run_analysis(job_id: int) -> None:
    """Execute stub analysis for *job_id* and persist the result.

    Opens its own DB session so it can run safely in a background thread
    without sharing the request-scoped session.
    """
    db: Session = SessionLocal()
    try:
        job: Job | None = db.get(Job, job_id)
        if job is None:
            logger.error("run_analysis: job %d not found", job_id)
            return

        # ── processing ────────────────────────────────────────────────────────
        job.status = JobStatus.processing
        job.updated_at = datetime.utcnow()
        db.commit()

        # Stub: real analysis (pose extraction, DTW, scoring) goes here.
        _write_stub_scores(db, job_id)

        # ── completed ─────────────────────────────────────────────────────────
        job.status = JobStatus.completed
        job.updated_at = datetime.utcnow()
        db.commit()

        logger.info("run_analysis: job %d completed", job_id)

    except Exception:
        logger.exception("run_analysis: job %d failed", job_id)
        try:
            db.rollback()
            job = db.get(Job, job_id)
            if job is not None:
                job.status = JobStatus.failed
                job.error_message = "Analysis failed; see server logs."
                job.updated_at = datetime.utcnow()
                db.commit()
        except Exception:
            logger.exception("run_analysis: could not persist failure for job %d", job_id)
    finally:
        db.close()


def _write_stub_scores(db: Session, job_id: int) -> None:
    """Insert placeholder Score rows so the schema is exercised end-to-end."""
    for criterion, value, reference_delta, label in _STUB_CRITERIA:
        db.add(
            Score(
                job_id=job_id,
                criterion=criterion,
                value=value,
                reference_delta=reference_delta,
                label=label,
            )
        )
    db.commit()
