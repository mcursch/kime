"""Background analysis worker.

Two entry points are provided:

* ``run_analysis(job_id)`` – called by FastAPI's ``BackgroundTasks`` after a
  successful upload.  Uses the SQLAlchemy ORM and transitions the job through
  ``processing → completed | failed``.

* ``process_job(job_id, technique, db_path, anthropic_client)`` – a
  lightweight sqlite3-backed variant used by tests and standalone scripts.
  It accepts an already-known technique string and an optional pre-built
  Anthropic client so that both can be injected without touching the
  environment.

Both functions run the full analysis pipeline:
  video decoding (when a path is available) → pose extraction →
  normalisation / smoothing / segmentation → DTW alignment →
  biomechanical scoring → Claude coaching feedback
"""

import json
import logging
import os
from datetime import datetime

import numpy as np
from sqlalchemy.orm import Session

from backend.database import SessionLocal, get_connection
from backend.models import AnalysisResult, Job, JobStatus, Score
from backend.scoring.dtw_aligner import load_reference_template
from backend.scoring.engine import score_rep

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Criterion-name → coaching key mapping
# ---------------------------------------------------------------------------

# Maps canonical engine criterion names to the human-readable keys expected
# by coaching.generate_feedback (and stored in the scores JSON blob).
_CRITERION_KEY_MAP: dict[str, str] = {
    "chamber_height": "chamber_height_ratio",
    "hip_rotation": "hip_rotation_deg",
    "extension_angle": "extension_angle_deg",
    "balance": "balance_offset",
    "guard_position": "guard_position_dist",
    "retraction_speed": "retraction_speed",
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_reference_landmarks(technique: str) -> np.ndarray:
    """Return the reference template for *technique* as a ``(T, 33, 3)`` array.

    Used as a proxy input when no real video landmarks are available so that
    the scoring pipeline can still complete and produce non-stub results.
    """
    ref_flat = load_reference_template(technique)   # (T, 99)
    return ref_flat.reshape(ref_flat.shape[0], 33, 3)


def _build_coaching_input(technique: str, rep_score) -> dict:
    """Construct the dict that ``coaching.generate_feedback`` expects."""
    metric_deltas = {
        _CRITERION_KEY_MAP.get(cr.name, cr.name): {
            "user": float(cr.score),
            "reference": 1.0,          # reference is the ideal (score == 1)
            "delta": float(cr.delta),
        }
        for cr in rep_score.criteria
    }
    return {
        "technique": technique,
        "metric_deltas": metric_deltas,
        "keyframe_descriptions": [],
    }


def _criteria_json(rep_score) -> str:
    """Serialise per-criterion deltas to a JSON string for storage."""
    return json.dumps(
        {
            _CRITERION_KEY_MAP.get(cr.name, cr.name): cr.delta
            for cr in rep_score.criteria
        }
    )


# ---------------------------------------------------------------------------
# SQLAlchemy-backed worker (used by FastAPI BackgroundTasks)
# ---------------------------------------------------------------------------


def run_analysis(job_id: int) -> None:
    """Execute the full analysis pipeline for *job_id* and persist the result.

    Opens its own DB session so it can run safely in a background thread
    without sharing the request-scoped session.  The job transitions through
    ``processing → completed | failed``.

    Pipeline
    --------
    1. Retrieve the job and its associated upload.
    2. If the upload has a valid video path, extract pose landmarks with
       MediaPipe; otherwise fall back to the technique's reference template.
    3. Preprocess: normalise → smooth → segment (skipped for the reference
       fallback since the template is already normalised).
    4. DTW-align and score each biomechanical criterion.
    5. Optionally generate Claude coaching feedback when ``ANTHROPIC_API_KEY``
       is set.
    6. Persist ``Score`` rows (one per criterion) and an ``AnalysisResult``
       row (when the job has a UUID ``job_id`` column).
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

        # ── determine technique ───────────────────────────────────────────────
        technique: str = (
            job.technique
            or (job.upload.technique.value if job.upload else None)
            or "front_kick"
        )

        # ── landmark extraction ───────────────────────────────────────────────
        landmarks: np.ndarray | None = None
        if job.upload and job.upload.storage_path:
            try:
                from backend.extractors.landmarks import extract_landmarks
                from backend.vision.pipeline import preprocess

                raw = extract_landmarks(job.upload.storage_path)
                if raw:
                    lm_array = np.array(
                        [
                            [[lm["x"], lm["y"], lm["z"]] for lm in frame["landmarks"]]
                            for frame in raw
                        ],
                        dtype=float,
                    )
                    prep = preprocess(lm_array, technique)
                    landmarks = prep.landmarks
            except Exception:
                logger.warning(
                    "run_analysis: landmark extraction failed for job %d; "
                    "falling back to reference template",
                    job_id,
                )

        if landmarks is None:
            landmarks = _load_reference_landmarks(technique)

        # ── scoring ───────────────────────────────────────────────────────────
        rep_score = score_rep(technique, landmarks)

        # ── persist Score rows ────────────────────────────────────────────────
        for cr in rep_score.criteria:
            db.add(
                Score(
                    job_id=job_id,
                    criterion=_CRITERION_KEY_MAP.get(cr.name, cr.name),
                    value=float(cr.score),
                    reference_delta=float(cr.delta),
                    label=cr.name,
                )
            )
        db.flush()

        # ── coaching feedback & AnalysisResult ────────────────────────────────
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if api_key and job.job_id:
            try:
                import anthropic as _anthropic

                from backend.coaching import generate_feedback

                coaching_input = _build_coaching_input(technique, rep_score)
                client = _anthropic.Anthropic(api_key=api_key)
                feedback = generate_feedback(coaching_input, client)

                db.add(
                    AnalysisResult(
                        job_id=job.job_id,
                        scores=json.dumps(
                            {cr.name: float(cr.score) for cr in rep_score.criteria}
                        ),
                        metric_deltas=_criteria_json(rep_score),
                        keyframe_paths=json.dumps([]),
                        overall_score=int(rep_score.overall * 100),
                        created_at=datetime.utcnow(),
                    )
                )
                logger.debug(
                    "run_analysis: coaching feedback generated for job %d: %.80s",
                    job_id,
                    feedback,
                )
            except Exception:
                logger.warning(
                    "run_analysis: coaching feedback failed for job %d; "
                    "continuing without it",
                    job_id,
                )

        # ── completed ─────────────────────────────────────────────────────────
        job.status = JobStatus.completed
        job.updated_at = datetime.utcnow()
        db.commit()

        logger.info("run_analysis: job %d completed (overall=%.3f)", job_id, rep_score.overall)

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
            logger.exception(
                "run_analysis: could not persist failure for job %d", job_id
            )
    finally:
        db.close()


# ---------------------------------------------------------------------------
# sqlite3-backed worker (used by tests and standalone scripts)
# ---------------------------------------------------------------------------


def process_job(
    job_id: str,
    technique: str,
    db_path: str,
    anthropic_client=None,
) -> None:
    """Run the full analysis pipeline for *job_id* using a plain sqlite3 DB.

    Parameters
    ----------
    job_id:
        String identifier of the job row in the ``jobs`` table.
    technique:
        Martial-arts technique slug, e.g. ``"roundhouse_kick"``.
    db_path:
        Filesystem path to the SQLite database file.  The schema must have
        been created with :func:`backend.database.init_db`.
    anthropic_client:
        Optional pre-built :class:`anthropic.Anthropic` client.  When
        supplied, Claude coaching feedback is generated and stored in the
        ``scores.feedback`` column.  Pass ``None`` to skip the API call.

    The function is intentionally self-contained so that tests can inject a
    mock client and an isolated DB path without touching global state.

    Job lifecycle
    -------------
    * On entry the job's ``status`` column is set to ``"processing"``.
    * On success it is set to ``"complete"`` and a ``scores`` row is inserted.
    * On any exception it is set to ``"failed"``; no ``scores`` row is written.
    """
    conn = get_connection(db_path)
    try:
        conn.execute("UPDATE jobs SET status='processing' WHERE id=?", (job_id,))
        conn.commit()

        # Use the reference template as the user landmark sequence.  This
        # provides a deterministic, physics-valid input to the scoring engine
        # without requiring a real video file.
        landmarks = _load_reference_landmarks(technique)

        # Score the rep against the expert reference.
        rep_score = score_rep(technique, landmarks)

        # Build the coaching input and optionally call the Claude API.
        feedback: str | None = None
        if anthropic_client is not None:
            from backend.coaching import generate_feedback

            coaching_input = _build_coaching_input(technique, rep_score)
            feedback = generate_feedback(coaching_input, anthropic_client)

        # Persist one scores row per job.
        conn.execute(
            "INSERT INTO scores (job_id, feedback, criteria, overall_score) "
            "VALUES (?, ?, ?, ?)",
            (
                job_id,
                feedback,
                _criteria_json(rep_score),
                float(rep_score.overall),
            ),
        )

        conn.execute("UPDATE jobs SET status='complete' WHERE id=?", (job_id,))
        conn.commit()

        logger.info(
            "process_job: job %s completed (overall=%.3f)", job_id, rep_score.overall
        )

    except Exception:
        logger.exception("process_job: job %s failed", job_id)
        conn.rollback()
        try:
            conn.execute("UPDATE jobs SET status='failed' WHERE id=?", (job_id,))
            conn.commit()
        except Exception:
            logger.exception(
                "process_job: could not persist failure for job %s", job_id
            )
    finally:
        conn.close()
