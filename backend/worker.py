"""
Analysis worker: run the pose-scoring pipeline and generate coaching feedback.

The worker is invoked synchronously inside a background thread (FastAPI
:class:`~fastapi.BackgroundTasks`).  It updates the job's status at each
stage and persists scores + coaching feedback when complete.

Pipeline
--------
1. Mark job ``processing``.
2. Run pose-extraction and scoring (currently a stub — real pipeline in Phase 3–4).
3. Persist the scores row with ``feedback = NULL``.
4. Generate coaching feedback via :func:`backend.coaching.generate_feedback`.
   *Skipped entirely if the job has already transitioned to* ``failed``.
5. Patch the scores row with the generated feedback and mark the job ``complete``.

Dependency injection
--------------------
The Anthropic client is passed in by the caller (``main.py`` lifespan) so
this module is independently testable: a test can substitute a mock client
without monkey-patching module globals.
"""

import json
import logging
from typing import Optional

import anthropic

from backend.coaching import generate_feedback
from backend.database import get_connection

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stub scoring pipeline
# ---------------------------------------------------------------------------

def _compute_scores(job_id: str, technique: str) -> dict:
    """Placeholder for the real pose-extraction + DTW scoring pipeline.

    Returns a scores dict conforming to the structure expected by
    :func:`backend.coaching.generate_feedback`.  This stub will be replaced
    by the real biomechanics engine in Phase 3–4.

    Raises:
        RuntimeError: If scoring fails (simulated by an empty technique name).
    """
    if not technique:
        raise RuntimeError(f"Cannot score job {job_id}: technique is empty")

    return {
        "technique": technique,
        "metric_deltas": {
            "hip_rotation_deg": {
                "user": 15.2,
                "reference": 55.4,
                "delta": -40.2,
            },
            "chamber_height_ratio": {
                "user": 0.72,
                "reference": 0.91,
                "delta": -0.19,
            },
            "extension_angle_deg": {
                "user": 148.0,
                "reference": 165.0,
                "delta": -17.0,
            },
            "guard_drop_deg": {
                "user": 22.0,
                "reference": 5.0,
                "delta": 17.0,
            },
        },
        "keyframe_descriptions": [
            "Chamber: knee raised to hip height; guard hands relaxed at sides",
            "Impact: hip barely rotated; striking leg not fully extended",
        ],
        "overall_score": 58.0,
    }


# ---------------------------------------------------------------------------
# Main worker entry point
# ---------------------------------------------------------------------------

def process_job(
    job_id: str,
    technique: str,
    db_path: str,
    anthropic_client: Optional[anthropic.Anthropic] = None,
) -> None:
    """Run the full analysis pipeline for *job_id* and persist the results.

    Args:
        job_id: UUID string identifying the job row.
        technique: Technique name (e.g. ``"roundhouse_kick"``).
        db_path: Filesystem path to the SQLite database.
        anthropic_client: Injected :class:`anthropic.Anthropic` instance used
            to generate coaching feedback.  Pass ``None`` to skip feedback
            generation (e.g. in scoring-only integration tests).
    """
    conn = get_connection(db_path)
    try:
        # ------------------------------------------------------------------ #
        # 1. Mark job as processing                                            #
        # ------------------------------------------------------------------ #
        conn.execute(
            "UPDATE jobs SET status='processing', updated_at=CURRENT_TIMESTAMP WHERE id=?",
            (job_id,),
        )
        conn.commit()

        # ------------------------------------------------------------------ #
        # 2. Run the scoring pipeline                                          #
        # ------------------------------------------------------------------ #
        try:
            scores = _compute_scores(job_id, technique)
        except Exception:
            logger.exception("Scoring pipeline failed for job %s", job_id)
            conn.execute(
                "UPDATE jobs SET status='failed', updated_at=CURRENT_TIMESTAMP WHERE id=?",
                (job_id,),
            )
            conn.commit()
            # No coaching API call when the job transitions to 'failed'.
            return

        # ------------------------------------------------------------------ #
        # 3. Persist scores row (feedback NULL until generated)                #
        # ------------------------------------------------------------------ #
        cursor = conn.execute(
            """
            INSERT INTO scores (job_id, technique, criteria, overall_score, feedback)
            VALUES (?, ?, ?, ?, NULL)
            """,
            (
                job_id,
                technique,
                json.dumps(scores.get("metric_deltas", {})),
                scores.get("overall_score"),
            ),
        )
        scores_row_id: int = cursor.lastrowid  # type: ignore[assignment]
        conn.commit()

        # ------------------------------------------------------------------ #
        # 4. Generate coaching feedback (only when client is available)        #
        # ------------------------------------------------------------------ #
        feedback: Optional[str] = None
        if anthropic_client is not None:
            try:
                feedback = generate_feedback(scores, anthropic_client)
            except Exception:
                logger.exception(
                    "Coaching feedback generation failed for job %s — "
                    "storing scores without feedback",
                    job_id,
                )
                # A feedback failure does not fail the whole job; we store
                # the scores and mark the job complete anyway.

        # ------------------------------------------------------------------ #
        # 5. Patch feedback + mark job complete                                #
        # ------------------------------------------------------------------ #
        conn.execute(
            "UPDATE scores SET feedback=? WHERE id=?",
            (feedback, scores_row_id),
        )
        conn.execute(
            "UPDATE jobs SET status='complete', updated_at=CURRENT_TIMESTAMP WHERE id=?",
            (job_id,),
        )
        conn.commit()

        logger.info(
            "Job %s complete — overall_score=%.1f, feedback_len=%s",
            job_id,
            scores.get("overall_score", 0),
            len(feedback) if feedback else 0,
        )

    finally:
        conn.close()
