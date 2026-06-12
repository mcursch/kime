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
from pathlib import Path

import numpy as np
from sqlalchemy.orm import Session

import backend.database as _db
from backend.config import UPLOAD_DIR
from backend.database import get_connection
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


def _build_coaching_input(
    technique: str,
    rep_score,
    keyframe_descriptions: list[str] | None = None,
) -> dict:
    """Construct the dict that ``coaching.generate_feedback`` expects.

    Parameters
    ----------
    technique:
        Martial-arts technique slug.
    rep_score:
        :class:`backend.scoring.engine.RepScore` produced by the scoring engine.
    keyframe_descriptions:
        Optional list of text descriptions for the chamber, extension, and
        retraction keyframes.  When omitted or ``None`` the list is empty.
    """
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
        "keyframe_descriptions": keyframe_descriptions or [],
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
# Keyframe extraction
# ---------------------------------------------------------------------------

# MediaPipe landmark indices needed for keyframe descriptions.
_LM_NOSE = 0
_LM_LEFT_WRIST, _LM_RIGHT_WRIST = 15, 16
_LM_LEFT_HIP, _LM_RIGHT_HIP = 23, 24
_LM_LEFT_KNEE, _LM_RIGHT_KNEE = 25, 26
_LM_LEFT_ANKLE, _LM_RIGHT_ANKLE = 27, 28


def _get_pose_connections():
    """Return MediaPipe Pose landmark connections, or an empty set on failure."""
    try:
        import mediapipe as mp  # noqa: PLC0415
        return mp.solutions.pose.POSE_CONNECTIONS
    except Exception:
        return set()


def _describe_keyframe(
    landmarks: np.ndarray,
    frame_idx: int,
    phase: str,
) -> str:
    """Produce a human-readable description of a single keyframe.

    Parameters
    ----------
    landmarks:
        Full normalised landmark sequence, shape ``(T, 33, 3)``.  Coordinates
        are hip-centred and torso-scale-normalised (torso length == 1.0).
    frame_idx:
        Index of the keyframe within *landmarks*.
    phase:
        ``"chamber"``, ``"extension"``, or ``"retraction"``.

    Returns
    -------
    str
        A concise description suitable for inclusion in the coaching prompt.
    """
    frame = landmarks[frame_idx]

    # Raised knee: in hip-centred MediaPipe coordinates y increases downward,
    # so the most-raised knee has the most-negative y value → use min().
    # Negate when reporting so positive means "above hip centre".
    raised_knee_y = float(min(frame[_LM_LEFT_KNEE, 1], frame[_LM_RIGHT_KNEE, 1]))

    if phase == "chamber":
        nose = frame[_LM_NOSE, :]
        guard_dist = float(min(
            np.linalg.norm(frame[_LM_LEFT_WRIST, :] - nose),
            np.linalg.norm(frame[_LM_RIGHT_WRIST, :] - nose),
        ))
        return (
            f"Chamber (frame {frame_idx}): raised knee at {-raised_knee_y:+.2f} "
            f"torso-lengths above hip centre; nearest guard wrist "
            f"{guard_dist:.2f} torso-lengths from nose"
        )

    elif phase == "extension":
        # Hip yaw from the hip-to-hip vector in the x–z plane
        hip_vec = frame[_LM_RIGHT_HIP, :] - frame[_LM_LEFT_HIP, :]
        hip_yaw = float(np.degrees(np.arctan2(hip_vec[2], hip_vec[0])))

        # Active end-effector: wrist or ankle with greatest peak displacement
        ee_candidates = [
            _LM_LEFT_WRIST, _LM_RIGHT_WRIST,
            _LM_LEFT_ANKLE, _LM_RIGHT_ANKLE,
        ]
        peak_disps = [
            float(np.max(np.linalg.norm(
                landmarks[:, idx, :] - landmarks[0, idx, :], axis=-1
            )))
            for idx in ee_candidates
        ]
        ee_idx = ee_candidates[int(np.argmax(peak_disps))]
        ee_height = float(frame[ee_idx, 1])
        return (
            f"Extension/Impact (frame {frame_idx}): hip yaw {hip_yaw:.1f}°; "
            f"active end-effector at height {ee_height:+.2f} torso-lengths"
        )

    else:  # retraction
        return (
            f"Retraction (frame {frame_idx}): raised knee at "
            f"{-raised_knee_y:+.2f} torso-lengths above hip centre"
        )


def _annotate_and_save_frame(
    video_path: str,
    video_frame_num: int,
    raw_lm_entry: dict,
    output_path: Path,
) -> bool:
    """Grab a video frame, draw the pose skeleton, and write it to *output_path*.

    Returns ``True`` on success, ``False`` if any step fails.
    """
    try:
        import cv2  # noqa: PLC0415

        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, video_frame_num)
        ok, bgr_frame = cap.read()
        cap.release()

        if not ok or bgr_frame is None:
            logger.warning(
                "_annotate_and_save_frame: could not read frame %d from %s",
                video_frame_num,
                video_path,
            )
            return False

        h, w = bgr_frame.shape[:2]
        lms = raw_lm_entry["landmarks"]  # list of 33 dicts (x, y ∈ [0, 1])
        pts: list[tuple[int, int]] = [
            (int(lm["x"] * w), int(lm["y"] * h)) for lm in lms
        ]

        # Draw skeleton edges
        for a_idx, b_idx in _get_pose_connections():
            if a_idx < len(pts) and b_idx < len(pts):
                cv2.line(bgr_frame, pts[a_idx], pts[b_idx], (0, 255, 0), 2)

        # Draw landmark dots
        for pt in pts:
            cv2.circle(bgr_frame, pt, 4, (0, 0, 255), -1)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), bgr_frame)
        return True

    except Exception:
        logger.warning(
            "_annotate_and_save_frame: failed to annotate frame %d",
            video_frame_num,
            exc_info=True,
        )
        return False


def _compute_keyframe_indices(
    landmarks: np.ndarray,
) -> tuple[int, int, int]:
    """Compute chamber, extension, and retraction frame indices using NumPy only.

    This is a lightweight alternative to
    :func:`backend.vision.segment.find_rep_window` that avoids the scipy
    dependency.  It uses simple peak-displacement heuristics:

    * **chamber**   — frame of maximum knee height in the first half of the
      sequence (pre-strike loading position).
    * **extension** — frame of peak active end-effector displacement from the
      starting position (moment of impact).
    * **retraction** — first local-minimum end-effector displacement frame
      after *extension* (limb returned toward guard).

    Parameters
    ----------
    landmarks:
        Normalised landmark sequence, shape ``(T, 33, 3)``.

    Returns
    -------
    (chamber_idx, extension_idx, retraction_idx)
        All indices are clamped to ``[0, T-1]``.
    """
    n = len(landmarks)

    # Chamber: highest knee in the first half (pre-strike load).
    # In hip-centred MediaPipe coordinates y increases *downward*, so the most
    # raised knee has the most-negative y value.  Use min + argmin accordingly.
    half = max(n // 2, 1)
    knee_heights = landmarks[:half, [_LM_LEFT_KNEE, _LM_RIGHT_KNEE], 1].min(axis=1)
    chamber_idx = int(np.argmin(knee_heights))

    # Extension: peak displacement of the most-active end-effector
    ee_candidates = [_LM_LEFT_WRIST, _LM_RIGHT_WRIST, _LM_LEFT_ANKLE, _LM_RIGHT_ANKLE]
    peak_disps = [
        float(np.max(np.linalg.norm(
            landmarks[:, idx, :] - landmarks[0, idx, :], axis=-1
        )))
        for idx in ee_candidates
    ]
    ee_idx = ee_candidates[int(np.argmax(peak_disps))]
    ee_disp = np.linalg.norm(
        landmarks[:, ee_idx, :] - landmarks[0, ee_idx, :], axis=-1
    )
    extension_idx = int(np.argmax(ee_disp))

    # Retraction: minimum displacement in the tail (after extension)
    tail_start = min(extension_idx + 1, n - 1)
    tail_disp = ee_disp[tail_start:]
    retraction_idx = (
        int(np.argmin(tail_disp)) + tail_start if len(tail_disp) > 0 else n - 1
    )

    return chamber_idx, extension_idx, retraction_idx


def _extract_keyframes(
    landmarks: np.ndarray,
    chamber_idx: int,
    extension_idx: int,
    retraction_idx: int,
    video_path: str | None = None,
    raw_frames: list | None = None,
    output_dir: Path | None = None,
) -> tuple[list[str], list[str]]:
    """Extract text descriptions and optional annotated images for the three key phases.

    Parameters
    ----------
    landmarks:
        Normalised landmark sequence, shape ``(T, 33, 3)``.
    chamber_idx, extension_idx, retraction_idx:
        Frame indices within *landmarks* for the chamber, peak-extension, and
        retraction phases as returned by
        :func:`backend.vision.pipeline.preprocess`.
    video_path:
        Optional path to the source video file.  When provided along with
        *raw_frames* and *output_dir*, annotated JPEG keyframe images are
        written to *output_dir*.
    raw_frames:
        Output of :func:`backend.extractors.landmarks.extract_landmarks` — one
        dict per pose-detected frame, each carrying a ``"frame"`` key that
        gives the original video frame index.
    output_dir:
        Directory where keyframe images are saved.  Created automatically if
        it does not exist.

    Returns
    -------
    (keyframe_descriptions, keyframe_paths)
        Both lists have exactly three entries (chamber / extension /
        retraction).  Entries in *keyframe_paths* are absolute filesystem
        paths when an annotated image was saved, or an empty string when no
        video was available.
    """
    phases = [
        ("chamber",    chamber_idx),
        ("extension",  extension_idx),
        ("retraction", retraction_idx),
    ]

    descriptions: list[str] = []
    paths: list[str] = []

    can_annotate = (
        video_path is not None
        and raw_frames is not None
        and output_dir is not None
    )

    n_frames = len(landmarks)
    n_raw = len(raw_frames) if raw_frames else 0

    for phase_name, frame_idx in phases:
        # Clamp to valid range in case segmentation returned a boundary index.
        frame_idx = max(0, min(frame_idx, n_frames - 1))

        desc = _describe_keyframe(landmarks, frame_idx, phase_name)
        descriptions.append(desc)

        saved_path = ""
        if can_annotate and frame_idx < n_raw:
            raw_entry = raw_frames[frame_idx]
            video_frame_num = raw_entry.get("frame", frame_idx)
            img_path = output_dir / f"{phase_name}.jpg"
            if _annotate_and_save_frame(
                video_path, video_frame_num, raw_entry, img_path
            ):
                saved_path = str(img_path)
        paths.append(saved_path)

    return descriptions, paths


# ---------------------------------------------------------------------------
# SQLAlchemy-backed worker (used by FastAPI BackgroundTasks)
# ---------------------------------------------------------------------------


def run_analysis(job_id: int, anthropic_client=None) -> None:
    """Execute the full analysis pipeline for *job_id* and persist the result.

    Opens its own DB session so it can run safely in a background thread
    without sharing the request-scoped session.  The job transitions through
    ``processing → completed | failed``.

    Parameters
    ----------
    job_id:
        Primary key of the :class:`backend.models.Job` row to process.
    anthropic_client:
        Optional pre-built :class:`anthropic.Anthropic` client.  When
        supplied it is used directly for coaching feedback, bypassing the
        environment-variable lookup.  When ``None`` the function falls back
        to constructing a client from ``ANTHROPIC_API_KEY``.

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
    db: Session = _db.SessionLocal()
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
        _extraction_failed = False
        _prep = None          # PreprocessResult (holds chamber/extension/retraction)
        _raw_frames: list | None = None  # raw landmark frames from extract_landmarks
        _video_path: str | None = None
        if job.upload and job.upload.storage_path:
            _video_path = job.upload.storage_path
            try:
                from backend.extractors.landmarks import extract_landmarks  # noqa: PLC0415
                from backend.vision.pipeline import preprocess              # noqa: PLC0415

                raw = extract_landmarks(job.upload.storage_path)
                if raw:
                    lm_array = np.array(
                        [
                            [[lm["x"], lm["y"], lm["z"]] for lm in frame["landmarks"]]
                            for frame in raw
                        ],
                        dtype=float,
                    )
                    _prep = preprocess(lm_array, technique)
                    landmarks = _prep.landmarks
                    _raw_frames = raw
            except Exception:
                logger.warning(
                    "run_analysis: landmark extraction failed for job %d; "
                    "falling back to reference template",
                    job_id,
                )
                _extraction_failed = True

        if landmarks is None:
            landmarks = _load_reference_landmarks(technique)

        # ── scoring ───────────────────────────────────────────────────────────
        rep_score = score_rep(technique, landmarks)

        # ── keyframe extraction ───────────────────────────────────────────────
        # Generate text descriptions and (when a video is available) annotated
        # images for the chamber, extension, and retraction phases identified
        # by the preprocessing pipeline.
        _keyframe_descriptions: list[str] = []
        _keyframe_paths_list: list[str] = []
        if _prep is not None:
            _kf_output_dir: Path | None = None
            if job.job_id:
                _kf_output_dir = UPLOAD_DIR / "keyframes" / str(job.job_id)
            try:
                _keyframe_descriptions, _keyframe_paths_list = _extract_keyframes(
                    landmarks=landmarks,
                    chamber_idx=_prep.chamber,
                    extension_idx=_prep.extension,
                    retraction_idx=_prep.retraction,
                    video_path=_video_path,
                    raw_frames=_raw_frames,
                    output_dir=_kf_output_dir,
                )
                logger.debug(
                    "run_analysis: extracted %d keyframe descriptions for job %d",
                    len(_keyframe_descriptions),
                    job_id,
                )
            except Exception:
                logger.warning(
                    "run_analysis: keyframe extraction failed for job %d; "
                    "continuing without keyframes",
                    job_id,
                    exc_info=True,
                )
        else:
            # No preprocessing result (landmark extraction failed; reference
            # template used).  Derive keyframe indices directly from the
            # reference-template landmarks so the coaching prompt still
            # receives visual grounding, mirroring what process_job does.
            try:
                _ref_chamber, _ref_ext, _ref_ret = _compute_keyframe_indices(landmarks)
                _keyframe_descriptions, _keyframe_paths_list = _extract_keyframes(
                    landmarks=landmarks,
                    chamber_idx=_ref_chamber,
                    extension_idx=_ref_ext,
                    retraction_idx=_ref_ret,
                )
                logger.debug(
                    "run_analysis: extracted %d keyframe descriptions from "
                    "reference template for job %d",
                    len(_keyframe_descriptions),
                    job_id,
                )
            except Exception:
                logger.warning(
                    "run_analysis: reference-template keyframe extraction failed "
                    "for job %d; continuing without keyframes",
                    job_id,
                    exc_info=True,
                )

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

        # ── AnalysisResult (written unconditionally when job has a UUID) ──────
        # The row is created here regardless of whether coaching feedback is
        # available so that the results endpoint never encounters a completed
        # job without a corresponding AnalysisResult row.
        analysis_result: AnalysisResult | None = None
        if job.job_id:
            # Derive the public video URL from the upload's storage path so
            # the results page can stream the original upload back to the user.
            video_url: str | None = None
            if job.upload and job.upload.storage_path:
                video_url = f"/uploads/{Path(job.upload.storage_path).name}"

            analysis_result = AnalysisResult(
                job_id=job.job_id,
                scores=json.dumps(
                    {cr.name: float(cr.score) for cr in rep_score.criteria}
                ),
                metric_deltas=_criteria_json(rep_score),
                keyframe_paths=json.dumps(_keyframe_paths_list),
                overall_score=int(rep_score.overall * 100),
                video_url=video_url,
                created_at=datetime.utcnow(),
            )
            db.add(analysis_result)

        # ── coaching feedback (optional) ──────────────────────────────────────
        # Prefer an injected client; fall back to constructing one from the env.
        _coaching_client = anthropic_client
        if _coaching_client is None:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if api_key:
                import anthropic as _anthropic  # noqa: PLC0415
                _coaching_client = _anthropic.Anthropic(api_key=api_key)

        if _coaching_client is not None and analysis_result is not None:
            try:
                from backend.coaching import generate_feedback  # noqa: PLC0415

                coaching_input = _build_coaching_input(
                    technique, rep_score, _keyframe_descriptions
                )
                feedback = generate_feedback(coaching_input, _coaching_client)
                analysis_result.feedback = feedback
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

        # ── degraded-result warning ───────────────────────────────────────────
        # When landmark extraction failed and scores were derived from the
        # reference template, record a human-readable note so callers can
        # surface the degraded result to the user.
        if _extraction_failed:
            job.error_message = (
                "Landmark extraction failed; scores are based on the reference template"
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

        # Derive keyframe descriptions from the reference-template landmarks so
        # the coaching prompt receives visual grounding even without a video.
        _kf_descriptions: list[str] = []
        try:
            chamber_idx, extension_idx, retraction_idx = _compute_keyframe_indices(
                landmarks
            )
            _kf_descriptions, _ = _extract_keyframes(
                landmarks=landmarks,
                chamber_idx=chamber_idx,
                extension_idx=extension_idx,
                retraction_idx=retraction_idx,
            )
        except Exception:
            logger.warning(
                "process_job: keyframe description extraction failed for job %s; "
                "continuing without keyframes",
                job_id,
                exc_info=True,
            )

        # Build the coaching input and optionally call the Claude API.
        feedback: str | None = None
        if anthropic_client is not None:
            from backend.coaching import generate_feedback  # noqa: PLC0415

            coaching_input = _build_coaching_input(technique, rep_score, _kf_descriptions)
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
