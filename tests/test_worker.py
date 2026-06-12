"""
Unit tests for backend/worker.py.

Tests cover:
- feedback is stored in the scores table after a successful job
- the Anthropic client is NOT called when the job transitions to 'failed'
- keyframe descriptions are extracted and included in the coaching prompt
"""

import json
import sqlite3
import tempfile
import os
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import anthropic
import numpy as np
import pytest

from backend.database import init_db, get_connection
from backend.worker import (
    _build_coaching_input,
    _compute_keyframe_indices,
    _describe_keyframe,
    _extract_keyframes,
    process_job,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_client(feedback_text: str = "Your hip rotation needs work.") -> MagicMock:
    content_block = MagicMock()
    content_block.text = feedback_text

    message = MagicMock()
    message.content = [content_block]

    client = MagicMock(spec=anthropic.Anthropic)
    client.messages.create.return_value = message
    return client


def _job_row(conn: sqlite3.Connection, job_id: str) -> sqlite3.Row:
    return conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()


def _scores_row(conn: sqlite3.Connection, job_id: str) -> sqlite3.Row:
    return conn.execute(
        "SELECT * FROM scores WHERE job_id=? ORDER BY id DESC LIMIT 1", (job_id,)
    ).fetchone()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_db(tmp_path):
    """Yield a path to a fresh, initialised SQLite database."""
    db_path = str(tmp_path / "test_kime.db")
    init_db(db_path)
    return db_path


@pytest.fixture
def seeded_job(tmp_db):
    """Insert a pending job and return (db_path, job_id)."""
    job_id = "test-job-1111-2222-3333"
    conn = get_connection(tmp_db)
    conn.execute(
        "INSERT INTO jobs (id, status, technique) VALUES (?, 'pending', ?)",
        (job_id, "roundhouse_kick"),
    )
    conn.commit()
    conn.close()
    return tmp_db, job_id


# ---------------------------------------------------------------------------
# Tests: happy path
# ---------------------------------------------------------------------------


class TestProcessJobSuccess:
    def test_job_status_is_complete(self, seeded_job):
        db_path, job_id = seeded_job
        client = _make_mock_client()
        process_job(job_id, "roundhouse_kick", db_path, client)

        conn = get_connection(db_path)
        row = _job_row(conn, job_id)
        conn.close()
        assert row["status"] == "complete"

    def test_scores_row_is_created(self, seeded_job):
        db_path, job_id = seeded_job
        client = _make_mock_client()
        process_job(job_id, "roundhouse_kick", db_path, client)

        conn = get_connection(db_path)
        row = _scores_row(conn, job_id)
        conn.close()
        assert row is not None

    def test_feedback_is_stored(self, seeded_job):
        db_path, job_id = seeded_job
        feedback_text = "Your hip rotation is 40° short of the reference."
        client = _make_mock_client(feedback_text)
        process_job(job_id, "roundhouse_kick", db_path, client)

        conn = get_connection(db_path)
        row = _scores_row(conn, job_id)
        conn.close()
        assert row["feedback"] == feedback_text

    def test_criteria_json_stored(self, seeded_job):
        db_path, job_id = seeded_job
        client = _make_mock_client()
        process_job(job_id, "roundhouse_kick", db_path, client)

        conn = get_connection(db_path)
        row = _scores_row(conn, job_id)
        conn.close()
        criteria = json.loads(row["criteria"])
        assert "hip_rotation_deg" in criteria

    def test_overall_score_stored(self, seeded_job):
        db_path, job_id = seeded_job
        client = _make_mock_client()
        process_job(job_id, "roundhouse_kick", db_path, client)

        conn = get_connection(db_path)
        row = _scores_row(conn, job_id)
        conn.close()
        assert row["overall_score"] is not None
        assert row["overall_score"] > 0


# ---------------------------------------------------------------------------
# Tests: no API call on failure
# ---------------------------------------------------------------------------


class TestProcessJobFailure:
    def test_job_marked_failed_on_scoring_error(self, seeded_job):
        """An empty technique name triggers the scoring stub to raise; job → failed."""
        db_path, job_id = seeded_job
        client = _make_mock_client()

        # Override technique with empty string to trigger stub failure path.
        conn = get_connection(db_path)
        conn.execute("UPDATE jobs SET technique='' WHERE id=?", (job_id,))
        conn.commit()
        conn.close()

        process_job(job_id, "", db_path, client)

        conn = get_connection(db_path)
        row = _job_row(conn, job_id)
        conn.close()
        assert row["status"] == "failed"

    def test_no_api_call_when_job_fails(self, seeded_job):
        """Acceptance criterion: the Anthropic client must NOT be called when
        the job transitions to 'failed'."""
        db_path, job_id = seeded_job
        client = _make_mock_client()

        process_job(job_id, "", db_path, client)  # empty technique → fail

        client.messages.create.assert_not_called()

    def test_no_scores_row_when_job_fails(self, seeded_job):
        """No partial scores row should be written when scoring fails."""
        db_path, job_id = seeded_job
        client = _make_mock_client()

        process_job(job_id, "", db_path, client)

        conn = get_connection(db_path)
        row = _scores_row(conn, job_id)
        conn.close()
        assert row is None


# ---------------------------------------------------------------------------
# Tests: no client supplied
# ---------------------------------------------------------------------------


class TestProcessJobNoClient:
    def test_job_complete_without_feedback(self, seeded_job):
        """With client=None the job should still complete; feedback stays NULL."""
        db_path, job_id = seeded_job
        process_job(job_id, "roundhouse_kick", db_path, anthropic_client=None)

        conn = get_connection(db_path)
        job_row = _job_row(conn, job_id)
        scores_row = _scores_row(conn, job_id)
        conn.close()

        assert job_row["status"] == "complete"
        assert scores_row is not None
        assert scores_row["feedback"] is None


# ---------------------------------------------------------------------------
# Tests: keyframe extraction helpers
# ---------------------------------------------------------------------------


def _make_synthetic_landmarks(n_frames: int = 30) -> np.ndarray:
    """Return a plausible (n_frames, 33, 3) normalised landmark sequence.

    The sequence simulates a basic kick: the right ankle rises to peak
    displacement at the midpoint and returns by the final frame.
    """
    rng = np.random.default_rng(42)
    lm = rng.uniform(-0.1, 0.1, (n_frames, 33, 3))

    # Right ankle (index 28): sweep up then back down
    t = np.linspace(0, np.pi, n_frames)
    lm[:, 28, 1] = np.sin(t)       # y rises to 1 then falls
    lm[:, 28, 0] = np.cos(t) * 0.3 # x swings outward

    return lm


class TestDescribeKeyframe:
    """Unit tests for the _describe_keyframe helper."""

    def test_chamber_contains_knee_height(self):
        lm = _make_synthetic_landmarks()
        desc = _describe_keyframe(lm, frame_idx=3, phase="chamber")
        assert "Chamber" in desc
        assert "torso-lengths" in desc

    def test_extension_contains_hip_yaw(self):
        lm = _make_synthetic_landmarks()
        desc = _describe_keyframe(lm, frame_idx=15, phase="extension")
        assert "Extension" in desc
        assert "°" in desc

    def test_retraction_contains_knee_height(self):
        lm = _make_synthetic_landmarks()
        desc = _describe_keyframe(lm, frame_idx=25, phase="retraction")
        assert "Retraction" in desc
        assert "torso-lengths" in desc

    def test_returns_string(self):
        lm = _make_synthetic_landmarks()
        for phase in ("chamber", "extension", "retraction"):
            assert isinstance(_describe_keyframe(lm, 10, phase), str)


class TestExtractKeyframes:
    """Unit tests for _extract_keyframes — text-only mode (no video)."""

    def test_returns_three_descriptions(self):
        lm = _make_synthetic_landmarks()
        descs, paths = _extract_keyframes(lm, 3, 15, 25)
        assert len(descs) == 3

    def test_returns_three_paths(self):
        lm = _make_synthetic_landmarks()
        descs, paths = _extract_keyframes(lm, 3, 15, 25)
        assert len(paths) == 3

    def test_paths_empty_without_video(self):
        lm = _make_synthetic_landmarks()
        _, paths = _extract_keyframes(lm, 3, 15, 25)
        assert all(p == "" for p in paths)

    def test_descriptions_are_non_empty_strings(self):
        lm = _make_synthetic_landmarks()
        descs, _ = _extract_keyframes(lm, 3, 15, 25)
        for desc in descs:
            assert isinstance(desc, str) and len(desc) > 0

    def test_clamping_out_of_range_indices(self):
        """Out-of-range indices must not raise; they are clamped to valid range."""
        lm = _make_synthetic_landmarks(n_frames=10)
        descs, paths = _extract_keyframes(lm, -5, 50, 100)
        assert len(descs) == 3
        assert len(paths) == 3


class TestBuildCoachingInput:
    """_build_coaching_input must propagate keyframe_descriptions correctly."""

    def _make_rep_score(self):
        """Return a minimal RepScore-like object."""
        from backend.scoring.engine import score_rep
        from backend.scoring.dtw_aligner import load_reference_template

        ref_flat = load_reference_template("roundhouse_kick")
        lm = ref_flat.reshape(ref_flat.shape[0], 33, 3)
        return score_rep("roundhouse_kick", lm)

    def test_empty_when_not_supplied(self):
        rep_score = self._make_rep_score()
        result = _build_coaching_input("roundhouse_kick", rep_score)
        assert result["keyframe_descriptions"] == []

    def test_empty_when_none(self):
        rep_score = self._make_rep_score()
        result = _build_coaching_input("roundhouse_kick", rep_score, None)
        assert result["keyframe_descriptions"] == []

    def test_descriptions_propagated(self):
        rep_score = self._make_rep_score()
        descs = ["Chamber desc", "Extension desc", "Retraction desc"]
        result = _build_coaching_input("roundhouse_kick", rep_score, descs)
        assert result["keyframe_descriptions"] == descs


class TestProcessJobKeyframesInPrompt:
    """Integration test: keyframe descriptions must appear in the coaching prompt."""

    def test_keyframe_descriptions_sent_to_claude(self, seeded_job):
        """process_job must include non-empty keyframe_descriptions in the prompt."""
        db_path, job_id = seeded_job
        client = _make_mock_client()

        process_job(job_id, "roundhouse_kick", db_path, client)

        assert client.messages.create.call_count == 1
        call_kwargs = client.messages.create.call_args.kwargs
        messages = call_kwargs.get("messages", [])
        prompt_text = "".join(
            m["content"] for m in messages
            if isinstance(m, dict) and m.get("role") == "user"
        )
        # The prompt must contain the "Annotated keyframes:" header that
        # coaching.build_prompt adds only when keyframe_descriptions is non-empty.
        assert "Annotated keyframes:" in prompt_text, (
            "coaching prompt is missing 'Annotated keyframes:' section — "
            "keyframe_descriptions was likely still empty\n"
            f"Prompt:\n{prompt_text}"
        )
