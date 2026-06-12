"""
Unit tests for backend/worker.py.

Tests cover:
- feedback is stored in the scores table after a successful job
- the Anthropic client is NOT called when the job transitions to 'failed'
"""

import json
import sqlite3
import tempfile
import os
from unittest.mock import MagicMock, patch

import anthropic
import pytest

from backend.database import init_db, get_connection
from backend.worker import process_job


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
