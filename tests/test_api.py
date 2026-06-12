"""
Integration tests for the FastAPI endpoints.

Verifies:
- POST /jobs/upload creates a job and returns a job_id.
- GET /jobs/{job_id}/results includes a non-empty ``feedback`` string once
  the job is complete.
- The app raises a clear startup error when ANTHROPIC_API_KEY is absent.

These tests replace the real Anthropic client with a mock so they run
offline.
"""

import os
import time
import sqlite3
import tempfile
from unittest.mock import MagicMock, patch

import anthropic
import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_client(feedback_text: str = "Your hip rotation is 40° short.") -> MagicMock:
    content_block = MagicMock()
    content_block.text = feedback_text

    message = MagicMock()
    message.content = [content_block]

    client = MagicMock(spec=anthropic.Anthropic)
    client.messages.create.return_value = message
    return client


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def app_with_client(tmp_path, monkeypatch):
    """
    Yield a TestClient whose lifespan has been patched so it uses:
    - an in-memory-equivalent SQLite DB in tmp_path
    - a mock Anthropic client
    - a fake ANTHROPIC_API_KEY
    """
    db_path = str(tmp_path / "test.db")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key")
    monkeypatch.setenv("KIME_DB_PATH", db_path)

    mock_client = _make_mock_client()

    # Patch the Anthropic constructor so lifespan uses our mock.
    with patch("backend.main.anthropic.Anthropic", return_value=mock_client):
        # Re-import after env vars are set so DB_PATH is picked up.
        import importlib
        import backend.database as db_module
        import backend.main as main_module

        importlib.reload(db_module)
        importlib.reload(main_module)

        from backend.main import app

        with TestClient(app, raise_server_exceptions=True) as client:
            yield client, mock_client, db_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestUploadEndpoint:
    def test_returns_202_and_job_id(self, app_with_client):
        client, _, _ = app_with_client
        resp = client.post("/jobs/upload?technique=roundhouse_kick")
        assert resp.status_code == 202
        body = resp.json()
        assert "job_id" in body
        assert body["status"] == "pending"


class TestResultsEndpoint:
    def test_returns_404_for_unknown_job(self, app_with_client):
        client, _, _ = app_with_client
        resp = client.get("/jobs/does-not-exist/results")
        assert resp.status_code == 404

    def test_complete_job_has_non_empty_feedback(self, app_with_client):
        """Acceptance criterion: GET /jobs/{id}/results includes non-empty feedback."""
        test_client, mock_anthropic_client, db_path = app_with_client
        feedback_text = "Your hip rotation is 40° short of the reference."
        content_block = MagicMock()
        content_block.text = feedback_text
        msg = MagicMock()
        msg.content = [content_block]
        mock_anthropic_client.messages.create.return_value = msg

        # Submit job.
        resp = test_client.post("/jobs/upload?technique=roundhouse_kick")
        assert resp.status_code == 202
        job_id = resp.json()["job_id"]

        # The TestClient runs background tasks synchronously before responding,
        # so the job should already be complete after the upload call.
        result_resp = test_client.get(f"/jobs/{job_id}/results")
        assert result_resp.status_code == 200
        body = result_resp.json()
        assert body["status"] == "complete"
        assert body["feedback"] is not None
        assert len(body["feedback"]) > 0

    def test_complete_job_has_criteria(self, app_with_client):
        test_client, _, _ = app_with_client
        resp = test_client.post("/jobs/upload?technique=roundhouse_kick")
        job_id = resp.json()["job_id"]

        result_resp = test_client.get(f"/jobs/{job_id}/results")
        body = result_resp.json()
        assert body["criteria"] is not None
        assert "hip_rotation_deg" in body["criteria"]


class TestStartupValidation:
    def test_raises_on_missing_api_key(self, tmp_path, monkeypatch):
        """App must raise RuntimeError at startup when ANTHROPIC_API_KEY is absent."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setenv("KIME_DB_PATH", str(tmp_path / "test.db"))

        import importlib
        import backend.database as db_module
        import backend.main as main_module

        importlib.reload(db_module)
        importlib.reload(main_module)

        from backend.main import app

        with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
            with TestClient(app):
                pass
