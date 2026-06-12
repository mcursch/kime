"""
Tests for GET /jobs/{job_id}/results and GET /history.

Uses an in-memory SQLite database so no file I/O or external dependencies
are needed.
"""

import json
import uuid
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.database import Base, get_db
from backend.main import app
from backend.models import AnalysisResult, Job, JobStatus

# ---------------------------------------------------------------------------
# In-memory SQLite engine shared across the test session.
# StaticPool ensures all sessions reuse the same underlying connection so
# in-memory tables created in setup are visible to the request handlers.
# ---------------------------------------------------------------------------

TEST_DATABASE_URL = "sqlite:///:memory:"

_engine = create_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
_TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)


def override_get_db():
    db = _TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture(autouse=True)
def setup_database():
    """Create all tables before each test and drop them after."""
    Base.metadata.create_all(bind=_engine)
    app.dependency_overrides[get_db] = override_get_db
    yield
    app.dependency_overrides.clear()
    Base.metadata.drop_all(bind=_engine)


@pytest.fixture()
def client():
    return TestClient(app)


@pytest.fixture()
def db():
    session = _TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_job(db, status: JobStatus, session_id: str = "sess-abc", technique: str = "front_kick") -> Job:
    job = Job(
        job_id=str(uuid.uuid4()),
        session_id=session_id,
        technique=technique,
        status=status,
        created_at=datetime.now(timezone.utc),
    )
    db.add(job)
    db.flush()
    return job


def attach_result(db, job: Job, video_url: str | None = None) -> AnalysisResult:
    result = AnalysisResult(
        job_id=job.job_id,
        scores=json.dumps({"chamber_height": 85, "hip_rotation": 72, "extension_angle": 90}),
        metric_deltas=json.dumps({"hip_rotation_delta": -18.3}),
        keyframe_paths=json.dumps(["frames/impact_001.jpg"]),
        overall_score=82,
        video_url=video_url,
        created_at=datetime.now(timezone.utc),
    )
    db.add(result)
    db.commit()
    db.refresh(job)
    return result


# ---------------------------------------------------------------------------
# GET /jobs/{job_id}/results
# ---------------------------------------------------------------------------

class TestGetJobResults:
    def test_completed_job_returns_200(self, client, db):
        job = make_job(db, JobStatus.completed)
        attach_result(db, job)

        resp = client.get(f"/jobs/{job.job_id}/results")

        assert resp.status_code == 200
        body = resp.json()
        assert body["job_id"] == job.job_id
        assert "scores" in body
        assert "created_at" in body
        assert body["scores"]["chamber_height"] == 85
        assert body["overall_score"] == 82
        assert body["keyframe_paths"] == ["frames/impact_001.jpg"]

    def test_pending_job_returns_202(self, client, db):
        job = make_job(db, JobStatus.pending)
        db.commit()

        resp = client.get(f"/jobs/{job.job_id}/results")

        assert resp.status_code == 202

    def test_processing_job_returns_202(self, client, db):
        job = make_job(db, JobStatus.processing)
        db.commit()

        resp = client.get(f"/jobs/{job.job_id}/results")

        assert resp.status_code == 202

    def test_failed_job_returns_422(self, client, db):
        job = make_job(db, JobStatus.failed)
        db.commit()

        resp = client.get(f"/jobs/{job.job_id}/results")

        assert resp.status_code == 422
        assert resp.json()["detail"] == "Job failed"

    def test_missing_job_returns_404(self, client):
        resp = client.get(f"/jobs/{uuid.uuid4()}/results")

        assert resp.status_code == 404
        assert resp.json()["detail"] == "Job not found"

    def test_completed_job_without_result_row_returns_500(self, db):
        """A completed job with no AnalysisResult row is a data-integrity error."""
        job = make_job(db, JobStatus.completed)
        db.commit()

        # raise_server_exceptions=False so we receive the 500 HTTP response
        # rather than having the test client re-raise the ValidationError.
        no_raise_client = TestClient(app, raise_server_exceptions=False)
        resp = no_raise_client.get(f"/jobs/{job.job_id}/results")

        assert resp.status_code == 500

    def test_video_url_is_returned_when_set(self, client, db):
        """video_url is included in the response when the AnalysisResult has one."""
        job = make_job(db, JobStatus.completed)
        attach_result(db, job, video_url="/uploads/abc123_kick.mp4")

        resp = client.get(f"/jobs/{job.job_id}/results")

        assert resp.status_code == 200
        body = resp.json()
        assert body["video_url"] == "/uploads/abc123_kick.mp4"

    def test_video_url_is_none_when_not_set(self, client, db):
        """video_url is null in the response when the AnalysisResult has no video."""
        job = make_job(db, JobStatus.completed)
        attach_result(db, job, video_url=None)

        resp = client.get(f"/jobs/{job.job_id}/results")

        assert resp.status_code == 200
        body = resp.json()
        assert body["video_url"] is None


# ---------------------------------------------------------------------------
# GET /history
# ---------------------------------------------------------------------------

class TestGetHistory:
    def test_returns_list_and_total(self, client, db):
        session_id = "user-session-xyz"
        for technique in ("front_kick", "roundhouse_kick", "straight_punch"):
            job = make_job(db, JobStatus.completed, session_id=session_id, technique=technique)
            attach_result(db, job)

        resp = client.get("/history", params={"session_id": session_id})

        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 3
        assert len(body["items"]) == 3

    def test_pagination(self, client, db):
        session_id = "paging-session"
        for _ in range(5):
            job = make_job(db, JobStatus.completed, session_id=session_id)
            attach_result(db, job)

        resp = client.get("/history", params={"session_id": session_id, "page": 2, "page_size": 2})

        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 5
        assert len(body["items"]) == 2
        assert body["page"] == 2
        assert body["page_size"] == 2

    def test_empty_history_for_unknown_session(self, client):
        resp = client.get("/history", params={"session_id": "no-such-session"})

        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 0
        assert body["items"] == []

    def test_isolates_by_session_id(self, client, db):
        job_a = make_job(db, JobStatus.completed, session_id="session-A")
        attach_result(db, job_a)
        job_b = make_job(db, JobStatus.completed, session_id="session-B")
        attach_result(db, job_b)

        resp = client.get("/history", params={"session_id": "session-A"})

        body = resp.json()
        assert body["total"] == 1
        assert body["items"][0]["job_id"] == job_a.job_id

    def test_session_id_required(self, client):
        resp = client.get("/history")

        assert resp.status_code == 422  # missing required query param

    def test_history_item_fields(self, client, db):
        session_id = "fields-session"
        job = make_job(db, JobStatus.completed, session_id=session_id, technique="straight_punch")
        attach_result(db, job)

        resp = client.get("/history", params={"session_id": session_id})

        item = resp.json()["items"][0]
        assert "job_id" in item
        assert "technique" in item
        assert "status" in item
        assert "overall_score" in item
        assert "created_at" in item
