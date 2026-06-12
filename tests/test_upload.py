"""Tests for POST /upload and GET /jobs/{job_id}."""

import io
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Point at an in-memory DB before importing any backend module so that
# database.py picks up the URL when it evaluates at import time.
os.environ["DATABASE_URL"] = "sqlite:///:memory:"

from backend.database import Base, SessionLocal, engine, get_db  # noqa: E402
from backend.main import app  # noqa: E402
from backend.models import Job, JobStatus, Score  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_db() -> None:
    """Create all tables before each test and drop them after.

    We reuse ``backend.database.engine`` (the in-memory DB the app was wired
    with at import time) so the app code, the dependency-override sessions,
    and the background worker all see the same schema.
    """
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture()
def tmp_upload_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Override UPLOAD_DIR to use a temp directory for each test."""
    monkeypatch.setattr("backend.routers.upload.UPLOAD_DIR", tmp_path)
    return tmp_path


@pytest.fixture()
def client(tmp_upload_dir: Path) -> TestClient:
    """Test client wired to the in-memory DB and temp upload dir."""
    # No get_db override needed: the app and the worker both use
    # backend.database.SessionLocal which is bound to the in-memory engine.
    return TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mp4_bytes() -> bytes:
    """Minimal payload with a video/mp4 MIME type (not a real decode target)."""
    return b"\x00\x00\x00\x18ftypisom" + b"\x00" * 100


# ---------------------------------------------------------------------------
# POST /upload
# ---------------------------------------------------------------------------


class TestUploadEndpoint:
    def test_valid_mp4_returns_202(self, client: TestClient) -> None:
        response = client.post(
            "/upload",
            files={"file": ("kick.mp4", io.BytesIO(_mp4_bytes()), "video/mp4")},
            data={"technique": "front_kick"},
        )
        assert response.status_code == 202

    def test_response_has_job_id_and_status(self, client: TestClient) -> None:
        response = client.post(
            "/upload",
            files={"file": ("kick.mp4", io.BytesIO(_mp4_bytes()), "video/mp4")},
            data={"technique": "front_kick"},
        )
        body = response.json()
        assert "job_id" in body
        assert isinstance(body["job_id"], int)
        assert body["status"] == "pending"

    def test_file_is_saved_to_upload_dir(
        self, client: TestClient, tmp_upload_dir: Path
    ) -> None:
        client.post(
            "/upload",
            files={"file": ("kick.mp4", io.BytesIO(_mp4_bytes()), "video/mp4")},
            data={"technique": "front_kick"},
        )
        saved = list(tmp_upload_dir.iterdir())
        assert len(saved) == 1
        assert saved[0].read_bytes() == _mp4_bytes()

    def test_non_video_mime_returns_422(self, client: TestClient) -> None:
        response = client.post(
            "/upload",
            files={"file": ("photo.jpg", io.BytesIO(b"fake-image"), "image/jpeg")},
            data={"technique": "front_kick"},
        )
        assert response.status_code == 422

    def test_text_mime_returns_422(self, client: TestClient) -> None:
        response = client.post(
            "/upload",
            files={"file": ("notes.txt", io.BytesIO(b"hello"), "text/plain")},
            data={"technique": "front_kick"},
        )
        assert response.status_code == 422

    def test_job_row_created_with_pending_status_immediately(
        self, client: TestClient
    ) -> None:
        """The job row must exist with status=pending right after the response.

        Because TestClient runs BackgroundTasks synchronously the status will
        already have advanced; we verify the row exists and has a valid status.
        """
        response = client.post(
            "/upload",
            files={"file": ("kick.mp4", io.BytesIO(_mp4_bytes()), "video/mp4")},
            data={"technique": "front_kick"},
        )
        job_id = response.json()["job_id"]
        db = SessionLocal()
        try:
            job = db.get(Job, job_id)
            assert job is not None
            assert job.status in (
                JobStatus.pending,
                JobStatus.processing,
                JobStatus.completed,
                JobStatus.failed,
            )
        finally:
            db.close()

    def test_default_technique_accepted(self, client: TestClient) -> None:
        """Omitting the technique form field should use the default."""
        response = client.post(
            "/upload",
            files={"file": ("kick.mp4", io.BytesIO(_mp4_bytes()), "video/mp4")},
        )
        assert response.status_code == 202

    def test_other_video_types_accepted(self, client: TestClient) -> None:
        for mime in ("video/quicktime", "video/x-msvideo", "video/webm"):
            response = client.post(
                "/upload",
                files={"file": ("clip.mov", io.BytesIO(_mp4_bytes()), mime)},
                data={"technique": "front_kick"},
            )
            assert response.status_code == 202, f"Expected 202 for {mime}"


# ---------------------------------------------------------------------------
# GET /jobs/{job_id}
# ---------------------------------------------------------------------------


class TestJobsEndpoint:
    def _upload(self, client: TestClient) -> int:
        response = client.post(
            "/upload",
            files={"file": ("kick.mp4", io.BytesIO(_mp4_bytes()), "video/mp4")},
            data={"technique": "front_kick"},
        )
        assert response.status_code == 202
        return response.json()["job_id"]

    def test_returns_job_status(self, client: TestClient) -> None:
        job_id = self._upload(client)
        response = client.get(f"/jobs/{job_id}")
        assert response.status_code == 200
        body = response.json()
        assert body["job_id"] == job_id
        assert body["status"] in {"pending", "processing", "completed", "failed"}

    def test_unknown_job_returns_404(self, client: TestClient) -> None:
        response = client.get("/jobs/999999")
        assert response.status_code == 404

    def test_background_task_completes_job(self, client: TestClient) -> None:
        """With TestClient background tasks run synchronously, so the job
        should be 'completed' by the time the status endpoint is polled."""
        job_id = self._upload(client)
        response = client.get(f"/jobs/{job_id}")
        assert response.status_code == 200
        assert response.json()["status"] == "completed"


# ---------------------------------------------------------------------------
# Worker – stub score rows
# ---------------------------------------------------------------------------


class TestWorker:
    def test_stub_scores_written_on_completion(self, client: TestClient) -> None:
        response = client.post(
            "/upload",
            files={"file": ("kick.mp4", io.BytesIO(_mp4_bytes()), "video/mp4")},
            data={"technique": "front_kick"},
        )
        job_id = response.json()["job_id"]
        # TestClient runs background tasks synchronously before returning.
        db = SessionLocal()
        try:
            scores = db.query(Score).filter(Score.job_id == job_id).all()
            assert len(scores) > 0
        finally:
            db.close()
