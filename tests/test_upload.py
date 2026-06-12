"""Tests for POST /upload and GET /jobs/{job_id}."""

import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# DATABASE_URL=sqlite:///:memory: is set in the root conftest.py before any
# backend module is imported, so backend.database.engine is already bound to
# the in-memory DB by the time this module is collected.
from backend.database import Base, SessionLocal, engine, get_db  # noqa: E402
from backend.main import app  # noqa: E402
from backend.models import Job, JobStatus, Score  # noqa: E402


# ---------------------------------------------------------------------------
# Regression test: LIN-172
# ---------------------------------------------------------------------------


def test_database_engine_is_in_memory() -> None:
    """backend.database.engine must be bound to an in-memory SQLite DB.

    This is a regression test for LIN-172: previously ``tests/test_upload.py``
    set ``os.environ["DATABASE_URL"]`` at module level to point at
    ``sqlite:///:memory:``, but ``backend/tests/test_analyze_endpoint.py`` is
    collected first (``backend/tests`` precedes ``tests`` in pytest.ini
    testpaths), causing ``backend.database`` to be imported with whatever URL
    was in the environment at *that* point — typically the real file-based
    ``kime.db``.

    The fix is to set ``DATABASE_URL`` in the root ``conftest.py`` *before*
    any test module is imported, so ``backend.database.engine`` is always
    wired to the in-memory DB during a test run.
    """
    assert "memory" in str(engine.url), (
        f"Expected in-memory SQLite DB; got engine URL: {engine.url!r}.\n"
        "This means DATABASE_URL was not set before backend.database was "
        "first imported.  Check that conftest.py sets DATABASE_URL via "
        "os.environ.setdefault before any backend module is imported."
    )


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

    def test_spoofed_content_type_rejected_by_magic_bytes(
        self, client: TestClient
    ) -> None:
        """A non-video payload sent with Content-Type: video/mp4 must be rejected.

        This validates that the server inspects actual file magic bytes rather
        than trusting the client-supplied MIME type.
        """
        # Shell-script-like bytes with a spoofed video/mp4 Content-Type header.
        shell_script = b"#!/bin/bash\nrm -rf /\n"
        response = client.post(
            "/upload",
            files={"file": ("exploit.mp4", io.BytesIO(shell_script), "video/mp4")},
            data={"technique": "front_kick"},
        )
        assert response.status_code == 422
        detail = response.json().get("detail", "")
        assert "video" in detail.lower()

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

    def test_path_traversal_filename_is_contained_in_upload_dir(
        self, client: TestClient, tmp_upload_dir: Path
    ) -> None:
        """A filename containing path separators must not escape the upload dir.

        A malicious client could supply a filename like ``x/../../evil.sh`` to
        write a file outside UPLOAD_DIR.  The server must strip directory
        components so the saved file stays inside the configured upload
        directory.
        """
        response = client.post(
            "/upload",
            files={
                "file": (
                    "x/../../evil.sh",
                    io.BytesIO(_mp4_bytes()),
                    "video/mp4",
                )
            },
            data={"technique": "front_kick"},
        )
        assert response.status_code == 202
        saved = list(tmp_upload_dir.iterdir())
        assert len(saved) == 1, "Expected exactly one file saved"
        # The file must reside directly inside the upload dir, not somewhere
        # that has escaped via path traversal.
        saved_path = saved[0].resolve()
        assert saved_path.parent == tmp_upload_dir.resolve(), (
            f"File was written outside the upload directory: {saved_path}"
        )

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
