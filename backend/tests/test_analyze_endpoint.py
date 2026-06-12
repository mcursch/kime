"""Tests for the POST/GET /api/analyze endpoints.

The landmark extractor is mocked so these tests have no MediaPipe dependency
and run in CI without a real video file.
"""

from __future__ import annotations

import asyncio
import io
from unittest.mock import patch

import pytest
import pytest_asyncio
from httpx2 import ASGITransport, AsyncClient

from backend.main import app
from backend.routers.analyze import Job, JobStore, _jobs

# ---------------------------------------------------------------------------
# Fake landmark data returned by the mocked extractor
# ---------------------------------------------------------------------------

_FAKE_LANDMARKS = [
    {
        "frame": i,
        "landmarks": [
            {"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 0.99, "presence": 1.0}
            for _ in range(33)
        ],
    }
    for i in range(5)
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clear_job_store():
    """Wipe the in-memory job store between tests."""
    _jobs.clear()
    yield
    _jobs.clear()


@pytest.fixture
def fake_video_bytes() -> bytes:
    """A tiny byte payload used as the 'video' upload."""
    return b"\x00" * 128


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_submit_returns_202_with_job_id(fake_video_bytes: bytes) -> None:
    """POST /api/analyze returns 202 and a job_id."""
    with patch(
        "backend.routers.analyze.extract_landmarks",
        return_value=_FAKE_LANDMARKS,
    ):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/analyze",
                files={"file": ("kick.mp4", io.BytesIO(fake_video_bytes), "video/mp4")},
            )

    assert response.status_code == 202
    body = response.json()
    assert "job_id" in body
    assert isinstance(body["job_id"], str)
    assert len(body["job_id"]) > 0


@pytest.mark.asyncio
async def test_poll_eventually_returns_complete(fake_video_bytes: bytes) -> None:
    """GET /api/analyze/{job_id} returns status=complete with landmarks."""
    with patch(
        "backend.routers.analyze.extract_landmarks",
        return_value=_FAKE_LANDMARKS,
    ):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            submit = await client.post(
                "/api/analyze",
                files={"file": ("kick.mp4", io.BytesIO(fake_video_bytes), "video/mp4")},
            )
            assert submit.status_code == 202
            job_id = submit.json()["job_id"]

            # Poll up to 5 seconds for completion.
            deadline = asyncio.get_event_loop().time() + 5.0
            status = "pending"
            while asyncio.get_event_loop().time() < deadline and status == "pending":
                await asyncio.sleep(0.05)
                poll = await client.get(f"/api/analyze/{job_id}")
                assert poll.status_code == 200
                status = poll.json()["status"]

    assert status == "complete"
    body = poll.json()
    assert "landmarks" in body
    assert isinstance(body["landmarks"], list)
    assert len(body["landmarks"]) == len(_FAKE_LANDMARKS)


@pytest.mark.asyncio
async def test_poll_returns_error_when_extraction_fails(fake_video_bytes: bytes) -> None:
    """GET /api/analyze/{job_id} returns status=error when extraction raises."""
    with patch(
        "backend.routers.analyze.extract_landmarks",
        side_effect=RuntimeError("simulated extraction failure"),
    ):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            submit = await client.post(
                "/api/analyze",
                files={"file": ("kick.mp4", io.BytesIO(fake_video_bytes), "video/mp4")},
            )
            assert submit.status_code == 202
            job_id = submit.json()["job_id"]

            # Poll up to 5 seconds for a terminal state.
            deadline = asyncio.get_event_loop().time() + 5.0
            job_status = "pending"
            while asyncio.get_event_loop().time() < deadline and job_status == "pending":
                await asyncio.sleep(0.05)
                poll = await client.get(f"/api/analyze/{job_id}")
                assert poll.status_code == 200
                job_status = poll.json()["status"]

    assert job_status == "error"
    body = poll.json()
    assert "simulated extraction failure" in (body.get("error") or "")


# ---------------------------------------------------------------------------
# JobStore TTL eviction tests
# ---------------------------------------------------------------------------


def test_jobstore_evict_expired_removes_old_entries() -> None:
    """evict_expired() removes entries whose timestamp exceeds the TTL."""
    store = JobStore(ttl=60.0)
    job = Job(job_id="abc", status="complete")
    store["abc"] = job

    # Manually backdate the timestamp so the entry looks expired.
    store._timestamps["abc"] = store._timestamps["abc"] - 61.0

    removed = store.evict_expired()
    assert removed == 1
    assert store.get("abc") is None
    assert len(store) == 0


def test_jobstore_evict_expired_keeps_recent_entries() -> None:
    """evict_expired() leaves entries that are still within the TTL window."""
    store = JobStore(ttl=60.0)
    job = Job(job_id="xyz", status="pending")
    store["xyz"] = job

    # Entry was just written — well within the TTL.
    removed = store.evict_expired()
    assert removed == 0
    assert store.get("xyz") is not None


def test_jobstore_clear_resets_timestamps() -> None:
    """clear() removes both job data and timestamps so nothing leaks."""
    store = JobStore(ttl=60.0)
    store["j1"] = Job(job_id="j1", status="complete")
    store["j2"] = Job(job_id="j2", status="error")

    store.clear()

    assert len(store) == 0
    # Timestamps dict should also be empty so a subsequent evict is a no-op.
    assert store.evict_expired() == 0


@pytest.mark.asyncio
async def test_unknown_job_id_returns_404() -> None:
    """GET /api/analyze/{job_id} returns 404 for an unknown job_id."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/api/analyze/does-not-exist")

    assert response.status_code == 404
    assert response.json()["detail"] == "Job not found"
