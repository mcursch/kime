"""Smoke tests for the FastAPI app and ORM models."""

import os

import pytest
from fastapi.testclient import TestClient

# Use an in-memory SQLite database so tests never touch the dev DB.
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")


def test_app_importable() -> None:
    """The FastAPI app can be imported without errors."""
    from backend.main import app  # noqa: F401


def test_models_importable() -> None:
    """All four ORM models are importable from backend.models."""
    from backend.models import History, Job, Score, Upload  # noqa: F401


def test_health_check() -> None:
    """GET /health returns 200 with {"status": "ok"}."""
    from backend.main import app

    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
