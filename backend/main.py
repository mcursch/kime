"""Kime backend – FastAPI application entry point."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

import anthropic
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from backend.config import UPLOAD_DIR
from backend.database import init_db
from backend.routers.analyze import router as analyze_router
from backend.routers.jobs import router as jobs_router
from backend.routers.results import router as results_router
from backend.routers.upload import router as upload_router

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler.

    Validates that ``ANTHROPIC_API_KEY`` is present and creates a shared
    :class:`anthropic.Anthropic` client stored in ``app.state`` so that
    background workers can reuse it without constructing a new client on
    every request.

    Raises:
        RuntimeError: When ``ANTHROPIC_API_KEY`` is not set.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY environment variable is not set. "
            "Please set it before starting the server."
        )

    app.state.anthropic_client = anthropic.Anthropic(api_key=api_key)

    # Ensure ORM tables exist in the configured database.
    init_db()

    yield

    # Teardown: nothing to clean up for the Anthropic client.


app = FastAPI(
    title="Kime",
    description="Martial-arts technique analyser API",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(analyze_router)
app.include_router(upload_router)
app.include_router(jobs_router)
app.include_router(results_router)

# Serve uploaded video files at /uploads/<filename>.
# The directory is created on demand by the upload router, so ensure it
# exists before mounting to avoid a StaticFiles startup error.
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")


@app.get("/health")
async def health() -> dict:
    """Liveness check — returns HTTP 200 when the server is up."""
    return {"status": "ok"}
