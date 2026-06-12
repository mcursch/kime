"""Kime backend – FastAPI application entry point."""

from __future__ import annotations

import logging

from fastapi import FastAPI

from backend.routers.analyze import router as analyze_router
from backend.routers.jobs import router as jobs_router
from backend.routers.results import router as results_router
from backend.routers.upload import router as upload_router

logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="Kime",
    description="Martial-arts technique analyser API",
    version="0.1.0",
)

app.include_router(analyze_router)
app.include_router(upload_router)
app.include_router(jobs_router)
app.include_router(results_router)


@app.get("/health")
async def health() -> dict:
    """Liveness check — returns HTTP 200 when the server is up."""
    return {"status": "ok"}
