"""FastAPI application entry point for Kime."""

from fastapi import FastAPI

from backend.database import init_db
from backend.routers import jobs, upload

app = FastAPI(title="Kime", description="Martial-arts technique analyser", version="0.1.0")

# Ensure tables exist on startup (idempotent; production uses Alembic instead).
init_db()

app.include_router(upload.router)
app.include_router(jobs.router)


@app.get("/health", tags=["meta"])
def health_check() -> dict:
    """Liveness probe — returns 200 when the app is running."""
    return {"status": "ok"}
