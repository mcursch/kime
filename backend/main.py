"""FastAPI application entry point for Kime."""

from fastapi import FastAPI

app = FastAPI(title="Kime", description="Martial-arts technique analyser", version="0.1.0")


@app.get("/health", tags=["meta"])
def health_check() -> dict:
    """Liveness probe — returns 200 when the app is running."""
    return {"status": "ok"}
