"""FastAPI application entry point for Kime backend."""

from fastapi import FastAPI

from .database import Base, engine
from .routers.results import router as results_router

# Create tables on startup (for development; migrations handle prod).
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Kime API", version="0.1.0")

app.include_router(results_router)
