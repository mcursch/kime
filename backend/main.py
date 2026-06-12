"""Kime backend – FastAPI application entry point."""

from __future__ import annotations

import logging

from fastapi import FastAPI

from backend.routers.analyze import router as analyze_router

logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="Kime",
    description="Martial-arts technique analyser API",
    version="0.1.0",
)

app.include_router(analyze_router)
