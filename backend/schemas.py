"""Pydantic response/request schemas for Kime API."""

import json
from datetime import datetime
from typing import Any

from pydantic import BaseModel, field_validator, model_validator

from .models import AnalysisResult, Job, JobStatus


class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    technique: str
    session_id: str
    created_at: datetime

    model_config = {"from_attributes": True}


class AnalysisResultResponse(BaseModel):
    job_id: str
    status: JobStatus
    technique: str
    session_id: str
    scores: dict[str, Any]
    metric_deltas: dict[str, Any]
    keyframe_paths: list[str]
    overall_score: int | None
    created_at: datetime

    model_config = {"from_attributes": True}

    @model_validator(mode="before")
    @classmethod
    def _flatten(cls, data: Any) -> Any:
        """Merge Job + AnalysisResult ORM objects into a single dict."""
        if isinstance(data, Job):
            result = data.result
            if result is None and data.status == JobStatus.completed:
                raise ValueError(
                    f"Data integrity error: completed job {data.job_id!r} has no AnalysisResult row"
                )
            out: dict[str, Any] = {
                "job_id": data.job_id,
                "status": data.status,
                "technique": data.technique,
                "session_id": data.session_id,
                "created_at": result.created_at if result else data.created_at,
                "scores": json.loads(result.scores) if result else {},
                "metric_deltas": json.loads(result.metric_deltas) if result else {},
                "keyframe_paths": json.loads(result.keyframe_paths) if result else [],
                "overall_score": result.overall_score if result else None,
            }
            return out
        return data


class HistoryItem(BaseModel):
    job_id: str
    technique: str
    status: JobStatus
    overall_score: int | None
    created_at: datetime

    model_config = {"from_attributes": True}

    @model_validator(mode="before")
    @classmethod
    def _flatten(cls, data: Any) -> Any:
        if isinstance(data, Job):
            return {
                "job_id": data.job_id,
                "technique": data.technique,
                "status": data.status,
                "overall_score": data.result.overall_score if data.result else None,
                "created_at": data.created_at,
            }
        return data


class HistoryResponse(BaseModel):
    items: list[HistoryItem]
    total: int
    page: int
    page_size: int
