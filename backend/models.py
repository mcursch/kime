"""SQLAlchemy ORM models for Kime.

Tables
------
uploads   – a video file submitted by the user for analysis.
jobs      – an async analysis job tied to an upload.
scores    – per-criterion scores produced by a completed job.
history   – aggregate score snapshots used for the progress-over-time view.
"""

import enum
from datetime import datetime

from sqlalchemy import (
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.database import Base


class TechniqueType(str, enum.Enum):
    front_kick = "front_kick"
    roundhouse_kick = "roundhouse_kick"
    straight_punch = "straight_punch"


class JobStatus(str, enum.Enum):
    pending = "pending"
    processing = "processing"
    completed = "completed"
    failed = "failed"


class Upload(Base):
    """A video file submitted by the user."""

    __tablename__ = "uploads"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    storage_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    technique: Mapped[TechniqueType] = mapped_column(
        Enum(TechniqueType), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )

    jobs: Mapped[list["Job"]] = relationship("Job", back_populates="upload")


class Job(Base):
    """An async analysis job tied to an upload."""

    __tablename__ = "jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    # UUID string assigned at creation time; used by the results/history API.
    job_id: Mapped[str | None] = mapped_column(
        String(36), unique=True, index=True, nullable=True
    )
    # upload_id is nullable so that jobs created outside the upload flow
    # (e.g. in tests for the results router) don't require an Upload row.
    upload_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("uploads.id"), nullable=True, index=True
    )
    session_id: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    technique: Mapped[str | None] = mapped_column(String(64), nullable=True)
    status: Mapped[JobStatus] = mapped_column(
        Enum(JobStatus), default=JobStatus.pending, nullable=False
    )
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    upload: Mapped["Upload | None"] = relationship("Upload", back_populates="jobs")
    scores: Mapped[list["Score"]] = relationship("Score", back_populates="job")
    result: Mapped["AnalysisResult | None"] = relationship(
        "AnalysisResult", back_populates="job", uselist=False
    )


class Score(Base):
    """A per-criterion score produced by a completed job."""

    __tablename__ = "scores"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    job_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("jobs.id"), nullable=False, index=True
    )
    criterion: Mapped[str] = mapped_column(String(128), nullable=False)
    value: Mapped[float] = mapped_column(Float, nullable=False)
    # Raw numeric delta from the reference template (e.g. hip-rotation degrees)
    reference_delta: Mapped[float | None] = mapped_column(Float, nullable=True)
    # Short machine-readable label so the frontend can group/display criteria
    label: Mapped[str | None] = mapped_column(String(256), nullable=True)

    job: Mapped["Job"] = relationship("Job", back_populates="scores")


class AnalysisResult(Base):
    """Detailed analysis result produced by a completed job."""

    __tablename__ = "analysis_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    # FK to jobs.job_id (the UUID string column) rather than jobs.id so that
    # the results API can resolve results using the same string job_id.
    job_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("jobs.job_id"), nullable=False, unique=True, index=True
    )
    scores: Mapped[str] = mapped_column(Text, nullable=False)         # JSON blob
    metric_deltas: Mapped[str] = mapped_column(Text, nullable=False)  # JSON blob
    keyframe_paths: Mapped[str] = mapped_column(Text, nullable=False) # JSON blob
    overall_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)

    job: Mapped["Job"] = relationship("Job", back_populates="result")


class History(Base):
    """Aggregate score snapshot for the progress-over-time view."""

    __tablename__ = "history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    upload_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("uploads.id"), nullable=False, index=True
    )
    technique: Mapped[TechniqueType] = mapped_column(
        Enum(TechniqueType), nullable=False
    )
    overall_score: Mapped[float] = mapped_column(Float, nullable=False)
    recorded_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )
