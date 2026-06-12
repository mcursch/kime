"""SQLAlchemy ORM models for Kime."""

import enum
import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, Enum, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .database import Base


class JobStatus(str, enum.Enum):
    pending = "pending"
    processing = "processing"
    completed = "completed"
    failed = "failed"


class Job(Base):
    """Represents an async analysis job submitted for a video upload."""

    __tablename__ = "jobs"

    job_id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    session_id: Mapped[str] = mapped_column(String(256), nullable=False, index=True)
    technique: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[JobStatus] = mapped_column(
        Enum(JobStatus), nullable=False, default=JobStatus.pending
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    # One job has at most one result row once completed.
    result: Mapped["AnalysisResult | None"] = relationship(
        "AnalysisResult", back_populates="job", uselist=False
    )


class AnalysisResult(Base):
    """Stores scoring output for a completed analysis job."""

    __tablename__ = "analysis_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("jobs.job_id"), nullable=False, unique=True, index=True
    )

    # JSON-serialised dicts stored as TEXT (SQLite has no native JSON column;
    # in production Postgres we would use JSON/JSONB).
    scores: Mapped[str] = mapped_column(Text, nullable=False)
    metric_deltas: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    keyframe_paths: Mapped[str] = mapped_column(Text, nullable=False, default="[]")

    overall_score: Mapped[int | None] = mapped_column(Integer, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    job: Mapped["Job"] = relationship("Job", back_populates="result")
