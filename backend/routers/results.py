"""
Endpoints for retrieving analysis results and session history.

Routes
------
GET /jobs/{job_id}/results
    - 200  completed job → AnalysisResultResponse
    - 202  job exists but is not yet completed
    - 404  job not found

GET /history
    - 200  paginated list of past jobs for a session_id
"""

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from ..database import get_db
from ..models import AnalysisResult, Job, JobStatus
from ..schemas import AnalysisResultResponse, HistoryResponse, HistoryItem

router = APIRouter(tags=["results"])


@router.get(
    "/jobs/{job_id}/results",
    response_model=AnalysisResultResponse,
    responses={
        202: {"description": "Job is still being processed"},
        404: {"description": "Job not found"},
    },
)
def get_job_results(
    job_id: str,
    db: Session = Depends(get_db),
) -> AnalysisResultResponse:
    """Return scores for a completed analysis job.

    Returns HTTP 202 when the job is pending/processing, HTTP 404 when the
    job_id does not exist.
    """
    job: Job | None = db.scalar(
        select(Job)
        .where(Job.job_id == job_id)
        .options(selectinload(Job.result))
    )

    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")

    if job.status != JobStatus.completed:
        raise HTTPException(
            status_code=status.HTTP_202_ACCEPTED,
            detail=f"Job is {job.status.value}",
        )

    return AnalysisResultResponse.model_validate(job)


@router.get(
    "/history",
    response_model=HistoryResponse,
)
def get_history(
    session_id: str = Query(..., description="Session identifier for the user"),
    page: int = Query(1, ge=1, description="1-based page number"),
    page_size: int = Query(10, ge=1, le=100, description="Items per page"),
    db: Session = Depends(get_db),
) -> HistoryResponse:
    """Return paginated analysis history for a session.

    Items are ordered newest-first.
    """
    base_query = select(Job).where(Job.session_id == session_id)

    total: int = db.scalar(
        select(func.count()).select_from(base_query.subquery())
    ) or 0

    offset = (page - 1) * page_size
    jobs = db.scalars(
        base_query
        .options(selectinload(Job.result))
        .order_by(Job.created_at.desc())
        .offset(offset)
        .limit(page_size)
    ).all()

    return HistoryResponse(
        items=[HistoryItem.model_validate(job) for job in jobs],
        total=total,
        page=page,
        page_size=page_size,
    )
