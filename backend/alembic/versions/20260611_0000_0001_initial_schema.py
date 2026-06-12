"""Initial schema – uploads, jobs, scores, history tables.

Revision ID: 0001
Revises:
Create Date: 2026-06-11 00:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ------------------------------------------------------------------ uploads
    op.create_table(
        "uploads",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("filename", sa.String(length=255), nullable=False),
        sa.Column("storage_path", sa.String(length=1024), nullable=False),
        sa.Column(
            "technique",
            sa.Enum("front_kick", "roundhouse_kick", "straight_punch", name="techniquetype"),
            nullable=False,
        ),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_uploads_id"), "uploads", ["id"], unique=False)

    # ------------------------------------------------------------------ jobs
    op.create_table(
        "jobs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("upload_id", sa.Integer(), nullable=False),
        sa.Column(
            "status",
            sa.Enum("pending", "processing", "completed", "failed", name="jobstatus"),
            nullable=False,
        ),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["upload_id"], ["uploads.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_jobs_id"), "jobs", ["id"], unique=False)
    op.create_index(op.f("ix_jobs_upload_id"), "jobs", ["upload_id"], unique=False)

    # ------------------------------------------------------------------ scores
    op.create_table(
        "scores",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("job_id", sa.Integer(), nullable=False),
        sa.Column("criterion", sa.String(length=128), nullable=False),
        sa.Column("value", sa.Float(), nullable=False),
        sa.Column("reference_delta", sa.Float(), nullable=True),
        sa.Column("label", sa.String(length=256), nullable=True),
        sa.ForeignKeyConstraint(["job_id"], ["jobs.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_scores_id"), "scores", ["id"], unique=False)
    op.create_index(op.f("ix_scores_job_id"), "scores", ["job_id"], unique=False)

    # ------------------------------------------------------------------ history
    op.create_table(
        "history",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("upload_id", sa.Integer(), nullable=False),
        sa.Column(
            "technique",
            sa.Enum("front_kick", "roundhouse_kick", "straight_punch", name="techniquetype"),
            nullable=False,
        ),
        sa.Column("overall_score", sa.Float(), nullable=False),
        sa.Column("recorded_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["upload_id"], ["uploads.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_history_id"), "history", ["id"], unique=False)
    op.create_index(op.f("ix_history_upload_id"), "history", ["upload_id"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_history_upload_id"), table_name="history")
    op.drop_index(op.f("ix_history_id"), table_name="history")
    op.drop_table("history")

    op.drop_index(op.f("ix_scores_job_id"), table_name="scores")
    op.drop_index(op.f("ix_scores_id"), table_name="scores")
    op.drop_table("scores")

    op.drop_index(op.f("ix_jobs_upload_id"), table_name="jobs")
    op.drop_index(op.f("ix_jobs_id"), table_name="jobs")
    op.drop_table("jobs")

    op.drop_index(op.f("ix_uploads_id"), table_name="uploads")
    op.drop_table("uploads")
