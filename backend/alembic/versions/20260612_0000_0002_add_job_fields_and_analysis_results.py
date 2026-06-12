"""Add job_id/session_id/technique to jobs and create analysis_results table.

Revision ID: 0002
Revises: 0001
Create Date: 2026-06-12 00:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ------------------------------------------------------------------ jobs
    # Make upload_id nullable (jobs can now be created without an upload row).
    with op.batch_alter_table("jobs") as batch_op:
        batch_op.alter_column("upload_id", existing_type=sa.Integer(), nullable=True)
        batch_op.add_column(sa.Column("job_id", sa.String(length=36), nullable=True))
        batch_op.add_column(sa.Column("session_id", sa.String(length=255), nullable=True))
        batch_op.add_column(sa.Column("technique", sa.String(length=64), nullable=True))
        batch_op.create_unique_constraint("uq_jobs_job_id", ["job_id"])
        batch_op.create_index("ix_jobs_job_id", ["job_id"], unique=True)
        batch_op.create_index("ix_jobs_session_id", ["session_id"], unique=False)

    # ------------------------------------------------------------------ analysis_results
    op.create_table(
        "analysis_results",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("job_id", sa.String(length=36), nullable=False),
        sa.Column("scores", sa.Text(), nullable=False),
        sa.Column("metric_deltas", sa.Text(), nullable=False),
        sa.Column("keyframe_paths", sa.Text(), nullable=False),
        sa.Column("overall_score", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["job_id"], ["jobs.job_id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("job_id"),
    )
    op.create_index(op.f("ix_analysis_results_id"), "analysis_results", ["id"], unique=False)
    op.create_index(op.f("ix_analysis_results_job_id"), "analysis_results", ["job_id"], unique=True)


def downgrade() -> None:
    op.drop_index(op.f("ix_analysis_results_job_id"), table_name="analysis_results")
    op.drop_index(op.f("ix_analysis_results_id"), table_name="analysis_results")
    op.drop_table("analysis_results")

    with op.batch_alter_table("jobs") as batch_op:
        batch_op.drop_index("ix_jobs_session_id")
        batch_op.drop_index("ix_jobs_job_id")
        batch_op.drop_constraint("uq_jobs_job_id", type_="unique")
        batch_op.drop_column("technique")
        batch_op.drop_column("session_id")
        batch_op.drop_column("job_id")
        batch_op.alter_column("upload_id", existing_type=sa.Integer(), nullable=False)
