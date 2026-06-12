"""Add feedback column to analysis_results.

Revision ID: 0003
Revises: 0002
Create Date: 2026-06-12 00:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0003"
down_revision: Union[str, None] = "0002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("analysis_results") as batch_op:
        batch_op.add_column(sa.Column("feedback", sa.Text(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("analysis_results") as batch_op:
        batch_op.drop_column("feedback")
