"""Training examples: the lines people corrected.

Revision ID: 0002
Revises: 0001
"""

import sqlalchemy as sa
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "training_examples",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("event_id", sa.String(40), nullable=False, unique=True),
        sa.Column("workspace_id", sa.String(32), nullable=False),
        sa.Column("recording_id", sa.String(32), nullable=False),
        sa.Column("transcript_id", sa.String(32), nullable=False),
        sa.Column("segment_index", sa.Integer, nullable=False),
        sa.Column("layer", sa.String(8), nullable=False),
        sa.Column("before", sa.Text, nullable=False),
        sa.Column("after", sa.Text, nullable=False),
        sa.Column("start_seconds", sa.Float, nullable=False),
        sa.Column("end_seconds", sa.Float, nullable=False),
        sa.Column("text_script", sa.Text, nullable=False),
        sa.Column("text_roman", sa.Text, nullable=False),
        sa.Column("language", sa.String(8), nullable=False, server_default=""),
        sa.Column("user_id", sa.String(32), nullable=False, server_default=""),
        sa.Column("corrected_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "training_examples_line",
        "training_examples",
        ["workspace_id", "recording_id", "segment_index", "corrected_at"],
    )


def downgrade() -> None:
    op.drop_table("training_examples")
