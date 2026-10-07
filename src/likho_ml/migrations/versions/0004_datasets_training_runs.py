"""Datasets and fine-tuning runs.

Revision ID: 0004
Revises: 0003
"""

import sqlalchemy as sa
from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "datasets",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("workspace_id", sa.String(32), nullable=False, index=True),
        sa.Column("uri", sa.Text, nullable=False),
        sa.Column("examples", sa.Integer, nullable=False),
        sa.Column("audio_seconds", sa.Float, nullable=False),
        sa.Column("held_out_recordings", sa.Integer, nullable=False),
        sa.Column("created_by", sa.String(32), nullable=False, server_default=""),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_table(
        "training_runs",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("workspace_id", sa.String(32), nullable=False, index=True),
        sa.Column("dataset_id", sa.String(32), nullable=False),
        sa.Column("base_model_id", sa.String(32), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("launcher", sa.String(200), nullable=False, server_default=""),
        sa.Column("external_id", sa.String(200), nullable=False, server_default=""),
        sa.Column("model_id", sa.String(32), nullable=False, server_default=""),
        sa.Column("error", sa.Text, nullable=False, server_default=""),
        sa.Column("started_by", sa.String(32), nullable=False, server_default=""),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_table("training_runs")
    op.drop_table("datasets")
