"""The gold set and evaluations.

Revision ID: 0003
Revises: 0002
"""

import sqlalchemy as sa
from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "gold_items",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("workspace_id", sa.String(32), nullable=False),
        sa.Column("recording_id", sa.String(32), nullable=False),
        sa.Column("media_id", sa.String(32), nullable=False),
        sa.Column("transcript_id", sa.String(32), nullable=False),
        sa.Column("transcript_version", sa.Integer, nullable=False),
        sa.Column("language", sa.String(8), nullable=False, server_default=""),
        sa.Column("audio_seconds", sa.Float, nullable=False, server_default="0"),
        sa.Column("lines", sa.Integer, nullable=False, server_default="0"),
        sa.Column("reference_script", sa.Text, nullable=False),
        sa.Column("reference_roman", sa.Text, nullable=False),
        sa.Column("added_by", sa.String(32), nullable=False, server_default=""),
        sa.Column("added_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("gold_items_recording", "gold_items", ["workspace_id", "recording_id"], unique=True)

    op.create_table(
        "evaluations",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("workspace_id", sa.String(32), nullable=False),
        sa.Column("model_id", sa.String(32), nullable=False),
        sa.Column("registry_id", sa.String(200), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("wer_script", sa.Float, nullable=False, server_default="0"),
        sa.Column("cer_script", sa.Float, nullable=False, server_default="0"),
        sa.Column("wer_roman", sa.Float, nullable=False, server_default="0"),
        sa.Column("cer_roman", sa.Float, nullable=False, server_default="0"),
        sa.Column("items_total", sa.Integer, nullable=False, server_default="0"),
        sa.Column("items_done", sa.Integer, nullable=False, server_default="0"),
        sa.Column("audio_seconds", sa.Float, nullable=False, server_default="0"),
        sa.Column("elapsed_seconds", sa.Float, nullable=False, server_default="0"),
        sa.Column("error", sa.Text, nullable=False, server_default=""),
        sa.Column("started_by", sa.String(32), nullable=False, server_default=""),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("evaluations_model", "evaluations", ["model_id", "status", "finished_at"])

    op.create_table(
        "evaluation_items",
        sa.Column("evaluation_id", sa.String(32), primary_key=True),
        sa.Column("recording_id", sa.String(32), primary_key=True),
        sa.Column("reference_transcript_id", sa.String(32), nullable=False),
        sa.Column("wer_script", sa.Float, nullable=False, server_default="0"),
        sa.Column("cer_script", sa.Float, nullable=False, server_default="0"),
        sa.Column("wer_roman", sa.Float, nullable=False, server_default="0"),
        sa.Column("cer_roman", sa.Float, nullable=False, server_default="0"),
        sa.Column("counts", sa.Text, nullable=False, server_default=""),
        sa.Column("words", sa.Integer, nullable=False, server_default="0"),
        sa.Column("audio_seconds", sa.Float, nullable=False, server_default="0"),
        sa.Column("elapsed_seconds", sa.Float, nullable=False, server_default="0"),
        sa.Column("error", sa.Text, nullable=False, server_default=""),
    )


def downgrade() -> None:
    op.drop_table("evaluation_items")
    op.drop_table("evaluations")
    op.drop_table("gold_items")
