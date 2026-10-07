"""The model registry.

Revision ID: 0001
Revises:
"""

import sqlalchemy as sa
from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "models",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("registry_id", sa.String(200), nullable=False, unique=True),
        sa.Column("engine", sa.String(64), nullable=False),
        sa.Column("name", sa.String(128), nullable=False),
        sa.Column("description", sa.Text, nullable=False, server_default=""),
        sa.Column("languages", sa.ARRAY(sa.String(8)), nullable=False, server_default="{}"),
        sa.Column("artifact_uri", sa.Text, nullable=False, server_default=""),
        sa.Column("base_model_id", sa.String(32), nullable=False, server_default=""),
        sa.Column("status", sa.String(16), nullable=False, server_default="available"),
        sa.Column("is_default", sa.Boolean, nullable=False, server_default=sa.false()),
        sa.Column("created_by", sa.String(32), nullable=False, server_default=""),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("retired_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("models_one_default", "models", ["is_default"], unique=True, postgresql_where=sa.text("is_default"))


def downgrade() -> None:
    op.drop_table("models")
