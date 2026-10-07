"""The tables of likho_ml. Only this service reads or writes them."""

from datetime import datetime

from sqlalchemy import ARRAY, Boolean, DateTime, Index, String, Text, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class ModelRow(Base):
    """A speech model the transcription service can load, by its registry id."""

    __tablename__ = "models"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    # "<engine>/<name>", e.g. faster-whisper/turbo.
    registry_id: Mapped[str] = mapped_column(String(200), unique=True)
    engine: Mapped[str] = mapped_column(String(64))
    name: Mapped[str] = mapped_column(String(128))
    description: Mapped[str] = mapped_column(Text, default="")
    languages: Mapped[list[str]] = mapped_column(ARRAY(String(8)), default=list)
    # Empty for an engine's own published sizes; s3://likho-models/... for a fine-tuned one.
    artifact_uri: Mapped[str] = mapped_column(Text, default="")
    base_model_id: Mapped[str] = mapped_column(String(32), default="")
    # available | retired
    status: Mapped[str] = mapped_column(String(16), default="available")
    is_default: Mapped[bool] = mapped_column(Boolean, default=False)
    created_by: Mapped[str] = mapped_column(String(32), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    retired_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        # At most one default: the database refuses a second, whatever two callers do at once.
        Index("models_one_default", "is_default", unique=True, postgresql_where=text("is_default")),
    )
