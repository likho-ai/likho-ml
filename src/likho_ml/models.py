"""The tables of likho_ml. Only this service reads or writes them."""

from datetime import datetime

from sqlalchemy import ARRAY, Boolean, DateTime, Float, Index, Integer, String, Text, text
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


class TrainingExampleRow(Base):
    """One line a person corrected: the audio span and the text as it is after the correction.

    Every correction is kept (the event id makes a redelivered event change nothing); when a line
    is corrected again, the newest row is the one that counts (stats, datasets).
    """

    __tablename__ = "training_examples"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    # The likho.transcript.corrected event this row came from.
    event_id: Mapped[str] = mapped_column(String(40), unique=True)
    workspace_id: Mapped[str] = mapped_column(String(32))
    recording_id: Mapped[str] = mapped_column(String(32))
    # The corrected transcript version, and the line in it.
    transcript_id: Mapped[str] = mapped_column(String(32))
    segment_index: Mapped[int] = mapped_column(Integer)
    # The layer the person changed: script or roman.
    layer: Mapped[str] = mapped_column(String(8))
    before: Mapped[str] = mapped_column(Text)
    after: Mapped[str] = mapped_column(Text)
    # The line's audio span and both layers as they are now (the training target).
    start_seconds: Mapped[float] = mapped_column(Float)
    end_seconds: Mapped[float] = mapped_column(Float)
    text_script: Mapped[str] = mapped_column(Text)
    text_roman: Mapped[str] = mapped_column(Text)
    language: Mapped[str] = mapped_column(String(8), default="")
    user_id: Mapped[str] = mapped_column(String(32), default="")
    corrected_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))

    __table_args__ = (Index("training_examples_line", "workspace_id", "recording_id", "segment_index", "corrected_at"),)
