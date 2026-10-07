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


class GoldItemRow(Base):
    """A recording of a workspace's gold set, with its reference text copied in.

    The reference is the transcript as it was when the recording was added: a later correction
    does not move the goalposts of evaluations already made (adding it again takes the new text).
    """

    __tablename__ = "gold_items"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    workspace_id: Mapped[str] = mapped_column(String(32))
    recording_id: Mapped[str] = mapped_column(String(32))
    media_id: Mapped[str] = mapped_column(String(32))
    transcript_id: Mapped[str] = mapped_column(String(32))
    transcript_version: Mapped[int] = mapped_column(Integer)
    language: Mapped[str] = mapped_column(String(8), default="")
    audio_seconds: Mapped[float] = mapped_column(Float, default=0.0)
    lines: Mapped[int] = mapped_column(Integer, default=0)
    reference_script: Mapped[str] = mapped_column(Text)
    reference_roman: Mapped[str] = mapped_column(Text)
    added_by: Mapped[str] = mapped_column(String(32), default="")
    added_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))

    __table_args__ = (Index("gold_items_recording", "workspace_id", "recording_id", unique=True),)


class EvaluationRow(Base):
    """A model scored on a workspace's gold set. status: queued, running, completed, failed."""

    __tablename__ = "evaluations"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    workspace_id: Mapped[str] = mapped_column(String(32))
    model_id: Mapped[str] = mapped_column(String(32))
    registry_id: Mapped[str] = mapped_column(String(200))
    status: Mapped[str] = mapped_column(String(16))
    wer_script: Mapped[float] = mapped_column(Float, default=0.0)
    cer_script: Mapped[float] = mapped_column(Float, default=0.0)
    wer_roman: Mapped[float] = mapped_column(Float, default=0.0)
    cer_roman: Mapped[float] = mapped_column(Float, default=0.0)
    items_total: Mapped[int] = mapped_column(Integer, default=0)
    items_done: Mapped[int] = mapped_column(Integer, default=0)
    audio_seconds: Mapped[float] = mapped_column(Float, default=0.0)
    elapsed_seconds: Mapped[float] = mapped_column(Float, default=0.0)
    error: Mapped[str] = mapped_column(Text, default="")
    started_by: Mapped[str] = mapped_column(String(32), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (Index("evaluations_model", "model_id", "status", "finished_at"),)


class DatasetRow(Base):
    """Training examples written to the models bucket for a fine-tuning run."""

    __tablename__ = "datasets"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    workspace_id: Mapped[str] = mapped_column(String(32))
    uri: Mapped[str] = mapped_column(Text)
    examples: Mapped[int] = mapped_column(Integer)
    audio_seconds: Mapped[float] = mapped_column(Float)
    held_out_recordings: Mapped[int] = mapped_column(Integer)
    created_by: Mapped[str] = mapped_column(String(32), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class TrainingRunRow(Base):
    """A fine-tuning run. status: pending, running, completed, failed."""

    __tablename__ = "training_runs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    workspace_id: Mapped[str] = mapped_column(String(32))
    dataset_id: Mapped[str] = mapped_column(String(32))
    base_model_id: Mapped[str] = mapped_column(String(32))
    status: Mapped[str] = mapped_column(String(16))
    launcher: Mapped[str] = mapped_column(String(200), default="")
    external_id: Mapped[str] = mapped_column(String(200), default="")
    model_id: Mapped[str] = mapped_column(String(32), default="")
    error: Mapped[str] = mapped_column(Text, default="")
    started_by: Mapped[str] = mapped_column(String(32), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class EvaluationItemRow(Base):
    """One gold recording of an evaluation, transcribed and scored."""

    __tablename__ = "evaluation_items"

    evaluation_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    recording_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    reference_transcript_id: Mapped[str] = mapped_column(String(32))
    wer_script: Mapped[float] = mapped_column(Float, default=0.0)
    cer_script: Mapped[float] = mapped_column(Float, default=0.0)
    wer_roman: Mapped[float] = mapped_column(Float, default=0.0)
    cer_roman: Mapped[float] = mapped_column(Float, default=0.0)
    # Edits and reference lengths, so the corpus rates can be summed again later.
    counts: Mapped[str] = mapped_column(Text, default="")
    words: Mapped[int] = mapped_column(Integer, default=0)
    audio_seconds: Mapped[float] = mapped_column(Float, default=0.0)
    elapsed_seconds: Mapped[float] = mapped_column(Float, default=0.0)
    error: Mapped[str] = mapped_column(Text, default="")
