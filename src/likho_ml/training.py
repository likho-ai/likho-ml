"""Training examples from people's corrections.

likho-transcription publishes `likho.transcript.corrected.v1` (stream LIKHO_KEEP, never expired)
each time a person fixes a line. This module takes those events with a durable consumer, asks
likho-transcription for the line's audio span and both layers as they are after the fix, and
keeps them as a training example. A line corrected again keeps every row; the newest counts.
"""

import asyncio
import contextlib
import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from nats.errors import TimeoutError as NatsTimeoutError
from nats.js.api import AckPolicy, ConsumerConfig, DeliverPolicy
from sqlalchemy import func, select, text
from sqlalchemy.dialects.postgresql import distinct_on, insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from likho_ml.ids import new_id
from likho_ml.models import TrainingExampleRow
from likho_ml.transcripts import LineNotFoundError, TranscriptionClient

log = logging.getLogger(__name__)

CORRECTED_SUBJECT = "likho.transcript.corrected"
CORRECTED_STREAM = "LIKHO_KEEP"


@dataclass(frozen=True)
class Stats:
    examples: int
    script_examples: int
    roman_examples: int
    recordings: int
    audio_seconds: float
    last_example_at: datetime | None


def corrected_at(event: dict[str, Any]) -> datetime:
    value = event.get("time")
    if isinstance(value, str):
        with contextlib.suppress(ValueError):
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    return datetime.now(UTC)


class TrainingStore:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def add(self, row: TrainingExampleRow) -> bool:
        """Keeps the example; False when its event was kept already (a redelivery)."""
        values = {column.name: getattr(row, column.name) for column in TrainingExampleRow.__table__.columns}
        statement = insert(TrainingExampleRow).values(values).on_conflict_do_nothing(index_elements=["event_id"])
        async with self._sessions() as session, session.begin():
            result = await session.execute(statement)
        return bool(result.rowcount)  # type: ignore[attr-defined]

    async def stats(self, workspace_id: str) -> Stats:
        """Counted over the newest correction of each line and layer."""
        latest = (
            select(TrainingExampleRow)
            .where(TrainingExampleRow.workspace_id == workspace_id)
            .ext(
                distinct_on(TrainingExampleRow.recording_id, TrainingExampleRow.segment_index, TrainingExampleRow.layer)
            )
            .order_by(
                TrainingExampleRow.recording_id,
                TrainingExampleRow.segment_index,
                TrainingExampleRow.layer,
                TrainingExampleRow.corrected_at.desc(),
            )
            .subquery()
        )
        lines = (
            select(
                latest.c.recording_id, latest.c.segment_index, func.max(latest.c.end_seconds - latest.c.start_seconds)
            )
            .group_by(latest.c.recording_id, latest.c.segment_index)
            .subquery()
        )
        async with self._sessions() as session:
            counts = (
                await session.execute(
                    select(
                        func.count(),
                        func.count().filter(latest.c.layer == "script"),
                        func.count().filter(latest.c.layer == "roman"),
                        func.count(latest.c.recording_id.distinct()),
                        func.max(latest.c.corrected_at),
                    ).select_from(latest)
                )
            ).one()
            audio = await session.scalar(select(func.coalesce(func.sum(lines.c[2]), 0.0)))
        return Stats(
            examples=int(counts[0]),
            script_examples=int(counts[1]),
            roman_examples=int(counts[2]),
            recordings=int(counts[3]),
            audio_seconds=float(audio or 0.0),
            last_example_at=counts[4],
        )

    async def delete_workspace(self, workspace_id: str) -> None:
        async with self._sessions() as session, session.begin():
            await session.execute(text("delete from training_examples where workspace_id = :w"), {"w": workspace_id})


class CorrectionConsumer:
    """Takes corrections from the bus and keeps them as training examples."""

    def __init__(
        self,
        js: Any,
        store: TrainingStore,
        transcripts: TranscriptionClient,
        durable: str,
        start: str,
        metrics: Any = None,
    ) -> None:
        self._js = js
        self._store = store
        self._transcripts = transcripts
        self._durable = durable
        self._start = start
        self._metrics = metrics
        self._subscription: Any = None

    async def start(self) -> None:
        config = ConsumerConfig(
            durable_name=self._durable,
            ack_policy=AckPolicy.EXPLICIT,
            ack_wait=60,
            max_deliver=-1,  # a correction is never given up on: transcription may be down a while
            deliver_policy=DeliverPolicy.NEW if self._start == "new" else DeliverPolicy.ALL,
            filter_subject=CORRECTED_SUBJECT,
        )
        self._subscription = await self._js.pull_subscribe(
            CORRECTED_SUBJECT, durable=self._durable, stream=CORRECTED_STREAM, config=config
        )
        log.info("keeping corrections from %s as %s", CORRECTED_SUBJECT, self._durable)

    async def run(self, stop: asyncio.Event) -> None:
        while not stop.is_set():
            try:
                messages = await self._subscription.fetch(20, timeout=1)
            except (NatsTimeoutError, TimeoutError):
                continue
            except Exception:
                log.exception("could not fetch corrections; trying again")
                await asyncio.sleep(1)
                continue
            for message in messages:
                await self.handle(message)
        with contextlib.suppress(Exception):
            await self._subscription.unsubscribe()

    def _count(self, outcome: str) -> None:
        if self._metrics is not None:
            self._metrics.events_handled.add(1, {"subject": CORRECTED_SUBJECT, "outcome": outcome})

    async def handle(self, message: Any) -> None:
        try:
            event = json.loads(message.data)
            data = event["data"]
            event_id = str(event["id"])
            transcript_id = str(data["transcript_id"])
            index = int(data["segment_index"])
            layer = str(data["layer"])
            if layer not in ("script", "roman"):
                raise ValueError(f"layer {layer}")
        except (ValueError, KeyError, TypeError) as error:
            log.error("dropping a message that is not a correction: %s", error)
            self._count("dropped")
            await message.term()
            return
        try:
            line = await self._transcripts.line(transcript_id, index)
        except LineNotFoundError as gone:
            log.warning("correction %s: %s; not kept", event_id, gone)
            self._count("skipped")
            await message.ack()
            return
        except Exception:
            log.exception("correction %s: transcription not reachable; trying again", event_id)
            self._count("retry")
            await message.nak(delay=10)
            return
        row = TrainingExampleRow(
            id=new_id("tex"),
            event_id=event_id,
            workspace_id=str(data["workspace_id"]),
            recording_id=str(data["recording_id"]),
            transcript_id=transcript_id,
            segment_index=index,
            layer=layer,
            before=str(data.get("before") or ""),
            after=str(data.get("after") or ""),
            start_seconds=line.start_seconds,
            end_seconds=line.end_seconds,
            text_script=line.text_script,
            text_roman=line.text_roman,
            language=line.language,
            user_id=str(data.get("user_id") or ""),
            corrected_at=corrected_at(event),
        )
        try:
            kept = await self._store.add(row)
        except Exception:
            log.exception("correction %s: could not keep it; trying again", event_id)
            self._count("retry")
            await message.nak(delay=5)
            return
        self._count("ok" if kept else "duplicate")
        await message.ack()
