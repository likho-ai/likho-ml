"""Evaluations: a model transcribes every recording of a workspace's gold set (without keeping
the result) and is scored against the references.

StartEvaluation answers at once with a queued evaluation; a background task runs the queue, one
evaluation at a time (each recording is a full transcription). An evaluation found queued or
running when the service starts was cut off by a stop and is failed, not left hanging.

The queue lives in the process: likho-ml runs as one instance (likho-deploy keeps replicas at 1).
"""

import asyncio
import json
import logging
import time
from datetime import UTC, datetime

from sqlalchemy import delete, select, update
from sqlalchemy.dialects.postgresql import distinct_on
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from likho_ml.events import EVALUATION_COMPLETED, EVALUATION_FAILED, Publisher
from likho_ml.gold import GoldSet
from likho_ml.ids import new_id
from likho_ml.models import EvaluationItemRow, EvaluationRow, GoldItemRow
from likho_ml.registry import Registry, RegistryError
from likho_ml.scoring import Counts, Scores, score
from likho_ml.transcripts import TranscriptionClient

log = logging.getLogger(__name__)


class Evaluations:
    def __init__(
        self,
        sessions: async_sessionmaker[AsyncSession],
        registry: Registry,
        gold: GoldSet,
        transcripts: TranscriptionClient,
        publisher: Publisher,
    ) -> None:
        self._sessions = sessions
        self._registry = registry
        self._gold = gold
        self._transcripts = transcripts
        self._publisher = publisher
        self._queue: asyncio.Queue[str] = asyncio.Queue()

    # ------------------------------------------------------------------------------ requests

    async def start(self, workspace_id: str, model_id: str, user_id: str = "") -> EvaluationRow:
        if not workspace_id:
            raise RegistryError("invalid", "Name the workspace.")
        model = await self._registry.get(model_id=model_id)
        if model.status != "available":
            raise RegistryError("failed_precondition", f"{model.registry_id} is retired.")
        items = await self._gold.items(workspace_id)
        row = EvaluationRow(
            id=new_id("evl"),
            workspace_id=workspace_id,
            model_id=model.id,
            registry_id=model.registry_id,
            status="queued",
            items_total=len(items),
            started_by=user_id,
            created_at=datetime.now(UTC),
        )
        async with self._sessions() as session, session.begin():
            session.add(row)
        self._queue.put_nowait(row.id)
        return row

    async def get(self, evaluation_id: str) -> tuple[EvaluationRow, list[EvaluationItemRow]]:
        async with self._sessions() as session:
            row = await session.get(EvaluationRow, evaluation_id)
            if row is None:
                raise RegistryError("not_found", f"No evaluation {evaluation_id}.")
            items = await session.scalars(
                select(EvaluationItemRow)
                .where(EvaluationItemRow.evaluation_id == evaluation_id)
                .order_by(EvaluationItemRow.recording_id)
            )
            return row, list(items)

    async def list_evaluations(self, workspace_id: str, model_id: str = "", limit: int = 50) -> list[EvaluationRow]:
        async with self._sessions() as session:
            query = select(EvaluationRow).where(EvaluationRow.workspace_id == workspace_id)
            if model_id:
                query = query.where(EvaluationRow.model_id == model_id)
            rows = await session.scalars(query.order_by(EvaluationRow.created_at.desc()).limit(limit or 50))
            return list(rows)

    async def latest_scores(self) -> dict[str, EvaluationRow]:
        """The newest completed evaluation of each model (any workspace), for the model list."""
        async with self._sessions() as session:
            rows = await session.scalars(
                select(EvaluationRow)
                .where(EvaluationRow.status == "completed")
                .ext(distinct_on(EvaluationRow.model_id))
                .order_by(EvaluationRow.model_id, EvaluationRow.finished_at.desc())
            )
            return {row.model_id: row for row in rows}

    # ------------------------------------------------------------------------------ the worker

    async def recover(self) -> None:
        """Fails what a stop cut off; called once at start, before the worker runs."""
        async with self._sessions() as session, session.begin():
            cut = await session.execute(
                update(EvaluationRow)
                .where(EvaluationRow.status.in_(("queued", "running")))
                .values(
                    status="failed",
                    error="Interrupted by a restart of likho-ml; start it again.",
                    finished_at=datetime.now(UTC),
                )
                .returning(EvaluationRow.id)
            )
            ids = list(cut.scalars())
        for evaluation_id in ids:
            log.warning("evaluation %s was cut off by a restart: failed", evaluation_id)

    async def run(self, stop: asyncio.Event) -> None:
        while not stop.is_set():
            try:
                evaluation_id = await asyncio.wait_for(self._queue.get(), timeout=1)
            except TimeoutError:
                continue
            try:
                await self._evaluate(evaluation_id)
            except Exception as error:
                log.exception("evaluation %s failed", evaluation_id)
                await self._fail(evaluation_id, "internal", f"The evaluation stopped: {error}")

    async def _evaluate(self, evaluation_id: str) -> None:
        async with self._sessions() as session, session.begin():
            row = await session.get(EvaluationRow, evaluation_id, with_for_update=True)
            if row is None or row.status != "queued":
                return
            row.status = "running"
            workspace_id, model_id, registry_id = row.workspace_id, row.model_id, row.registry_id
        items = await self._gold.items(workspace_id)
        if not items:
            await self._fail(evaluation_id, "empty_gold_set", "The workspace has no gold recordings yet.")
            return
        async with self._sessions() as session, session.begin():
            await session.execute(
                update(EvaluationRow).where(EvaluationRow.id == evaluation_id).values(items_total=len(items))
            )

        total = Scores()
        scored = failed = 0
        audio = elapsed = 0.0
        for item in items:
            result = await self._item(evaluation_id, item, registry_id)
            if result.error:
                failed += 1
                if "model_unavailable" in result.error:
                    await self._fail(evaluation_id, "model_unavailable", result.error)
                    return
            else:
                scored += 1
                counts = json.loads(result.counts)
                total.add(Scores(script=Counts(**counts["script"]), roman=Counts(**counts["roman"])))
                audio += result.audio_seconds
                elapsed += result.elapsed_seconds
            async with self._sessions() as session, session.begin():
                await session.execute(
                    update(EvaluationRow).where(EvaluationRow.id == evaluation_id).values(items_done=scored + failed)
                )

        if scored == 0:
            await self._fail(evaluation_id, "all_failed", "No gold recording could be transcribed with this model.")
            return
        rates = total.rates()
        async with self._sessions() as session, session.begin():
            await session.execute(
                update(EvaluationRow)
                .where(EvaluationRow.id == evaluation_id)
                .values(
                    status="completed",
                    audio_seconds=audio,
                    elapsed_seconds=elapsed,
                    finished_at=datetime.now(UTC),
                    **rates,
                )
            )
        log.info("evaluation %s of %s: %s", evaluation_id, registry_id, rates)
        await self._publisher.publish(
            EVALUATION_COMPLETED,
            evaluation_id,
            {
                "evaluation_id": evaluation_id,
                "workspace_id": workspace_id,
                "model_id": model_id,
                "registry_id": registry_id,
                "items_scored": scored,
                "items_failed": failed,
                "audio_seconds": audio,
                "realtime_factor": audio / elapsed if elapsed else 0.0,
                **rates,
            },
        )

    async def _item(self, evaluation_id: str, item: GoldItemRow, registry_id: str) -> EvaluationItemRow:
        started = time.perf_counter()
        row = EvaluationItemRow(
            evaluation_id=evaluation_id, recording_id=item.recording_id, reference_transcript_id=item.transcript_id
        )
        try:
            hypothesis = await self._transcripts.evaluate(
                item.recording_id, item.media_id, item.workspace_id, registry_id
            )
            scores = score(item.reference_script, hypothesis.script, item.reference_roman, hypothesis.roman)
            row.wer_script, row.cer_script = scores.script.wer, scores.script.cer
            row.wer_roman, row.cer_roman = scores.roman.wer, scores.roman.cer
            row.counts = json.dumps({"script": vars(scores.script), "roman": vars(scores.roman)})
            row.words = scores.script.words
            row.audio_seconds = hypothesis.audio_seconds or item.audio_seconds
            row.elapsed_seconds = hypothesis.elapsed_seconds or (time.perf_counter() - started)
            row.error = ""
        except Exception as error:
            details = getattr(error, "details", None)
            row.error = (details() if callable(details) else "") or str(error) or type(error).__name__
            log.warning("evaluation %s: %s failed: %s", evaluation_id, item.recording_id, row.error)
        async with self._sessions() as session, session.begin():
            await session.merge(row)
        return row

    async def _fail(self, evaluation_id: str, code: str, message: str) -> None:
        async with self._sessions() as session, session.begin():
            row = await session.get(EvaluationRow, evaluation_id, with_for_update=True)
            if row is None:
                return
            row.status = "failed"
            row.error = message
            row.finished_at = datetime.now(UTC)
            data = {
                "evaluation_id": row.id,
                "workspace_id": row.workspace_id,
                "model_id": row.model_id,
                "registry_id": row.registry_id,
                "code": code,
                "message": message,
            }
        await self._publisher.publish(EVALUATION_FAILED, evaluation_id, data)

    async def delete_workspace(self, workspace_id: str) -> None:
        async with self._sessions() as session, session.begin():
            ids = select(EvaluationRow.id).where(EvaluationRow.workspace_id == workspace_id)
            await session.execute(delete(EvaluationItemRow).where(EvaluationItemRow.evaluation_id.in_(ids)))
            await session.execute(delete(EvaluationRow).where(EvaluationRow.workspace_id == workspace_id))
