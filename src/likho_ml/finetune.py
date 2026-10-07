"""Fine-tuning: training examples written out as a dataset, and runs handed to a launcher.

A dataset is s3://<bucket>/datasets/<id>/manifest.jsonl - one line per corrected line, with its
newest text in both layers and the audio span (recording, start, end) - and dataset.json, what
it holds. The gold set's recordings are left out, so a model trained on it is still scored on
calls it never saw.

A run is recorded, then handed to the launcher (LAUNCHER_URL: a GPU machine's agent or a cloud
job's front) as JSON. The job reports back with ReportTrainingRun; COMPLETED registers the new
model, fine-tuned from the run's base. Without a launcher a run waits as pending.
"""

import asyncio
import json
import logging
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import distinct_on
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from likho_ml.ids import new_id
from likho_ml.models import DatasetRow, GoldItemRow, TrainingExampleRow, TrainingRunRow
from likho_ml.registry import NewModel, Registry, RegistryError

log = logging.getLogger(__name__)

STATUSES = ("pending", "running", "completed", "failed")


@dataclass(frozen=True)
class Bucket:
    """Where datasets go; `put` writes one object (blocking, run in a thread)."""

    endpoint: str
    region: str
    access_key: str
    secret_key: str
    name: str

    def put(self, key: str, body: bytes, content_type: str) -> None:
        import boto3  # only the exporting process needs it

        client = boto3.client(
            "s3",
            endpoint_url=self.endpoint or None,
            region_name=self.region,
            aws_access_key_id=self.access_key or None,
            aws_secret_access_key=self.secret_key or None,
        )
        client.put_object(Bucket=self.name, Key=key, Body=body, ContentType=content_type)


class FineTuning:
    def __init__(
        self,
        sessions: async_sessionmaker[AsyncSession],
        registry: Registry,
        bucket: Bucket,
        launcher_url: str = "",
        report_address: str = "",
    ) -> None:
        self._sessions = sessions
        self._registry = registry
        self._bucket = bucket
        self._launcher_url = launcher_url
        self._report_address = report_address

    # ------------------------------------------------------------------------------ datasets

    async def export(self, workspace_id: str, user_id: str = "") -> DatasetRow:
        if not workspace_id:
            raise RegistryError("invalid", "Name the workspace.")
        async with self._sessions() as session:
            gold = set(
                await session.scalars(select(GoldItemRow.recording_id).where(GoldItemRow.workspace_id == workspace_id))
            )
            # The newest correction of each line (either layer): its texts are the line as it is now.
            latest = await session.scalars(
                select(TrainingExampleRow)
                .where(TrainingExampleRow.workspace_id == workspace_id)
                .ext(distinct_on(TrainingExampleRow.recording_id, TrainingExampleRow.segment_index))
                .order_by(
                    TrainingExampleRow.recording_id,
                    TrainingExampleRow.segment_index,
                    TrainingExampleRow.corrected_at.desc(),
                )
            )
            lines = [row for row in latest if row.recording_id not in gold]
        if not lines:
            raise RegistryError(
                "failed_precondition", "There are no corrected lines to train on yet (gold recordings are left out)."
            )
        dataset_id = new_id("dst")
        prefix = f"datasets/{dataset_id}/"
        manifest = "".join(
            json.dumps(
                {
                    "recording_id": row.recording_id,
                    "transcript_id": row.transcript_id,
                    "segment_index": row.segment_index,
                    "start": row.start_seconds,
                    "end": row.end_seconds,
                    "text_script": row.text_script,
                    "text_roman": row.text_roman,
                    "language": row.language,
                },
                ensure_ascii=False,
            )
            + "\n"
            for row in lines
        ).encode()
        audio = sum(max(0.0, row.end_seconds - row.start_seconds) for row in lines)
        about = {
            "id": dataset_id,
            "workspace_id": workspace_id,
            "examples": len(lines),
            "audio_seconds": audio,
            "held_out_recordings": sorted(gold),
            "created_at": datetime.now(UTC).isoformat(),
            "manifest": "manifest.jsonl: recording_id, start and end (seconds) of the audio; text_script and "
            "text_roman as people corrected them; the audio is the recording's (likho-media).",
        }
        await asyncio.to_thread(self._bucket.put, prefix + "manifest.jsonl", manifest, "application/x-ndjson")
        await asyncio.to_thread(
            self._bucket.put, prefix + "dataset.json", json.dumps(about, indent=2).encode(), "application/json"
        )
        row = DatasetRow(
            id=dataset_id,
            workspace_id=workspace_id,
            uri=f"s3://{self._bucket.name}/{prefix}",
            examples=len(lines),
            audio_seconds=audio,
            held_out_recordings=len(gold),
            created_by=user_id,
            created_at=datetime.now(UTC),
        )
        async with self._sessions() as session, session.begin():
            session.add(row)
        log.info(
            "dataset %s: %d lines, %.0f s of audio, %d gold recordings held out", row.id, len(lines), audio, len(gold)
        )
        return row

    # ------------------------------------------------------------------------------ runs

    async def start(self, workspace_id: str, dataset_id: str, base_model_id: str, user_id: str = "") -> TrainingRunRow:
        async with self._sessions() as session:
            dataset = await session.get(DatasetRow, dataset_id)
        if dataset is None or dataset.workspace_id != workspace_id:
            raise RegistryError("not_found", f"No dataset {dataset_id}.")
        base = await self._registry.get(model_id=base_model_id)
        run = TrainingRunRow(
            id=new_id("trr"),
            workspace_id=workspace_id,
            dataset_id=dataset_id,
            base_model_id=base.id,
            status="pending",
            launcher=self._launcher_url,
            started_by=user_id,
            created_at=datetime.now(UTC),
        )
        async with self._sessions() as session, session.begin():
            session.add(run)
        if self._launcher_url:
            try:
                answer = await asyncio.to_thread(self._hand_over, run, dataset.uri, base.registry_id, base.artifact_uri)
                run = await self._update(run.id, status="running", external_id=str(answer.get("external_id") or ""))
            except Exception as error:
                log.exception("training run %s: the launcher did not take it", run.id)
                run = await self._update(
                    run.id, status="failed", error=f"The launcher did not take the run: {error}", finished=True
                )
        return run

    def _hand_over(
        self, run: TrainingRunRow, dataset_uri: str, base_registry_id: str, base_artifact: str
    ) -> dict[str, Any]:
        body = json.dumps(
            {
                "run_id": run.id,
                "workspace_id": run.workspace_id,
                "dataset_uri": dataset_uri,
                "base_model": {"id": run.base_model_id, "registry_id": base_registry_id, "artifact_uri": base_artifact},
                # Where to report back: likho.ml.v1.MlService.ReportTrainingRun on this gRPC address.
                "report_to": self._report_address,
            }
        ).encode()
        request = urllib.request.Request(
            self._launcher_url, data=body, method="POST", headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            raw = response.read()
        return json.loads(raw) if raw else {}

    async def report(
        self,
        run_id: str,
        status: str,
        external_id: str = "",
        registry_id: str = "",
        artifact_uri: str = "",
        error: str = "",
    ) -> TrainingRunRow:
        if status not in ("running", "completed", "failed"):
            raise RegistryError("invalid", "Report RUNNING, COMPLETED or FAILED.")
        async with self._sessions() as session:
            run = await session.get(TrainingRunRow, run_id)
        if run is None:
            raise RegistryError("not_found", f"No training run {run_id}.")
        if run.status in ("completed", "failed"):
            raise RegistryError("failed_precondition", f"Run {run_id} is {run.status} already.")
        if status == "running":
            return await self._update(run_id, status="running", external_id=external_id or run.external_id)
        if status == "failed":
            return await self._update(run_id, status="failed", error=error or "The job failed.", finished=True)
        if not registry_id or not artifact_uri:
            raise RegistryError("invalid", "A completed run names its model's registry id and where the weights are.")
        model = await self._registry.register(
            NewModel(
                registry_id=registry_id,
                description=f"Fine-tuned from the run {run_id} on dataset {run.dataset_id}",
                artifact_uri=artifact_uri,
                base_model_id=run.base_model_id,
                user_id=run.started_by,
            )
        )
        return await self._update(run_id, status="completed", model_id=model.id, finished=True)

    async def runs(self, workspace_id: str) -> list[TrainingRunRow]:
        async with self._sessions() as session:
            rows = await session.scalars(
                select(TrainingRunRow)
                .where(TrainingRunRow.workspace_id == workspace_id)
                .order_by(TrainingRunRow.created_at.desc())
            )
            return list(rows)

    async def _update(self, run_id: str, finished: bool = False, **values: Any) -> TrainingRunRow:
        async with self._sessions() as session, session.begin():
            run = await session.get(TrainingRunRow, run_id, with_for_update=True)
            assert run is not None
            for key, value in values.items():
                setattr(run, key, value)
            if finished:
                run.finished_at = datetime.now(UTC)
        return run
