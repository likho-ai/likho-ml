"""Datasets and fine-tuning runs: real database and S3 (the stack's SeaweedFS), a stand-in launcher."""

import json
import threading
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, ClassVar

import grpc
import pytest
from likho.ml.v1 import ml_pb2

from likho_ml.db import make_engine, make_sessions
from likho_ml.events import NullPublisher
from likho_ml.finetune import Bucket, FineTuning
from likho_ml.ids import new_id
from likho_ml.models import TrainingExampleRow
from likho_ml.registry import Registry
from likho_ml.training import TrainingStore
from tests.conftest import Service

pytestmark = pytest.mark.integration


def bucket_of(service: Service) -> Bucket:
    s = service.settings
    return Bucket(s.s3_endpoint, s.s3_region, s.s3_access_key, s.s3_secret_key, s.s3_bucket_models)


def read_object(service: Service, uri: str, name: str) -> bytes:
    import boto3

    s = service.settings
    client = boto3.client(
        "s3",
        endpoint_url=s.s3_endpoint,
        region_name=s.s3_region,
        aws_access_key_id=s.s3_access_key,
        aws_secret_access_key=s.s3_secret_key,
    )
    key = uri.removeprefix(f"s3://{s.s3_bucket_models}/") + name
    return client.get_object(Bucket=s.s3_bucket_models, Key=key)["Body"].read()


def example(workspace_id: str, recording_id: str, index: int, layer: str, roman: str, at: int) -> TrainingExampleRow:
    return TrainingExampleRow(
        id=new_id("tex"),
        event_id=new_id("evt"),
        workspace_id=workspace_id,
        recording_id=recording_id,
        transcript_id=new_id("trn"),
        segment_index=index,
        layer=layer,
        before="galat",
        after=roman,
        start_seconds=float(index * 3),
        end_seconds=float(index * 3 + 2),
        text_script="नमस्ते",
        text_roman=roman,
        language="hi",
        user_id="usr_x",
        corrected_at=datetime(2026, 10, 7, 10, at, tzinfo=UTC),
    )


class Launcher(BaseHTTPRequestHandler):
    received: ClassVar[list[dict[str, Any]]] = []

    def do_POST(self) -> None:
        body = self.rfile.read(int(self.headers["Content-Length"]))
        Launcher.received.append(json.loads(body))
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"external_id": "gpu-job-7"}).encode())

    def log_message(self, *args: Any) -> None:
        pass


async def test_a_dataset_leaves_out_the_gold_set_and_a_run_waits_without_a_launcher(service: Service) -> None:
    stub, fake = service.stub, service.transcripts
    workspace_id = new_id("wsp")
    rec_train, rec_gold = new_id("rec"), new_id("rec")
    store = TrainingStore(make_sessions(make_engine(service.settings.database_url)))
    try:
        # One line corrected twice (the newest counts) and another line; the gold recording's line is left out.
        for row in (
            example(workspace_id, rec_train, 0, "roman", "namaste ji", 1),
            example(workspace_id, rec_train, 0, "script", "namaste jee", 2),
            example(workspace_id, rec_train, 1, "roman", "aapka order", 3),
            example(workspace_id, rec_gold, 0, "roman", "paise wapas", 4),
        ):
            assert await store.add(row)
        fake.put(new_id("trn"), rec_gold, [(0.0, 4.0, "पैसे वापस", "paise wapas")])
        await stub.AddToGoldSet(
            ml_pb2.AddToGoldSetRequest(workspace_id=workspace_id, recording_id=rec_gold, media_id="med")
        )

        dataset = (
            await stub.ExportDataset(ml_pb2.ExportDatasetRequest(workspace_id=workspace_id, user_id="usr_x"))
        ).dataset
        assert dataset.examples == 2 and dataset.held_out_recordings == 1 and dataset.audio_seconds == 4.0
        assert dataset.uri.startswith("s3://likho-models/datasets/dst_")
        lines = [json.loads(line) for line in read_object(service, dataset.uri, "manifest.jsonl").splitlines()]
        assert [(line["recording_id"], line["segment_index"], line["text_roman"]) for line in lines] == [
            (rec_train, 0, "namaste jee"),
            (rec_train, 1, "aapka order"),
        ]
        about = json.loads(read_object(service, dataset.uri, "dataset.json"))
        assert about["held_out_recordings"] == [rec_gold]

        base = (await stub.GetDefault(ml_pb2.GetDefaultRequest())).model
        run = (
            await stub.StartTrainingRun(
                ml_pb2.StartTrainingRunRequest(workspace_id=workspace_id, dataset_id=dataset.id, base_model_id=base.id)
            )
        ).run
        assert run.status == ml_pb2.TRAINING_RUN_STATUS_PENDING and run.launcher == ""

        # The job reports back: running, then completed with its weights - the model is registered.
        registry_id = f"faster-whisper/likho-test-{new_id('x')[-8:].lower()}"
        report = ml_pb2.ReportTrainingRunRequest
        running = await stub.ReportTrainingRun(
            report(run_id=run.id, status=ml_pb2.TRAINING_RUN_STATUS_RUNNING, external_id="j1")
        )
        assert running.run.status == ml_pb2.TRAINING_RUN_STATUS_RUNNING and running.run.external_id == "j1"
        done = (
            await stub.ReportTrainingRun(
                report(
                    run_id=run.id,
                    status=ml_pb2.TRAINING_RUN_STATUS_COMPLETED,
                    registry_id=registry_id,
                    artifact_uri="s3://likho-models/models/test/",
                )
            )
        ).run
        assert done.status == ml_pb2.TRAINING_RUN_STATUS_COMPLETED and done.model_id.startswith("mdl_")
        model = (await stub.GetModel(ml_pb2.GetModelRequest(model_id=done.model_id))).model
        assert model.registry_id == registry_id and model.base_model_id == base.id
        listed = (await stub.ListTrainingRuns(ml_pb2.ListTrainingRunsRequest(workspace_id=workspace_id))).runs
        assert [r.id for r in listed] == [run.id]

        with pytest.raises(grpc.aio.AioRpcError) as again:
            await stub.ReportTrainingRun(report(run_id=run.id, status=ml_pb2.TRAINING_RUN_STATUS_FAILED, error="x"))
        assert again.value.code() == grpc.StatusCode.FAILED_PRECONDITION
        await stub.RetireModel(ml_pb2.RetireModelRequest(model_id=model.id))
    finally:
        await store.delete_workspace(workspace_id)


async def test_a_run_goes_to_the_launcher(service: Service) -> None:
    server = HTTPServer(("127.0.0.1", 0), Launcher)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    sessions = make_sessions(make_engine(service.settings.database_url))
    workspace_id = new_id("wsp")
    store = TrainingStore(sessions)
    try:
        await store.add(example(workspace_id, new_id("rec"), 0, "roman", "theek hai", 1))
        tuning = FineTuning(
            sessions,
            Registry(sessions, NullPublisher()),
            bucket_of(service),
            launcher_url=f"http://127.0.0.1:{server.server_port}/runs",
            report_address="likho-ml:5080",
        )
        dataset = await tuning.export(workspace_id)
        base = await Registry(sessions, NullPublisher()).default()
        run = await tuning.start(workspace_id, dataset.id, base.id, "usr_x")
        assert run.status == "running" and run.external_id == "gpu-job-7"
        sent = Launcher.received[-1]
        assert sent["run_id"] == run.id and sent["dataset_uri"] == dataset.uri
        assert sent["base_model"]["registry_id"] == base.registry_id and sent["report_to"] == "likho-ml:5080"
    finally:
        server.shutdown()
        await store.delete_workspace(workspace_id)


async def test_what_fine_tuning_refuses(service: Service) -> None:
    stub = service.stub
    with pytest.raises(grpc.aio.AioRpcError) as nothing:
        await stub.ExportDataset(ml_pb2.ExportDatasetRequest(workspace_id=new_id("wsp")))
    assert nothing.value.code() == grpc.StatusCode.FAILED_PRECONDITION
    with pytest.raises(grpc.aio.AioRpcError) as missing:
        await stub.StartTrainingRun(
            ml_pb2.StartTrainingRunRequest(workspace_id=new_id("wsp"), dataset_id=new_id("dst"), base_model_id="x")
        )
    assert missing.value.code() == grpc.StatusCode.NOT_FOUND
