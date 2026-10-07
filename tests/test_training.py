"""Corrections become training examples: real event bus and database, a stand-in transcription service."""

import asyncio
import json
from datetime import UTC, datetime

import grpc
import jsonschema
import pytest
from likho.ml.v1 import ml_pb2

from likho_ml.db import make_engine, make_sessions
from likho_ml.ids import new_id
from likho_ml.training import TrainingStore
from tests.conftest import Service

pytestmark = pytest.mark.integration

SCHEMAS = __import__("pathlib").Path(__file__).parent / "contracts"


def correction(workspace_id: str, recording_id: str, transcript_id: str, index: int, layer: str, after: str) -> dict:
    event = {
        "specversion": "1.0",
        "id": new_id("evt"),
        "source": "likho-transcription",
        "type": "likho.transcript.corrected.v1",
        "time": datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
        "subject": recording_id,
        "datacontenttype": "application/json",
        "data": {
            "transcript_id": transcript_id,
            "recording_id": recording_id,
            "workspace_id": workspace_id,
            "user_id": new_id("usr"),
            "segment_index": index,
            "layer": layer,
            "before": "galat",
            "after": after,
        },
    }
    schema = json.loads((SCHEMAS / "likho.transcript.corrected.v1.schema.json").read_text(encoding="utf-8"))
    jsonschema.validate(event["data"], schema)
    return event


async def publish(service: Service, event: dict) -> None:
    import nats

    connection = await nats.connect(service.settings.nats_url)
    try:
        await connection.jetstream().publish(
            "likho.transcript.corrected", json.dumps(event).encode(), headers={"Nats-Msg-Id": event["id"]}
        )
    finally:
        await connection.close()


async def stats_when(service: Service, workspace_id: str, examples: int) -> ml_pb2.GetTrainingStatsResponse:
    for _ in range(100):
        stats = await service.stub.GetTrainingStats(ml_pb2.GetTrainingStatsRequest(workspace_id=workspace_id))
        if stats.examples == examples:
            return stats
        await asyncio.sleep(0.1)
    raise AssertionError(f"expected {examples} examples, have {stats.examples}")


async def test_a_correction_becomes_a_training_example(service: Service) -> None:
    workspace_id, recording_id = new_id("wsp"), new_id("rec")
    corrected = new_id("trn")
    service.transcripts.put(
        corrected,
        recording_id,
        [(0.0, 2.5, "नमस्ते जी", "namaste ji"), (2.5, 6.0, "आपका ऑर्डर कल आएगा", "aapka order kal aayega")],
    )
    store = TrainingStore(make_sessions(make_engine(service.settings.database_url)))
    try:
        first = correction(workspace_id, recording_id, corrected, 1, "roman", "aapka order kal aayega")
        await publish(service, first)
        stats = await stats_when(service, workspace_id, 1)
        assert stats.roman_examples == 1 and stats.script_examples == 0 and stats.recordings == 1
        assert stats.audio_seconds == pytest.approx(3.5)
        assert stats.last_example_at.seconds > 0

        # The same event again (a redelivery) changes nothing; the same line again counts once.
        await publish(service, first)
        await publish(
            service, correction(workspace_id, recording_id, corrected, 1, "roman", "aapka order kal aa jayega")
        )
        # Another layer of the same line is another example, but not more audio.
        await publish(service, correction(workspace_id, recording_id, corrected, 1, "script", "आपका ऑर्डर कल आ जाएगा"))
        stats = await stats_when(service, workspace_id, 2)
        assert stats.script_examples == 1 and stats.roman_examples == 1
        assert stats.audio_seconds == pytest.approx(3.5)

        # A correction of a transcript transcription no longer has is skipped, not retried for ever.
        await publish(service, correction(workspace_id, recording_id, new_id("trn"), 0, "roman", "x"))
        await asyncio.sleep(1)
        assert (await stats_when(service, workspace_id, 2)).recordings == 1
    finally:
        await store.delete_workspace(workspace_id)


async def test_stats_need_a_workspace(service: Service) -> None:
    with pytest.raises(grpc.aio.AioRpcError) as missing:
        await service.stub.GetTrainingStats(ml_pb2.GetTrainingStatsRequest())
    assert missing.value.code() == grpc.StatusCode.INVALID_ARGUMENT
    empty = await service.stub.GetTrainingStats(ml_pb2.GetTrainingStatsRequest(workspace_id=new_id("wsp")))
    assert empty.examples == 0 and empty.audio_seconds == 0
