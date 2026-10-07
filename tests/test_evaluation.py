"""The gold set and evaluations: real database and event bus, a stand-in transcription service."""

import asyncio

import grpc
import pytest
from likho.ml.v1 import ml_pb2

from likho_ml.db import make_engine, make_sessions
from likho_ml.evaluation import Evaluations
from likho_ml.events import NullPublisher
from likho_ml.gold import GoldSet
from likho_ml.ids import new_id
from likho_ml.registry import Registry
from likho_ml.transcripts import TranscriptionClient
from tests.conftest import Events, Service

pytestmark = pytest.mark.integration

REFERENCE_A = [(0.0, 3.0, "नमस्ते जी।", "Namaste ji."), (3.0, 8.0, "आपका ऑर्डर कल आएगा", "aapka order kal aayega")]
REFERENCE_B = [(0.0, 4.0, "पैसे वापस मिलेंगे", "paise wapas milenge")]


async def finished(service: Service, evaluation_id: str) -> ml_pb2.Evaluation:
    for _ in range(100):
        reply = await service.stub.GetEvaluation(ml_pb2.GetEvaluationRequest(evaluation_id=evaluation_id))
        if reply.evaluation.status in (ml_pb2.EVALUATION_STATUS_COMPLETED, ml_pb2.EVALUATION_STATUS_FAILED):
            return reply.evaluation
        await asyncio.sleep(0.1)
    raise AssertionError(f"evaluation {evaluation_id} did not finish")


async def register(service: Service, name: str) -> ml_pb2.Model:
    registry_id = f"faster-whisper/{name}-{new_id('x')[-8:].lower()}"
    return (await service.stub.RegisterModel(ml_pb2.RegisterModelRequest(registry_id=registry_id))).model


async def cleanup(service: Service, workspace_id: str) -> None:
    sessions = make_sessions(make_engine(service.settings.database_url))
    transcripts = TranscriptionClient(service.settings.transcription_grpc_addr)
    gold = GoldSet(sessions, transcripts)
    await Evaluations(
        sessions, Registry(sessions, NullPublisher()), gold, transcripts, NullPublisher()
    ).delete_workspace(workspace_id)
    await gold.delete_workspace(workspace_id)
    await transcripts.close()


async def test_a_model_is_scored_on_the_gold_set(service: Service, events: Events) -> None:
    stub, fake = service.stub, service.transcripts
    workspace_id = new_id("wsp")
    rec_a, rec_b = new_id("rec"), new_id("rec")
    fake.put(new_id("trn"), rec_a, [(0.0, 3.0, "पुराना", "purana")], version=1)  # an older version
    latest_a = new_id("trn")
    fake.put(latest_a, rec_a, REFERENCE_A, version=2)
    fake.put(ref_b := new_id("trn"), rec_b, REFERENCE_B)
    model = await register(service, "candidate")
    # The model hears rec_a perfectly (but for punctuation and an "aapka/apka" spelling) and
    # gets one word of rec_b's three wrong in both layers.
    fake.hypotheses[(rec_a, model.registry_id)] = [
        (0.0, 3.0, "नमस्ते जी", "namaste ji"),
        (3.0, 8.0, "आपका ऑर्डर कल आएगा", "apka order kal aayega"),
    ]
    fake.hypotheses[(rec_b, model.registry_id)] = [(0.0, 4.0, "पैसे वापस मिलेगा", "paise wapas milega")]
    try:
        item = (
            await stub.AddToGoldSet(
                ml_pb2.AddToGoldSetRequest(
                    workspace_id=workspace_id, recording_id=rec_a, media_id="med_a", user_id="usr_x"
                )
            )
        ).item
        assert item.transcript_id == latest_a and item.transcript_version == 2, "the latest version is the reference"
        assert item.lines == 2 and item.audio_seconds == 8.0 and item.language == "hi"
        await stub.AddToGoldSet(
            ml_pb2.AddToGoldSetRequest(
                workspace_id=workspace_id, recording_id=rec_b, transcript_id=ref_b, media_id="med_b"
            )
        )
        listed = await stub.ListGoldSet(ml_pb2.ListGoldSetRequest(workspace_id=workspace_id))
        assert len(listed.items) == 2 and listed.audio_seconds == 12.0
        # The reference was copied: a later edit of the transcript does not move it.
        fake.put(latest_a, rec_a, [(0.0, 8.0, "कुछ और", "kuch aur")], version=2)

        started = await stub.StartEvaluation(
            ml_pb2.StartEvaluationRequest(workspace_id=workspace_id, model_id=model.id)
        )
        assert started.evaluation.status == ml_pb2.EVALUATION_STATUS_QUEUED and started.evaluation.items_total == 2
        evaluation = await finished(service, started.evaluation.id)
        assert evaluation.status == ml_pb2.EVALUATION_STATUS_COMPLETED, evaluation.error
        assert evaluation.items_done == 2 and len(evaluation.items) == 2
        # 1 wrong word of 9 reference words (6 + 3) over both recordings - a corpus rate - in both layers.
        assert evaluation.scores.wer_script == pytest.approx(1 / 9)
        assert evaluation.scores.wer_roman == pytest.approx(1 / 9)
        per_recording = {item.recording_id: item for item in evaluation.items}
        assert per_recording[rec_a].scores.wer_roman == 0.0, "aapka/apka and punctuation are not errors"
        assert per_recording[rec_b].scores.wer_script == pytest.approx(1 / 3)
        assert all(request.evaluation for request in fake.evaluations_asked), "nothing is kept by transcription"

        event = await events.wait_for("likho.evaluation.completed.v1", "evaluation_id", evaluation.id)
        assert event["data"]["items_scored"] == 2 and event["data"]["wer_script"] == pytest.approx(1 / 9)

        models = (await stub.ListModels(ml_pb2.ListModelsRequest())).models
        scored = next(m for m in models if m.id == model.id)
        assert scored.latest_evaluation_id == evaluation.id
        assert scored.latest_scores.wer_roman == pytest.approx(1 / 9)
        history = (await stub.ListEvaluations(ml_pb2.ListEvaluationsRequest(workspace_id=workspace_id))).evaluations
        assert [e.id for e in history] == [evaluation.id]

        await stub.RemoveFromGoldSet(ml_pb2.RemoveFromGoldSetRequest(workspace_id=workspace_id, recording_id=rec_b))
        assert len((await stub.ListGoldSet(ml_pb2.ListGoldSetRequest(workspace_id=workspace_id))).items) == 1
    finally:
        await cleanup(service, workspace_id)
        await stub.RetireModel(ml_pb2.RetireModelRequest(model_id=model.id))


async def test_evaluations_that_cannot_score_fail_with_a_reason(service: Service, events: Events) -> None:
    stub, fake = service.stub, service.transcripts
    workspace_id = new_id("wsp")
    model = await register(service, "candidate")
    broken = await register(service, "broken")
    try:
        empty = await stub.StartEvaluation(ml_pb2.StartEvaluationRequest(workspace_id=workspace_id, model_id=model.id))
        failed = await finished(service, empty.evaluation.id)
        assert failed.status == ml_pb2.EVALUATION_STATUS_FAILED and "no gold recordings" in failed.error
        event = await events.wait_for("likho.evaluation.failed.v1", "evaluation_id", failed.id)
        assert event["data"]["code"] == "empty_gold_set"

        rec = new_id("rec")
        fake.put(new_id("trn"), rec, REFERENCE_B)
        await stub.AddToGoldSet(ml_pb2.AddToGoldSetRequest(workspace_id=workspace_id, recording_id=rec, media_id="med"))
        cannot_load = await stub.StartEvaluation(
            ml_pb2.StartEvaluationRequest(workspace_id=workspace_id, model_id=broken.id)
        )
        failed = await finished(service, cannot_load.evaluation.id)
        assert failed.status == ml_pb2.EVALUATION_STATUS_FAILED
        assert (await events.wait_for("likho.evaluation.failed.v1", "evaluation_id", failed.id))["data"][
            "code"
        ] == "model_unavailable"

        unreadable = await stub.StartEvaluation(
            ml_pb2.StartEvaluationRequest(workspace_id=workspace_id, model_id=model.id)
        )
        failed = await finished(service, unreadable.evaluation.id)
        assert (await events.wait_for("likho.evaluation.failed.v1", "evaluation_id", failed.id))["data"][
            "code"
        ] == "all_failed"
        assert (
            "audio_unreadable"
            in (await stub.GetEvaluation(ml_pb2.GetEvaluationRequest(evaluation_id=failed.id)))
            .evaluation.items[0]
            .error
        )
    finally:
        await cleanup(service, workspace_id)
        for m in (model, broken):
            await stub.RetireModel(ml_pb2.RetireModelRequest(model_id=m.id))


async def test_what_the_gold_set_refuses(service: Service) -> None:
    stub = service.stub
    with pytest.raises(grpc.aio.AioRpcError) as no_audio:
        await stub.AddToGoldSet(ml_pb2.AddToGoldSetRequest(workspace_id=new_id("wsp"), recording_id=new_id("rec")))
    assert no_audio.value.code() == grpc.StatusCode.INVALID_ARGUMENT
    with pytest.raises(grpc.aio.AioRpcError) as unknown:
        await stub.AddToGoldSet(
            ml_pb2.AddToGoldSetRequest(workspace_id=new_id("wsp"), recording_id=new_id("rec"), media_id="med")
        )
    assert unknown.value.code() == grpc.StatusCode.NOT_FOUND
    with pytest.raises(grpc.aio.AioRpcError) as no_model:
        await stub.StartEvaluation(ml_pb2.StartEvaluationRequest(workspace_id=new_id("wsp"), model_id=new_id("mdl")))
    assert no_model.value.code() == grpc.StatusCode.NOT_FOUND
