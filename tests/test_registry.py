"""The registry through gRPC, against the real database and event bus."""

import grpc
import pytest
from likho.ml.v1 import ml_pb2

from likho_ml.ids import new_id
from likho_ml.registry import RegistryError, parse_registry_id
from tests.conftest import Events, Service

pytestmark = pytest.mark.integration


def unique_registry_id() -> str:
    return f"faster-whisper/test-{new_id('x')[-10:].lower()}"


async def test_the_published_sizes_are_there_and_one_is_the_default(service: Service) -> None:
    models = (await service.stub.ListModels(ml_pb2.ListModelsRequest())).models
    ids = {model.registry_id for model in models}
    assert {"faster-whisper/turbo", "faster-whisper/large-v3"} <= ids
    assert models[0].is_default, "the default comes first"
    assert sum(model.is_default for model in models) == 1
    default = (await service.stub.GetDefault(ml_pb2.GetDefaultRequest())).model
    assert default.id == models[0].id
    turbo = next(model for model in models if model.registry_id == "faster-whisper/turbo")
    assert turbo.engine == "faster-whisper" and turbo.name == "turbo" and "turbo" in turbo.description


async def test_register_choose_and_retire(service: Service, events: Events) -> None:
    stub = service.stub
    before = (await stub.GetDefault(ml_pb2.GetDefaultRequest())).model
    registry_id = unique_registry_id()
    model = (
        await stub.RegisterModel(
            ml_pb2.RegisterModelRequest(
                registry_id=registry_id,
                description="fine-tuned on corrections",
                languages=["hi", "en"],
                artifact_uri="s3://likho-models/models/test/",
                base_model_id=before.id,
                user_id="usr_01JB7Z5K3M9Q2W4X6Y8A0C1E3G",
            )
        )
    ).model
    assert model.id.startswith("mdl_") and model.status == ml_pb2.MODEL_STATUS_AVAILABLE and not model.is_default
    registered = await events.wait_for("likho.model.registered.v1", "model_id", model.id)
    assert registered["data"]["base_model_id"] == before.id

    try:
        chosen = (await stub.SetDefault(ml_pb2.SetDefaultRequest(model_id=model.id, user_id="usr_x"))).model
        assert chosen.is_default
        assert (await stub.GetDefault(ml_pb2.GetDefaultRequest())).model.id == model.id
        event = await events.wait_for("likho.model.chosen.v1", "model_id", model.id)
        assert event["data"]["previous_registry_id"] == before.registry_id

        with pytest.raises(grpc.aio.AioRpcError) as refused:
            await stub.RetireModel(ml_pb2.RetireModelRequest(model_id=model.id))
        assert refused.value.code() == grpc.StatusCode.FAILED_PRECONDITION
    finally:
        await stub.SetDefault(ml_pb2.SetDefaultRequest(model_id=before.id))

    retired = (await stub.RetireModel(ml_pb2.RetireModelRequest(model_id=model.id))).model
    assert retired.status == ml_pb2.MODEL_STATUS_RETIRED
    listed = {m.id for m in (await stub.ListModels(ml_pb2.ListModelsRequest())).models}
    assert model.id not in listed
    every = {m.id for m in (await stub.ListModels(ml_pb2.ListModelsRequest(include_retired=True))).models}
    assert model.id in every
    with pytest.raises(grpc.aio.AioRpcError) as again:
        await stub.SetDefault(ml_pb2.SetDefaultRequest(model_id=model.id))
    assert again.value.code() == grpc.StatusCode.FAILED_PRECONDITION


async def test_what_the_registry_refuses(service: Service) -> None:
    stub = service.stub
    with pytest.raises(grpc.aio.AioRpcError) as bad:
        await stub.RegisterModel(ml_pb2.RegisterModelRequest(registry_id="no-slash"))
    assert bad.value.code() == grpc.StatusCode.INVALID_ARGUMENT
    with pytest.raises(grpc.aio.AioRpcError) as twice:
        await stub.RegisterModel(ml_pb2.RegisterModelRequest(registry_id="faster-whisper/turbo"))
    assert twice.value.code() == grpc.StatusCode.ALREADY_EXISTS
    with pytest.raises(grpc.aio.AioRpcError) as language:
        await stub.RegisterModel(ml_pb2.RegisterModelRequest(registry_id=unique_registry_id(), languages=["hindi"]))
    assert language.value.code() == grpc.StatusCode.INVALID_ARGUMENT
    with pytest.raises(grpc.aio.AioRpcError) as missing:
        await stub.GetModel(ml_pb2.GetModelRequest(model_id="mdl_01JB7Z5K3M9Q2W4X6Y8A0C1E3G"))
    assert missing.value.code() == grpc.StatusCode.NOT_FOUND
    by_name = (await stub.GetModel(ml_pb2.GetModelRequest(registry_id="faster-whisper/turbo"))).model
    assert by_name.registry_id == "faster-whisper/turbo"


async def test_a_call_of_a_later_step_says_so(service: Service) -> None:
    with pytest.raises(grpc.aio.AioRpcError) as later:
        await service.stub.ListGoldSet(ml_pb2.ListGoldSetRequest(workspace_id="wsp_x"))
    assert later.value.code() == grpc.StatusCode.UNIMPLEMENTED


def test_registry_ids() -> None:
    assert parse_registry_id("faster-whisper/turbo") == ("faster-whisper", "turbo")
    assert parse_registry_id(" faster-whisper/likho-2026.10 ") == ("faster-whisper", "likho-2026.10")
    for bad in ("turbo", "/turbo", "faster-whisper/", "Faster/turbo", "a/b/c", "a/-b"):
        with pytest.raises(RegistryError):
            parse_registry_id(bad)
