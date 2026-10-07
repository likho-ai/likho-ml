"""likho.ml.v1.MlService over gRPC. The calls of later steps answer UNIMPLEMENTED until they land."""

import logging
from collections.abc import Awaitable, Callable
from typing import Any

import grpc
from google.protobuf.timestamp_pb2 import Timestamp
from likho.ml.v1 import ml_pb2, ml_pb2_grpc

from likho_ml.models import ModelRow
from likho_ml.registry import NewModel, Registry, RegistryError
from likho_ml.training import TrainingStore

log = logging.getLogger(__name__)

STATUS = {
    "invalid": grpc.StatusCode.INVALID_ARGUMENT,
    "not_found": grpc.StatusCode.NOT_FOUND,
    "conflict": grpc.StatusCode.ALREADY_EXISTS,
    "failed_precondition": grpc.StatusCode.FAILED_PRECONDITION,
}

MODEL_STATUS = {"available": ml_pb2.MODEL_STATUS_AVAILABLE, "retired": ml_pb2.MODEL_STATUS_RETIRED}


def timestamp(value: Any) -> Timestamp:
    stamp = Timestamp()
    if value is not None:
        stamp.FromDatetime(value)
    return stamp


def model_message(row: ModelRow) -> ml_pb2.Model:
    return ml_pb2.Model(
        id=row.id,
        registry_id=row.registry_id,
        engine=row.engine,
        name=row.name,
        description=row.description,
        languages=list(row.languages or []),
        artifact_uri=row.artifact_uri,
        base_model_id=row.base_model_id,
        status=MODEL_STATUS.get(row.status, ml_pb2.MODEL_STATUS_UNSPECIFIED),
        is_default=row.is_default,
        created_by=row.created_by,
        created_at=timestamp(row.created_at),
    )


async def answer[T](context: grpc.aio.ServicerContext, call: Callable[[], Awaitable[T]]) -> T:
    """Runs a call; a RegistryError becomes the matching gRPC status with its message."""
    try:
        return await call()
    except RegistryError as error:
        await context.abort(STATUS.get(error.code, grpc.StatusCode.INTERNAL), str(error))
        raise  # abort raises; this satisfies the type checker


class MlServicer(ml_pb2_grpc.MlServiceServicer):
    def __init__(self, registry: Registry, training: TrainingStore) -> None:
        self._registry = registry
        self._training = training

    async def GetTrainingStats(self, request: ml_pb2.GetTrainingStatsRequest, context: grpc.aio.ServicerContext) -> Any:
        if not request.workspace_id:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Name the workspace.")
        stats = await self._training.stats(request.workspace_id)
        reply = ml_pb2.GetTrainingStatsResponse(
            examples=stats.examples,
            script_examples=stats.script_examples,
            roman_examples=stats.roman_examples,
            recordings=stats.recordings,
            audio_seconds=stats.audio_seconds,
        )
        if stats.last_example_at is not None:
            reply.last_example_at.CopyFrom(timestamp(stats.last_example_at))
        return reply

    async def ListModels(self, request: ml_pb2.ListModelsRequest, context: grpc.aio.ServicerContext) -> Any:
        rows = await self._registry.list_models(include_retired=request.include_retired)
        return ml_pb2.ListModelsResponse(models=[model_message(row) for row in rows])

    async def GetModel(self, request: ml_pb2.GetModelRequest, context: grpc.aio.ServicerContext) -> Any:
        row = await answer(context, lambda: self._registry.get(request.model_id, request.registry_id))
        return ml_pb2.GetModelResponse(model=model_message(row))

    async def GetDefault(self, request: ml_pb2.GetDefaultRequest, context: grpc.aio.ServicerContext) -> Any:
        row = await answer(context, self._registry.default)
        return ml_pb2.GetDefaultResponse(model=model_message(row))

    async def RegisterModel(self, request: ml_pb2.RegisterModelRequest, context: grpc.aio.ServicerContext) -> Any:
        new = NewModel(
            registry_id=request.registry_id,
            description=request.description,
            languages=tuple(request.languages),
            artifact_uri=request.artifact_uri,
            base_model_id=request.base_model_id,
            user_id=request.user_id,
        )
        row = await answer(context, lambda: self._registry.register(new))
        return ml_pb2.RegisterModelResponse(model=model_message(row))

    async def SetDefault(self, request: ml_pb2.SetDefaultRequest, context: grpc.aio.ServicerContext) -> Any:
        row = await answer(context, lambda: self._registry.set_default(request.model_id, request.user_id))
        return ml_pb2.SetDefaultResponse(model=model_message(row))

    async def RetireModel(self, request: ml_pb2.RetireModelRequest, context: grpc.aio.ServicerContext) -> Any:
        row = await answer(context, lambda: self._registry.retire(request.model_id, request.user_id))
        return ml_pb2.RetireModelResponse(model=model_message(row))
