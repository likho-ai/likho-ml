"""likho.ml.v1.MlService over gRPC. The calls of later steps answer UNIMPLEMENTED until they land."""

import logging
from collections.abc import Awaitable, Callable
from typing import Any

import grpc
from google.protobuf.timestamp_pb2 import Timestamp
from likho.ml.v1 import ml_pb2, ml_pb2_grpc

from likho_ml.evaluation import Evaluations
from likho_ml.gold import GoldSet
from likho_ml.models import EvaluationItemRow, EvaluationRow, GoldItemRow, ModelRow
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
EVALUATION_STATUS = {
    "queued": ml_pb2.EVALUATION_STATUS_QUEUED,
    "running": ml_pb2.EVALUATION_STATUS_RUNNING,
    "completed": ml_pb2.EVALUATION_STATUS_COMPLETED,
    "failed": ml_pb2.EVALUATION_STATUS_FAILED,
}


def timestamp(value: Any) -> Timestamp:
    stamp = Timestamp()
    if value is not None:
        stamp.FromDatetime(value)
    return stamp


def scores_of(row: EvaluationRow | EvaluationItemRow) -> ml_pb2.Scores:
    return ml_pb2.Scores(
        wer_script=row.wer_script, cer_script=row.cer_script, wer_roman=row.wer_roman, cer_roman=row.cer_roman
    )


def model_message(row: ModelRow, latest: EvaluationRow | None = None) -> ml_pb2.Model:
    model = ml_pb2.Model(
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
    if latest is not None:
        model.latest_evaluation_id = latest.id
        model.latest_scores.CopyFrom(scores_of(latest))
    return model


def gold_message(row: GoldItemRow) -> ml_pb2.GoldItem:
    return ml_pb2.GoldItem(
        id=row.id,
        workspace_id=row.workspace_id,
        recording_id=row.recording_id,
        transcript_id=row.transcript_id,
        transcript_version=row.transcript_version,
        language=row.language,
        audio_seconds=row.audio_seconds,
        lines=row.lines,
        added_by=row.added_by,
        added_at=timestamp(row.added_at),
        media_id=row.media_id,
    )


def evaluation_message(row: EvaluationRow, items: list[EvaluationItemRow] | None = None) -> ml_pb2.Evaluation:
    message = ml_pb2.Evaluation(
        id=row.id,
        workspace_id=row.workspace_id,
        model_id=row.model_id,
        registry_id=row.registry_id,
        status=EVALUATION_STATUS.get(row.status, ml_pb2.EVALUATION_STATUS_UNSPECIFIED),
        scores=scores_of(row),
        items_total=row.items_total,
        items_done=row.items_done,
        audio_seconds=row.audio_seconds,
        realtime_factor=row.audio_seconds / row.elapsed_seconds if row.elapsed_seconds else 0.0,
        error=row.error,
        started_by=row.started_by,
        created_at=timestamp(row.created_at),
    )
    if row.finished_at is not None:
        message.finished_at.CopyFrom(timestamp(row.finished_at))
    for item in items or []:
        message.items.append(
            ml_pb2.EvaluationItem(
                recording_id=item.recording_id,
                reference_transcript_id=item.reference_transcript_id,
                scores=scores_of(item),
                words=item.words,
                audio_seconds=item.audio_seconds,
                elapsed_seconds=item.elapsed_seconds,
                error=item.error,
            )
        )
    return message


async def answer[T](context: grpc.aio.ServicerContext, call: Callable[[], Awaitable[T]]) -> T:
    """Runs a call; a RegistryError becomes the matching gRPC status with its message."""
    try:
        return await call()
    except RegistryError as error:
        await context.abort(STATUS.get(error.code, grpc.StatusCode.INTERNAL), str(error))
        raise  # abort raises; this satisfies the type checker


class MlServicer(ml_pb2_grpc.MlServiceServicer):
    def __init__(self, registry: Registry, training: TrainingStore, gold: GoldSet, evaluations: Evaluations) -> None:
        self._registry = registry
        self._training = training
        self._gold = gold
        self._evaluations = evaluations

    # ------------------------------------------------------------------------------ models

    async def ListModels(self, request: ml_pb2.ListModelsRequest, context: grpc.aio.ServicerContext) -> Any:
        rows = await self._registry.list_models(include_retired=request.include_retired)
        latest = await self._evaluations.latest_scores()
        return ml_pb2.ListModelsResponse(models=[model_message(row, latest.get(row.id)) for row in rows])

    async def GetModel(self, request: ml_pb2.GetModelRequest, context: grpc.aio.ServicerContext) -> Any:
        row = await answer(context, lambda: self._registry.get(request.model_id, request.registry_id))
        latest = await self._evaluations.latest_scores()
        return ml_pb2.GetModelResponse(model=model_message(row, latest.get(row.id)))

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

    # ------------------------------------------------------------------------------ gold set

    async def AddToGoldSet(self, request: ml_pb2.AddToGoldSetRequest, context: grpc.aio.ServicerContext) -> Any:
        row = await answer(
            context,
            lambda: self._gold.add(
                request.workspace_id, request.recording_id, request.media_id, request.transcript_id, request.user_id
            ),
        )
        return ml_pb2.AddToGoldSetResponse(item=gold_message(row))

    async def RemoveFromGoldSet(
        self, request: ml_pb2.RemoveFromGoldSetRequest, context: grpc.aio.ServicerContext
    ) -> Any:
        await self._gold.remove(request.workspace_id, request.recording_id)
        return ml_pb2.RemoveFromGoldSetResponse()

    async def ListGoldSet(self, request: ml_pb2.ListGoldSetRequest, context: grpc.aio.ServicerContext) -> Any:
        items = await self._gold.items(request.workspace_id)
        return ml_pb2.ListGoldSetResponse(
            items=[gold_message(item) for item in items], audio_seconds=sum(item.audio_seconds for item in items)
        )

    # ------------------------------------------------------------------------------ evaluations

    async def StartEvaluation(self, request: ml_pb2.StartEvaluationRequest, context: grpc.aio.ServicerContext) -> Any:
        row = await answer(
            context, lambda: self._evaluations.start(request.workspace_id, request.model_id, request.user_id)
        )
        return ml_pb2.StartEvaluationResponse(evaluation=evaluation_message(row))

    async def GetEvaluation(self, request: ml_pb2.GetEvaluationRequest, context: grpc.aio.ServicerContext) -> Any:
        row, items = await answer(context, lambda: self._evaluations.get(request.evaluation_id))
        return ml_pb2.GetEvaluationResponse(evaluation=evaluation_message(row, items))

    async def ListEvaluations(self, request: ml_pb2.ListEvaluationsRequest, context: grpc.aio.ServicerContext) -> Any:
        rows = await self._evaluations.list_evaluations(request.workspace_id, request.model_id, request.limit)
        return ml_pb2.ListEvaluationsResponse(evaluations=[evaluation_message(row) for row in rows])

    # ------------------------------------------------------------------------------ training

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
