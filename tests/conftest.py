"""Fixtures for the integration tests: a real service on free ports against the likho-infra stack.

Start the stack first:  likho-infra> bash scripts/up.sh   (or .\\stack.ps1 up)
Without it these tests are skipped locally; with LIKHO_REQUIRE_STACK=1 (set in CI) they fail instead.
"""

import asyncio
import json
import os
import socket
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import grpc
import pytest
import pytest_asyncio
from likho.common.v1 import common_pb2
from likho.ml.v1 import ml_pb2_grpc
from likho.transcription.v1 import transcription_pb2, transcription_pb2_grpc

from likho_ml.__main__ import serve
from likho_ml.ids import new_id
from likho_ml.settings import Settings


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _reachable(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


class FakeTranscripts(transcription_pb2_grpc.TranscriptionServiceServicer):
    """A stand-in for likho-transcription: GetTranscript answers the transcripts a test put here."""

    def __init__(self) -> None:
        self.transcripts: dict[str, transcription_pb2.Transcript] = {}
        # (recording, model registry id) -> the lines that model "hears", for evaluations.
        self.hypotheses: dict[tuple[str, str], list[tuple[float, float, str, str]]] = {}
        self.evaluations_asked: list[transcription_pb2.TranscribeRequest] = []

    @staticmethod
    def _transcript(
        transcript_id: str, recording_id: str, version: int, lines: list[tuple[float, float, str, str]]
    ) -> transcription_pb2.Transcript:
        return transcription_pb2.Transcript(
            id=transcript_id,
            recording_id=recording_id,
            version=version,
            language=common_pb2.LanguageDetection(detected="hi", decoded_as="hi"),
            segments=[
                common_pb2.Segment(index=i, start_seconds=start, end_seconds=end, text_script=script, text_roman=roman)
                for i, (start, end, script, roman) in enumerate(lines)
            ],
            stats=transcription_pb2.TranscriptStats(
                audio_seconds=max((end for _, end, _, _ in lines), default=0.0), elapsed_seconds=1.0
            ),
        )

    def put(
        self, transcript_id: str, recording_id: str, lines: list[tuple[float, float, str, str]], version: int = 2
    ) -> None:
        self.transcripts[transcript_id] = self._transcript(transcript_id, recording_id, version, lines)

    async def GetTranscript(self, request: Any, context: grpc.aio.ServicerContext) -> Any:  # noqa: N802
        if request.id not in self.transcripts:
            await context.abort(grpc.StatusCode.NOT_FOUND, f"no transcript {request.id}")
        return transcription_pb2.GetTranscriptResponse(transcript=self.transcripts[request.id])

    async def ListTranscripts(self, request: Any, context: grpc.aio.ServicerContext) -> Any:  # noqa: N802
        found = [t for t in self.transcripts.values() if t.recording_id == request.recording_id]
        listed = [transcription_pb2.Transcript(id=t.id, recording_id=t.recording_id, version=t.version) for t in found]
        return transcription_pb2.ListTranscriptsResponse(transcripts=listed)

    async def Transcribe(self, request: Any, context: grpc.aio.ServicerContext) -> Any:  # noqa: N802
        self.evaluations_asked.append(request)
        if "broken" in request.model_registry_id:
            await context.abort(
                grpc.StatusCode.FAILED_PRECONDITION, f"model_unavailable: {request.model_registry_id} cannot be loaded"
            )
        lines = self.hypotheses.get((request.recording_id, request.model_registry_id))
        if lines is None:
            await context.abort(grpc.StatusCode.INTERNAL, "audio_unreadable: the audio file could not be read")
        transcript = self._transcript(new_id("trn"), request.recording_id, 0, lines or [])
        yield transcription_pb2.TranscribeResponse(started=transcription_pb2.TranscribeStarted(audio_seconds=10))
        for segment in transcript.segments:
            yield transcription_pb2.TranscribeResponse(segment=segment)
        yield transcription_pb2.TranscribeResponse(completed=transcript)


@dataclass
class Service:
    settings: Settings
    stub: ml_pb2_grpc.MlServiceStub
    channel: grpc.aio.Channel
    transcripts: FakeTranscripts


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def service() -> AsyncIterator[Service]:
    fake = FakeTranscripts()
    fake_server = grpc.aio.server()
    transcription_pb2_grpc.add_TranscriptionServiceServicer_to_server(fake, fake_server)
    fake_port = fake_server.add_insecure_port("127.0.0.1:0")
    await fake_server.start()
    settings = Settings(
        grpc_port=_free_port(),
        http_port=_free_port(),
        log_level="WARNING",
        transcription_grpc_addr=f"127.0.0.1:{fake_port}",
        corrected_durable="test-" + new_id("ml")[-12:].lower(),  # this run's own consumer
        corrected_start="new",
    )
    # postgresql+asyncpg://user:pass@host:port/db  and  nats://host:port
    db_host, db_port = settings.database_url.rsplit("@", 1)[1].split("/", 1)[0].split(":")
    nats_host, nats_port = settings.nats_url.split("//", 1)[1].split(":")
    missing = [
        name
        for name, host, port in (("PostgreSQL", db_host, db_port), ("NATS", nats_host, nats_port))
        if not _reachable(host, int(port))
    ]
    if missing:
        message = f"{' and '.join(missing)} not reachable; start the likho-infra stack"
        if os.environ.get("LIKHO_REQUIRE_STACK") == "1":
            pytest.fail(message)
        pytest.skip(message)

    stop = asyncio.Event()
    task = asyncio.create_task(serve(settings, stop))
    channel = grpc.aio.insecure_channel(f"127.0.0.1:{settings.grpc_port}")
    try:
        await asyncio.wait_for(channel.channel_ready(), timeout=30)
    except TimeoutError:
        stop.set()
        await task  # surfaces the start-up error
        raise

    yield Service(settings, ml_pb2_grpc.MlServiceStub(channel), channel, fake)

    await channel.close()
    stop.set()
    await asyncio.wait_for(task, timeout=30)
    await fake_server.stop(grace=1)
    # The durable consumer would otherwise stay on the server.
    import nats

    connection = await nats.connect(settings.nats_url)
    try:
        await connection.jetstream().delete_consumer("LIKHO_KEEP", settings.corrected_durable)
    except Exception:
        pass
    finally:
        await connection.close()


class Events:
    """The events published on some subjects while a test runs (a plain NATS subscription)."""

    def __init__(self) -> None:
        self.received: list[dict[str, Any]] = []

    async def wait_for(self, event_type: str, key: str, value: str, within: float = 5) -> dict[str, Any]:
        async def find() -> dict[str, Any]:
            while True:
                for event in self.received:
                    if event["type"] == event_type and event["data"].get(key) == value:
                        return event
                await asyncio.sleep(0.05)

        return await asyncio.wait_for(find(), within)


@pytest_asyncio.fixture(loop_scope="session")
async def events(service: Service) -> AsyncIterator[Events]:
    import nats

    collected = Events()
    connection = await nats.connect(service.settings.nats_url)

    async def keep(message: Any) -> None:
        collected.received.append(json.loads(message.data))

    await connection.subscribe("likho.model.*", cb=keep)
    await connection.subscribe("likho.evaluation.*", cb=keep)
    yield collected
    await connection.close()
