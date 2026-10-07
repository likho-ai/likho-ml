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
from likho.ml.v1 import ml_pb2_grpc

from likho_ml.__main__ import serve
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


@dataclass
class Service:
    settings: Settings
    stub: ml_pb2_grpc.MlServiceStub
    channel: grpc.aio.Channel


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def service() -> AsyncIterator[Service]:
    settings = Settings(grpc_port=_free_port(), http_port=_free_port(), log_level="WARNING")
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

    yield Service(settings, ml_pb2_grpc.MlServiceStub(channel), channel)

    await channel.close()
    stop.set()
    await asyncio.wait_for(task, timeout=30)


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
