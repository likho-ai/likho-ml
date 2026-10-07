"""Unit tests: no database, no event bus."""

import asyncio
import json
import re
from pathlib import Path

from likho_ml.events import MODEL_CHOSEN, NullPublisher, subject_of
from likho_ml.health import start_health_server
from likho_ml.ids import new_id

ULID = re.compile(r"^mdl_[0-9A-HJKMNP-TV-Z]{26}$")


def test_ids_are_prefixed_ulids_that_sort_by_time() -> None:
    first = new_id("mdl")
    second = new_id("mdl")
    assert ULID.match(first) and ULID.match(second)
    assert first[:14] <= second[:14]


def test_the_subject_is_the_type_without_its_version() -> None:
    assert subject_of(MODEL_CHOSEN) == "likho.model.chosen"
    assert subject_of("likho.evaluation.completed.v1") == "likho.evaluation.completed"


async def test_events_are_cloud_events() -> None:
    publisher = NullPublisher()
    await publisher.publish(MODEL_CHOSEN, "mdl_x", {"model_id": "mdl_x"})
    event = publisher.events[0]
    assert event["type"] == MODEL_CHOSEN and event["source"] == "likho-ml" and event["subject"] == "mdl_x"
    assert event["id"].startswith("evt_") and event["time"].endswith("Z")
    json.dumps(event)


async def _get(port: int, path: str) -> tuple[str, str]:
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    writer.write(f"GET {path} HTTP/1.1\r\nHost: x\r\n\r\n".encode())
    await writer.drain()
    response = (await reader.read()).decode()
    writer.close()
    status, _, rest = response.partition("\r\n")
    return status, rest.split("\r\n\r\n", 1)[1]


async def test_health_ready_and_metrics() -> None:
    ready = True

    async def check() -> bool:
        return ready

    server = await start_health_server(0, check, lambda: ("text/plain", b"likho_ml_requests 0\n"))
    port = server.sockets[0].getsockname()[1]
    try:
        assert (await _get(port, "/healthz"))[0].endswith("200 OK")
        assert (await _get(port, "/readyz"))[0].endswith("200 OK")
        ready = False
        assert (await _get(port, "/readyz"))[0].endswith("503 Service Unavailable")
        assert "likho_ml_requests" in (await _get(port, "/metrics"))[1]
        assert (await _get(port, "/nothing"))[0].endswith("404 Not Found")
    finally:
        server.close()
        await server.wait_closed()


def test_the_example_events_match_the_service() -> None:
    """The event types this service publishes are the ones the contracts describe."""
    contracts = Path(__file__).parent / "contracts"
    for name in ("likho.model.registered.v1", "likho.model.chosen.v1", "likho.evaluation.completed.v1"):
        assert (contracts / f"{name}.schema.json").exists(), name
