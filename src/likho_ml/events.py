"""Publishes this service's events to NATS JetStream as CloudEvents.

The contracts are likho-contracts/events/likho.model.*.v1 and likho.evaluation.*.v1. A failed
publish is logged and does not undo the change that caused it: the registry is the truth, and
its readers (likho-transcription asks for the default now and then) do not depend on the event.
"""

import asyncio
import json
import logging
from datetime import UTC, datetime
from typing import Any, Protocol

from likho_ml.ids import new_id

log = logging.getLogger(__name__)

SOURCE = "likho-ml"

MODEL_REGISTERED = "likho.model.registered.v1"
MODEL_CHOSEN = "likho.model.chosen.v1"
EVALUATION_COMPLETED = "likho.evaluation.completed.v1"
EVALUATION_FAILED = "likho.evaluation.failed.v1"


def subject_of(event_type: str) -> str:
    """The NATS subject of an event type: the type without its version (likho.model.chosen)."""
    return event_type.rsplit(".", 1)[0]


def cloud_event(event_type: str, subject: str, data: dict[str, Any]) -> dict[str, Any]:
    return {
        "specversion": "1.0",
        "id": new_id("evt"),
        "source": SOURCE,
        "type": event_type,
        "time": datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
        "subject": subject,
        "datacontenttype": "application/json",
        "data": data,
    }


class Publisher(Protocol):
    async def publish(self, event_type: str, subject: str, data: dict[str, Any]) -> None: ...


class NullPublisher:
    """Used when no event bus is configured (unit tests)."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    async def publish(self, event_type: str, subject: str, data: dict[str, Any]) -> None:
        self.events.append(cloud_event(event_type, subject, data))


class NatsPublisher:
    def __init__(self, url: str) -> None:
        self._url = url
        self._nc: Any = None
        self._js: Any = None

    async def connect(self, timeout_seconds: float = 0.0) -> None:
        """Connects, trying again while NATS is not there yet, for `timeout_seconds` (0 = one try)."""
        import nats

        deadline = asyncio.get_running_loop().time() + timeout_seconds
        wait = 1.0
        while True:
            try:
                self._nc = await nats.connect(self._url, name=SOURCE, max_reconnect_attempts=-1, connect_timeout=5)
                self._js = self._nc.jetstream()
                return
            except Exception as error:
                if asyncio.get_running_loop().time() + wait > deadline:
                    raise
                log.warning("event bus not ready (%s); trying again in %.0f s", error, wait)
                await asyncio.sleep(wait)
                wait = min(wait * 2, 10.0)

    @property
    def connected(self) -> bool:
        return self._nc is not None and self._nc.is_connected

    @property
    def js(self) -> Any:
        """The JetStream context, for consumers; None while not connected."""
        return self._js

    async def close(self) -> None:
        if self._nc is not None:
            await self._nc.drain()
            self._nc = None

    async def publish(self, event_type: str, subject: str, data: dict[str, Any]) -> None:
        event = cloud_event(event_type, subject, data)
        if self._js is None:
            log.warning("event bus not connected; %s for %s not published", event_type, subject)
            return
        try:
            # The event id is the message id, so a retried publish is stored once.
            await self._js.publish(
                subject_of(event_type),
                json.dumps(event, ensure_ascii=False).encode(),
                headers={"Nats-Msg-Id": event["id"]},
                timeout=3,
            )
        except Exception:
            log.exception("could not publish %s for %s", event_type, subject)
