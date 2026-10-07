"""The model registry: which speech models exist and which one transcribes by default.

A model is named by its registry id, "<engine>/<name>". At the first start the engine's own
published sizes are registered (settings.seed_models) and the first of them becomes the default,
so a new installation transcribes as before until an admin chooses otherwise.
"""

import contextlib
import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from likho_ml.events import MODEL_CHOSEN, MODEL_REGISTERED, Publisher
from likho_ml.ids import new_id
from likho_ml.models import ModelRow

log = logging.getLogger(__name__)

REGISTRY_ID = re.compile(r"^(?P<engine>[a-z0-9][a-z0-9._-]*)/(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)$")
LANGUAGE = re.compile(r"^[a-z]{2,3}$")

# Descriptions of the engine's published sizes, for the admin screen.
KNOWN = {
    "faster-whisper/turbo": "Whisper large-v3 turbo (CTranslate2): the fast default, near large-v3 quality",
    "faster-whisper/large-v3": "Whisper large-v3 (CTranslate2): the most accurate published size, slowest",
    "faster-whisper/medium": "Whisper medium (CTranslate2): smaller and faster, less accurate on Hinglish",
    "faster-whisper/small": "Whisper small (CTranslate2): for a machine without much memory",
    "faster-whisper/base": "Whisper base (CTranslate2): for trying things out",
}


class RegistryError(Exception):
    """A request the registry refuses; `code` maps to a gRPC status."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code  # invalid | not_found | conflict | failed_precondition


def parse_registry_id(registry_id: str) -> tuple[str, str]:
    """(engine, name) of a registry id, or RegistryError."""
    match = REGISTRY_ID.match(registry_id.strip())
    if not match:
        raise RegistryError("invalid", f"'{registry_id}' is not a registry id: write it as <engine>/<name>")
    return match["engine"], match["name"]


@dataclass(frozen=True)
class NewModel:
    registry_id: str
    description: str = ""
    languages: tuple[str, ...] = ()
    artifact_uri: str = ""
    base_model_id: str = ""
    user_id: str = ""


class Registry:
    def __init__(self, sessions: async_sessionmaker[AsyncSession], publisher: Publisher) -> None:
        self._sessions = sessions
        self._publisher = publisher

    # ------------------------------------------------------------------------------ reading

    async def list_models(self, include_retired: bool = False) -> list[ModelRow]:
        """The default first, then by registry id."""
        async with self._sessions() as session:
            query = select(ModelRow)
            if not include_retired:
                query = query.where(ModelRow.status == "available")
            rows = (await session.scalars(query.order_by(ModelRow.is_default.desc(), ModelRow.registry_id))).all()
            return list(rows)

    async def get(self, model_id: str = "", registry_id: str = "") -> ModelRow:
        async with self._sessions() as session:
            if model_id:
                row = await session.get(ModelRow, model_id)
            elif registry_id:
                row = await session.scalar(select(ModelRow).where(ModelRow.registry_id == registry_id.strip()))
            else:
                raise RegistryError("invalid", "Name the model by its id or its registry id.")
            if row is None:
                raise RegistryError("not_found", f"No model {model_id or registry_id}.")
            return row

    async def default(self) -> ModelRow:
        async with self._sessions() as session:
            row = await session.scalar(select(ModelRow).where(ModelRow.is_default))
            if row is None:
                raise RegistryError("failed_precondition", "No model is the default yet.")
            return row

    # ------------------------------------------------------------------------------ changing

    async def register(self, new: NewModel) -> ModelRow:
        engine, name = parse_registry_id(new.registry_id)
        languages = [code.strip().lower() for code in new.languages if code.strip()]
        for code in languages:
            if not LANGUAGE.match(code):
                raise RegistryError("invalid", f"'{code}' is not a language code (ISO 639-1, e.g. hi).")
        if new.base_model_id:
            await self.get(model_id=new.base_model_id)
        row = ModelRow(
            id=new_id("mdl"),
            registry_id=f"{engine}/{name}",
            engine=engine,
            name=name,
            description=new.description.strip() or KNOWN.get(f"{engine}/{name}", ""),
            languages=languages,
            artifact_uri=new.artifact_uri.strip(),
            base_model_id=new.base_model_id,
            status="available",
            is_default=False,
            created_by=new.user_id,
            created_at=datetime.now(UTC),
        )
        try:
            async with self._sessions() as session, session.begin():
                session.add(row)
        except IntegrityError as error:
            raise RegistryError("conflict", f"{row.registry_id} is registered already.") from error
        log.info("registered %s (%s)", row.registry_id, row.id)
        data = {"model_id": row.id, "registry_id": row.registry_id, "engine": row.engine}
        if row.base_model_id:
            data["base_model_id"] = row.base_model_id
        if row.artifact_uri:
            data["artifact_uri"] = row.artifact_uri
        if row.created_by:
            data["user_id"] = row.created_by
        await self._publisher.publish(MODEL_REGISTERED, row.id, data)
        return row

    async def set_default(self, model_id: str, user_id: str = "") -> ModelRow:
        async with self._sessions() as session, session.begin():
            row = await session.get(ModelRow, model_id, with_for_update=True)
            if row is None:
                raise RegistryError("not_found", f"No model {model_id}.")
            if row.status != "available":
                raise RegistryError("failed_precondition", f"{row.registry_id} is retired; it cannot be the default.")
            if row.is_default:
                return row
            previous = await session.scalar(select(ModelRow).where(ModelRow.is_default).with_for_update())
            if previous is not None:
                previous.is_default = False
                await session.flush()  # the one-default index sees the old default go first
            row.is_default = True
            previous_registry_id = previous.registry_id if previous is not None else ""
        log.info("default model: %s (was %s)", row.registry_id, previous_registry_id or "none")
        data = {"model_id": row.id, "registry_id": row.registry_id, "previous_registry_id": previous_registry_id}
        if user_id:
            data["user_id"] = user_id
        await self._publisher.publish(MODEL_CHOSEN, row.id, data)
        return row

    async def retire(self, model_id: str, user_id: str = "") -> ModelRow:
        async with self._sessions() as session, session.begin():
            row = await session.get(ModelRow, model_id, with_for_update=True)
            if row is None:
                raise RegistryError("not_found", f"No model {model_id}.")
            if row.is_default:
                raise RegistryError("failed_precondition", "The default model cannot be retired; choose another first.")
            if row.status != "retired":
                row.status = "retired"
                row.retired_at = datetime.now(UTC)
        log.info("retired %s (by %s)", row.registry_id, user_id or "?")
        return row

    # ------------------------------------------------------------------------------ first start

    async def seed(self, registry_ids: list[str]) -> None:
        """Registers the engine's published sizes that are missing, and makes the first one the
        default when there is no default yet. Safe at every start, from several instances."""
        for registry_id in registry_ids:
            async with self._sessions() as session:
                exists = await session.scalar(select(ModelRow.id).where(ModelRow.registry_id == registry_id))
            if exists is None:
                try:
                    await self.register(NewModel(registry_id=registry_id))
                except RegistryError as error:
                    if error.code != "conflict":  # another instance registered it a moment ago
                        raise
        if not registry_ids:
            return
        async with self._sessions() as session:
            has_default = await session.scalar(select(ModelRow.id).where(ModelRow.is_default))
        if has_default is None:
            first = await self.get(registry_id=registry_ids[0])
            # Another instance may choose it at the same moment; the one-default index keeps one.
            with contextlib.suppress(IntegrityError):
                await self.set_default(first.id)
