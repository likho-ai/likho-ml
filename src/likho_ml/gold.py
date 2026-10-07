"""A workspace's gold set: recordings whose transcript a person has checked line by line."""

from datetime import UTC, datetime

from sqlalchemy import delete, func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from likho_ml.ids import new_id
from likho_ml.models import GoldItemRow
from likho_ml.registry import RegistryError
from likho_ml.transcripts import LineNotFoundError, TranscriptionClient


class GoldSet:
    def __init__(self, sessions: async_sessionmaker[AsyncSession], transcripts: TranscriptionClient) -> None:
        self._sessions = sessions
        self._transcripts = transcripts

    async def add(
        self, workspace_id: str, recording_id: str, media_id: str, transcript_id: str = "", user_id: str = ""
    ) -> GoldItemRow:
        """Adds the recording with its transcript (the latest when none is named) as the reference.
        Adding it again takes the newer text."""
        for name, value in (("workspace", workspace_id), ("recording", recording_id), ("audio (media_id)", media_id)):
            if not value:
                raise RegistryError("invalid", f"Name the {name}.")
        try:
            text = await self._transcripts.transcript(transcript_id=transcript_id, recording_id=recording_id)
        except LineNotFoundError as missing:
            raise RegistryError("not_found", str(missing)) from missing
        if not text.roman.strip() and not text.script.strip():
            raise RegistryError("failed_precondition", "That transcript has no text to compare with.")
        values = {
            "id": new_id("gld"),
            "workspace_id": workspace_id,
            "recording_id": recording_id,
            "media_id": media_id,
            "transcript_id": text.transcript_id,
            "transcript_version": text.version,
            "language": text.language,
            "audio_seconds": text.audio_seconds,
            "lines": text.lines,
            "reference_script": text.script,
            "reference_roman": text.roman,
            "added_by": user_id,
            "added_at": datetime.now(UTC),
        }
        refreshed = {k: v for k, v in values.items() if k not in ("id", "workspace_id", "recording_id")}
        statement = (
            insert(GoldItemRow)
            .values(values)
            .on_conflict_do_update(index_elements=["workspace_id", "recording_id"], set_=refreshed)
            .returning(GoldItemRow)
        )
        async with self._sessions() as session, session.begin():
            row = (await session.execute(statement)).scalar_one()
        return row

    async def remove(self, workspace_id: str, recording_id: str) -> None:
        async with self._sessions() as session, session.begin():
            await session.execute(
                delete(GoldItemRow).where(
                    GoldItemRow.workspace_id == workspace_id, GoldItemRow.recording_id == recording_id
                )
            )

    async def items(self, workspace_id: str) -> list[GoldItemRow]:
        async with self._sessions() as session:
            rows = await session.scalars(
                select(GoldItemRow).where(GoldItemRow.workspace_id == workspace_id).order_by(GoldItemRow.added_at)
            )
            return list(rows)

    async def audio_seconds(self, workspace_id: str) -> float:
        async with self._sessions() as session:
            total = await session.scalar(
                select(func.coalesce(func.sum(GoldItemRow.audio_seconds), 0.0)).where(
                    GoldItemRow.workspace_id == workspace_id
                )
            )
            return float(total or 0.0)

    async def delete_workspace(self, workspace_id: str) -> None:
        async with self._sessions() as session, session.begin():
            await session.execute(delete(GoldItemRow).where(GoldItemRow.workspace_id == workspace_id))
