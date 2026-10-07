"""The transcription service, as this service needs it: a line of a transcript version."""

from dataclasses import dataclass

import grpc
from likho.transcription.v1 import transcription_pb2, transcription_pb2_grpc


@dataclass(frozen=True)
class Line:
    start_seconds: float
    end_seconds: float
    text_script: str
    text_roman: str
    language: str


class LineNotFoundError(Exception):
    """The transcript or the line does not exist (a deleted recording): nothing to learn from."""


class TranscriptionClient:
    def __init__(self, address: str, timeout_seconds: float = 10.0) -> None:
        self._channel = grpc.aio.insecure_channel(address)
        self._stub = transcription_pb2_grpc.TranscriptionServiceStub(self._channel)
        self._timeout = timeout_seconds

    async def line(self, transcript_id: str, index: int) -> Line:
        """A line of a transcript version. LineNotFoundError when it is gone; other errors are transient."""
        try:
            reply = await self._stub.GetTranscript(
                transcription_pb2.GetTranscriptRequest(id=transcript_id), timeout=self._timeout
            )
        except grpc.aio.AioRpcError as error:
            if error.code() == grpc.StatusCode.NOT_FOUND:
                raise LineNotFoundError(f"transcript {transcript_id} not found") from error
            raise
        transcript = reply.transcript
        for segment in transcript.segments:
            if segment.index == index:
                return Line(
                    start_seconds=segment.start_seconds,
                    end_seconds=segment.end_seconds,
                    text_script=segment.text_script,
                    text_roman=segment.text_roman,
                    language=transcript.language.decoded_as or transcript.language.detected,
                )
        raise LineNotFoundError(f"transcript {transcript_id} has no line {index}")

    async def transcript(self, transcript_id: str = "", recording_id: str = "") -> "Text":
        """A transcript version (or a recording's latest) as whole texts of both layers."""
        if not transcript_id:
            listed = await self._stub.ListTranscripts(
                transcription_pb2.ListTranscriptsRequest(recording_id=recording_id), timeout=self._timeout
            )
            if not listed.transcripts:
                raise LineNotFoundError(f"recording {recording_id} has no transcript")
            transcript_id = max(listed.transcripts, key=lambda t: t.version).id
        try:
            reply = await self._stub.GetTranscript(
                transcription_pb2.GetTranscriptRequest(id=transcript_id), timeout=self._timeout
            )
        except grpc.aio.AioRpcError as error:
            if error.code() == grpc.StatusCode.NOT_FOUND:
                raise LineNotFoundError(f"transcript {transcript_id} not found") from error
            raise
        return Text.of(reply.transcript)

    async def evaluate(self, recording_id: str, media_id: str, workspace_id: str, registry_id: str) -> "Text":
        """Transcribes a recording with a model without keeping the result (an evaluation)."""
        request = transcription_pb2.TranscribeRequest(
            recording_id=recording_id,
            media_id=media_id,
            workspace_id=workspace_id,
            model_registry_id=registry_id,
            evaluation=True,
        )
        completed = None
        async for reply in self._stub.Transcribe(request):
            if reply.WhichOneof("event") == "completed":
                completed = reply.completed
        if completed is None:
            raise RuntimeError(f"the transcription of {recording_id} ended without a result")
        return Text.of(completed)

    async def close(self) -> None:
        await self._channel.close()


@dataclass(frozen=True)
class Text:
    """A transcript as whole texts, for scoring."""

    transcript_id: str
    version: int
    script: str
    roman: str
    lines: int
    language: str
    audio_seconds: float
    elapsed_seconds: float

    @staticmethod
    def of(transcript: transcription_pb2.Transcript) -> "Text":
        segments = sorted(transcript.segments, key=lambda s: s.index)
        return Text(
            transcript_id=transcript.id,
            version=transcript.version,
            script=" ".join(s.text_script for s in segments),
            roman=" ".join(s.text_roman for s in segments),
            lines=len(segments),
            language=transcript.language.decoded_as or transcript.language.detected,
            audio_seconds=transcript.stats.audio_seconds,
            elapsed_seconds=transcript.stats.elapsed_seconds,
        )
