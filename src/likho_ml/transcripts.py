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

    async def close(self) -> None:
        await self._channel.close()
