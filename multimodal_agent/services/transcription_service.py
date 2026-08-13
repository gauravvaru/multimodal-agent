"""Audio transcription service."""

from typing import Any


class TranscriptionService:
    """Interface for audio transcription."""

    async def transcribe(self, source: str) -> dict[str, Any]:
        """Transcribe audio from a file path or byte source."""
        raise NotImplementedError
