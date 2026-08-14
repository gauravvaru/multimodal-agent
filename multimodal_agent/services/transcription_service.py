from typing import Any

class TranscriptionService:
    async def transcribe(self, source: str) -> dict[str, Any]:
        raise NotImplementedError
