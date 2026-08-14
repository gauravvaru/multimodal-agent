from typing import Any

from pydantic import BaseModel, Field


class NormalizedArtifact(BaseModel):
    artifact_id: str
    artifact_type: str
    source: str | None = None
    content_type: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    content: Any = None
