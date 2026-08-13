"""Normalized input artifact contracts."""

from typing import Any

from pydantic import BaseModel, Field


class NormalizedArtifact(BaseModel):
    """Canonical representation of a user-provided input."""

    artifact_id: str
    artifact_type: str
    source: str | None = None
    content_type: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    content: Any = None
