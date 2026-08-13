"""Inbound API request models."""

from pydantic import BaseModel, Field


class InputArtifact(BaseModel):
    """Reference to a user-provided input artifact."""

    filename: str | None = None
    content_type: str | None = None
    source: str | None = None


class AgentRequest(BaseModel):
    """Primary agent invocation request."""

    query: str = Field(min_length=1)
    artifacts: list[InputArtifact] = Field(default_factory=list)
