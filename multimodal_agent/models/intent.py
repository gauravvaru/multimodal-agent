"""Intent detection contracts."""

from pydantic import BaseModel, Field


class Intent(BaseModel):
    """Detected user intent and routing metadata."""

    name: str
    confidence: float = Field(ge=0.0, le=1.0)
    requires_clarification: bool = False
    rationale: str | None = None
