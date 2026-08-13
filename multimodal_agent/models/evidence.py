"""Evidence contracts."""

from typing import Any

from pydantic import BaseModel, Field


class Evidence(BaseModel):
    """Grounded evidence item with provenance."""

    source_id: str | None = None
    filename: str | None = None
    source_type: str
    page: int | None = None
    chunk_id: str | None = None
    extracted_text: str
    extraction_confidence: float | None = None
    relevance_score: float | None = None
    timestamp_start: float | None = None
    timestamp_end: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
