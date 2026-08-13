"""Evidence contracts."""

from pydantic import BaseModel, Field


class Evidence(BaseModel):
    """Grounded evidence item with provenance."""

    document: str
    page: int | None = None
    chunk: str
    text: str
    retrieval_score: float = Field(ge=0.0)
