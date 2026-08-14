from pydantic import BaseModel, Field

class Intent(BaseModel):
    name: str
    confidence: float = Field(ge=0.0, le=1.0)
    requires_clarification: bool = False
    rationale: str | None = None
