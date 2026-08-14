from pydantic import BaseModel, Field

class InputArtifact(BaseModel):
    filename: str | None = None
    content_type: str | None = None
    source: str | None = None


class AgentRequest(BaseModel):
    query: str = Field(min_length=1)
    artifacts: list[InputArtifact] = Field(default_factory=list)
