from typing import Any
from pydantic import BaseModel, Field
from multimodal_agent.models.tools import ToolResult

class AgentResponse(BaseModel):
    request_id: str
    answer: str
    clarification_required: bool = False
    clarification_question: str | None = None
    trace: list[Any] = Field(default_factory=list)
    tool_results: list[ToolResult] = Field(default_factory=list)
    evidence: list[Any] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
