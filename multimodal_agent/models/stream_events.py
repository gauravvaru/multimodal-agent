from __future__ import annotations
from typing import Any, Literal
from pydantic import BaseModel, Field
StreamEventType = Literal['node_update', 'tool_update', 'status_update', 'final_answer', 'complete', 'error']
NodeStatus = Literal['started', 'completed', 'failed']
ToolStatus = Literal['running', 'completed', 'failed']

class AgentStreamEvent(BaseModel):
    type: StreamEventType
    request_id: str | None = None
    node: str | None = None
    tool: str | None = None
    status: str | None = None
    message: str | None = None
    latency_ms: float | None = Field(default=None, ge=0.0)
    final_answer: str | None = None
    response: dict[str, Any] | None = None
    errors: list[str] = Field(default_factory=list)

    def to_sse_data(self) -> str:
        return self.model_dump_json(exclude_none=True, exclude_defaults=True)