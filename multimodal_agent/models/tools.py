"""Tool execution contract."""

from typing import Any

from pydantic import BaseModel, Field


class ToolResult(BaseModel):
    """Standard return shape for every tool."""

    tool_name: str
    status: str
    data: Any = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    latency_ms: float | None = Field(default=None, ge=0.0)
    error: str | None = None
