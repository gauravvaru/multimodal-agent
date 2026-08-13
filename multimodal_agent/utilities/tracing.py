"""Execution trace helpers."""

from typing import Any

from pydantic import BaseModel, Field


class TraceEvent(BaseModel):
    """Single step in an agent execution trace."""

    step: str
    detail: dict[str, Any] = Field(default_factory=dict)


def append_trace_event(trace: list[Any], step: str, **detail: Any) -> list[Any]:
    """Append a trace event and return the updated trace."""
    trace.append(TraceEvent(step=step, detail=detail).model_dump())
    return trace
