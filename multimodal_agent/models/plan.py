"""Planner contracts."""

from pydantic import BaseModel, Field


class PlanStep(BaseModel):
    """Single step in an execution plan."""

    step_id: str
    tool_name: str
    inputs: dict[str, str] = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list)


class Plan(BaseModel):
    """Validated tool execution plan."""

    steps: list[PlanStep] = Field(default_factory=list)
    max_retries: int = Field(default=1, ge=0)
