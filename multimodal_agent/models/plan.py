from pydantic import BaseModel, Field

class PlanStep(BaseModel):
    step_id: str
    tool_name: str
    inputs: dict[str, str] = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list)


class Plan(BaseModel):
    steps: list[PlanStep] = Field(default_factory=list)
    max_retries: int = Field(default=1, ge=0)
