from __future__ import annotations
from typing import Protocol, TypeVar
from pydantic import BaseModel
from multimodal_agent.models.state import AgentState

T = TypeVar("T", bound=BaseModel)

class SynthesisContext(BaseModel):
    user_query: str
    intent_name: str
    tool_results: list[dict[str, object]] = []
    evidence: list[dict[str, object]] = []
    normalized_contents: list[dict[str, object]] = []


class SynthesisLLMClient(Protocol):
    def synthesize_structured(self, context: SynthesisContext, output_model: type[T]) -> T:
        """Return structured synthesis output for the given schema."""

class SynthesisService:
    def synthesize(self, state: AgentState) -> str:
        raise NotImplementedError
