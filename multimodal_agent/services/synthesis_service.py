"""Final response synthesis service."""

from __future__ import annotations

from typing import Protocol, TypeVar

from pydantic import BaseModel

from multimodal_agent.models.state import AgentState

T = TypeVar("T", bound=BaseModel)


class SynthesisContext(BaseModel):
    """Grounded inputs passed to the synthesis model."""

    user_query: str
    intent_name: str
    tool_results: list[dict[str, object]] = []
    evidence: list[dict[str, object]] = []
    normalized_contents: list[dict[str, object]] = []


class SynthesisLLMClient(Protocol):
    """Sync structured synthesis LLM interface."""

    def synthesize_structured(self, context: SynthesisContext, output_model: type[T]) -> T:
        """Return structured synthesis output for the given schema."""


class SynthesisService:
    """Generate the user-facing final answer."""

    def synthesize(self, state: AgentState) -> str:
        """Produce a text-only response grounded in available evidence."""
        raise NotImplementedError
