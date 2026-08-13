"""Tool result validation service interface."""

from multimodal_agent.models.state import AgentState
from multimodal_agent.models.tools import ToolResult


class ResultValidationService:
    """Semantic or domain-specific tool result validation."""

    def validate(self, state: AgentState, result: ToolResult) -> tuple[bool, str | None]:
        """Return whether the result is usable and an optional reason."""
        raise NotImplementedError
