"""Clarification assessment service interface."""

from multimodal_agent.models.state import AgentState


class ClarificationService:
    """Semantic clarification assessment."""

    def assess(self, state: AgentState) -> tuple[bool, str | None]:
        """Return whether clarification is required and an optional question."""
        raise NotImplementedError
