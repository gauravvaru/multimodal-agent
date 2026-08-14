from multimodal_agent.models.state import AgentState
from multimodal_agent.models.tools import ToolResult


class ResultValidationService:
    def validate(self, state: AgentState, result: ToolResult) -> tuple[bool, str | None]:
        raise NotImplementedError
