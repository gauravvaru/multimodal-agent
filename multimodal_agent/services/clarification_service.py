from multimodal_agent.models.state import AgentState

class ClarificationService:
    def assess(self, state: AgentState) -> tuple[bool, str | None]:
        raise NotImplementedError
