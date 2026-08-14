from multimodal_agent.models.evidence import Evidence
from multimodal_agent.models.state import AgentState

class EvidenceService:

    def build(self, state: AgentState) -> list[Evidence]:
        raise NotImplementedError
