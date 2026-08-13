"""Evidence construction service interface."""

from multimodal_agent.models.evidence import Evidence
from multimodal_agent.models.state import AgentState


class EvidenceService:
    """Construct grounded evidence from tool and retrieval outputs."""

    def build(self, state: AgentState) -> list[Evidence]:
        """Assemble evidence items from the current agent state."""
        raise NotImplementedError
