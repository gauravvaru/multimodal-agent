"""Intent detection service interface."""

from multimodal_agent.models.artifacts import NormalizedArtifact
from multimodal_agent.models.intent import Intent


class IntentService:
    """Semantic intent detection backed by an LLM or classifier."""

    def detect(
        self,
        query: str,
        normalized_contents: list[NormalizedArtifact],
    ) -> Intent:
        """Infer user intent when deterministic rules are insufficient."""
        raise NotImplementedError
