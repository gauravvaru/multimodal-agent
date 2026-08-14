from multimodal_agent.models.artifacts import NormalizedArtifact
from multimodal_agent.models.intent import Intent

class IntentService:
    def detect(
        self,
        query: str,
        normalized_contents: list[NormalizedArtifact],
    ) -> Intent:
        raise NotImplementedError
