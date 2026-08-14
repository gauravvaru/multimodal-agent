from typing import Any
from multimodal_agent.models.requests import InputArtifact

class InputNormalizer:
    def normalize(self, query: str, artifacts: list[InputArtifact]) -> dict[str, Any]:
        raise NotImplementedError
