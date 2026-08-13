"""Input normalization pipeline."""

from typing import Any

from multimodal_agent.models.requests import InputArtifact


class InputNormalizer:
    """Normalize heterogeneous user inputs into a canonical representation."""

    def normalize(self, query: str, artifacts: list[InputArtifact]) -> dict[str, Any]:
        """Normalize text, files, and URLs into structured content."""
        raise NotImplementedError
