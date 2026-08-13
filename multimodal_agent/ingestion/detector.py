"""Deterministic input type detection."""

from multimodal_agent.models.requests import InputArtifact


def detect_artifact_type(artifact: InputArtifact) -> str:
    """Detect artifact type from filename, MIME type, and source."""
    raise NotImplementedError
