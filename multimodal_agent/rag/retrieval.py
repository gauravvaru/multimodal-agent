"""Hybrid retrieval."""

from multimodal_agent.models.evidence import Evidence


def retrieve(query: str, *, top_k: int = 5) -> list[Evidence]:
    """Run dense and sparse retrieval with result fusion."""
    raise NotImplementedError
