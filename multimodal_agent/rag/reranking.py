"""Retrieval reranking."""

from multimodal_agent.models.evidence import Evidence


def rerank(query: str, candidates: list[Evidence]) -> list[Evidence]:
    """Rerank retrieved candidates for a query."""
    raise NotImplementedError
