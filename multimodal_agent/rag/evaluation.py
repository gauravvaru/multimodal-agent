"""RAG evaluation helpers."""

from typing import Any


def evaluate_retrieval(query: str, evidence: list[Any]) -> dict[str, float]:
    """Evaluate retrieval quality for observability."""
    raise NotImplementedError
