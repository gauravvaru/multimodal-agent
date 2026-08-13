"""Document chunking."""

from typing import Any


def chunk_document(document: Any) -> list[dict[str, Any]]:
    """Split a parsed document into retrieval-ready chunks."""
    raise NotImplementedError
