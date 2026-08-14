from __future__ import annotations
import time
from multimodal_agent.config.settings import get_settings
from multimodal_agent.models.tools import ToolResult
from multimodal_agent.rag.pipeline import RAGPipeline

_TOOL_NAME = "rag"


def retrieve_evidence(query: str, context: str = "") -> ToolResult:
    started = time.perf_counter()
    normalized_query = (query or "").strip()
    if not normalized_query:
        return ToolResult(
            tool_name=_TOOL_NAME,
            status="failed",
            error="Query is required for retrieval",
            latency_ms=_elapsed_ms(started),
        )

    if not context.strip():
        return ToolResult(
            tool_name=_TOOL_NAME,
            status="partial",
            data={"evidence": []},
            error="No documents available for retrieval",
            latency_ms=_elapsed_ms(started),
        )

    pipeline = RAGPipeline()
    pipeline.index_documents([{"text": context, "document": "session"}])
    items = pipeline.retrieve(normalized_query, top_k=5)
    settings = get_settings()

    evidence_payload = [item.model_dump() for item in items]
    if len(items) < settings.min_rag_evidence_items:
        return ToolResult(
            tool_name=_TOOL_NAME,
            status="partial",
            data={"evidence": evidence_payload},
            error="Insufficient retrieval evidence",
            latency_ms=_elapsed_ms(started),
        )

    return ToolResult(
        tool_name=_TOOL_NAME,
        status="success",
        data={"evidence": evidence_payload},
        confidence=max((item.retrieval_score for item in items), default=0.0),
        latency_ms=_elapsed_ms(started),
    )


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 2)
