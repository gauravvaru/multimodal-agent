"""Evidence construction node."""

from __future__ import annotations

from typing import Any

from multimodal_agent.agent.types import StateUpdate
from multimodal_agent.models.artifacts import NormalizedArtifact
from multimodal_agent.models.evidence import Evidence
from multimodal_agent.models.state import AgentState
from multimodal_agent.models.tools import ToolResult
from multimodal_agent.services.evidence_service import EvidenceService
from multimodal_agent.utilities.tracing import TraceEvent

_USABLE_TOOL_STATUSES = frozenset({"success", "partial"})


def build_evidence(
    state: AgentState,
    *,
    evidence_service: EvidenceService | None = None,
) -> StateUpdate:
    """Assemble grounded evidence from tool and retrieval results."""
    if evidence_service is not None:
        evidence = evidence_service.build(state)
    else:
        evidence = collect_evidence(state)

    return {
        "evidence": evidence,
        "trace": [
            TraceEvent(
                step="build_evidence",
                detail={"evidence_count": len(evidence), "status": "ok"},
            )
        ],
    }


def collect_evidence(state: AgentState) -> list[Evidence]:
    """Collect evidence from tool results using normalized artifact context."""
    artifact_lookup = _build_artifact_lookup(state.normalized_contents)
    evidence: list[Evidence] = []

    for result in state.tool_results:
        if result.status not in _USABLE_TOOL_STATUSES or result.data is None:
            continue
        evidence.extend(_extract_evidence_from_tool_result(result, artifact_lookup))

    return dedupe_evidence(evidence)


def dedupe_evidence(items: list[Evidence]) -> list[Evidence]:
    """Remove duplicate evidence entries while preserving order."""
    seen: set[tuple[Any, ...]] = set()
    unique: list[Evidence] = []
    for item in items:
        key = _dedupe_key(item)
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def _extract_evidence_from_tool_result(
    result: ToolResult,
    artifact_lookup: dict[str, NormalizedArtifact],
) -> list[Evidence]:
    extractors = {
        "pdf_extract": _extract_pdf_evidence,
        "ocr": _extract_ocr_evidence,
        "audio_transcribe": _extract_audio_evidence,
        "rag": _extract_rag_evidence,
    }
    extractor = extractors.get(result.tool_name, _extract_generic_evidence)
    return extractor(result, artifact_lookup)


def _extract_pdf_evidence(
    result: ToolResult,
    artifact_lookup: dict[str, NormalizedArtifact],
) -> list[Evidence]:
    data = result.data
    if not isinstance(data, dict):
        return []

    source_id = _as_optional_str(data.get("artifact_id") or data.get("source_id"))
    filename = _resolve_filename(data, artifact_lookup, source_id)
    items: list[Evidence] = []

    pages = data.get("pages")
    if isinstance(pages, list):
        for index, page_item in enumerate(pages):
            if not isinstance(page_item, dict):
                continue
            text = _extract_text_value(page_item)
            if text is None:
                continue
            items.append(
                Evidence(
                    source_id=source_id,
                    filename=filename or _as_optional_str(page_item.get("filename")),
                    source_type="pdf",
                    page=_as_optional_int(page_item.get("page")),
                    chunk_id=_as_optional_str(page_item.get("chunk_id")) or str(index),
                    extracted_text=text,
                    extraction_confidence=result.confidence,
                    metadata=_copy_metadata(page_item),
                )
            )
        return items

    text = _extract_text_value(data)
    if text is None:
        return []

    return [
        Evidence(
            source_id=source_id,
            filename=filename,
            source_type="pdf",
            page=_as_optional_int(data.get("page")),
            extracted_text=text,
            extraction_confidence=result.confidence,
            metadata=_copy_metadata(data),
        )
    ]


def _extract_ocr_evidence(
    result: ToolResult,
    artifact_lookup: dict[str, NormalizedArtifact],
) -> list[Evidence]:
    data = result.data
    if not isinstance(data, dict):
        return []

    text = _extract_text_value(data)
    if text is None:
        return []

    source_id = _as_optional_str(data.get("artifact_id") or data.get("source_id"))
    confidence = data.get("confidence")
    if confidence is None:
        confidence = result.confidence

    return [
        Evidence(
            source_id=source_id,
            filename=_resolve_filename(data, artifact_lookup, source_id),
            source_type="ocr",
            extracted_text=text,
            extraction_confidence=_as_optional_float(confidence),
            metadata=_copy_metadata(data),
        )
    ]


def _extract_audio_evidence(
    result: ToolResult,
    artifact_lookup: dict[str, NormalizedArtifact],
) -> list[Evidence]:
    data = result.data
    if not isinstance(data, dict):
        return []

    source_id = _as_optional_str(data.get("artifact_id") or data.get("source_id"))
    filename = _resolve_filename(data, artifact_lookup, source_id)
    items: list[Evidence] = []

    segments = data.get("segments")
    if isinstance(segments, list):
        for index, segment in enumerate(segments):
            if not isinstance(segment, dict):
                continue
            text = _extract_text_value(segment)
            if text is None:
                continue
            items.append(
                Evidence(
                    source_id=source_id,
                    filename=filename,
                    source_type="audio",
                    chunk_id=_as_optional_str(segment.get("chunk_id")) or str(index),
                    timestamp_start=_as_optional_float(segment.get("timestamp_start", segment.get("start"))),
                    timestamp_end=_as_optional_float(segment.get("timestamp_end", segment.get("end"))),
                    extracted_text=text,
                    extraction_confidence=result.confidence,
                    metadata=_copy_metadata(segment),
                )
            )
        return items

    text = _extract_text_value(data)
    if text is None:
        return []

    return [
        Evidence(
            source_id=source_id,
            filename=filename,
            source_type="audio",
            timestamp_start=_as_optional_float(data.get("timestamp_start")),
            timestamp_end=_as_optional_float(data.get("timestamp_end")),
            extracted_text=text,
            extraction_confidence=result.confidence,
            metadata=_copy_metadata(data),
        )
    ]


def _extract_rag_evidence(
    result: ToolResult,
    artifact_lookup: dict[str, NormalizedArtifact],
) -> list[Evidence]:
    data = result.data
    if not isinstance(data, dict):
        return []

    raw_items = data.get("evidence")
    if not isinstance(raw_items, list):
        return []

    items: list[Evidence] = []
    for index, raw_item in enumerate(raw_items):
        if not isinstance(raw_item, dict):
            continue
        text = _extract_text_value(raw_item)
        if text is None:
            continue

        source_id = _as_optional_str(raw_item.get("source_id") or raw_item.get("artifact_id"))
        filename = _as_optional_str(
            raw_item.get("filename") or raw_item.get("document") or raw_item.get("source")
        )
        if filename is None and source_id is not None:
            filename = _filename_from_artifact(source_id, artifact_lookup)

        relevance = raw_item.get("relevance_score", raw_item.get("retrieval_score"))
        items.append(
            Evidence(
                source_id=source_id,
                filename=filename,
                source_type="rag",
                page=_as_optional_int(raw_item.get("page")),
                chunk_id=_as_optional_str(raw_item.get("chunk_id", raw_item.get("chunk"))) or str(index),
                extracted_text=text,
                relevance_score=_as_optional_float(relevance),
                extraction_confidence=result.confidence,
                metadata=_copy_metadata(raw_item),
            )
        )
    return items


def _extract_generic_evidence(
    result: ToolResult,
    artifact_lookup: dict[str, NormalizedArtifact],
) -> list[Evidence]:
    data = result.data
    if isinstance(data, dict):
        source_id = _as_optional_str(data.get("artifact_id") or data.get("source_id"))
        text = _extract_text_value(data)
        if text is None:
            return []
        return [
            Evidence(
                source_id=source_id,
                filename=_resolve_filename(data, artifact_lookup, source_id),
                source_type=result.tool_name,
                extracted_text=text,
                extraction_confidence=result.confidence,
                metadata=_copy_metadata(data),
            )
        ]

    if isinstance(data, str) and data.strip():
        return [
            Evidence(
                source_type=result.tool_name,
                extracted_text=data.strip(),
                extraction_confidence=result.confidence,
            )
        ]

    return []


def _build_artifact_lookup(normalized_contents: list[NormalizedArtifact]) -> dict[str, NormalizedArtifact]:
    return {artifact.artifact_id: artifact for artifact in normalized_contents}


def _resolve_filename(
    data: dict[str, Any],
    artifact_lookup: dict[str, NormalizedArtifact],
    source_id: str | None,
) -> str | None:
    filename = _as_optional_str(data.get("filename") or data.get("document") or data.get("source"))
    if filename is not None:
        return filename.split("/")[-1]
    if source_id is not None:
        return _filename_from_artifact(source_id, artifact_lookup)
    return None


def _filename_from_artifact(source_id: str, artifact_lookup: dict[str, NormalizedArtifact]) -> str | None:
    artifact = artifact_lookup.get(source_id)
    if artifact is None:
        return None
    display_name = artifact.metadata.get("filename")
    if isinstance(display_name, str) and display_name.strip():
        return display_name
    if not artifact.source:
        return None
    return artifact.source.removeprefix("upload://").split("/")[-1]


def _extract_text_value(payload: dict[str, Any]) -> str | None:
    for key in ("extracted_text", "text", "content", "transcript", "transcription", "summary"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _dedupe_key(item: Evidence) -> tuple[Any, ...]:
    return (
        item.source_id or "",
        item.filename or "",
        item.source_type or "",
        item.page,
        item.chunk_id or "",
        item.timestamp_start,
        item.timestamp_end,
        item.extracted_text.strip(),
    )


def _copy_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        return dict(metadata)
    excluded = {
        "extracted_text",
        "text",
        "content",
        "transcript",
        "transcription",
        "summary",
        "filename",
        "document",
        "source",
        "artifact_id",
        "source_id",
        "page",
        "chunk_id",
        "chunk",
        "timestamp_start",
        "timestamp_end",
        "start",
        "end",
        "confidence",
        "relevance_score",
        "retrieval_score",
        "pages",
        "segments",
        "evidence",
        "metadata",
    }
    return {key: value for key, value in payload.items() if key not in excluded}


def _as_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _as_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
