"""Tool registration and lookup."""

from __future__ import annotations

from collections.abc import Callable

from multimodal_agent.models.tools import ToolResult
from multimodal_agent.tools.audio import transcribe_audio
from multimodal_agent.tools.code_analysis import explain_code
from multimodal_agent.tools.comparison import compare_inputs
from multimodal_agent.tools.ocr import run_ocr
from multimodal_agent.tools.pdf import extract_pdf
from multimodal_agent.tools.rag import retrieve_evidence
from multimodal_agent.tools.sentiment import analyze_sentiment
from multimodal_agent.tools.summarization import summarize_text
from multimodal_agent.tools.youtube import fetch_youtube_transcript


class ToolRegistry:
    """Registry for deterministic and LLM-backed tools."""

    def __init__(self) -> None:
        self._tools: dict[str, Callable[..., ToolResult]] = {}

    def register(self, name: str, tool: Callable[..., ToolResult]) -> None:
        """Register a tool by name."""
        self._tools[name] = tool

    def get(self, name: str) -> Callable[..., ToolResult]:
        """Return a registered tool."""
        return self._tools[name]

    def names(self) -> list[str]:
        """Return registered tool names."""
        return sorted(self._tools)


def _pdf_extract_tool(**kwargs: str) -> ToolResult:
    source = kwargs.get("source") or kwargs.get("artifact_id", "")
    return extract_pdf(source, artifact_id=kwargs.get("artifact_id"))


def _summarize_tool(**kwargs: str) -> ToolResult:
    return summarize_text(kwargs.get("text", ""))


def _youtube_transcript_tool(**kwargs: str) -> ToolResult:
    return fetch_youtube_transcript(kwargs.get("url", ""))


def _audio_transcribe_tool(**kwargs: str) -> ToolResult:
    source = kwargs.get("source") or kwargs.get("artifact_id", "")
    return transcribe_audio(source)


def _compare_tool(**kwargs: str) -> ToolResult:
    sources = kwargs.get("sources", "")
    return compare_inputs(sources.split(",") if sources else [])


def _ocr_tool(**kwargs: str) -> ToolResult:
    source = kwargs.get("source") or kwargs.get("artifact_id", "")
    return run_ocr(source)


def _sentiment_tool(**kwargs: str) -> ToolResult:
    return analyze_sentiment(kwargs.get("text", ""))


def _rag_tool(**kwargs: str) -> ToolResult:
    return retrieve_evidence(kwargs.get("query", ""))


def _code_analysis_tool(**kwargs: str) -> ToolResult:
    code = kwargs.get("text") or kwargs.get("source", "")
    return explain_code(code)


def create_default_tool_registry() -> ToolRegistry:
    """Return a registry with the canonical agent tool names."""
    registry = ToolRegistry()
    registry.register("pdf_extract", _pdf_extract_tool)
    registry.register("summarize", _summarize_tool)
    registry.register("youtube_transcript", _youtube_transcript_tool)
    registry.register("audio_transcribe", _audio_transcribe_tool)
    registry.register("compare", _compare_tool)
    registry.register("ocr", _ocr_tool)
    registry.register("sentiment", _sentiment_tool)
    registry.register("rag", _rag_tool)
    registry.register("code_analysis", _code_analysis_tool)
    return registry
