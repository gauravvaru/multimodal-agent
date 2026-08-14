from __future__ import annotations
import time
from multimodal_agent.config.settings import get_settings
from multimodal_agent.models.synthesis import SummaryResponse, format_summary_response
from multimodal_agent.models.tools import ToolResult
from multimodal_agent.services.llm_provider import (
    LLMInvocationError,
    LLMNotConfiguredError,
    invoke_structured,
    is_llm_configured,
)

_TOOL_NAME = "summarize"


def summarize_text(text: str) -> ToolResult:
    started = time.perf_counter()

    if text is None or not str(text).strip():
        return ToolResult(
            tool_name=_TOOL_NAME,
            status="failed",
            error="No text provided for summarization",
            latency_ms=_elapsed_ms(started),
        )

    normalized = str(text).strip()
    if not is_llm_configured():
        return ToolResult(
            tool_name=_TOOL_NAME,
            status="failed",
            error="LLM is not configured. Set LLM_API_KEY to enable summarization.",
            latency_ms=_elapsed_ms(started),
        )

    try:
        summary = _summarize_with_llm(normalized)
    except LLMNotConfiguredError as exc:
        return ToolResult(
            tool_name=_TOOL_NAME,
            status="failed",
            error=str(exc),
            latency_ms=_elapsed_ms(started),
        )
    except LLMInvocationError as exc:
        return ToolResult(
            tool_name=_TOOL_NAME,
            status="failed",
            error=str(exc),
            latency_ms=_elapsed_ms(started),
        )

    formatted = format_summary_response(summary)
    return ToolResult(
        tool_name=_TOOL_NAME,
        status="success",
        data={
            "text": formatted,
            "summary": summary.one_line_summary,
            "bullets": summary.bullets,
            "sentences": summary.sentences,
        },
        confidence=0.9,
        latency_ms=_elapsed_ms(started),
    )


def _summarize_with_llm(text: str) -> SummaryResponse:
    settings = get_settings()
    if len(text) <= settings.llm_max_input_chars:
        return _invoke_summary(text)

    chunks = _split_text(text, max_chars=min(4000, settings.llm_max_input_chars))
    if len(chunks) == 1:
        return _invoke_summary(chunks[0])

    partials = [_invoke_summary(chunk).one_line_summary for chunk in chunks[:5]]
    merged_prompt = "Combine the following section summaries into one cohesive summary:\n" + "\n".join(
        f"- {item}" for item in partials
    )
    return _invoke_summary(merged_prompt)


def _invoke_summary(text: str) -> SummaryResponse:
    prompt = (
        "Summarize the following text. Preserve factual content only.\n\n"
        f"{text}"
    )
    system = (
        "You produce concise factual summaries. Do not invent information. "
        "Return structured output matching the requested schema."
    )
    return invoke_structured(prompt, SummaryResponse, system=system)


def _split_text(text: str, *, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    start = 0
    overlap = min(200, max_chars // 10)
    while start < len(text):
        end = min(start + max_chars, len(text))
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 2)
