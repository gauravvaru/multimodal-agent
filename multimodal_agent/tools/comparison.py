"""Cross-input comparison tool."""

from __future__ import annotations

import time

from multimodal_agent.models.synthesis import ComparisonResponse, format_comparison_response
from multimodal_agent.models.tools import ToolResult
from multimodal_agent.services.llm_provider import (
    LLMInvocationError,
    LLMNotConfiguredError,
    invoke_structured,
    is_llm_configured,
)
from multimodal_agent.tools._tool_utils import elapsed_ms

_TOOL_NAME = "compare"


def compare_inputs(sources: list[str]) -> ToolResult:
    """Compare multiple normalized inputs."""
    started = time.perf_counter()
    normalized = [item.strip() for item in sources if isinstance(item, str) and item.strip()]

    if len(normalized) < 2:
        return ToolResult(
            tool_name=_TOOL_NAME,
            status="failed",
            error="At least two non-empty sources are required for comparison",
            latency_ms=elapsed_ms(started),
        )

    if is_llm_configured():
        try:
            comparison = _compare_with_llm(normalized)
        except LLMNotConfiguredError as exc:
            return ToolResult(
                tool_name=_TOOL_NAME,
                status="failed",
                error=str(exc),
                latency_ms=elapsed_ms(started),
            )
        except LLMInvocationError as exc:
            return ToolResult(
                tool_name=_TOOL_NAME,
                status="failed",
                error=str(exc),
                latency_ms=elapsed_ms(started),
            )
        formatted = format_comparison_response(comparison)
        return ToolResult(
            tool_name=_TOOL_NAME,
            status="success",
            data={
                "text": formatted,
                "conclusion": comparison.conclusion,
                "similarities": comparison.similarities,
                "differences": comparison.differences,
                "contradictions": comparison.contradictions,
                "evidence": comparison.evidence,
            },
            confidence=0.88,
            latency_ms=elapsed_ms(started),
        )

    comparison = _compare_deterministic(normalized)
    formatted = format_comparison_response(comparison)
    return ToolResult(
        tool_name=_TOOL_NAME,
        status="success",
        data={
            "text": formatted,
            "conclusion": comparison.conclusion,
            "similarities": comparison.similarities,
            "differences": comparison.differences,
            "contradictions": comparison.contradictions,
            "evidence": comparison.evidence,
        },
        confidence=0.75,
        latency_ms=elapsed_ms(started),
    )


def _compare_with_llm(sources: list[str]) -> ComparisonResponse:
    joined = "\n\n---\n\n".join(f"Source {index + 1}:\n{source}" for index, source in enumerate(sources))
    prompt = (
        "Compare the following sources. Identify similarities, differences, and contradictions. "
        "Use only information present in the sources.\n\n"
        f"{joined}"
    )
    system = (
        "You compare inputs factually. Do not invent content. "
        "Return structured output matching the requested schema."
    )
    return invoke_structured(prompt, ComparisonResponse, system=system)


def _compare_deterministic(sources: list[str]) -> ComparisonResponse:
    left_lines = _line_set(sources[0])
    right_lines = _line_set(sources[1])
    shared = sorted(left_lines & right_lines)
    left_only = sorted(left_lines - right_lines)[:5]
    right_only = sorted(right_lines - left_lines)[:5]

    conclusion = "Deterministic comparison completed using shared and unique lines."
    if shared:
        conclusion = "Both sources share overlapping content with some unique lines."

    return ComparisonResponse(
        conclusion=conclusion,
        similarities=shared[:5] or ["No exact shared lines detected"],
        differences=[f"Unique to source 1: {line}" for line in left_only]
        + [f"Unique to source 2: {line}" for line in right_only],
        contradictions=[],
        evidence=[f"source-{index + 1}" for index in range(min(len(sources), 2))],
    )


def _line_set(text: str) -> set[str]:
    return {line.strip() for line in text.splitlines() if line.strip()}
