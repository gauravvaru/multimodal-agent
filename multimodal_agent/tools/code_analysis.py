"""Code explanation tool."""

from __future__ import annotations

import time

from multimodal_agent.models.synthesis import CodeExplanationResponse, format_code_explanation_response
from multimodal_agent.models.tools import ToolResult
from multimodal_agent.services.llm_provider import (
    LLMInvocationError,
    LLMNotConfiguredError,
    invoke_structured,
    is_llm_configured,
)
from multimodal_agent.tools._tool_utils import elapsed_ms

_TOOL_NAME = "code_analysis"
_FORBIDDEN_CODE_PATTERNS = (
    "__import__",
    "eval(",
    "exec(",
    "compile(",
    "os.system(",
    "subprocess.",
    "Popen(",
)


def explain_code(code: str) -> ToolResult:
    """Explain provided source code. Analysis only — never executes code."""
    started = time.perf_counter()

    if not code or not code.strip():
        return ToolResult(
            tool_name=_TOOL_NAME,
            status="failed",
            error="Code input is empty",
            latency_ms=elapsed_ms(started),
        )

    lowered = code.lower()
    for pattern in _FORBIDDEN_CODE_PATTERNS:
        if pattern.lower() in lowered:
            return ToolResult(
                tool_name=_TOOL_NAME,
                status="failed",
                error="Code input contains disallowed execution patterns",
                latency_ms=elapsed_ms(started),
            )

    if not is_llm_configured():
        return ToolResult(
            tool_name=_TOOL_NAME,
            status="failed",
            error="LLM is not configured. Set LLM_API_KEY to enable code analysis.",
            latency_ms=elapsed_ms(started),
        )

    prompt = (
        "Explain the following source code without executing it. "
        "Identify language, behavior, potential issues, and complexity.\n\n"
        f"{code.strip()}"
    )
    system = (
        "You analyze code statically. Never suggest executing untrusted code. "
        "Return structured output matching the requested schema."
    )

    try:
        explanation = invoke_structured(prompt, CodeExplanationResponse, system=system)
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

    formatted = format_code_explanation_response(explanation)
    return ToolResult(
        tool_name=_TOOL_NAME,
        status="success",
        data={
            "text": formatted,
            "language": explanation.language,
            "explanation": explanation.explanation,
            "bugs_issues": explanation.bugs_issues,
            "time_complexity": explanation.time_complexity,
            "space_complexity": explanation.space_complexity,
        },
        confidence=0.88,
        latency_ms=elapsed_ms(started),
    )
