from __future__ import annotations
import time
from collections.abc import Callable
from typing import Any
from pydantic import BaseModel
from multimodal_agent.models.tools import ToolResult
from multimodal_agent.services.llm_provider import LLMInvocationError, LLMNotConfiguredError, invoke_structured, is_llm_configured

def elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 2)

def llm_structured_tool_result(*, tool_name: str, started: float, prompt: str, output_model: type[BaseModel], system: str, data_builder: Callable[[BaseModel], dict[str, Any]], confidence: float=0.9) -> ToolResult:
    if not is_llm_configured():
        return ToolResult(tool_name=tool_name, status='failed', error='LLM is not configured. Set LLM_API_KEY to enable this tool.', latency_ms=elapsed_ms(started))
    try:
        structured = invoke_structured(prompt, output_model, system=system)
    except LLMNotConfiguredError as exc:
        return ToolResult(tool_name=tool_name, status='failed', error=str(exc), latency_ms=elapsed_ms(started))
    except LLMInvocationError as exc:
        return ToolResult(tool_name=tool_name, status='failed', error=str(exc), latency_ms=elapsed_ms(started))
    return ToolResult(tool_name=tool_name, status='success', data=data_builder(structured), confidence=confidence, latency_ms=elapsed_ms(started))