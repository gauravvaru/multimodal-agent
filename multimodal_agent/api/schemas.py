from __future__ import annotations
from typing import Any, Literal
from pydantic import BaseModel, Field
from multimodal_agent.models.responses import AgentResponse
from multimodal_agent.models.tools import ToolResult

AgentRunStatus = Literal["success", "clarification_required", "error", "partial"]

class HealthResponse(BaseModel):
    status: str

class AgentRunResponse(BaseModel):
    request_id: str
    status: AgentRunStatus
    final_answer: str
    extracted_content: list[dict[str, Any]] = Field(default_factory=list)
    trace: list[Any] = Field(default_factory=list)
    evidence: list[Any] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)

class ErrorResponse(BaseModel):
    detail: str | list[str]
    request_id: str | None = None

def agent_response_to_run_response(response: AgentResponse) -> AgentRunResponse:
    status = derive_run_status(response)
    final_answer = resolve_final_answer(response)

    return AgentRunResponse(
        request_id=response.request_id,
        status=status,
        final_answer=final_answer,
        extracted_content=build_extracted_content(response.tool_results),
        trace=response.trace,
        evidence=response.evidence,
        errors=response.errors,
    )

def derive_run_status(response: AgentResponse) -> AgentRunStatus:
    if response.clarification_required:
        return "clarification_required"

    if response.errors:
        if response.answer.strip():
            return "partial"
        if any("unexpected error" in error.lower() for error in response.errors):
            return "error"
        return "error"
    return "success"

def resolve_final_answer(response: AgentResponse) -> str:
    if response.clarification_required and response.clarification_question:
        return response.clarification_question
    return response.answer

def build_extracted_content(tool_results: list[ToolResult]) -> list[dict[str, Any]]:
    extracted: list[dict[str, Any]] = []
    for result in tool_results:
        if result.status not in {"success", "partial"} or result.data is None:
            continue
        extracted.append(
            {
                "tool_name": result.tool_name,
                "status": result.status,
                "content": result.data,
                "confidence": result.confidence,
            }
        )
    return extracted

def is_input_validation_error(errors: list[str]) -> bool:
    validation_markers = (
        "query is required",
        "filename or source is required",
        "filename cannot be blank",
        "unsupported file type",
        "file exceeds maximum size",
        "uploaded file must have a filename",
    )
    return any(any(marker in error for marker in validation_markers) for error in errors)

__all__ = [
    "AgentRunResponse",
    "AgentRunStatus",
    "ErrorResponse",
    "HealthResponse",
    "agent_response_to_run_response",
    "build_extracted_content",
    "derive_run_status",
    "is_input_validation_error",
    "resolve_final_answer",
]
