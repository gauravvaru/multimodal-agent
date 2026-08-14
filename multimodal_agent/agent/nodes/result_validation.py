from __future__ import annotations
from typing import Any
from multimodal_agent.agent.types import StateUpdate
from multimodal_agent.config.settings import Settings, get_settings
from multimodal_agent.models.state import AgentState
from multimodal_agent.models.tools import ToolResult
from multimodal_agent.models.validation import ResultValidationOutcome, ValidationStatus
from multimodal_agent.utilities.tracing import TraceEvent

_USABLE_STATUSES = frozenset({ValidationStatus.SUCCESS, ValidationStatus.PARTIAL})

def validate_results(
    state: AgentState,
    *,
    settings: Settings | None = None,
) -> StateUpdate:
    if not state.tool_results:
        return {
            "errors": ["Tool results are required before result validation"],
            "trace": [TraceEvent(step="validate_results", detail={"status": "missing_results"})],
        }

    active_settings = settings or get_settings()
    result = state.tool_results[-1]
    outcome = validate_tool_result(
        result,
        retry_count=state.retry_count,
        max_retries=get_max_tool_retries(state, settings=active_settings),
        settings=active_settings,
    )

    constraints = dict(state.constraints)
    constraints["last_result_usable"] = outcome.validation_status in _USABLE_STATUSES
    if result.tool_name:
        constraints["retry_tool_name"] = result.tool_name

    trace_detail: dict[str, object] = {
        "tool_name": result.tool_name,
        "status": result.status,
        "validation_status": outcome.validation_status,
        "retry_required": outcome.retry_required,
        "no_evidence": outcome.no_evidence,
    }
    update: StateUpdate = {
        "validation_status": outcome.validation_status,
        "retry_required": outcome.retry_required,
        "retry_count": _next_retry_count(state.retry_count, outcome),
        "validation_error": outcome.validation_error,
        "no_evidence": outcome.no_evidence,
        "constraints": constraints,
        "trace": [TraceEvent(step="validate_results", detail=trace_detail)],
    }

    if outcome.validation_error and outcome.validation_status == ValidationStatus.FATAL_FAILURE:
        update["errors"] = [outcome.validation_error]

    if outcome.validation_status in _USABLE_STATUSES:
        update["retry_count"] = 0

    if outcome.validation_error:
        trace_detail["validation_error"] = outcome.validation_error

    trace_detail["retry_count"] = update["retry_count"]
    trace_detail["max_retries"] = get_max_tool_retries(state, settings=active_settings)
    return update


def get_max_tool_retries(state: AgentState, *, settings: Settings | None = None) -> int:
    """Return the configured retry limit for tool result validation."""
    active_settings = settings or get_settings()
    if state.plan is not None:
        return min(state.plan.max_retries, active_settings.max_tool_retries)
    return active_settings.max_tool_retries


def validate_tool_result(
    result: ToolResult,
    *,
    retry_count: int,
    max_retries: int,
    settings: Settings | None = None,
) -> ResultValidationOutcome:
    """Validate a tool result using deterministic rules."""
    active_settings = settings or get_settings()
    raw_outcome = _validate_by_tool(result, settings=active_settings)
    return _apply_retry_policy(raw_outcome, retry_count=retry_count, max_retries=max_retries)


def _validate_by_tool(result: ToolResult, *, settings: Settings) -> ResultValidationOutcome:
    if result.status == "failed":
        return ResultValidationOutcome(
            validation_status=ValidationStatus.RETRYABLE_FAILURE,
            validation_error=result.error or f"Tool '{result.tool_name}' failed",
        )

    validators = {
        "pdf_extract": _validate_pdf_result,
        "ocr": _validate_ocr_result,
        "youtube_transcript": _validate_youtube_result,
        "audio_transcribe": _validate_audio_result,
        "rag": _validate_rag_result,
    }
    validator = validators.get(result.tool_name, _validate_generic_result)
    return validator(result, settings=settings)


def _validate_pdf_result(result: ToolResult, *, settings: Settings) -> ResultValidationOutcome:
    text = _extract_text(result.data)
    if len(text) >= settings.min_pdf_text_chars:
        return ResultValidationOutcome(validation_status=ValidationStatus.SUCCESS)

    return ResultValidationOutcome(
        validation_status=ValidationStatus.RETRYABLE_FAILURE,
        validation_error="PDF extraction returned no usable text",
    )


def _validate_ocr_result(result: ToolResult, *, settings: Settings) -> ResultValidationOutcome:
    text = _extract_text(result.data)
    confidence = result.confidence

    if not text:
        return ResultValidationOutcome(
            validation_status=ValidationStatus.RETRYABLE_FAILURE,
            validation_error="OCR returned no extracted text",
        )

    if result.status == "partial" or (
        confidence is not None and confidence < settings.min_ocr_confidence
    ):
        message = "OCR confidence is below threshold"
        if result.status == "partial":
            message = "OCR returned partial text with low confidence"
        return ResultValidationOutcome(
            validation_status=ValidationStatus.PARTIAL,
            validation_error=message,
        )

    return ResultValidationOutcome(validation_status=ValidationStatus.SUCCESS)


def _validate_youtube_result(result: ToolResult, *, settings: Settings) -> ResultValidationOutcome:
    _ = settings
    transcript = _extract_text(result.data)
    if transcript:
        return ResultValidationOutcome(validation_status=ValidationStatus.SUCCESS)

    unavailable = _data_flag(result.data, "transcript_unavailable")
    message = "YouTube transcript is unavailable"
    if isinstance(result.data, dict) and isinstance(result.data.get("message"), str):
        message = result.data["message"]

    if unavailable or result.status == "partial":
        return ResultValidationOutcome(
            validation_status=ValidationStatus.PARTIAL,
            validation_error=message,
        )

    return ResultValidationOutcome(
        validation_status=ValidationStatus.FATAL_FAILURE,
        validation_error=message,
    )

def _validate_audio_result(result: ToolResult, *, settings: Settings) -> ResultValidationOutcome:
    _ = settings
    transcript = _extract_text(result.data)
    if transcript:
        return ResultValidationOutcome(validation_status=ValidationStatus.SUCCESS)

    return ResultValidationOutcome(
        validation_status=ValidationStatus.RETRYABLE_FAILURE,
        validation_error="Audio transcription returned no text",
    )

def _validate_rag_result(result: ToolResult, *, settings: Settings) -> ResultValidationOutcome:
    evidence_items = _extract_evidence_items(result.data)
    if len(evidence_items) >= settings.min_rag_evidence_items:
        return ResultValidationOutcome(validation_status=ValidationStatus.SUCCESS)

    return ResultValidationOutcome(
        validation_status=ValidationStatus.PARTIAL,
        validation_error="Insufficient retrieval evidence",
        no_evidence=True,
    )


def _validate_generic_result(result: ToolResult, *, settings: Settings) -> ResultValidationOutcome:
    _ = settings
    if result.status in {"success", "partial"}:
        if result.status == "partial":
            return ResultValidationOutcome(
                validation_status=ValidationStatus.PARTIAL,
                validation_error=result.error,
            )
        return ResultValidationOutcome(validation_status=ValidationStatus.SUCCESS)

    return ResultValidationOutcome(
        validation_status=ValidationStatus.FATAL_FAILURE,
        validation_error=result.error or f"Tool '{result.tool_name}' returned an invalid status",
    )

def _apply_retry_policy(
    outcome: ResultValidationOutcome,
    *,
    retry_count: int,
    max_retries: int,
) -> ResultValidationOutcome:
    if outcome.validation_status != ValidationStatus.RETRYABLE_FAILURE:
        return outcome.model_copy(update={"retry_required": False})

    next_retry_count = retry_count + 1
    if next_retry_count <= max_retries:
        return outcome.model_copy(update={"retry_required": True})

    return ResultValidationOutcome(
        validation_status=ValidationStatus.FATAL_FAILURE,
        retry_required=False,
        validation_error=outcome.validation_error or "Maximum retry limit reached",
        no_evidence=outcome.no_evidence,
    )


def _next_retry_count(current_retry_count: int, outcome: ResultValidationOutcome) -> int:
    if outcome.validation_status in _USABLE_STATUSES:
        return 0
    return current_retry_count + 1


def _extract_text(data: Any) -> str:
    if isinstance(data, str) and data.strip():
        return data.strip()
    if isinstance(data, dict):
        for key in ("text", "content", "transcript", "transcription", "summary"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def _extract_evidence_items(data: Any) -> list[Any]:
    if isinstance(data, dict) and isinstance(data.get("evidence"), list):
        return data["evidence"]
    if isinstance(data, list):
        return data
    return []


def _data_flag(data: Any, key: str) -> bool:
    return isinstance(data, dict) and bool(data.get(key))
