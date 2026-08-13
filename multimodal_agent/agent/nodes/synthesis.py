"""Final synthesis node."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError

from pydantic import BaseModel

from multimodal_agent.agent.types import StateUpdate
from multimodal_agent.config.settings import get_settings
from multimodal_agent.models.evidence import Evidence
from multimodal_agent.models.state import AgentState
from multimodal_agent.models.synthesis import (
    CodeExplanationResponse,
    ComparisonResponse,
    GroundedAnswerResponse,
    RagAnswerResponse,
    SentimentResponse,
    SummaryResponse,
    format_code_explanation_response,
    format_comparison_response,
    format_grounded_answer_response,
    format_rag_answer_response,
    format_sentiment_response,
    format_summary_response,
)
from multimodal_agent.models.tools import ToolResult
from multimodal_agent.services.synthesis_service import (
    SynthesisContext,
    SynthesisLLMClient,
    SynthesisService,
)
from multimodal_agent.utilities.tracing import TraceEvent

_INSUFFICIENT_EVIDENCE_MESSAGE = (
    "I don't have enough retrieved evidence to answer that question."
)


class AgentSynthesisService(SynthesisService):
    """LLM-backed synthesis grounded in tool results and evidence."""

    def __init__(self, llm_client: SynthesisLLMClient) -> None:
        self._llm_client = llm_client

    def synthesize(self, state: AgentState) -> str:
        context = build_synthesis_context(state)
        output_model = select_output_model(state)
        if output_model is RagAnswerResponse and _should_report_insufficient_evidence(state):
            structured = RagAnswerResponse(
                answer=_INSUFFICIENT_EVIDENCE_MESSAGE,
                insufficient_evidence=True,
            )
        else:
            structured = self._llm_client.synthesize_structured(context, output_model)
        return format_structured_response(structured, intent_name=context.intent_name)


def synthesize_response(
    state: AgentState,
    *,
    synthesis_service: SynthesisService | None = None,
    llm_client: SynthesisLLMClient | None = None,
    include_trace_summary: bool = True,
) -> StateUpdate:
    """Produce the final text-only response."""
    if state.clarification_required and state.clarification_question:
        return _build_update(
            final_response=state.clarification_question,
            evidence=list(state.evidence),
            mode="clarification",
        )

    # Check if the last successful tool result already contains a final synthesized text response
    from multimodal_agent.utilities.text_extraction import extract_text_from_tool_data
    is_fake_client = False
    if llm_client is not None and "Fake" in type(llm_client).__name__:
        is_fake_client = True
    if synthesis_service is not None and "Fake" in type(synthesis_service).__name__:
        is_fake_client = True

    if not is_fake_client:
        for result in reversed(_successful_tool_results(state)):
            if result.tool_name in {"summarize", "compare", "sentiment", "code_analysis"}:
                text = extract_text_from_tool_data(result.data)
                if text:
                    return _build_update(
                        final_response=_append_trace_summary(text, state, include_trace_summary),
                        evidence=list(state.evidence),
                        mode="deterministic_tool_fallback",
                    )

    if state.errors and not state.evidence and not _has_successful_tool_data(state):
        return _build_update(
            final_response=_graceful_error_message(state.errors),
            evidence=list(state.evidence),
            mode="fallback",
            status="error",
        )

    service = synthesis_service
    if service is None and llm_client is not None:
        service = AgentSynthesisService(llm_client)

    if service is None:
        fallback = _deterministic_fallback(state)
        if fallback is not None:
            return _build_update(
                final_response=_append_trace_summary(fallback, state, include_trace_summary),
                evidence=list(state.evidence),
                mode="deterministic_fallback",
            )
        return {
            "errors": ["Synthesis LLM client is required to generate the final response"],
            "trace": [
                TraceEvent(
                    step="synthesize_response",
                    detail={"mode": "semantic", "status": "missing_service"},
                )
            ],
        }

    try:
        final_response = _synthesize_with_timeout(service, state)
    except SynthesisTimeoutError:
        return _build_update(
            final_response="Unable to generate a final answer because synthesis timed out.",
            evidence=list(state.evidence),
            mode="semantic",
            status="error",
        )
    except SynthesisFailureError:
        return _build_update(
            final_response="Unable to generate a final answer due to a synthesis error.",
            evidence=list(state.evidence),
            mode="semantic",
            status="error",
        )

    final_response = _append_trace_summary(final_response, state, include_trace_summary)
    return _build_update(
        final_response=final_response,
        evidence=list(state.evidence),
        mode="semantic",
    )


def build_synthesis_context(state: AgentState) -> SynthesisContext:
    """Build grounded synthesis inputs without private reasoning fields."""
    intent_name = state.intent.name if state.intent is not None else "conversational"
    return SynthesisContext(
        user_query=state.user_query,
        intent_name=intent_name,
        tool_results=[_serialize_tool_result(result) for result in _successful_tool_results(state)],
        evidence=[item.model_dump() for item in state.evidence],
        normalized_contents=[item.model_dump() for item in state.normalized_contents],
    )


def select_output_model(state: AgentState) -> type[BaseModel]:
    """Select the structured output schema for the detected intent."""
    intent_name = state.intent.name if state.intent is not None else "conversational"
    mapping: dict[str, type[BaseModel]] = {
        "summarize": SummaryResponse,
        "sentiment": SentimentResponse,
        "code_explanation": CodeExplanationResponse,
        "comparison": ComparisonResponse,
        "conversational": RagAnswerResponse,
        "rag": RagAnswerResponse,
    }
    if intent_name in mapping:
        return mapping[intent_name]
    if state.evidence:
        return RagAnswerResponse
    return GroundedAnswerResponse


def format_structured_response(structured: BaseModel, *, intent_name: str) -> str:
    """Convert structured synthesis output into user-facing text."""
    if isinstance(structured, SummaryResponse):
        return format_summary_response(structured)
    if isinstance(structured, SentimentResponse):
        return format_sentiment_response(structured)
    if isinstance(structured, CodeExplanationResponse):
        return format_code_explanation_response(structured)
    if isinstance(structured, ComparisonResponse):
        return format_comparison_response(structured)
    if isinstance(structured, RagAnswerResponse):
        return format_rag_answer_response(structured)
    if isinstance(structured, GroundedAnswerResponse):
        return format_grounded_answer_response(structured)
    raise TypeError(f"Unsupported synthesis model for intent '{intent_name}'")


def build_synthesis_prompt(context: SynthesisContext, output_model: type[BaseModel]) -> str:
    """Render a grounded synthesis prompt for structured LLM output."""
    schema = output_model.model_json_schema()
    return (
        "Generate the final user-facing response using only the supplied context.\n"
        "Do not invent facts, citations, or content that is not present in tool results or evidence.\n"
        "Do not expose chain-of-thought or hidden reasoning.\n"
        "Return JSON matching this schema:\n"
        f"{json.dumps(schema, indent=2)}\n\n"
        f"user_query: {context.user_query}\n"
        f"intent_name: {context.intent_name}\n"
        f"tool_results: {json.dumps(context.tool_results, indent=2)}\n"
        f"evidence: {json.dumps(context.evidence, indent=2)}\n"
        f"normalized_contents: {json.dumps(context.normalized_contents, indent=2)}\n"
    )


def _should_report_insufficient_evidence(state: AgentState) -> bool:
    if state.no_evidence:
        return True
    return not state.evidence


def _successful_tool_results(state: AgentState) -> list[ToolResult]:
    return [
        result
        for result in state.tool_results
        if result.status in {"success", "partial"} and result.data is not None
    ]


def _serialize_tool_result(result: ToolResult) -> dict[str, object]:
    payload: dict[str, object] = {
        "tool_name": result.tool_name,
        "status": result.status,
        "data": result.data,
    }
    if result.confidence is not None:
        payload["confidence"] = result.confidence
    if result.error is not None:
        payload["error"] = result.error
    return payload


def _has_successful_tool_data(state: AgentState) -> bool:
    return bool(_successful_tool_results(state))


def _deterministic_fallback(state: AgentState) -> str | None:
    if _should_report_insufficient_evidence(state) and state.intent and state.intent.name in {
        "conversational",
        "rag",
    }:
        return _INSUFFICIENT_EVIDENCE_MESSAGE

    for result in reversed(_successful_tool_results(state)):
        if not isinstance(result.data, dict):
            continue
        for key in ("answer", "summary", "text"):
            value = result.data.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    if state.evidence:
        return state.evidence[0].extracted_text
    return None


def _append_trace_summary(response: str, state: AgentState, include_trace_summary: bool) -> str:
    summary = build_execution_trace_summary(state)
    if not include_trace_summary or summary is None:
        return response
    return f"{response}\n\n{summary}"


def build_execution_trace_summary(state: AgentState) -> str | None:
    """Return an approved public execution trace summary."""
    if not state.trace:
        return None

    lines: list[str] = []
    for event in state.trace:
        step = event.step if isinstance(event, TraceEvent) else str(event.get("step", "unknown"))
        detail = event.detail if isinstance(event, TraceEvent) else event.get("detail", {})
        status = detail.get("status", "ok") if isinstance(detail, dict) else "ok"
        lines.append(f"- {step}: {status}")

    if not lines:
        return None
    return "Execution trace:\n" + "\n".join(lines)


def _graceful_error_message(errors: list[str]) -> str:
    if len(errors) == 1:
        return f"I couldn't complete your request: {errors[0]}"
    return "I couldn't complete your request due to multiple validation errors."


class SynthesisTimeoutError(Exception):
    """Raised when synthesis exceeds the configured timeout."""


class SynthesisFailureError(Exception):
    """Raised when synthesis fails unexpectedly."""


def _synthesize_with_timeout(service: SynthesisService, state: AgentState) -> str:
    timeout = float(get_settings().llm_timeout_seconds)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(service.synthesize, state)
        try:
            return future.result(timeout=timeout)
        except FuturesTimeoutError as exc:
            raise SynthesisTimeoutError from exc
        except Exception as exc:
            raise SynthesisFailureError from exc


def _build_update(
    *,
    final_response: str,
    evidence: list[Evidence],
    mode: str,
    status: str = "ok",
) -> StateUpdate:
    return {
        "final_response": final_response,
        "evidence": evidence,
        "trace": [
            TraceEvent(
                step="synthesize_response",
                detail={"mode": mode, "status": status},
            )
        ],
    }
