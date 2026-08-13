"""Agent invocation service."""

from __future__ import annotations

import uuid
from collections.abc import Callable, Iterator
from typing import Any

from multimodal_agent.agent.graph import GraphDependencies
from multimodal_agent.agent.streaming import iter_graph_stream_events
from multimodal_agent.config.langsmith import (
    invoke_graph_with_observability,
    stream_graph_with_observability,
)
from multimodal_agent.models.evidence import Evidence
from multimodal_agent.models.requests import InputArtifact
from multimodal_agent.models.responses import AgentResponse
from multimodal_agent.models.state import AgentState
from multimodal_agent.models.stream_events import AgentStreamEvent
from multimodal_agent.models.tools import ToolResult
from multimodal_agent.services.dependencies import build_default_graph_dependencies
from multimodal_agent.utilities.security import safe_client_error_message
from multimodal_agent.utilities.tracing import TraceEvent

GraphInvoker = Callable[[AgentState], dict[str, Any]]
GraphStreamer = Callable[[AgentState], Iterator[Any]]


class AgentService:
    """Coordinate request handling outside FastAPI route handlers."""

    def __init__(
        self,
        *,
        graph_dependencies: GraphDependencies | None = None,
        graph_invoker: GraphInvoker | None = None,
        graph_streamer: GraphStreamer | None = None,
        request_id_factory: Callable[[], str] | None = None,
    ) -> None:
        self._graph_dependencies = graph_dependencies
        self._graph_invoker = graph_invoker or self._default_graph_invoker
        self._graph_streamer = graph_streamer or self._default_graph_streamer
        self._request_id_factory = request_id_factory or (lambda: str(uuid.uuid4()))

    def run(
        self,
        query: str,
        files: list[InputArtifact] | None = None,
    ) -> AgentResponse:
        """Execute the agent workflow for a user query and optional files."""
        request_id = self._request_id_factory()
        artifacts = list(files or [])

        validation_errors = validate_run_inputs(query, artifacts)
        if validation_errors:
            return build_error_response(
                request_id=request_id,
                errors=validation_errors,
                answer="Unable to process the request due to invalid input.",
            )

        initial_state = AgentState(
            request_id=request_id,
            user_query=query.strip(),
            input_artifacts=artifacts,
        )

        try:
            final_state = self._graph_invoker(initial_state)
        except Exception:  # noqa: BLE001 - graph failures must return a stable API response
            import traceback
            traceback.print_exc()
            return build_error_response(
                request_id=request_id,
                errors=[safe_client_error_message(context="agent run")],
                answer="Unable to complete the request due to an unexpected error.",
            )

        return state_to_response(final_state, request_id=request_id)

    def run_stream(
        self,
        query: str,
        files: list[InputArtifact] | None = None,
    ) -> Iterator[AgentStreamEvent]:
        """Execute the agent workflow and yield safe streaming events."""
        request_id = self._request_id_factory()
        artifacts = list(files or [])

        validation_errors = validate_run_inputs(query, artifacts)
        if validation_errors:
            yield AgentStreamEvent(
                type="error",
                request_id=request_id,
                message="Unable to process the request due to invalid input.",
                errors=validation_errors,
            )
            yield AgentStreamEvent(
                type="complete",
                request_id=request_id,
                response=_validation_error_payload(request_id, validation_errors),
                errors=validation_errors,
            )
            return

        initial_state = AgentState(
            request_id=request_id,
            user_query=query.strip(),
            input_artifacts=artifacts,
        )

        try:
            stream = self._graph_streamer(initial_state)
            yield from iter_graph_stream_events(stream, request_id=request_id)
        except Exception as exc:
            import logging
            logger = logging.getLogger(__name__)
            logger.exception("Unhandled exception during agent stream execution")
            message = f"{safe_client_error_message(context='agent stream')}: {exc}"
            yield AgentStreamEvent(
                type="error",
                request_id=request_id,
                message=message,
                errors=[message],
            )
            yield AgentStreamEvent(
                type="complete",
                request_id=request_id,
                errors=[message],
            )

    def _resolve_graph_dependencies(self) -> GraphDependencies:
        if self._graph_dependencies is not None:
            return self._graph_dependencies
        return build_default_graph_dependencies()

    def _default_graph_invoker(self, state: AgentState) -> dict[str, Any]:
        return invoke_graph_with_observability(state, self._resolve_graph_dependencies())

    def _default_graph_streamer(self, state: AgentState) -> Iterator[Any]:
        return stream_graph_with_observability(state, self._resolve_graph_dependencies())


def validate_run_inputs(query: str, files: list[InputArtifact]) -> list[str]:
    """Validate inbound service inputs before graph invocation."""
    errors: list[str] = []

    if not query or not query.strip():
        errors.append("query is required")

    for index, artifact in enumerate(files):
        prefix = f"files[{index}]"
        if not artifact.filename and not artifact.source:
            errors.append(f"{prefix}: filename or source is required")
        if artifact.filename is not None and not artifact.filename.strip():
            errors.append(f"{prefix}: filename cannot be blank")

    return errors


def state_to_response(
    final_state: dict[str, Any] | AgentState,
    *,
    request_id: str,
) -> AgentResponse:
    """Convert final LangGraph state into the public response model."""
    state_dict = final_state.model_dump() if isinstance(final_state, AgentState) else final_state

    clarification_required = bool(state_dict.get("clarification_required"))
    clarification_question = state_dict.get("clarification_question")
    errors = list(state_dict.get("errors") or [])
    trace = serialize_trace(state_dict.get("trace") or [])
    tool_results = normalize_tool_results(state_dict.get("tool_results") or [])
    evidence = serialize_evidence(state_dict.get("evidence") or [])
    answer = resolve_answer(
        state_dict,
        clarification_required=clarification_required,
        errors=errors,
    )

    return AgentResponse(
        request_id=str(state_dict.get("request_id") or request_id),
        answer=answer,
        clarification_required=clarification_required,
        clarification_question=clarification_question,
        trace=trace,
        tool_results=tool_results,
        evidence=evidence,
        errors=errors,
    )


def build_error_response(
    *,
    request_id: str,
    errors: list[str],
    answer: str,
) -> AgentResponse:
    """Return a stable error response without invoking the graph."""
    return AgentResponse(
        request_id=request_id,
        answer=answer,
        errors=errors,
    )


def resolve_answer(
    state_dict: dict[str, Any],
    *,
    clarification_required: bool,
    errors: list[str],
) -> str:
    """Derive the public answer string from final graph state."""
    final_response = state_dict.get("final_response")
    if isinstance(final_response, str) and final_response.strip():
        return final_response

    if clarification_required:
        return ""

    if errors:
        return "Unable to complete the request."

    return ""


def serialize_trace(trace: list[Any]) -> list[Any]:
    """Normalize trace events for API serialization."""
    serialized: list[Any] = []
    for item in trace:
        if isinstance(item, TraceEvent):
            serialized.append(item.model_dump())
        elif isinstance(item, dict):
            serialized.append(item)
        else:
            serialized.append({"step": str(item), "detail": {}})
    return serialized


def serialize_evidence(evidence: list[Any]) -> list[Any]:
    """Normalize evidence items for API serialization."""
    serialized: list[Any] = []
    for item in evidence:
        if isinstance(item, Evidence):
            serialized.append(item.model_dump())
        elif isinstance(item, dict):
            serialized.append(item)
        else:
            serialized.append(item)
    return serialized


def normalize_tool_results(tool_results: list[Any]) -> list[ToolResult]:
    """Normalize tool results into the public response contract."""
    normalized: list[ToolResult] = []
    for item in tool_results:
        if isinstance(item, ToolResult):
            normalized.append(item)
        elif isinstance(item, dict):
            normalized.append(ToolResult.model_validate(item))
    return normalized


def _validation_error_payload(request_id: str, errors: list[str]) -> dict[str, Any]:
    response = build_error_response(
        request_id=request_id,
        errors=errors,
        answer="Unable to process the request due to invalid input.",
    )
    from multimodal_agent.api.schemas import agent_response_to_run_response

    return agent_response_to_run_response(response).model_dump()
