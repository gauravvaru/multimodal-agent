"""LangGraph stream event translation."""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from typing import Any

from multimodal_agent.agent import routing
from multimodal_agent.models.stream_events import AgentStreamEvent
from multimodal_agent.utilities.tracing import TraceEvent

_SAFE_STATE_KEYS = frozenset(
    {
        "clarification_required",
        "clarification_question",
        "current_step",
        "errors",
        "evidence",
        "final_response",
        "request_id",
        "retry_required",
        "tool_results",
        "trace",
        "validation_status",
    }
)

_NODE_MESSAGES: dict[str, str] = {
    routing.VALIDATION: "Validating request input",
    routing.NORMALIZATION: "Normalizing uploaded artifacts",
    routing.INTENT: "Detecting user intent",
    routing.CLARIFICATION: "Evaluating clarification requirements",
    routing.PLANNER: "Building execution plan",
    routing.VALIDATE_PLAN: "Validating execution plan",
    routing.EXECUTION: "Executing planned tools",
    routing.RESULT_VALIDATION: "Validating tool results",
    routing.EVIDENCE: "Building grounded evidence",
    routing.SYNTHESIS: "Synthesizing final answer",
}


class GraphStreamTranslator:
    """Translate LangGraph stream chunks into safe public events."""

    def __init__(self, *, request_id: str) -> None:
        self._request_id = request_id
        self._active_node: str | None = None
        self._node_started_at: dict[str, float] = {}
        self._seen_tool_starts: set[str] = set()
        self._accumulated_state: dict[str, Any] = {"request_id": request_id}
        self._emitted_final_answer = False

    def translate_chunk(self, chunk: Any) -> list[AgentStreamEvent]:
        if not isinstance(chunk, dict) or not chunk:
            return []

        events: list[AgentStreamEvent] = []
        for node_name, update in chunk.items():
            if not isinstance(node_name, str):
                continue
            normalized_update = _normalize_update(update)
            events.extend(self._begin_node(node_name))
            events.extend(self._events_from_update(node_name, normalized_update))
            self._merge_state(normalized_update)
            events.extend(self._maybe_emit_final_answer())
        return events

    def complete(self) -> list[AgentStreamEvent]:
        from multimodal_agent.services.agent_service import state_to_response

        events: list[AgentStreamEvent] = []
        if self._active_node is not None:
            events.extend(self._complete_node(self._active_node))
            self._active_node = None
        events.extend(self._maybe_emit_final_answer(force=True))
        response = state_to_response(self._accumulated_state, request_id=self._request_id)
        events.append(
            AgentStreamEvent(
                type="complete",
                request_id=self._request_id,
                response=_safe_response_payload(response),
                errors=response.errors,
            )
        )
        return events

    def error_event(self, message: str) -> AgentStreamEvent:
        return AgentStreamEvent(
            type="error",
            request_id=self._request_id,
            message=message,
            errors=[message],
        )

    def _begin_node(self, node_name: str) -> list[AgentStreamEvent]:
        if self._active_node == node_name:
            return []

        events: list[AgentStreamEvent] = []
        if self._active_node is not None:
            events.extend(self._complete_node(self._active_node))

        self._active_node = node_name
        self._node_started_at[node_name] = time.perf_counter()
        return [
            *events,
            AgentStreamEvent(
                type="node_update",
                request_id=self._request_id,
                node=node_name,
                status="started",
                message=_NODE_MESSAGES.get(node_name, f"Running {node_name}"),
            ),
            AgentStreamEvent(
                type="status_update",
                request_id=self._request_id,
                message=_NODE_MESSAGES.get(node_name, f"Running {node_name}"),
            ),
        ]

    def _complete_node(self, node_name: str) -> list[AgentStreamEvent]:
        started = self._node_started_at.pop(node_name, None)
        latency_ms = round((time.perf_counter() - started) * 1000, 2) if started else None
        if self._active_node == node_name:
            self._active_node = None
        return [
            AgentStreamEvent(
                type="node_update",
                request_id=self._request_id,
                node=node_name,
                status="completed",
                latency_ms=latency_ms,
                message=f"Completed {node_name}",
            )
        ]

    def _events_from_update(self, node_name: str, update: dict[str, Any]) -> list[AgentStreamEvent]:
        events: list[AgentStreamEvent] = []

        if node_name == routing.EXECUTION:
            events.extend(self._tool_events_from_execution(update))

        trace_items = update.get("trace") or []
        if isinstance(trace_items, list):
            for item in trace_items:
                detail = _trace_detail(item)
                if detail.get("status") == "failed" and node_name == routing.EXECUTION:
                    tool_name = detail.get("tool_name")
                    if isinstance(tool_name, str):
                        events.append(
                            AgentStreamEvent(
                                type="tool_update",
                                request_id=self._request_id,
                                tool=tool_name,
                                status="failed",
                                message=f"Tool {tool_name} failed",
                            )
                        )

        if update.get("errors"):
            events.append(
                AgentStreamEvent(
                    type="status_update",
                    request_id=self._request_id,
                    message="Execution reported errors",
                )
            )

        return events

    def _tool_events_from_execution(self, update: dict[str, Any]) -> list[AgentStreamEvent]:
        events: list[AgentStreamEvent] = []
        trace_items = update.get("trace") or []
        tool_names: list[str] = []

        if isinstance(trace_items, list):
            for item in trace_items:
                detail = _trace_detail(item)
                tool_name = detail.get("tool_name")
                if isinstance(tool_name, str):
                    tool_names.append(tool_name)

        tool_results = update.get("tool_results") or []
        if isinstance(tool_results, list):
            for result in tool_results:
                if isinstance(result, dict):
                    tool_name = result.get("tool_name")
                    if isinstance(tool_name, str):
                        tool_names.append(tool_name)

        for tool_name in tool_names:
            if tool_name not in self._seen_tool_starts:
                self._seen_tool_starts.add(tool_name)
                events.append(
                    AgentStreamEvent(
                        type="tool_update",
                        request_id=self._request_id,
                        tool=tool_name,
                        status="running",
                        message=f"Running tool {tool_name}",
                    )
                )

        if isinstance(tool_results, list):
            for result in tool_results:
                result_data = _normalize_tool_result(result)
                if result_data is None:
                    continue
                tool_name = result_data.get("tool_name")
                if not isinstance(tool_name, str):
                    continue
                status = str(result_data.get("status") or "completed")
                mapped_status = "failed" if status == "failed" else "completed"
                events.append(
                    AgentStreamEvent(
                        type="tool_update",
                        request_id=self._request_id,
                        tool=tool_name,
                        status=mapped_status,
                        latency_ms=_maybe_float(result_data.get("latency_ms")),
                        message=f"Tool {tool_name} {mapped_status}",
                    )
                )
        return events

    def _maybe_emit_final_answer(self, *, force: bool = False) -> list[AgentStreamEvent]:
        if self._emitted_final_answer:
            return []

        final_response = self._accumulated_state.get("final_response")
        if not isinstance(final_response, str) or not final_response.strip():
            if not force:
                return []
            clarification_required = bool(self._accumulated_state.get("clarification_required"))
            clarification_question = self._accumulated_state.get("clarification_question")
            if clarification_required and isinstance(clarification_question, str):
                final_response = clarification_question
            else:
                return []

        self._emitted_final_answer = True
        return [
            AgentStreamEvent(
                type="final_answer",
                request_id=self._request_id,
                final_answer=final_response,
                message="Final answer available",
            )
        ]

    def _merge_state(self, update: dict[str, Any]) -> None:
        for key, value in update.items():
            if key not in _SAFE_STATE_KEYS:
                continue
            if key in self._accumulated_state and isinstance(self._accumulated_state[key], list) and isinstance(value, list):
                self._accumulated_state[key] = [*self._accumulated_state[key], *value]
            else:
                self._accumulated_state[key] = value


def iter_graph_stream_events(
    stream: Iterator[Any],
    *,
    request_id: str,
) -> Iterator[AgentStreamEvent]:
    """Convert a LangGraph stream iterator into public SSE events."""
    translator = GraphStreamTranslator(request_id=request_id)
    for chunk in stream:
        yield from translator.translate_chunk(chunk)
    yield from translator.complete()


def serialize_stream_event(event: AgentStreamEvent) -> str:
    """Serialize one stream event as an SSE data line payload."""
    return event.to_sse_data()


def format_sse(event: AgentStreamEvent) -> str:
    """Format a stream event for Server-Sent Events transport."""
    return f"data: {serialize_stream_event(event)}\n\n"


def _normalize_update(update: Any) -> dict[str, Any]:
    if isinstance(update, dict):
        return update
    if hasattr(update, "model_dump"):
        return update.model_dump()
    return {}


def _trace_detail(item: Any) -> dict[str, Any]:
    if isinstance(item, TraceEvent):
        return dict(item.detail)
    if isinstance(item, dict):
        detail = item.get("detail")
        if isinstance(detail, dict):
            return detail
    return {}


def _maybe_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _normalize_tool_result(result: Any) -> dict[str, Any] | None:
    if isinstance(result, dict):
        return result
    if hasattr(result, "model_dump"):
        return result.model_dump()
    return None


def _safe_response_payload(response: Any) -> dict[str, Any]:
    from multimodal_agent.api.schemas import agent_response_to_run_response

    run_response = agent_response_to_run_response(response)
    return json.loads(run_response.model_dump_json())
