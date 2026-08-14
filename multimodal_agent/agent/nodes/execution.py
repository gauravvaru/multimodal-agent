from __future__ import annotations
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from multimodal_agent.agent.nodes.result_validation import get_max_tool_retries
from multimodal_agent.agent.plan_validation import get_max_agent_steps
from multimodal_agent.agent.routing import AGENT_STEP_COUNT_KEY
from multimodal_agent.agent.types import StateUpdate
from multimodal_agent.config.settings import get_settings
from multimodal_agent.models.plan import PlanStep
from multimodal_agent.models.state import AgentState
from multimodal_agent.models.tools import ToolResult
from multimodal_agent.tools.registry import ToolRegistry, create_default_tool_registry
from multimodal_agent.tools.specs import ALLOWED_TOOL_INPUTS
from multimodal_agent.utilities.text_extraction import extract_text_from_tool_data
from multimodal_agent.utilities.tracing import TraceEvent

_EXECUTOR_SUCCESS_STATUSES = frozenset({"success", "partial"})

class ToolExecutor:
    def __init__(
        self,
        registry: ToolRegistry | None = None,
        *,
        timeout_seconds: float | None = None,
        max_agent_steps: int | None = None,
    ) -> None:
        self._registry = registry or create_default_tool_registry()
        self._timeout_seconds = timeout_seconds
        self._max_agent_steps = max_agent_steps

    def execute_current_step(self, state: AgentState) -> tuple[ToolResult, PlanStep | None]:
        if state.plan is None or not state.plan.steps:
            raise ExecutionPreconditionError("No plan available for execution")

        if state.current_step >= len(state.plan.steps):
            raise ExecutionPreconditionError("All plan steps have already been executed")

        if self._is_max_agent_steps_reached(state):
            raise MaxAgentStepsExceededError(get_max_agent_steps())

        if self._retry_limit_exceeded(state):
            raise RetryLimitExceededError(state.plan.max_retries)

        step = state.plan.steps[state.current_step]
        started = time.perf_counter()

        if step.tool_name not in self._registry.names():
            return (
                _failed_result(
                    step.tool_name,
                    error=f"Tool '{step.tool_name}' is not registered",
                    latency_ms=_elapsed_ms(started),
                ),
                step,
            )

        input_errors = validate_tool_arguments(step.tool_name, step.inputs)
        if input_errors:
            return (
                _failed_result(
                    step.tool_name,
                    error="; ".join(input_errors),
                    latency_ms=_elapsed_ms(started),
                ),
                step,
            )
        tool = self._registry.get(step.tool_name)
        resolved_inputs = resolve_tool_inputs(step, state)

        try:
            raw_result = self._invoke_tool(tool, resolved_inputs)
        except ToolExecutionTimeoutError:
            raise
        except Exception as exc:
            return (
                _failed_result(
                    step.tool_name,
                    error=str(exc),
                    latency_ms=_elapsed_ms(started),
                ),
                step,
            )
        return (
            normalize_tool_result(
                raw_result,
                tool_name=step.tool_name,
                latency_ms=_elapsed_ms(started),
            ),
            step,
        )

    def _invoke_tool(
        self,
        tool: Callable[..., ToolResult],
        inputs: dict[str, str],
    ) -> ToolResult:
        timeout = self._timeout_seconds
        if timeout is None:
            timeout = float(get_settings().request_timeout_seconds)
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(tool, **inputs)
            try:
                return future.result(timeout=timeout)
            except FuturesTimeoutError as exc:
                raise ToolExecutionTimeoutError(timeout) from exc

    def _is_max_agent_steps_reached(self, state: AgentState) -> bool:
        limit = self._max_agent_steps if self._max_agent_steps is not None else get_max_agent_steps()
        return _agent_step_count(state) >= limit
    @staticmethod
    def _retry_limit_exceeded(state: AgentState) -> bool:
        if state.plan is None:
            return False
        max_retries = get_max_tool_retries(state)
        retry_count = state.retry_count or int(state.constraints.get("retry_count", 0))
        return retry_count > max_retries

class ExecutionPreconditionError(Exception):
    """Raised when the executor cannot run a step."""
class MaxAgentStepsExceededError(Exception):
    """Raised when the configured agent step limit is reached."""
    def __init__(self, limit: int) -> None:
        self.limit = limit
        super().__init__(f"Maximum agent steps ({limit}) reached")

class RetryLimitExceededError(Exception):
    def __init__(self, limit: int) -> None:
        self.limit = limit
        super().__init__(f"Retry limit ({limit}) exceeded")

class ToolExecutionTimeoutError(Exception):
    def __init__(self, timeout_seconds: float) -> None:
        self.timeout_seconds = timeout_seconds
        super().__init__("Tool execution timed out")

def execute_tools(
    state: AgentState,
    *,
    registry: ToolRegistry | None = None,
    timeout_seconds: float | None = None,
    max_agent_steps: int | None = None,
) -> StateUpdate:
    executor = ToolExecutor(
        registry=registry,
        timeout_seconds=timeout_seconds,
        max_agent_steps=max_agent_steps,
    )
    if state.plan is None or not state.plan.steps:
        return _execution_error_update(
            errors=["No plan available for execution"],
            trace_detail={"status": "missing_plan"},
        )

    if state.current_step >= len(state.plan.steps):
        return {
            "trace": [
                TraceEvent(
                    step="execute_tools",
                    detail={"status": "complete", "current_step": state.current_step},
                )
            ],
        }
    try:
        result, step = executor.execute_current_step(state)
    except MaxAgentStepsExceededError as exc:
        return _execution_error_update(
            errors=[str(exc)],
            tool_results=[
                ToolResult(
                    tool_name=state.plan.steps[state.current_step].tool_name,
                    status="failed",
                    error=str(exc),
                )
            ],
            current_step=state.current_step + 1,
            constraints=_increment_agent_step_count(state),
            trace_detail={
                "status": "max_agent_steps",
                "step_id": state.plan.steps[state.current_step].step_id,
            },
        )
    except RetryLimitExceededError as exc:
        step = state.plan.steps[state.current_step]
        return _execution_error_update(
            errors=[str(exc)],
            tool_results=[
                ToolResult(
                    tool_name=step.tool_name,
                    status="failed",
                    error=str(exc),
                )
            ],
            constraints=_increment_agent_step_count(state),
            trace_detail={"status": "retry_limit", "step_id": step.step_id},
        )
    except ExecutionPreconditionError as exc:
        return _execution_error_update(
            errors=[str(exc)],
            trace_detail={"status": "precondition_failed"},
        )
    except ToolExecutionTimeoutError:
        step = state.plan.steps[state.current_step]
        return {
            "tool_results": [
                ToolResult(
                    tool_name=step.tool_name,
                    status="failed",
                    error="Tool execution timed out",
                    latency_ms=None,
                )
            ],
            "current_step": state.current_step + 1,
            "constraints": _increment_agent_step_count(state),
            "trace": [
                TraceEvent(
                    step="execute_tools",
                    detail={
                        "step_id": step.step_id,
                        "tool_name": step.tool_name,
                        "status": "failed",
                        "reason": "timeout",
                    },
                )
            ],
        }
    assert step is not None
    return {
        "tool_results": [result],
        "current_step": state.current_step + 1,
        "constraints": _increment_agent_step_count(state),
        "trace": [
            TraceEvent(
                step="execute_tools",
                detail={
                    "step_id": step.step_id,
                    "tool_name": step.tool_name,
                    "status": result.status,
                    "latency_ms": result.latency_ms,
                },
            )
        ],
    }

def validate_tool_arguments(tool_name: str, inputs: dict[str, str]) -> list[str]:
    allowed = ALLOWED_TOOL_INPUTS.get(tool_name)
    if allowed is None:
        return [f"tool '{tool_name}' has no registered input specification"]

    errors: list[str] = []
    for key, value in inputs.items():
        if key not in allowed:
            errors.append(f"invalid argument '{key}' for tool '{tool_name}'")
        if not isinstance(value, str):
            errors.append(f"argument '{key}' must be a string")

    return errors

def resolve_tool_inputs(step: PlanStep, state: AgentState) -> dict[str, str]:
    allowed = ALLOWED_TOOL_INPUTS.get(step.tool_name, frozenset())
    resolved = {key: value for key, value in step.inputs.items() if key in allowed}
    artifact_id = resolved.get("artifact_id")
    if artifact_id:
        for artifact in state.normalized_contents:
            if artifact.artifact_id == artifact_id:
                if artifact.source and "source" in allowed:
                    resolved.setdefault("source", artifact.source)
                if artifact.metadata.get("youtube_url") and "url" in allowed:
                    url = artifact.metadata["youtube_url"]
                    if isinstance(url, str):
                        resolved.setdefault("url", url)
    upstream_text = _text_from_prior_tool_results(state)
    if "text" in allowed and "text" not in resolved and upstream_text:
        resolved["text"] = upstream_text
    
    if step.tool_name == "code_analysis" and upstream_text:
        resolved["text"] = upstream_text
        resolved.pop("code", None)
        resolved.pop("source", None)

    if step.tool_name == "rag" and "query" in allowed:
        resolved.setdefault("query", state.user_query.strip())
        if "context" in allowed and upstream_text:
            resolved.setdefault("context", upstream_text)

    if step.tool_name == "compare" and "sources" in allowed and "sources" not in resolved:
        sources = _compare_sources_from_prior_results(state)
        if sources:
            resolved["sources"] = ",".join(sources)

    if step.tool_name == "youtube_transcript" and "url" not in resolved:
        url = _youtube_url_from_prior_results(state)
        if url:
            resolved["url"] = url

    return resolved

def _text_from_prior_tool_results(state: AgentState) -> str:
    parts: list[str] = []
    for result in state.tool_results:
        if result.status not in _EXECUTOR_SUCCESS_STATUSES or result.data is None:
            continue
        text = extract_text_from_tool_data(result.data)
        if text:
            parts.append(text)
    return "\n\n".join(parts)

def _compare_sources_from_prior_results(state: AgentState) -> list[str]:
    sources: list[str] = []
    for result in state.tool_results:
        if result.status not in _EXECUTOR_SUCCESS_STATUSES or result.data is None:
            continue
        text = extract_text_from_tool_data(result.data)
        if text:
            sources.append(text)
    return sources

def _youtube_url_from_prior_results(state: AgentState) -> str | None:
    from multimodal_agent.utilities.urls import extract_urls, validate_youtube_url

    for result in state.tool_results:
        if result.status not in _EXECUTOR_SUCCESS_STATUSES or result.data is None:
            continue
        text = extract_text_from_tool_data(result.data)
        if not text:
            continue
        for url in extract_urls(text):
            if not validate_youtube_url(url):
                return url
    return None

def normalize_tool_result(
    raw_result: ToolResult,
    *,
    tool_name: str,
    latency_ms: float,
) -> ToolResult:
    if not isinstance(raw_result, ToolResult):
        return ToolResult(
            tool_name=tool_name,
            status="failed",
            error="Tool returned an invalid result type",
            latency_ms=latency_ms,
        )
    if raw_result.status in _EXECUTOR_SUCCESS_STATUSES:
        status = raw_result.status
    else:
        status = "failed"

    return raw_result.model_copy(
        update={
            "tool_name": tool_name,
            "status": status,
            "latency_ms": raw_result.latency_ms if raw_result.latency_ms is not None else latency_ms,
        }
    )

def _failed_result(tool_name: str, *, error: str, latency_ms: float) -> ToolResult:
    return ToolResult(
        tool_name=tool_name,
        status="failed",
        error=error,
        latency_ms=latency_ms,
    )

def _execution_error_update(
    *,
    errors: list[str],
    trace_detail: dict[str, object],
    tool_results: list[ToolResult] | None = None,
    current_step: int | None = None,
    constraints: dict[str, str | int | float | bool | None] | None = None,
) -> StateUpdate:
    update: StateUpdate = {
        "errors": errors,
        "trace": [TraceEvent(step="execute_tools", detail=trace_detail)],
    }
    if tool_results is not None:
        update["tool_results"] = tool_results
    if current_step is not None:
        update["current_step"] = current_step
    if constraints is not None:
        update["constraints"] = constraints
    return update

def _agent_step_count(state: AgentState) -> int:
    value = state.constraints.get(AGENT_STEP_COUNT_KEY, 0)
    return int(value) if value is not None else 0

def _increment_agent_step_count(state: AgentState) -> dict[str, str | int | float | bool | None]:
    constraints = dict(state.constraints)
    constraints[AGENT_STEP_COUNT_KEY] = _agent_step_count(state) + 1
    return constraints

def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 2)
