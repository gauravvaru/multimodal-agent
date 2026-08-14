from __future__ import annotations
import os
import time
from functools import lru_cache
from typing import Any
from pydantic import BaseModel, Field
from multimodal_agent.models.state import AgentState

LANGSMITH_TRACING_ENV = "LANGSMITH_TRACING"
LANGSMITH_API_KEY_ENV = "LANGSMITH_API_KEY"
LANGSMITH_PROJECT_ENV = "LANGSMITH_PROJECT"

LANGCHAIN_TRACING_ENV = "LANGCHAIN_TRACING_V2"
LANGCHAIN_API_KEY_ENV = "LANGCHAIN_API_KEY"
LANGCHAIN_PROJECT_ENV = "LANGCHAIN_PROJECT"

_SENSITIVE_METADATA_KEYS = frozenset(
    {
        "user_query",
        "query",
        "prompt",
        "file_content",
        "raw_content",
        "extracted_text",
        "api_key",
        "authorization",
        "password",
        "secret",
    }
)


class LangSmithSettings(BaseModel):
    """LangSmith tracing settings loaded from environment variables."""
    tracing_enabled: bool = False
    api_key: str | None = None
    project: str = Field(default="multimodal-agent", min_length=1)

@lru_cache
def get_langsmith_settings() -> LangSmithSettings:
    return LangSmithSettings(
        tracing_enabled=_env_flag(LANGSMITH_TRACING_ENV),
        api_key=os.environ.get(LANGSMITH_API_KEY_ENV) or None,
        project=os.environ.get(LANGSMITH_PROJECT_ENV, "multimodal-agent"),
    )

def is_langsmith_tracing_active(settings: LangSmithSettings | None = None) -> bool:
    resolved = settings or get_langsmith_settings()
    return resolved.tracing_enabled and bool(resolved.api_key)

def configure_langsmith_tracing(settings: LangSmithSettings | None = None) -> bool:
    resolved = settings or get_langsmith_settings()
    active = is_langsmith_tracing_active(resolved)

    if not active:
        os.environ[LANGSMITH_TRACING_ENV] = "false"
        os.environ[LANGCHAIN_TRACING_ENV] = "false"
        return False

    os.environ[LANGSMITH_TRACING_ENV] = "true"
    os.environ[LANGCHAIN_TRACING_ENV] = "true"
    os.environ[LANGSMITH_API_KEY_ENV] = resolved.api_key or ""
    os.environ[LANGCHAIN_API_KEY_ENV] = resolved.api_key or ""
    os.environ[LANGSMITH_PROJECT_ENV] = resolved.project
    os.environ[LANGCHAIN_PROJECT_ENV] = resolved.project
    return True

def build_graph_run_config(
    state: AgentState,
    *,
    recursion_limit: int = 100,
) -> dict[str, Any]:
    metadata = build_initial_run_metadata(state)
    return {
        "run_name": f"agent-run:{state.request_id}",
        "metadata": metadata,
        "tags": ["multimodal-agent", metadata.get("task_type", "unknown")],
        "recursion_limit": recursion_limit,
    }

def build_initial_run_metadata(state: AgentState) -> dict[str, Any]:
    return sanitize_metadata(
        {
            "request_id": state.request_id,
            "task_type": _infer_task_type(state),
            "input_count": len(state.input_artifacts),
        }
    )

def build_final_run_metadata(
    final_state: dict[str, Any] | AgentState,
    *,
    latency_ms: float,
) -> dict[str, Any]:
    state_dict = final_state.model_dump() if isinstance(final_state, AgentState) else final_state
    tool_results = state_dict.get("tool_results") or []
    errors = list(state_dict.get("errors") or [])
    final_response = state_dict.get("final_response")
    clarification_required = bool(state_dict.get("clarification_required"))

    return sanitize_metadata(
        {
            "request_id": str(state_dict.get("request_id") or ""),
            "task_type": _infer_task_type_from_dict(state_dict),
            "input_count": len(state_dict.get("input_artifacts") or []),
            "tool_count": len(tool_results),
            "success": _infer_success(
                errors=errors,
                final_response=final_response,
                clarification_required=clarification_required,
            ),
            "failure": bool(errors) and not _has_answer(final_response, clarification_required),
            "total_latency_ms": round(latency_ms, 2),
        }
    )

def invoke_graph_with_observability(
    state: AgentState,
    dependencies: Any,
    *,
    recursion_limit: int = 100,
) -> dict[str, Any]:
    from multimodal_agent.agent.graph import invoke_agent_graph

    run_config = build_graph_run_config(state, recursion_limit=recursion_limit)
    started = time.perf_counter()

    if is_langsmith_tracing_active():
        return _traced_invoke(
            state,
            dependencies,
            run_config=run_config,
            started=started,
            recursion_limit=recursion_limit,
        )

    result = invoke_agent_graph(
        state,
        dependencies=dependencies,
        run_config=run_config,
        recursion_limit=recursion_limit,
    )
    _record_local_run_metadata(result, started=started)
    return result


def stream_graph_with_observability(
    state: AgentState,
    dependencies: Any,
    *,
    recursion_limit: int = 100,
):
    from multimodal_agent.agent.graph import stream_agent_graph

    run_config = build_graph_run_config(state, recursion_limit=recursion_limit)
    yield from stream_agent_graph(
        state,
        dependencies=dependencies,
        run_config=run_config,
        recursion_limit=recursion_limit,
    )

def sanitize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    sanitized: dict[str, Any] = {}
    for key, value in metadata.items():
        lowered = key.lower()
        if lowered in _SENSITIVE_METADATA_KEYS:
            continue
        if any(marker in lowered for marker in ("secret", "password", "token", "api_key")):
            continue
        if isinstance(value, str) and _looks_like_secret(value):
            sanitized[key] = "[redacted]"
            continue
        sanitized[key] = value
    return sanitized

def _traced_invoke(
    state: AgentState,
    dependencies: Any,
    *,
    run_config: dict[str, Any],
    started: float,
    recursion_limit: int,
) -> dict[str, Any]:
    from langsmith import traceable
    from multimodal_agent.agent.graph import invoke_agent_graph
    @traceable(
        name="multimodal_agent_run",
        run_type="chain",
        metadata=run_config.get("metadata"),
        tags=run_config.get("tags"),
    )
    def _invoke() -> dict[str, Any]:
        return invoke_agent_graph(
            state,
            dependencies=dependencies,
            run_config=run_config,
            recursion_limit=recursion_limit,
        )
    result = _invoke()
    _record_local_run_metadata(result, started=started)
    return result

def _record_local_run_metadata(
    final_state: dict[str, Any] | AgentState,
    *,
    started: float,
) -> None:
    latency_ms = (time.perf_counter() - started) * 1000
    metadata = build_final_run_metadata(final_state, latency_ms=latency_ms)
    if not is_langsmith_tracing_active():
        return
    try:
        from langsmith.run_helpers import get_current_run_tree
    except ImportError:
        return
    run_tree = get_current_run_tree()
    if run_tree is None:
        return
    run_tree.metadata.update(metadata)
    run_tree.end(
        outputs={
            "request_id": metadata.get("request_id"),
            "success": metadata.get("success"),
            "tool_count": metadata.get("tool_count"),
            "total_latency_ms": metadata.get("total_latency_ms"),
        }
    )

def _infer_task_type(state: AgentState) -> str:
    if state.intent is not None:
        return state.intent.name
    return "unknown"


def _infer_task_type_from_dict(state_dict: dict[str, Any]) -> str:
    intent = state_dict.get("intent")
    if isinstance(intent, dict):
        name = intent.get("name")
        if isinstance(name, str) and name:
            return name
    if intent is not None and hasattr(intent, "name"):
        return str(intent.name)
    return "unknown"


def _infer_success(
    *,
    errors: list[str],
    final_response: Any,
    clarification_required: bool,
) -> bool:
    if errors and not _has_answer(final_response, clarification_required):
        return False
    return _has_answer(final_response, clarification_required) or not errors


def _has_answer(final_response: Any, clarification_required: bool) -> bool:
    if clarification_required:
        return True
    return isinstance(final_response, str) and bool(final_response.strip())


def _env_flag(name: str) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return False
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _looks_like_secret(value: str) -> bool:
    lowered = value.lower()
    return lowered.startswith(("sk-", "lsv2_"))


def clear_langsmith_settings_cache() -> None:
    get_langsmith_settings.cache_clear()
