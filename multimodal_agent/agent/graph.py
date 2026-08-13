"""LangGraph workflow assembly."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from langgraph.graph import END, START, StateGraph

from multimodal_agent.agent import routing
from multimodal_agent.agent.nodes.clarification import clarification_gate
from multimodal_agent.agent.nodes.evidence import build_evidence
from multimodal_agent.agent.nodes.execution import execute_tools
from multimodal_agent.agent.nodes.intent import detect_intent
from multimodal_agent.agent.nodes.normalization import normalize_inputs
from multimodal_agent.agent.nodes.planner import create_plan, validate_plan
from multimodal_agent.agent.nodes.result_validation import validate_results
from multimodal_agent.agent.nodes.synthesis import synthesize_response
from multimodal_agent.agent.nodes.validation import validate_input
from multimodal_agent.agent.types import StateUpdate
from multimodal_agent.models.state import AgentState
from multimodal_agent.services.intent_service import IntentService
from multimodal_agent.services.planner_service import PlannerService
from multimodal_agent.services.synthesis_service import (
    SynthesisLLMClient,
    SynthesisService,
)
from multimodal_agent.tools.registry import ToolRegistry


@dataclass(kw_only=True)
class GraphDependencies:
    """Injected runtime dependencies for the agent graph."""

    registry: ToolRegistry | None = None
    llm_client: SynthesisLLMClient | None = None
    intent_service: IntentService | None = None
    planner_service: PlannerService | None = None
    synthesis_service: SynthesisService | None = None
    max_agent_steps: int = 15

def build_agent_graph(
    dependencies: GraphDependencies | None = None,
) -> Any:
    """Construct and compile the LangGraph agent workflow."""
    deps = dependencies or GraphDependencies()
    builder = StateGraph(AgentState)

    builder.add_node(routing.VALIDATION, _validation_node())
    builder.add_node(routing.NORMALIZATION, _normalization_node())
    builder.add_node(routing.INTENT, _intent_node(deps))
    builder.add_node(routing.CLARIFICATION, _clarification_node())
    builder.add_node(routing.PLANNER, _planner_node(deps))
    builder.add_node(routing.VALIDATE_PLAN, _validate_plan_node(deps))
    builder.add_node(routing.EXECUTION, _execution_node(deps))
    builder.add_node(routing.RESULT_VALIDATION, _result_validation_node())
    builder.add_node(routing.EVIDENCE, _evidence_node())
    builder.add_node(routing.SYNTHESIS, _synthesis_node(deps))

    builder.add_edge(START, routing.VALIDATION)
    builder.add_edge(routing.VALIDATION, routing.NORMALIZATION)
    builder.add_edge(routing.NORMALIZATION, routing.INTENT)
    builder.add_edge(routing.INTENT, routing.CLARIFICATION)
    builder.add_conditional_edges(
        routing.CLARIFICATION,
        _route(deps, routing.route_after_clarification),
        _clarification_targets(),
    )
    builder.add_edge(routing.PLANNER, routing.VALIDATE_PLAN)
    builder.add_conditional_edges(
        routing.VALIDATE_PLAN,
        _route(deps, routing.route_after_plan_validation),
        _plan_validation_targets(),
    )
    builder.add_conditional_edges(
        routing.EXECUTION,
        _route(deps, routing.route_after_execution),
        _execution_targets(),
    )
    builder.add_conditional_edges(
        routing.RESULT_VALIDATION,
        _route(deps, routing.route_after_result_validation),
        _result_validation_targets(),
    )
    builder.add_edge(routing.EVIDENCE, routing.SYNTHESIS)
    builder.add_edge(routing.SYNTHESIS, END)

    return builder.compile()


def invoke_agent_graph(
    state: AgentState,
    *,
    dependencies: GraphDependencies | None = None,
    recursion_limit: int = 100,
    run_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Invoke the compiled agent graph and return the final state."""
    graph = build_agent_graph(dependencies)
    config = _merge_run_config(run_config, recursion_limit=recursion_limit)
    return graph.invoke(state, config=config)


def stream_agent_graph(
    state: AgentState,
    *,
    dependencies: GraphDependencies | None = None,
    recursion_limit: int = 100,
    run_config: dict[str, Any] | None = None,
):
    """Stream LangGraph execution updates for the agent workflow."""
    graph = build_agent_graph(dependencies)
    config = _merge_run_config(run_config, recursion_limit=recursion_limit)
    yield from graph.stream(
        state,
        stream_mode="updates",
        config=config,
    )


def get_graph_topology(
    dependencies: GraphDependencies | None = None,
) -> Any:
    """Return the graph topology for inspection or rendering."""
    graph = build_agent_graph(dependencies)
    return graph.get_graph()


def _validation_node() -> Callable[[AgentState], StateUpdate]:
    def node(state: AgentState) -> StateUpdate:
        return validate_input(state)

    return node


def _normalization_node() -> Callable[[AgentState], StateUpdate]:
    def node(state: AgentState) -> StateUpdate:
        return normalize_inputs(state)

    return node


def _intent_node(deps: GraphDependencies) -> Callable[[AgentState], StateUpdate]:
    def node(state: AgentState) -> StateUpdate:
        return detect_intent(state, intent_service=deps.intent_service)

    return node


def _clarification_node() -> Callable[[AgentState], StateUpdate]:
    def node(state: AgentState) -> StateUpdate:
        return clarification_gate(state)

    return node


def _planner_node(deps: GraphDependencies) -> Callable[[AgentState], StateUpdate]:
    def node(state: AgentState) -> StateUpdate:
        return create_plan(
            state,
            planner_service=deps.planner_service,
            registry=deps.registry,
        )

    return node


def _validate_plan_node(deps: GraphDependencies) -> Callable[[AgentState], StateUpdate]:
    def node(state: AgentState) -> StateUpdate:
        return validate_plan(state, registry=deps.registry)

    return node


def _execution_node(deps: GraphDependencies) -> Callable[[AgentState], StateUpdate]:
    def node(state: AgentState) -> StateUpdate:
        working_state = state
        merged: StateUpdate = {}
        if state.retry_required:
            retry_update = routing.build_retry_execution_update(state)
            merged = _merge_updates(merged, retry_update)
            working_state = state.model_copy(update=retry_update)

        execution_update = execute_tools(
            working_state,
            registry=deps.registry,
            max_agent_steps=deps.max_agent_steps,
        )
        return _merge_updates(merged, execution_update)

    return node


def _result_validation_node() -> Callable[[AgentState], StateUpdate]:
    def node(state: AgentState) -> StateUpdate:
        return validate_results(state)

    return node


def _evidence_node() -> Callable[[AgentState], StateUpdate]:
    def node(state: AgentState) -> StateUpdate:
        return build_evidence(state)

    return node


def _synthesis_node(deps: GraphDependencies) -> Callable[[AgentState], StateUpdate]:
    def node(state: AgentState) -> StateUpdate:
        merged: StateUpdate = {}
        if routing.is_max_agent_steps_reached(state, max_agent_steps=deps.max_agent_steps):
            exceeded_update = routing.build_max_agent_steps_exceeded_update(state)
            merged = _merge_updates(merged, exceeded_update)
            state = state.model_copy(update=exceeded_update)

        synthesis_update = synthesize_response(
            state,
            llm_client=deps.llm_client,
            synthesis_service=deps.synthesis_service,
        )
        merged = _merge_updates(merged, synthesis_update)

        if (
            routing.is_max_agent_steps_reached(state, max_agent_steps=deps.max_agent_steps)
            and merged.get("final_response") is None
        ):
            merged["final_response"] = routing.build_max_agent_steps_response(state)

        return merged

    return node


def _route(
    deps: GraphDependencies,
    router: Callable[..., str],
) -> Callable[[AgentState], str]:
    def routed(state: AgentState) -> str:
        destination = router(state, max_agent_steps=deps.max_agent_steps)
        return END if destination == routing.END else destination

    return routed


def _clarification_targets() -> dict[str, str]:
    return {
        routing.PLANNER: routing.PLANNER,
        routing.SYNTHESIS: routing.SYNTHESIS,
        END: END,
    }


def _plan_validation_targets() -> dict[str, str]:
    return {
        routing.EXECUTION: routing.EXECUTION,
        routing.SYNTHESIS: routing.SYNTHESIS,
    }


def _execution_targets() -> dict[str, str]:
    return {
        routing.RESULT_VALIDATION: routing.RESULT_VALIDATION,
        routing.SYNTHESIS: routing.SYNTHESIS,
    }


def _result_validation_targets() -> dict[str, str]:
    return {
        routing.EXECUTION: routing.EXECUTION,
        routing.EVIDENCE: routing.EVIDENCE,
        routing.SYNTHESIS: routing.SYNTHESIS,
    }


def _merge_updates(base: StateUpdate, delta: StateUpdate) -> StateUpdate:
    merged = dict(base)
    for key, value in delta.items():
        if key in merged and isinstance(merged[key], list) and isinstance(value, list):
            merged[key] = [*merged[key], *value]
        else:
            merged[key] = value
    return merged


def _merge_run_config(
    run_config: dict[str, Any] | None,
    *,
    recursion_limit: int,
) -> dict[str, Any]:
    config = dict(run_config or {})
    config.setdefault("recursion_limit", recursion_limit)
    return config
