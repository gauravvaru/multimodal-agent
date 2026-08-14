from __future__ import annotations
from typing import Any
from multimodal_agent.agent.types import StateUpdate
from multimodal_agent.models.state import AgentState
from multimodal_agent.services.planner_service import AgentPlanner, PlannerService, PlanValidationError, validate_generated_plan
from multimodal_agent.tools.registry import ToolRegistry, create_default_tool_registry
from multimodal_agent.utilities.tracing import TraceEvent

def create_plan(
    state: AgentState,
    *,
    planner_service: PlannerService | None = None,
    registry: ToolRegistry | None = None,
    llm_client: Any = None,
) -> StateUpdate:
    if state.clarification_required:
        return {
            "trace": [
                TraceEvent(
                    step="create_plan",
                    detail={"status": "skipped", "reason": "clarification_required"},
                )
            ],
        }

    if state.intent is None:
        return {
            "errors": ["Intent is required before planning"],
            "trace": [TraceEvent(step="create_plan", detail={"status": "missing_intent"})],
        }

    planner = planner_service or AgentPlanner(registry=registry, llm_client=llm_client)
    try:
        plan = planner.create_plan(state)
    except PlanValidationError as exc:
        return {
            "errors": exc.errors,
            "trace": [
                TraceEvent(
                    step="create_plan",
                    detail={"status": "invalid_plan", "error_count": len(exc.errors)},
                )
            ],
        }

    return {
        "plan": plan,
        "current_step": 0,
        "trace": [
            TraceEvent(
                step="create_plan",
                detail={"step_count": len(plan.steps), "status": "ok"},
            )
        ],
    }

def validate_plan(
    state: AgentState,
    *,
    registry: ToolRegistry | None = None,
) -> StateUpdate:
    """Validate the proposed execution plan."""
    if state.plan is None:
        return {
            "errors": ["Plan is required before validation"],
            "trace": [TraceEvent(step="validate_plan", detail={"status": "missing_plan"})],
        }

    active_registry = registry or create_default_tool_registry()
    errors = validate_generated_plan(state.plan, registry=active_registry, state=state)
    trace = [
        TraceEvent(
            step="validate_plan",
            detail={"error_count": len(errors), "status": "failed" if errors else "ok"},
        )
    ]
    update: StateUpdate = {"trace": trace}
    if errors:
        update["errors"] = errors
    return update
