from __future__ import annotations
from typing import Final

from multimodal_agent.agent.plan_validation import (
    get_max_agent_steps,
    plan_has_validation_errors,
)
from multimodal_agent.agent.types import StateUpdate
from multimodal_agent.models.state import AgentState
from multimodal_agent.models.validation import ValidationStatus
from multimodal_agent.utilities.tracing import TraceEvent

# LangGraph route destinations
END: Final = "__end__"
VALIDATION: Final = "validation"
NORMALIZATION: Final = "normalization"
INTENT: Final = "intent"
CLARIFICATION: Final = "clarification"
PLANNER: Final = "planner"
VALIDATE_PLAN: Final = "validate_plan"
EXECUTION: Final = "execution"
RESULT_VALIDATION: Final = "result_validation"
EVIDENCE: Final = "evidence"
SYNTHESIS: Final = "synthesis"

AGENT_STEP_COUNT_KEY = "agent_step_count"
def agent_step_count(state: AgentState) -> int:
    value = state.constraints.get(AGENT_STEP_COUNT_KEY, 0)
    return int(value) if value is not None else 0

def is_max_agent_steps_reached(
    state: AgentState,
    *,
    max_agent_steps: int | None = None,
) -> bool:
    limit = max_agent_steps if max_agent_steps is not None else get_max_agent_steps()
    return agent_step_count(state) >= limit

def is_last_result_usable(state: AgentState) -> bool:
    if state.validation_status is not None:
        return state.validation_status in (ValidationStatus.SUCCESS, ValidationStatus.PARTIAL)
    return state.constraints.get("last_result_usable") is True

def is_retry_required(state: AgentState) -> bool:
    if state.retry_required:
        return True
    if is_last_result_usable(state):
        return False
    if state.plan is None:
        return False
    retry_count = state.retry_count or int(state.constraints.get("retry_count", 0))
    return retry_count <= state.plan.max_retries

def has_remaining_plan_steps(state: AgentState) -> bool:
    if state.plan is None:
        return False
    return state.current_step < len(state.plan.steps)

def route_after_clarification(
    state: AgentState,
    *,
    max_agent_steps: int | None = None,
) -> str:
    if is_max_agent_steps_reached(state, max_agent_steps=max_agent_steps):
        return SYNTHESIS
    if state.clarification_required:
        return END
    return PLANNER

def route_after_plan_validation(
    state: AgentState,
    *,
    max_agent_steps: int | None = None,
) -> str:
    if is_max_agent_steps_reached(state, max_agent_steps=max_agent_steps):
        return SYNTHESIS
    if plan_has_validation_errors(state):
        return SYNTHESIS
    return EXECUTION

def route_after_result_validation(
    state: AgentState,
    *,
    max_agent_steps: int | None = None,
) -> str:
    if is_max_agent_steps_reached(state, max_agent_steps=max_agent_steps):
        return SYNTHESIS
    if is_retry_required(state):
        return EXECUTION
    if is_last_result_usable(state):
        return route_execution_continuation(state, max_agent_steps=max_agent_steps)
    return SYNTHESIS

def route_after_execution(
    state: AgentState,
    *,
    max_agent_steps: int | None = None,
) -> str:
    if is_max_agent_steps_reached(state, max_agent_steps=max_agent_steps):
        return SYNTHESIS
    if state.errors and not state.tool_results:
        return SYNTHESIS
    if state.plan is None:
        return SYNTHESIS
    return RESULT_VALIDATION

def route_execution_continuation(
    state: AgentState,
    *,
    max_agent_steps: int | None = None,
) -> str:
    if is_max_agent_steps_reached(state, max_agent_steps=max_agent_steps):
        return SYNTHESIS
    if has_remaining_plan_steps(state):
        return EXECUTION
    return EVIDENCE

def route_final_termination(
    state: AgentState,
    *,
    max_agent_steps: int | None = None,
) -> str:
    if state.final_response is not None:
        return END
    if is_max_agent_steps_reached(state, max_agent_steps=max_agent_steps):
        return SYNTHESIS
    if state.clarification_required:
        return END
    return SYNTHESIS

def build_max_agent_steps_exceeded_update(state: AgentState) -> StateUpdate:
    message = (
        f"Maximum agent steps ({get_max_agent_steps()}) reached; "
        "unable to continue safely."
    )
    return {
        "errors": [message],
        "trace": [
            TraceEvent(
                step="route_max_agent_steps",
                detail={
                    "agent_step_count": agent_step_count(state),
                    "max_agent_steps": get_max_agent_steps(),
                },
            )
        ],
    }

def build_retry_execution_update(state: AgentState) -> StateUpdate:
    retry_step = max(state.current_step - 1, 0)
    return {
        "current_step": retry_step,
        "trace": [
            TraceEvent(
                step="route_retry_execution",
                detail={"current_step": retry_step},
            )
        ],
    }

def build_max_agent_steps_response(state: AgentState) -> str:
    """Return a user-facing response when the step limit is exceeded."""
    if state.errors:
        return (
            "I couldn't complete your request because the workflow exceeded "
            f"the maximum number of steps ({get_max_agent_steps()})."
        )
    return (
        "I couldn't complete your request because the workflow exceeded "
        f"the maximum number of steps ({get_max_agent_steps()})."
    )
