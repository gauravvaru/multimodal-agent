from __future__ import annotations
from multimodal_agent.config.settings import get_settings
from multimodal_agent.models.plan import Plan
from multimodal_agent.models.state import AgentState
from multimodal_agent.tools.registry import ToolRegistry

def get_max_agent_steps() -> int:
    return get_settings().max_graph_steps

def collect_plan_errors(plan: Plan) -> list[str]:
    errors: list[str] = []
    if not plan.steps:
        errors.append('plan must contain at least one step')
    step_ids = [step.step_id for step in plan.steps]
    if len(step_ids) != len(set(step_ids)):
        errors.append('plan step_id values must be unique')
    known_ids = set(step_ids)
    for step in plan.steps:
        if not step.tool_name.strip():
            errors.append(f'step {step.step_id}: tool_name is required')
        for dependency in step.depends_on:
            if dependency not in known_ids:
                errors.append(f'step {step.step_id}: unknown dependency {dependency}')
    return errors

def plan_has_validation_errors(state: AgentState) -> bool:
    if state.plan is None:
        return True
    return bool(collect_plan_errors(state.plan))

def plan_exceeds_step_limit(plan: Plan, *, max_steps: int | None=None) -> bool:
    limit = max_steps if max_steps is not None else get_max_agent_steps()
    return len(plan.steps) > limit

def validate_registered_tools(plan: Plan, registry: ToolRegistry) -> list[str]:
    registered_tools = set(registry.names())
    errors: list[str] = []
    for step in plan.steps:
        if step.tool_name not in registered_tools:
            errors.append(f"step {step.step_id}: unknown tool '{step.tool_name}'")
    return errors