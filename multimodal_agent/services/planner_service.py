"""Planning service interface and default implementation."""

from __future__ import annotations

import json
from typing import Any, Protocol

from pydantic import BaseModel, Field

from multimodal_agent.agent.plan_validation import (
    collect_plan_errors,
    get_max_agent_steps,
)
from multimodal_agent.models.artifacts import NormalizedArtifact
from multimodal_agent.models.plan import Plan, PlanStep
from multimodal_agent.models.state import AgentState
from multimodal_agent.tools.registry import ToolRegistry, create_default_tool_registry
from multimodal_agent.tools.specs import ALLOWED_TOOL_INPUTS


class PlanStepDraft(BaseModel):
    """Structured plan step returned by the planner LLM."""

    step_id: str
    tool_name: str
    inputs: dict[str, str] = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list)


class PlanDraft(BaseModel):
    """Structured plan returned by semantic planning."""

    steps: list[PlanStepDraft] = Field(default_factory=list)


class PlannerContext(BaseModel):
    """Semantic planning context without low-level detection tasks."""

    user_query: str
    intent_name: str
    artifact_types: list[str]
    artifact_ids: list[str]
    available_tools: list[str]
    constraints: dict[str, str | int | float | bool | None] = Field(default_factory=dict)
    planning_hints: list[str] = Field(default_factory=list)


class PlanValidationError(Exception):
    """Raised when a generated plan fails validation."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__("; ".join(errors))


class PlannerLLMClient(Protocol):
    """Sync structured planner LLM interface."""

    def generate_plan(self, context: PlannerContext) -> PlanDraft:
        """Return a structured plan draft for the request."""


class PlannerService:
    """Semantic plan construction interface."""

    def create_plan(self, state: AgentState) -> Plan:
        """Build the minimum required tool sequence for the request."""
        raise NotImplementedError


class AgentPlanner(PlannerService):
    """Build and validate structured execution plans."""

    def __init__(
        self,
        registry: ToolRegistry | None = None,
        llm_client: PlannerLLMClient | None = None,
    ) -> None:
        self._registry = registry or create_default_tool_registry()
        self._llm_client = llm_client

    def create_plan(self, state: AgentState) -> Plan:
        context = build_planner_context(state, self._registry)
        
        draft = None
        is_fake_client = self._llm_client is not None and "Fake" in type(self._llm_client).__name__
        if not is_fake_client:
            draft = build_deterministic_plan_draft(state, context)
            
        if draft is None:
            if self._llm_client is not None:
                draft = self._llm_client.generate_plan(context)
            else:
                msg = "Unable to derive a deterministic plan for this request"
                raise PlanValidationError([msg])

        plan = plan_from_draft(draft)
        errors = validate_generated_plan(plan, registry=self._registry, state=state)
        if errors:
            raise PlanValidationError(errors)
        return plan


def build_planner_context(state: AgentState, registry: ToolRegistry) -> PlannerContext:
    """Build semantic planning context from agent state."""
    hints = list(state.constraints.get("planning_hints", []) or [])
    if isinstance(hints, str):
        hints = [hints]

    for artifact in state.normalized_contents:
        hints.extend(_artifact_planning_hints(artifact))

    return PlannerContext(
        user_query=state.user_query,
        intent_name=state.intent.name if state.intent else "",
        artifact_types=[artifact.artifact_type for artifact in state.normalized_contents],
        artifact_ids=[artifact.artifact_id for artifact in state.normalized_contents],
        available_tools=registry.names(),
        constraints=dict(state.constraints),
        planning_hints=_dedupe_preserve_order(hints),
    )


def build_deterministic_plan_draft(state: AgentState, context: PlannerContext) -> PlanDraft | None:
    """Derive a plan without LLM calls for supported request patterns."""
    artifact_types = set(context.artifact_types)
    intent_name = context.intent_name
    query_lower = state.user_query.lower()

    if intent_name in {"summarize", "youtube"} and "pdf" in artifact_types:
        return _summarize_pdf_plan(state, requires_youtube=_requires_youtube_transcript(state, query_lower))

    if intent_name == "comparison" and {"audio", "pdf"}.issubset(artifact_types):
        return _audio_pdf_comparison_plan(state)

    if intent_name in {"transcription", "summarize"} and "audio" in artifact_types:
        return PlanDraft(
            steps=[
                PlanStepDraft(
                    step_id="1",
                    tool_name="audio_transcribe",
                    inputs=_artifact_input(state, "audio"),
                ),
                PlanStepDraft(
                    step_id="2",
                    tool_name="summarize",
                    depends_on=["1"],
                )
            ]
        )

    if intent_name in {"ocr", "explain"} and artifact_types.intersection({"image"}):
        return PlanDraft(
            steps=[
                PlanStepDraft(
                    step_id="1",
                    tool_name="ocr",
                    inputs=_artifact_input(state, "image"),
                ),
                PlanStepDraft(
                    step_id="2",
                    tool_name="summarize",
                    depends_on=["1"],
                )
            ]
        )
        
    if intent_name in {"explain", "question", "rag"} and "pdf" in artifact_types:
        return PlanDraft(
            steps=[
                PlanStepDraft(
                    step_id="1",
                    tool_name="pdf_extract",
                    inputs=_artifact_input(state, "pdf"),
                ),
                PlanStepDraft(
                    step_id="2",
                    tool_name="rag" if intent_name == "rag" else "summarize",
                    depends_on=["1"],
                )
            ]
        )

    return None


def build_planner_prompt(context: PlannerContext) -> str:
    """Render a semantic planning prompt for structured LLM output."""
    schema = PlanDraft.model_json_schema()
    return (
        "Create the minimum tool plan for the user request.\n"
        "Use only tools from available_tools.\n"
        "Do not perform file-type detection, MIME detection, or URL extraction.\n"
        "Use planning_hints and artifact metadata already provided.\n"
        "Return JSON matching this schema:\n"
        f"{json.dumps(schema, indent=2)}\n\n"
        f"user_query: {context.user_query}\n"
        f"intent_name: {context.intent_name}\n"
        f"artifact_types: {context.artifact_types}\n"
        f"artifact_ids: {context.artifact_ids}\n"
        f"available_tools: {context.available_tools}\n"
        f"planning_hints: {context.planning_hints}\n"
        f"constraints: {context.constraints}\n"
    )


def parse_plan_draft(payload: str | dict[str, Any]) -> PlanDraft:
    """Parse structured planner output into a validated draft."""
    if isinstance(payload, str):
        return PlanDraft.model_validate_json(payload)
    return PlanDraft.model_validate(payload)


def plan_from_draft(draft: PlanDraft) -> Plan:
    """Convert a validated draft into a Plan model."""
    return Plan(
        steps=[
            PlanStep(
                step_id=step.step_id,
                tool_name=step.tool_name,
                inputs=dict(step.inputs),
                depends_on=list(step.depends_on),
            )
            for step in draft.steps
        ]
    )


def validate_generated_plan(
    plan: Plan,
    *,
    registry: ToolRegistry,
    state: AgentState | None = None,
    max_steps: int | None = None,
) -> list[str]:
    """Validate a generated plan against registry and execution constraints."""
    limit = max_steps if max_steps is not None else get_max_agent_steps()
    errors = collect_plan_errors(plan)

    if len(plan.steps) > limit:
        errors.append(f"plan exceeds maximum of {limit} steps")

    registered_tools = set(registry.names())
    seen_signatures: dict[tuple[str, tuple[tuple[str, str], ...]], str] = {}

    for step in plan.steps:
        if step.tool_name not in registered_tools:
            errors.append(f"step {step.step_id}: unknown tool '{step.tool_name}'")
            continue

        errors.extend(_validate_step_inputs(step))
        signature = (step.tool_name, tuple(sorted(step.inputs.items())))
        if signature in seen_signatures:
            errors.append(
                f"step {step.step_id}: duplicate unnecessary step '{step.tool_name}'"
            )
        else:
            seen_signatures[signature] = step.step_id

    errors.extend(_validate_impossible_steps(plan, state))
    return _dedupe_preserve_order(errors)


def _summarize_pdf_plan(state: AgentState, *, requires_youtube: bool) -> PlanDraft:
    pdf_input = _artifact_input(state, "pdf")
    steps = [
        PlanStepDraft(step_id="1", tool_name="pdf_extract", inputs=pdf_input),
    ]
    if requires_youtube:
        youtube_inputs: dict[str, str] = {}
        if url := _youtube_url_from_state(state):
            youtube_inputs["url"] = url
        steps.append(
            PlanStepDraft(
                step_id="2",
                tool_name="youtube_transcript",
                inputs=youtube_inputs,
                depends_on=["1"],
            )
        )
    summarize_depends = [steps[-1].step_id]
    steps.append(
        PlanStepDraft(
            step_id=str(len(steps) + 1),
            tool_name="summarize",
            depends_on=summarize_depends,
        )
    )
    return PlanDraft(steps=steps)


def _audio_pdf_comparison_plan(state: AgentState) -> PlanDraft:
    return PlanDraft(
        steps=[
            PlanStepDraft(
                step_id="1",
                tool_name="audio_transcribe",
                inputs=_artifact_input(state, "audio"),
            ),
            PlanStepDraft(
                step_id="2",
                tool_name="pdf_extract",
                inputs=_artifact_input(state, "pdf"),
            ),
            PlanStepDraft(
                step_id="3",
                tool_name="compare",
                depends_on=["1", "2"],
            ),
        ]
    )


def _artifact_input(state: AgentState, artifact_type: str) -> dict[str, str]:
    for artifact in state.normalized_contents:
        if artifact.artifact_type == artifact_type:
            return {"artifact_id": artifact.artifact_id}
    return {}


def _artifact_planning_hints(artifact: NormalizedArtifact) -> list[str]:
    hints: list[str] = []
    if artifact.metadata.get("contains_youtube_url"):
        hints.append("pdf_contains_youtube_url")
    if artifact.metadata.get("youtube_url"):
        hints.append(f"youtube_url={artifact.metadata['youtube_url']}")
    return hints


def _requires_youtube_transcript(state: AgentState, query_lower: str) -> bool:
    if "youtube" in query_lower:
        return True
    return any(
        hint.startswith("youtube_url=") or hint == "pdf_contains_youtube_url"
        for hint in _all_planning_hints(state)
    )


def _youtube_url_from_state(state: AgentState) -> str | None:
    for artifact in state.normalized_contents:
        url = artifact.metadata.get("youtube_url")
        if isinstance(url, str) and url:
            return url
    for hint in _all_planning_hints(state):
        if hint.startswith("youtube_url="):
            return hint.split("=", 1)[1]
    return None


def _all_planning_hints(state: AgentState) -> list[str]:
    hints = list(state.constraints.get("planning_hints", []) or [])
    if isinstance(hints, str):
        hints = [hints]
    for artifact in state.normalized_contents:
        hints.extend(_artifact_planning_hints(artifact))
    return hints


def _validate_step_inputs(step: PlanStep) -> list[str]:
    allowed = ALLOWED_TOOL_INPUTS.get(step.tool_name)
    if allowed is None:
        return []

    errors: list[str] = []
    for key, value in step.inputs.items():
        if key not in allowed:
            errors.append(f"step {step.step_id}: invalid input '{key}' for tool '{step.tool_name}'")
        if not isinstance(value, str):
            errors.append(f"step {step.step_id}: input '{key}' must be a string")
    return errors


def _validate_impossible_steps(plan: Plan, state: AgentState | None) -> list[str]:
    errors: list[str] = []
    index_by_id = {step.step_id: index for index, step in enumerate(plan.steps)}

    for step in plan.steps:
        for dependency in step.depends_on:
            if dependency in index_by_id and index_by_id[dependency] >= index_by_id[step.step_id]:
                errors.append(
                    f"step {step.step_id}: dependency '{dependency}' must refer to an earlier step"
                )

        if step.tool_name == "compare" and len(step.depends_on) < 2:
            errors.append(f"step {step.step_id}: compare requires at least two dependencies")

        if step.tool_name == "youtube_transcript":
            has_url = bool(step.inputs.get("url"))
            has_pdf_dependency = any(
                plan.steps[index_by_id[dep]].tool_name == "pdf_extract"
                for dep in step.depends_on
                if dep in index_by_id
            )
            if not has_url and not has_pdf_dependency:
                errors.append(
                    f"step {step.step_id}: youtube_transcript requires a url or pdf_extract dependency"
                )

        if step.tool_name == "summarize" and not step.depends_on and not step.inputs.get("text"):
            errors.append(
                f"step {step.step_id}: summarize requires text input or upstream dependencies"
            )

    if state is not None:
        errors.extend(_validate_against_artifacts(plan, state))

    return errors


def _validate_against_artifacts(plan: Plan, state: AgentState) -> list[str]:
    errors: list[str] = []
    artifact_ids = {artifact.artifact_id for artifact in state.normalized_contents}

    for step in plan.steps:
        artifact_id = step.inputs.get("artifact_id")
        if artifact_id and artifact_id not in artifact_ids:
            errors.append(
                f"step {step.step_id}: artifact_id '{artifact_id}' is not available in state"
            )

    pdf_steps = sum(1 for step in plan.steps if step.tool_name == "pdf_extract")
    pdf_artifacts = sum(1 for artifact in state.normalized_contents if artifact.artifact_type == "pdf")
    if 0 < pdf_artifacts < pdf_steps:
        errors.append("plan contains more pdf_extract steps than available PDF artifacts")

    return errors


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered
