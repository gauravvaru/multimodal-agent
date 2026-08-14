from __future__ import annotations
from multimodal_agent.agent.types import StateUpdate
from multimodal_agent.models.state import AgentState
from multimodal_agent.services.clarification_service import ClarificationService
from multimodal_agent.utilities.tracing import TraceEvent

_DEFAULT_CONFIDENCE_THRESHOLD = 0.6


def clarification_gate(
    state: AgentState,
    *,
    clarification_service: ClarificationService | None = None,
    confidence_threshold: float = _DEFAULT_CONFIDENCE_THRESHOLD,
) -> StateUpdate:
    if state.intent is None:
        return {
            "errors": ["Intent is required before clarification"],
            "trace": [TraceEvent(step="clarification_gate", detail={"status": "missing_intent"})],
        }

    clarification_required = state.intent.requires_clarification
    clarification_question: str | None = None

    if clarification_required:
        clarification_question = _default_clarification_question(state)
    elif state.intent.confidence < confidence_threshold:
        if clarification_service is None:
            clarification_required = True
            clarification_question = "Could you provide more detail about what you need?"
        else:
            clarification_required, clarification_question = clarification_service.assess(state)

    return {
        "clarification_required": clarification_required,
        "clarification_question": clarification_question if clarification_required else None,
        "trace": [
            TraceEvent(
                step="clarification_gate",
                detail={
                    "clarification_required": clarification_required,
                    "confidence": state.intent.confidence,
                },
            )
        ],
    }


def _default_clarification_question(state: AgentState) -> str:
    if state.intent and state.intent.rationale:
        return f"Could you clarify your request? {state.intent.rationale}"
    return "Could you clarify your request?"
