from __future__ import annotations

from multimodal_agent.agent.intent_rules import (
    detect_intent_deterministic,
    requires_semantic_intent,
)
from multimodal_agent.agent.types import StateUpdate
from multimodal_agent.models.state import AgentState
from multimodal_agent.services.intent_service import IntentService
from multimodal_agent.utilities.tracing import TraceEvent


def detect_intent(
    state: AgentState,
    *,
    intent_service: IntentService | None = None,
) -> StateUpdate:
    if requires_semantic_intent(state):
        if intent_service is None:
            return {
                "errors": ["Intent service is required for semantic intent detection"],
                "trace": [
                    TraceEvent(
                        step="detect_intent",
                        detail={"mode": "semantic", "status": "missing_service"},
                    )
                ],
            }
        intent = intent_service.detect(state.user_query, list(state.normalized_contents))
        mode = "semantic"
    else:
        intent = detect_intent_deterministic(state)
        mode = "deterministic"

    if intent is None:
        return {
            "errors": ["Unable to determine intent"],
            "trace": [TraceEvent(step="detect_intent", detail={"mode": mode, "status": "failed"})],
        }

    return {
        "intent": intent,
        "trace": [
            TraceEvent(
                step="detect_intent",
                detail={"mode": mode, "intent": intent.name, "confidence": intent.confidence},
            )
        ],
    }
