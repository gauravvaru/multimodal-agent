from multimodal_agent.agent.types import StateUpdate
from multimodal_agent.models.state import AgentState
from multimodal_agent.utilities.tracing import TraceEvent

def validate_input(state: AgentState) -> StateUpdate:
    """Validate inbound request and artifacts."""
    if not state.user_query.strip():
        return {
            "errors": ["User query cannot be empty"],
            "trace": [TraceEvent(step="validation", detail={"status": "failed"})],
        }
        
    return {
        "validation_status": "ok",
        "trace": [TraceEvent(step="validation", detail={"status": "ok"})],
    }