"""Tests for Pydantic contracts."""

from multimodal_agent.models.requests import AgentRequest, InputArtifact
from multimodal_agent.models.responses import AgentResponse
from multimodal_agent.models.state import AgentState
from multimodal_agent.models.tools import ToolResult
from multimodal_agent.utilities.tracing import append_trace_event


def test_agent_state_defaults() -> None:
    state = AgentState(request_id="req-1", user_query="hello")
    assert state.clarification_required is False
    assert state.tool_results == []
    assert state.plan is None
    assert state.intent is None
    assert state.evidence == []
    assert state.final_response is None


def test_tool_result_contract() -> None:
    result = ToolResult(tool_name="ocr", status="success", confidence=0.9)
    assert result.error is None


def test_agent_request_validation() -> None:
    request = AgentRequest(
        query="Summarize this PDF",
        artifacts=[InputArtifact(filename="doc.pdf", content_type="application/pdf")],
    )
    assert request.artifacts[0].filename == "doc.pdf"


def test_agent_response_shape() -> None:
    response = AgentResponse(request_id="req-1", answer="Done.")
    assert response.trace == []


def test_append_trace_event() -> None:
    trace = append_trace_event([], "validate_input", status="ok")
    assert trace[0]["step"] == "validate_input"
