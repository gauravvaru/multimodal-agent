"""Tests for LangGraph AgentState contract."""

import operator

from multimodal_agent.models.artifacts import NormalizedArtifact
from multimodal_agent.models.evidence import Evidence
from multimodal_agent.models.intent import Intent
from multimodal_agent.models.plan import Plan, PlanStep
from multimodal_agent.models.requests import InputArtifact
from multimodal_agent.models.state import AgentState
from multimodal_agent.models.tools import ToolResult
from multimodal_agent.utilities.tracing import TraceEvent


def test_agent_state_can_be_constructed() -> None:
    state = AgentState(request_id="req-19", user_query="Summarize the PDF")
    assert state.request_id == "req-19"
    assert state.user_query == "Summarize the PDF"
    assert state.current_step == 0
    assert state.input_artifacts == []
    assert state.normalized_contents == []
    assert state.errors == []
    assert state.trace == []


def test_optional_fields_default_to_none_or_false() -> None:
    state = AgentState()
    assert state.intent is None
    assert state.plan is None
    assert state.clarification_required is False
    assert state.clarification_question is None
    assert state.final_response is None


def test_tool_results_can_be_stored() -> None:
    result = ToolResult(
        tool_name="pdf",
        status="success",
        data={"pages": 2},
        confidence=0.95,
        latency_ms=12.5,
    )
    state = AgentState(tool_results=[result])
    assert len(state.tool_results) == 1
    assert state.tool_results[0].tool_name == "pdf"
    assert state.tool_results[0].data == {"pages": 2}


def test_plan_can_be_stored() -> None:
    plan = Plan(
        steps=[
            PlanStep(step_id="1", tool_name="pdf", inputs={"source": "doc.pdf"}),
            PlanStep(step_id="2", tool_name="summarization", depends_on=["1"]),
        ]
    )
    state = AgentState(plan=plan, current_step=1)
    assert state.plan is not None
    assert len(state.plan.steps) == 2
    assert state.plan.steps[0].tool_name == "pdf"
    assert state.current_step == 1


def test_evidence_can_be_stored() -> None:
    item = Evidence(
        source_id="artifact-1",
        filename="report.pdf",
        source_type="pdf",
        page=3,
        extracted_text="Revenue grew 12%.",
        relevance_score=0.88,
    )
    state = AgentState(evidence=[item])
    assert state.evidence[0].filename == "report.pdf"
    assert state.evidence[0].relevance_score == 0.88


def test_final_response_can_be_stored() -> None:
    state = AgentState(final_response="The document summarizes quarterly growth.")
    assert state.final_response == "The document summarizes quarterly growth."


def test_intent_artifacts_and_trace_use_domain_models() -> None:
    state = AgentState(
        input_artifacts=[InputArtifact(filename="a.pdf", content_type="application/pdf")],
        normalized_contents=[
            NormalizedArtifact(
                artifact_id="a1",
                artifact_type="pdf",
                content_type="application/pdf",
            )
        ],
        intent=Intent(name="summarize", confidence=0.91),
        clarification_required=True,
        clarification_question="Which section should I summarize?",
        errors=["timeout"],
        trace=[TraceEvent(step="clarification_gate", detail={"reason": "ambiguous"})],
    )
    assert state.intent is not None
    assert state.intent.name == "summarize"
    assert state.input_artifacts[0].filename == "a.pdf"
    assert state.normalized_contents[0].artifact_type == "pdf"
    assert state.clarification_required is True
    assert state.clarification_question == "Which section should I summarize?"
    assert state.errors == ["timeout"]
    assert state.trace[0].step == "clarification_gate"


def test_list_fields_declare_operator_add_reducers() -> None:
    annotations = AgentState.model_fields
    for field_name in (
        "input_artifacts",
        "normalized_contents",
        "tool_results",
        "evidence",
        "errors",
        "trace",
    ):
        metadata = annotations[field_name].metadata
        assert operator.add in metadata, f"{field_name} missing operator.add reducer"
