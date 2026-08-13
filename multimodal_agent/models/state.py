"""LangGraph agent state contract."""

from __future__ import annotations

import operator
from typing import Annotated, Any

from pydantic import BaseModel, Field

from multimodal_agent.models.artifacts import NormalizedArtifact
from multimodal_agent.models.evidence import Evidence
from multimodal_agent.models.intent import Intent
from multimodal_agent.models.plan import Plan
from multimodal_agent.models.requests import InputArtifact
from multimodal_agent.models.tools import ToolResult
from multimodal_agent.utilities.tracing import TraceEvent

# Accumulator type aliases for LangGraph-compatible list updates.
InputArtifacts = Annotated[list[InputArtifact], operator.add]
NormalizedContents = Annotated[list[NormalizedArtifact], operator.add]
ToolResults = Annotated[list[ToolResult], operator.add]
EvidenceItems = Annotated[list[Evidence], operator.add]
ErrorMessages = Annotated[list[str], operator.add]
TraceEvents = Annotated[list[TraceEvent], operator.add]


class AgentState(BaseModel):
    """Shared state passed through the LangGraph workflow.

    Scalar and object fields use last-write-wins semantics.
    List fields annotated with ``operator.add`` accumulate updates so nodes
    can return deltas instead of rewriting the full list.
    """

    request_id: str = ""
    user_query: str = ""
    input_artifacts: InputArtifacts = Field(default_factory=list)
    normalized_contents: NormalizedContents = Field(default_factory=list)
    intent: Intent | None = None
    constraints: dict[str, str | int | float | bool | None] = Field(default_factory=dict)
    clarification_required: bool = False
    clarification_question: str | None = None
    plan: Plan | None = None
    current_step: int = 0
    tool_results: ToolResults = Field(default_factory=list)
    evidence: EvidenceItems = Field(default_factory=list)
    errors: ErrorMessages = Field(default_factory=list)
    trace: TraceEvents = Field(default_factory=list)
    final_response: str | None = None
    retry_required: bool = False
    retry_count: int = 0
    validation_status: Any | None = None
    no_evidence: bool = False
