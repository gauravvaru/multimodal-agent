from __future__ import annotations
from multimodal_agent.agent.graph import GraphDependencies
from multimodal_agent.agent.nodes.planner import AgentPlanner
from multimodal_agent.agent.nodes.synthesis import AgentSynthesisService
from multimodal_agent.services.llm_provider import (
    build_intent_service,
    build_planner_llm_client,
    build_synthesis_llm_client,
)
from multimodal_agent.tools.registry import create_default_tool_registry

def build_default_graph_dependencies() -> GraphDependencies:
    registry = create_default_tool_registry()
    llm_client = build_synthesis_llm_client()
    
    return GraphDependencies(
        registry=registry,
        llm_client=llm_client,
        intent_service=build_intent_service(),
        planner_service=AgentPlanner(registry=registry, llm_client=build_planner_llm_client()),
        synthesis_service=AgentSynthesisService(llm_client=llm_client) if llm_client else None,
    )
