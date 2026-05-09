"""Coordinator-specific StateGraph node factories for the multi-agent workflow."""

from __future__ import annotations

from typing import Any

from solo_agent.workflow.graph_state import (
    SoloGraphState,
    coordinator_state_from_graph_data,
    coordinator_state_to_graph_data,
)
from solo_agent.workflow.stages import (
    _deep_plan_stage,
    _plan_response_stage,
    _plan_self_review_stage,
)


def make_generate_research_plan_node(provider: Any, settings: Any):
    """Generate a research plan using the deep plan stage."""
    from solo_agent.workflow.graph_nodes import _make_node
    return _make_node(_deep_plan_stage, provider, settings)


def make_plan_self_review_node(provider: Any, settings: Any):
    """Review the research plan quality."""
    from solo_agent.workflow.graph_nodes import _make_node
    return _make_node(_plan_self_review_stage, provider, settings)


def make_plan_response_node(provider: Any, settings: Any):
    """Format the plan as response."""
    from solo_agent.workflow.graph_nodes import _make_node
    return _make_node(_plan_response_stage)


def make_delegate_to_supervisor_node(settings: Any):
    """Create SupervisorState from the research plan and pass to supervisor graph."""
    from solo_agent.workflow.graph_state import initial_supervisor_graph_state
    from solo_agent.workflow.state import WorkflowState

    async def node(graph_state: SoloGraphState) -> SoloGraphState:
        agent_state = coordinator_state_from_graph_data(graph_state["agent_state"])
        agent_state.loop_stage = "delegate_to_supervisor"

        supervisor_state = WorkflowState.from_agent_state(agent_state)
        gs = initial_supervisor_graph_state(supervisor_state)
        gs["agent_state"]["dispatched_tasks"] = [
            {
                "task_id": "research_1",
                "title": agent_state.deep_plan or agent_state.plan,
                "description": agent_state.user_input,
                "subagent_type": "general-purpose",
            }
        ]
        graph_state["supervisor_state"] = gs["agent_state"]
        graph_state["agent_state"] = coordinator_state_to_graph_data(agent_state)
        return graph_state
    return node


def make_aggregate_supervisor_results_node():
    """Aggregate results returned from the supervisor subgraph."""
    async def node(graph_state: SoloGraphState) -> SoloGraphState:
        agent_state = coordinator_state_from_graph_data(graph_state["agent_state"])
        agent_state.loop_stage = "aggregate_results"

        supervisor_data = graph_state.get("supervisor_state") or {}
        completed = supervisor_data.get("completed_results") or {}
        agent_state.aggregated_results = completed

        findings_lines = []
        for task_id, result in completed.items():
            if isinstance(result, dict):
                findings_lines.append(f"## {result.get('title', task_id)}")
                findings_lines.append(result.get("findings", str(result)))
        if findings_lines:
            agent_state.context.append({
                "source": "supervisor",
                "content": "\n\n".join(findings_lines),
            })

        graph_state["agent_state"] = coordinator_state_to_graph_data(agent_state)
        return graph_state
    return node
