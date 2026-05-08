from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph

from solo_agent.workflow.graph_nodes import (
    make_collect_context_node,
    make_context_guard_node,
    make_execute_tools_node,
    make_inspect_node,
    make_parallelism_gate_node,
    make_plan_node,
    make_propose_verified_patch_node,
    make_respond_node,
    make_select_tools_node,
    make_subdirectory_hint_node,
    make_task_state_node,
    parallel_dispatch_placeholder_node,
)
from solo_agent.workflow.graph_state import SoloGraphState


def build_text_provider_graph(
    *,
    provider: Any,
    deps: Any,
    settings: Any,
) -> StateGraph:
    """Build a StateGraph covering the text-provider strategy core.

    This graph handles the workflow from plan through respond.
    Shared prelude (receive_user_turn, memory, skill_context) and
    shared postlude (sync_memory, compress, persist_snapshot) are
    managed by the WorkflowRuntime — they are NOT in this graph.
    """
    graph = StateGraph(SoloGraphState)

    graph.add_node("plan", make_plan_node(provider, settings))
    graph.add_node("task_state", make_task_state_node())
    graph.add_node("parallelism_gate", make_parallelism_gate_node(settings))
    graph.add_node("parallel_dispatch_placeholder", parallel_dispatch_placeholder_node)
    graph.add_node("collect_context", make_collect_context_node(deps, settings))
    graph.add_node("inspect", make_inspect_node(deps))
    graph.add_node("select_tools", make_select_tools_node(deps, settings))
    graph.add_node("execute_tools", make_execute_tools_node(deps, settings))
    graph.add_node("propose_verified_patch", make_propose_verified_patch_node(provider, deps, settings))
    graph.add_node("subdirectory_hint", make_subdirectory_hint_node(settings))
    graph.add_node("context_guard_before_respond", make_context_guard_node(provider, deps, settings, phase="before_respond"))
    graph.add_node("respond", make_respond_node(provider, settings))

    graph.set_entry_point("plan")

    graph.add_edge("plan", "task_state")
    graph.add_edge("task_state", "parallelism_gate")

    graph.add_conditional_edges(
        "parallelism_gate",
        route_after_parallelism_gate,
        {
            END: END,
            "parallel_dispatch_placeholder": "parallel_dispatch_placeholder",
            "collect_context": "collect_context",
        },
    )

    graph.add_edge("parallel_dispatch_placeholder", "collect_context")
    graph.add_edge("collect_context", "inspect")

    graph.add_conditional_edges(
        "inspect",
        make_route_after_guard("select_tools"),
        {
            END: END,
            "select_tools": "select_tools",
        },
    )

    graph.add_edge("select_tools", "execute_tools")

    graph.add_conditional_edges(
        "execute_tools",
        route_after_execute_tools,
        {
            END: END,
            "propose_verified_patch": "propose_verified_patch",
        },
    )

    graph.add_conditional_edges(
        "propose_verified_patch",
        route_after_patch,
        {
            END: END,
            "subdirectory_hint": "subdirectory_hint",
        },
    )

    graph.add_edge("subdirectory_hint", "context_guard_before_respond")

    graph.add_conditional_edges(
        "context_guard_before_respond",
        make_route_after_guard("respond"),
        {
            END: END,
            "respond": "respond",
        },
    )

    graph.add_edge("respond", END)

    return graph


def make_route_after_guard(target: str):
    def router(state: SoloGraphState) -> str:
        agent_data = state.get("agent_state") or {}
        if agent_data.get("blocked", False):
            return END
        return target
    return router


def route_after_parallelism_gate(state: SoloGraphState) -> str:
    agent_data = state.get("agent_state") or {}
    if agent_data.get("blocked", False):
        return END
    strategy = agent_data.get("execution_strategy", "serial")
    if strategy == "parallel":
        return "parallel_dispatch_placeholder"
    return "collect_context"


def route_after_execute_tools(state: SoloGraphState) -> str:
    agent_data = state.get("agent_state") or {}
    if agent_data.get("awaiting_approval", False):
        return END
    return "propose_verified_patch"


def route_after_patch(state: SoloGraphState) -> str:
    agent_data = state.get("agent_state") or {}
    if agent_data.get("awaiting_approval", False):
        return END
    return "subdirectory_hint"



