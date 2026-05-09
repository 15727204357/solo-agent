"""Coordinator StateGraph builder — top-level multi-agent orchestration."""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph

from solo_agent.workflow.coordinator_nodes import (
    make_generate_research_plan_node,
    make_plan_response_node,
    make_plan_self_review_node,
)
from solo_agent.workflow.graph_nodes import (
    make_build_memory_context_node,
    make_context_guard_node,
    make_load_builtin_memory_node,
    make_prefetch_memory_node,
    make_receive_user_turn_node,
    make_respond_node,
    make_skill_context_node,
    make_skip_memory_node,
)
from solo_agent.workflow.graph_state import SoloGraphState
from solo_agent.workflow.supervisor_nodes import (
    make_dispatch_researchers_node,
    make_evaluate_results_node,
    make_wait_researchers_node,
)

# ---------------------------------------------------------------------------
# Route helpers
# ---------------------------------------------------------------------------

def _is_blocked(state: SoloGraphState) -> bool:
    return bool((state.get("agent_state") or {}).get("blocked", False))


def _memory_enabled_route(state: SoloGraphState) -> str:
    agent_data = state.get("agent_state") or {}
    return "load_builtin_memory" if agent_data.get("memory_enabled", True) else "skip_memory"


def _exploration_loop_route(state: SoloGraphState) -> str:
    """After LLM evaluation: loop back for more research or proceed to respond."""
    snapshots = (state.get("agent_state") or {}).get("snapshots") or {}
    decision = snapshots.get("exploration_decision", "summarize_return")
    if decision == "continue_explore":
        return "dispatch_researchers"
    return "context_guard_before_respond"


# ---------------------------------------------------------------------------
# build_coordinator_graph
# ---------------------------------------------------------------------------

def build_coordinator_graph(*, provider: Any, deps: Any, settings: Any) -> StateGraph:
    """Build the top-level coordinator LangGraph StateGraph.

    This graph orchestrates:
      - Memory prelude (load/prefetch/build memory context)
      - Research plan generation (deep plan => quality review)
      - Researcher dispatch + wait + evaluate (multi-agent loop)
      - Final response generation
      - Memory postlude (sync/compress/persist)
    """
    from solo_agent.workflow.graph_nodes import (
        make_compress_memory_node,
        make_persist_snapshot_node,
        make_queue_prefetch_node,
        make_sync_memory_node,
    )

    graph = StateGraph(SoloGraphState)

    # -----------------------------------------------------------------------
    # Prelude: receive, memory, skills
    # -----------------------------------------------------------------------
    graph.add_node("receive_user_turn", make_receive_user_turn_node(settings))
    graph.add_node("skip_memory", make_skip_memory_node())
    graph.add_node("load_builtin_memory", make_load_builtin_memory_node(deps))
    graph.add_node("prefetch_memory", make_prefetch_memory_node(deps, settings))
    graph.add_node("build_memory_context", make_build_memory_context_node())
    graph.add_node("skill_context", make_skill_context_node(deps, settings))
    graph.add_node("context_guard_before_plan", make_context_guard_node(provider, deps, settings, phase="before_plan"))

    # -----------------------------------------------------------------------
    # Research plan
    # -----------------------------------------------------------------------
    graph.add_node("generate_research_plan", make_generate_research_plan_node(provider, settings))
    graph.add_node("plan_self_review", make_plan_self_review_node(provider, settings))
    graph.add_node("plan_response", make_plan_response_node(provider, settings))

    # -----------------------------------------------------------------------
    # Supervisor: dispatch, wait, evaluate
    # -----------------------------------------------------------------------
    graph.add_node("dispatch_researchers", make_dispatch_researchers_node(deps, settings))
    graph.add_node("wait_researchers", make_wait_researchers_node(settings))
    graph.add_node("evaluate_results", make_evaluate_results_node(provider, settings))

    # -----------------------------------------------------------------------
    # Respond
    # -----------------------------------------------------------------------
    graph.add_node("respond", make_respond_node(provider, settings))
    graph.add_node("context_guard_before_respond", make_context_guard_node(provider, deps, settings, phase="before_respond"))

    # -----------------------------------------------------------------------
    # Postlude
    # -----------------------------------------------------------------------
    graph.add_node("sync_memory", make_sync_memory_node(deps))
    graph.add_node("queue_prefetch", make_queue_prefetch_node(deps, settings))
    graph.add_node("compress_memory", make_compress_memory_node(provider, deps, settings))
    graph.add_node("persist_snapshot", make_persist_snapshot_node())

    # =======================================================================
    # EDGES
    # =======================================================================

    graph.set_entry_point("receive_user_turn")

    # Prelude chain
    graph.add_edge("receive_user_turn", "skip_memory")
    graph.add_conditional_edges(
        "skip_memory", _memory_enabled_route,
        {"load_builtin_memory": "load_builtin_memory", "skip_memory": "skill_context"},
    )
    graph.add_edge("load_builtin_memory", "prefetch_memory")
    graph.add_edge("prefetch_memory", "build_memory_context")
    graph.add_edge("build_memory_context", "skill_context")
    graph.add_edge("skill_context", "context_guard_before_plan")

    # Plan chain
    graph.add_conditional_edges(
        "context_guard_before_plan",
        lambda s: END if _is_blocked(s) else "generate_research_plan",
        {END: END, "generate_research_plan": "generate_research_plan"},
    )
    graph.add_edge("generate_research_plan", "plan_self_review")
    graph.add_edge("plan_self_review", "plan_response")
    graph.add_edge("plan_response", "dispatch_researchers")

    # Supervisor dispatch chain
    graph.add_edge("dispatch_researchers", "wait_researchers")
    graph.add_edge("wait_researchers", "evaluate_results")

    # LLM-driven exploration loop
    graph.add_conditional_edges(
        "evaluate_results",
        _exploration_loop_route,
        {
            "dispatch_researchers": "dispatch_researchers",
            "context_guard_before_respond": "context_guard_before_respond",
        },
    )

    # Respond chain
    graph.add_conditional_edges(
        "context_guard_before_respond",
        lambda s: END if _is_blocked(s) else "respond",
        {END: END, "respond": "respond"},
    )

    # Postlude chain
    graph.add_conditional_edges(
        "respond",
        lambda s: "sync_memory" if not _is_blocked(s) else END,
        {"sync_memory": "sync_memory", END: END},
    )
    graph.add_edge("sync_memory", "queue_prefetch")
    graph.add_edge("queue_prefetch", "compress_memory")
    graph.add_edge("compress_memory", "persist_snapshot")
    graph.add_edge("persist_snapshot", END)

    return graph
