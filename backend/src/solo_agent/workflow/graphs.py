from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph

from solo_agent.workflow.graph_nodes import (
    make_architecture_failure_response_node,
    make_build_memory_context_node,
    make_classify_error_node,
    make_collect_context_node,
    make_compress_memory_node,
    make_context_guard_node,
    make_deep_plan_node,
    make_deep_plan_revision_node,
    make_environment_error_response_node,
    make_execute_tools_node,
    make_inspect_node,
    make_load_builtin_memory_node,
    make_parallelism_gate_node,
    make_persist_snapshot_node,
    make_plan_node,
    make_plan_quality_gate_node,
    make_plan_response_node,
    make_plan_self_review_node,
    make_prefetch_memory_node,
    make_propose_verified_patch_node,
    make_queue_prefetch_node,
    make_receive_user_turn_node,
    make_recovery_action_node,
    make_repetition_guard_node,
    make_respond_node,
    make_select_tools_node,
    make_skill_context_node,
    make_skip_memory_node,
    make_spec_compliance_review_node,
    make_subagent_dispatch_node,
    make_subdirectory_hint_node,
    make_supervisor_review_node,
    make_sync_memory_node,
    make_task_state_node,
    make_wait_subagents_node,
)
from solo_agent.workflow.graph_state import SoloGraphState

# ---------------------------------------------------------------------------
# Route helpers
# ---------------------------------------------------------------------------

def _is_blocked(state: SoloGraphState) -> bool:
    return bool((state.get("agent_state") or {}).get("blocked", False))


def _is_awaiting_approval(state: SoloGraphState) -> bool:
    return bool((state.get("agent_state") or {}).get("awaiting_approval", False))


def _has_error(state: SoloGraphState) -> bool:
    return bool(state.get("error"))


def _is_architecture_failure(state: SoloGraphState) -> bool:
    return bool((state.get("error_state") or {}).get("classification") == "architecture_failure")


def _error_aware_route(state: SoloGraphState, fallback: str) -> str:
    if _has_error(state):
        return "classify_error"
    return fallback


def _memory_enabled_route(state: SoloGraphState) -> str:
    agent_data = state.get("agent_state") or {}
    if agent_data.get("memory_enabled", True):
        return "load_builtin_memory"
    return "skip_memory"


def _run_mode_route(state: SoloGraphState) -> str:
    return "plan"


def _plan_quality_route(state: SoloGraphState) -> str:
    agent_data = state.get("agent_state") or {}
    report = agent_data.get("plan_quality_report") or {}
    if report.get("passed", True):
        return "plan_self_review"
    return "deep_plan_revision"


def _execution_strategy_route(state: SoloGraphState) -> str:
    agent_data = state.get("agent_state") or {}
    strategy = agent_data.get("execution_strategy", "serial")
    if strategy == "parallel":
        return "parallel_dispatch"
    return "collect_context"


def _patch_route(state: SoloGraphState) -> str:
    agent_data = state.get("agent_state") or {}
    patch = agent_data.get("patch_proposal")
    if patch is None:
        return "subdirectory_hint"
    if agent_data.get("awaiting_approval", False):
        return END
    return "propose_verified_patch"


def _supervisor_route(state: SoloGraphState) -> str:
    agent_data = state.get("agent_state") or {}
    report = agent_data.get("supervisor_report") or {}
    decision = report.get("decision", "continue_parallel_summary")
    if decision == "need_main_execution":
        return "collect_context"
    if decision == "fallback_serial":
        return "collect_context"
    if decision == "blocked":
        return END
    return "context_guard_before_respond"


def _recovery_route(state: SoloGraphState) -> str:
    error_state = state.get("error_state") or {}
    category = error_state.get("classification", "fatal")
    if category == "recoverable_error":
        return "recovery_action"
    if category == "policy_violation":
        return "blocked_response"
    if category == "environment_error":
        return "environment_error_response"
    return "architecture_failure_response"


# ---------------------------------------------------------------------------
# build_main_workflow_graph — the Single Graph
# ---------------------------------------------------------------------------

def build_main_workflow_graph(
    *,
    provider: Any,
    deps: Any,
    settings: Any,
) -> StateGraph:
    """Build the complete LangGraph StateGraph for the Solo Agent workflow.

    This is the SOLE orchestration graph covering all execution modes in a
    single unified topology:
      - plan mode (deep plan → quality gate → revision → self-review → response)
      - agent serial (plan → context → inspect → tools → review → respond)
      - agent parallel (plan → gate → dispatch → subagents → supervisor → respond)
      - verified editing (propose → await approval → apply → verify → review)
      - error recovery (classify → recover/block)
      - memory prelude/postlude
      - checkpoint persistence
    """
    graph = StateGraph(SoloGraphState)

    # -----------------------------------------------------------------------
    # Prelude: receive, memory, skills, context guard
    # -----------------------------------------------------------------------
    graph.add_node("receive_user_turn", make_receive_user_turn_node(settings))
    graph.add_node("skip_memory", make_skip_memory_node())
    graph.add_node("load_builtin_memory", make_load_builtin_memory_node(deps))
    graph.add_node("prefetch_memory", make_prefetch_memory_node(deps, settings))
    graph.add_node("build_memory_context", make_build_memory_context_node())
    graph.add_node("skill_context", make_skill_context_node(deps, settings))
    graph.add_node("context_guard_before_plan", make_context_guard_node(provider, deps, settings, phase="before_plan"))

    # -----------------------------------------------------------------------
    # Plan route
    # -----------------------------------------------------------------------
    graph.add_node("plan", make_plan_node(provider, settings))
    graph.add_node("task_state", make_task_state_node())
    graph.add_node("deep_plan", make_deep_plan_node(provider, settings))
    graph.add_node("plan_quality_gate", make_plan_quality_gate_node(settings))
    graph.add_node("deep_plan_revision", make_deep_plan_revision_node(provider, settings))
    graph.add_node("plan_self_review", make_plan_self_review_node(provider, settings))
    graph.add_node("plan_response", make_plan_response_node(provider, settings))

    # -----------------------------------------------------------------------
    # Agent serial path
    # -----------------------------------------------------------------------
    graph.add_node("parallelism_gate", make_parallelism_gate_node(settings))
    graph.add_node("collect_context", make_collect_context_node(deps, settings))
    graph.add_node("inspect", make_inspect_node(deps))
    graph.add_node("select_tools", make_select_tools_node(deps, settings))
    graph.add_node("execute_tools", make_execute_tools_node(deps, settings))

    # -----------------------------------------------------------------------
    # Agent parallel path (real dispatch, not placeholder)
    # -----------------------------------------------------------------------
    graph.add_node("parallel_dispatch", make_subagent_dispatch_node(deps, settings))
    graph.add_node("wait_subagents", make_wait_subagents_node(settings))
    graph.add_node("supervisor_review", make_supervisor_review_node(provider, settings))

    # -----------------------------------------------------------------------
    # Verified editing
    # -----------------------------------------------------------------------
    graph.add_node("propose_verified_patch", make_propose_verified_patch_node(provider, deps, settings))

    # -----------------------------------------------------------------------
    # Review layer
    # -----------------------------------------------------------------------
    graph.add_node("spec_compliance_review", make_spec_compliance_review_node(provider, settings))
    graph.add_node("subdirectory_hint", make_subdirectory_hint_node(settings))

    # -----------------------------------------------------------------------
    # Respond
    # -----------------------------------------------------------------------
    graph.add_node("context_guard_before_respond", make_context_guard_node(provider, deps, settings, phase="before_respond"))
    graph.add_node("respond", make_respond_node(provider, settings))

    # -----------------------------------------------------------------------
    # Error recovery
    # -----------------------------------------------------------------------
    graph.add_node("classify_error", make_classify_error_node())
    graph.add_node("repetition_guard", make_repetition_guard_node())
    graph.add_node("recovery_action", make_recovery_action_node(deps, settings))
    graph.add_node("blocked_response", make_respond_node(provider, settings))
    graph.add_node("environment_error_response", make_environment_error_response_node(provider, settings))
    graph.add_node("architecture_failure_response", make_architecture_failure_response_node(provider, settings))

    # -----------------------------------------------------------------------
    # Postlude: memory sync, persistence, emit completed
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
        "skip_memory",
        _memory_enabled_route,
        {
            "load_builtin_memory": "load_builtin_memory",
            "skip_memory": "skill_context",
        },
    )
    graph.add_edge("load_builtin_memory", "prefetch_memory")
    graph.add_edge("prefetch_memory", "build_memory_context")
    graph.add_edge("build_memory_context", "skill_context")
    graph.add_edge("skill_context", "context_guard_before_plan")

    # Plan vs agent route
    graph.add_conditional_edges(
        "context_guard_before_plan",
        _run_mode_route,
        {
            "deep_plan": "deep_plan",
            "plan": "plan",
        },
    )

    # Plan mode chain
    graph.add_conditional_edges(
        "deep_plan",
        lambda s: _error_aware_route(s, "plan_quality_gate"),
        {"plan_quality_gate": "plan_quality_gate", "classify_error": "classify_error"},
    )
    graph.add_conditional_edges(
        "plan_quality_gate",
        _plan_quality_route,
        {
            "plan_self_review": "plan_self_review",
            "deep_plan_revision": "deep_plan_revision",
        },
    )
    graph.add_conditional_edges(
        "deep_plan_revision",
        lambda s: _error_aware_route(s, "plan_quality_gate"),
        {"plan_quality_gate": "plan_quality_gate", "classify_error": "classify_error"},
    )
    graph.add_edge("plan_self_review", "plan_response")
    graph.add_edge("plan_response", "sync_memory")

    # Agent mode chain
    graph.add_conditional_edges(
        "plan",
        lambda s: _error_aware_route(s, "task_state"),
        {"task_state": "task_state", "classify_error": "classify_error"},
    )
    graph.add_edge("task_state", "parallelism_gate")

    graph.add_conditional_edges(
        "parallelism_gate",
        _execution_strategy_route,
        {
            "parallel_dispatch": "parallel_dispatch",
            "collect_context": "collect_context",
        },
    )

    # Serial path
    graph.add_edge("collect_context", "inspect")
    graph.add_conditional_edges(
        "inspect",
        lambda s: END if _is_blocked(s) else "select_tools",
        {END: END, "select_tools": "select_tools"},
    )
    graph.add_edge("select_tools", "execute_tools")
    graph.add_conditional_edges(
        "execute_tools",
        lambda s: "classify_error" if _has_error(s) else (END if _is_awaiting_approval(s) else "spec_compliance_review"),
        {END: END, "spec_compliance_review": "spec_compliance_review", "classify_error": "classify_error"},
    )
    # Always route through propose_verified_patch after spec compliance review
    # (the node itself skips if patch already generated)
    graph.add_edge("spec_compliance_review", "propose_verified_patch")

    graph.add_conditional_edges(
        "propose_verified_patch",
        lambda s: "classify_error" if _has_error(s) else _patch_route(s),
        {
            "classify_error": "classify_error",
            END: END,
            "subdirectory_hint": "subdirectory_hint",
            "propose_verified_patch": "propose_verified_patch",
        },
    )

    # Parallel path
    graph.add_conditional_edges(
        "parallel_dispatch",
        lambda s: _error_aware_route(s, "wait_subagents"),
        {"wait_subagents": "wait_subagents", "classify_error": "classify_error"},
    )
    graph.add_conditional_edges(
        "wait_subagents",
        lambda s: _error_aware_route(s, "supervisor_review"),
        {"supervisor_review": "supervisor_review", "classify_error": "classify_error"},
    )
    graph.add_conditional_edges(
        "supervisor_review",
        _supervisor_route,
        {
            "context_guard_before_respond": "context_guard_before_respond",
            "collect_context": "collect_context",
            END: END,
        },
    )

    # Respond path
    graph.add_edge("subdirectory_hint", "context_guard_before_respond")
    graph.add_conditional_edges(
        "context_guard_before_respond",
        lambda s: END if _is_blocked(s) else "respond",
        {END: END, "respond": "respond"},
    )

    graph.add_conditional_edges(
        "respond",
        lambda s: _error_aware_route(s, "sync_memory"),
        {"sync_memory": "sync_memory", "classify_error": "classify_error"},
    )

    # Error edges from execution nodes — catch exceptions and route to classify_error

    # Postlude with memory conditional
    graph.add_conditional_edges(
        "sync_memory",
        lambda s: "queue_prefetch" if (s.get("agent_state") or {}).get("memory_enabled", True) else "persist_snapshot",
        {"queue_prefetch": "queue_prefetch", "persist_snapshot": "persist_snapshot"},
    )
    graph.add_conditional_edges(
        "queue_prefetch",
        lambda s: "compress_memory" if (s.get("agent_state") or {}).get("memory_enabled", True) else "persist_snapshot",
        {"compress_memory": "compress_memory", "persist_snapshot": "persist_snapshot"},
    )
    graph.add_conditional_edges(
        "compress_memory",
        lambda s: "persist_snapshot" if (s.get("agent_state") or {}).get("memory_enabled", True) else "persist_snapshot",
        {"persist_snapshot": "persist_snapshot"},
    )
    graph.add_edge("persist_snapshot", END)

    # Error recovery routing (conditional entry)
    graph.add_conditional_edges(
        "classify_error",
        _recovery_route,
        {
            "recovery_action": "recovery_action",
            "blocked_response": "blocked_response",
            "environment_error_response": "environment_error_response",
            "architecture_failure_response": "architecture_failure_response",
        },
    )
    graph.add_edge("recovery_action", "repetition_guard")

    # Error terminal nodes all go to postlude
    for terminal in ("blocked_response", "environment_error_response", "architecture_failure_response"):
        graph.add_edge(terminal, "sync_memory")

    graph.add_conditional_edges(
        "repetition_guard",
        lambda s: "sync_memory" if (_is_blocked(s) or _is_architecture_failure(s)) else "recovery_action",
        {END: END, "recovery_action": "recovery_action", "sync_memory": "sync_memory"},
    )

    return graph


# ---------------------------------------------------------------------------
# Public route helpers (for testing)
# ---------------------------------------------------------------------------

def make_route_after_guard(target: str):
    def router(state: SoloGraphState) -> str:
        if _is_blocked(state):
            return END
        return target
    return router


def route_after_execute_tools(state: SoloGraphState) -> str:
    if _is_awaiting_approval(state):
        return END
    return "spec_compliance_review"


def route_after_parallelism_gate(state: SoloGraphState) -> str:
    if _is_blocked(state):
        return END
    agent_data = state.get("agent_state") or {}
    strategy = agent_data.get("execution_strategy", "serial")
    return "parallel_dispatch" if strategy == "parallel" else "collect_context"


def route_after_patch(state: SoloGraphState) -> str:
    if _is_awaiting_approval(state):
        return END
    return "subdirectory_hint"



