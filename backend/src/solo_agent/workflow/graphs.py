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
    make_environment_error_response_node,
    make_execute_tools_node,
    make_inspect_node,
    make_intent_route_node,
    make_load_builtin_memory_node,
    make_parallel_dispatch_node,
    make_parallelism_gate_node,
    make_persist_snapshot_node,
    make_plan_node,
    make_prefetch_memory_node,
    make_propose_verified_patch_node,
    make_queue_prefetch_node,
    make_receive_user_turn_node,
    make_recovery_action_node,
    make_repetition_guard_node,
    make_respond_node,
    make_select_tools_node,
    make_skill_context_node,
    make_skill_evolution_node,
    make_skip_memory_node,
    make_spec_compliance_review_node,
    make_subdirectory_hint_node,
    make_supervisor_review_node,
    make_sync_memory_node,
    make_task_state_node,
    make_team_develop_node,
    make_team_plan_node,
    make_team_supervisor_node,
    make_team_test_node,
    make_wait_subagents_node,
)
from solo_agent.workflow.graph_state import SoloGraphState
from solo_agent.workflow.intent_router import reroute_triggers_from_state

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


def _team_mode_enabled(state: SoloGraphState) -> bool:
    agent_data = state.get("agent_state") or {}
    run_mode = str(agent_data.get("run_mode") or "")
    return bool((run_mode == "plan" or agent_data.get("is_plan_mode")) and agent_data.get("subagent_enabled"))




def _patch_route(state: SoloGraphState) -> str:
    agent_data = state.get("agent_state") or {}
    patch = agent_data.get("patch_proposal")
    if patch is None:
        return "subdirectory_hint"
    if agent_data.get("awaiting_approval", False):
        return END
    return "propose_verified_patch"


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
# build_main_workflow_graph 鈥?the Single Graph
# ---------------------------------------------------------------------------

def build_main_workflow_graph(
    *,
    provider: Any,
    deps: Any,
    settings: Any,
) -> StateGraph:
    """Build the complete LangGraph StateGraph for the Solo Agent workflow.

    This is the sole orchestration graph covering all execution modes in a
    single unified topology. In team mode, ``parallelism_gate`` is the
    developer orchestration gate between planning and development; it does
    not dispatch subagents directly.
      - plan capability mode: same execution graph, stronger planning prompt
      - agent workflow: plan -> context -> inspect -> tools -> review -> respond
      - team workflow: plan -> team_plan -> parallelism_gate -> team_develop -> team_test -> team_supervisor
      - verified editing (propose -> await approval -> apply -> verify -> review)
      - error recovery (classify -> recover/block)
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
    graph.add_node("load_task_state", make_task_state_node(settings))
    graph.add_node("plan", make_plan_node(provider, settings))
    graph.add_node("task_state", make_task_state_node(settings))

    # -----------------------------------------------------------------------
    # Agent serial path
    # -----------------------------------------------------------------------
    graph.add_node("parallelism_gate", make_parallelism_gate_node(settings))
    graph.add_node("parallel_dispatch", make_parallel_dispatch_node(settings))
    graph.add_node("wait_subagents", make_wait_subagents_node(provider, deps, settings))
    graph.add_node("supervisor_review", make_supervisor_review_node(settings))
    graph.add_node("team_plan", make_team_plan_node(settings))
    graph.add_node("team_develop", make_team_develop_node(provider, deps, settings))
    graph.add_node("team_test", make_team_test_node(deps, settings))
    graph.add_node("team_supervisor", make_team_supervisor_node(deps, settings))
    graph.add_node("collect_context", make_collect_context_node(deps, settings))
    graph.add_node("inspect", make_inspect_node(deps))
    graph.add_node("intent_route", make_intent_route_node(deps, settings))
    graph.add_node("select_tools", make_select_tools_node(deps, settings))
    graph.add_node("execute_tools", make_execute_tools_node(deps, settings))

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
    graph.add_node("skill_evolution", make_skill_evolution_node(deps, settings))
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
    graph.add_edge("context_guard_before_plan", "load_task_state")
    graph.add_edge("load_task_state", "plan")

    # Agent mode chain
    graph.add_conditional_edges(
        "plan",
        lambda s: _error_aware_route(s, "task_state"),
        {"task_state": "task_state", "classify_error": "classify_error"},
    )
    graph.add_conditional_edges(
        "task_state",
        route_after_task_state,
        {"team_plan": "team_plan", "collect_context": "collect_context"},
    )

    # Legacy diagnostic fan-out nodes stay registered for compatibility, but
    # they are intentionally not wired into the main graph. Developer
    # parallelism is decided by parallelism_gate inside the team workflow.

    # Lightweight team workflow: planner -> developer parallelism gate -> developer pool -> tester -> supervisor.
    graph.add_edge("team_plan", "parallelism_gate")
    graph.add_conditional_edges(
        "parallelism_gate",
        route_after_parallelism_gate,
        {END: END, "team_develop": "team_develop"},
    )
    graph.add_edge("team_develop", "team_test")
    graph.add_conditional_edges(
        "team_test",
        route_after_team_test,
        {"team_develop": "team_develop", "team_supervisor": "team_supervisor"},
    )
    graph.add_conditional_edges(
        "team_supervisor",
        route_after_team_supervisor,
        {END: END, "collect_context": "collect_context", "classify_error": "classify_error"},
    )

    # Serial path
    graph.add_edge("collect_context", "inspect")
    graph.add_conditional_edges(
        "inspect",
        lambda s: END if _is_blocked(s) else "intent_route",
        {END: END, "intent_route": "intent_route"},
    )
    graph.add_edge("intent_route", "select_tools")
    graph.add_edge("select_tools", "execute_tools")
    graph.add_conditional_edges(
        "execute_tools",
        lambda s: "classify_error" if _has_error(s) else route_after_execute_tools(s),
        {
            END: END,
            "intent_route": "intent_route",
            "spec_compliance_review": "spec_compliance_review",
            "classify_error": "classify_error",
        },
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

    # Respond path
    graph.add_edge("subdirectory_hint", "context_guard_before_respond")
    graph.add_conditional_edges(
        "context_guard_before_respond",
        lambda s: END if _is_blocked(s) else "respond",
        {END: END, "respond": "respond"},
    )

    graph.add_conditional_edges(
        "respond",
        lambda s: _error_aware_route(s, "skill_evolution"),
        {"skill_evolution": "skill_evolution", "classify_error": "classify_error"},
    )

    # Error edges from execution nodes 鈥?catch exceptions and route to classify_error

    # Postlude with memory conditional
    graph.add_conditional_edges(
        "skill_evolution",
        lambda s: "classify_error" if _has_error(s) else (END if _is_awaiting_approval(s) else "sync_memory"),
        {END: END, "sync_memory": "sync_memory", "classify_error": "classify_error"},
    )
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
    if _should_reroute_after_tools(state):
        return "intent_route"
    return "spec_compliance_review"


def _should_reroute_after_tools(state: SoloGraphState) -> bool:
    agent_data = state.get("agent_state") or {}
    snapshots = agent_data.get("snapshots") or {}
    if isinstance(snapshots, dict) and snapshots.get("pending_reroute"):
        return True
    router_settings = snapshots.get("intent_router") if isinstance(snapshots, dict) else {}
    settings = router_settings if isinstance(router_settings, dict) else {}
    return bool(reroute_triggers_from_state(agent_data, settings))


def route_after_task_state(state: SoloGraphState) -> str:
    if _team_mode_enabled(state):
        return "team_plan"
    return "collect_context"


def route_after_parallelism_gate(state: SoloGraphState) -> str:
    if _is_blocked(state):
        return END
    return "team_develop"


def route_after_team_test(state: SoloGraphState) -> str:
    agent_data = state.get("agent_state") or {}
    review_reports = agent_data.get("review_reports") or {}
    snapshots = agent_data.get("snapshots") or {}
    report = review_reports.get("team_test") or snapshots.get("team_test_report") or {}
    if (
        isinstance(report, dict)
        and report.get("status") == "needs_fix"
        and int(report.get("iteration", 0)) < int(report.get("max_iterations", 2))
    ):
        return "team_develop"
    return "team_supervisor"


def route_after_team_supervisor(state: SoloGraphState) -> str:
    if _has_error(state):
        return "classify_error"
    if _is_awaiting_approval(state):
        return END
    return "collect_context"


def route_after_supervisor_review(state: SoloGraphState) -> str:
    if _has_error(state):
        return "classify_error"
    agent_data = state.get("agent_state") or {}
    report = agent_data.get("supervisor_report") or {}
    if isinstance(report, dict) and report.get("status") == "passed":
        return "spec_compliance_review"
    return "collect_context"


def route_after_patch(state: SoloGraphState) -> str:
    if _is_awaiting_approval(state):
        return END
    return "subdirectory_hint"
