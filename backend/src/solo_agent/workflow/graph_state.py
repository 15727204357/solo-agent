from __future__ import annotations

from typing import Any

from solo_agent.agent.state import AgentState, ToolCallRecord

SoloGraphState = dict[str, Any]


def append_events(
    existing: list[dict[str, Any]] | None,
    new: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    if existing is None:
        existing = []
    if new:
        existing.extend(new)
    return existing


def merge_dicts(
    existing: dict[str, Any] | None,
    new: dict[str, Any] | None,
) -> dict[str, Any]:
    if existing is None:
        existing = {}
    if new:
        existing.update(new)
    return existing


def agent_state_to_graph_data(state: AgentState) -> dict[str, Any]:
    return state.snapshot()


def agent_state_from_graph_data(data: dict[str, Any]) -> AgentState:
    common = _parse_common_fields(data)
    return AgentState(**common)


def _parse_common_fields(data: dict[str, Any]) -> dict[str, Any]:
    tool_calls_raw: list[dict[str, Any]] = data.get("tool_calls") or []
    tool_calls = [
        ToolCallRecord(
            name=c.get("name", ""),
            arguments=dict(c.get("arguments") or {}),
            result=c.get("result"),
            blocked=bool(c.get("blocked", False)),
            reason=c.get("reason"),
        )
        for c in tool_calls_raw
    ]
    return {
        "session_id": str(data.get("session_id", "")),
        "run_id": str(data.get("run_id", "")),
        "user_input": str(data.get("user_input", "")),
        "loop_stage": str(data.get("loop_stage", "initialized")),
        "run_mode": str(data.get("run_mode", "agent")),
        "tool_loop_mode": str(data.get("tool_loop_mode", "heuristic")),
        "approval_mode": str(data.get("approval_mode", "confirm")),
        "workspace_backend": str(data.get("workspace_backend", "copy")),
        "eval_suite_id": data.get("eval_suite_id"),
        "is_plan_mode": bool(data.get("is_plan_mode", False)),
        "subagent_policy": str(data.get("subagent_policy", "off")),
        "subagent_enabled": bool(data.get("subagent_enabled", False)),
        "memory_enabled": bool(data.get("memory_enabled", True)),
        "conversation_history_enabled": bool(data.get("conversation_history_enabled", True)),
        "memory_budget": dict(data.get("memory_budget") or {}),
        "summary_status": str(data.get("summary_status", "not_started")),
        "plan": str(data.get("plan", "")),
        "conversation_context": dict(data.get("conversation_context") or {}),
        "memory_context_block": str(data.get("memory_context_block", "")),
        "memory_warnings": list(data.get("memory_warnings") or []),
        "skills_index_block": str(data.get("skills_index_block", "")),
        "skill_recipes_block": str(data.get("skill_recipes_block", "")),
        "skill_context_block": str(data.get("skill_context_block", "")),
        "selected_skills": list(data.get("selected_skills") or []),
        "selected_recipes": list(data.get("selected_recipes") or []),
        "recipe_runs": list(data.get("recipe_runs") or []),
        "active_subflow": dict(data.get("active_subflow") or {}),
        "recipe_policy_snapshot": dict(data.get("recipe_policy_snapshot") or {}),
        "skill_budget": dict(data.get("skill_budget") or {}),
        "behavior_policy": dict(data.get("behavior_policy") or {}),
        "task_list": dict(data.get("task_list") or {}),
        "task_candidates": list(data.get("task_candidates") or []),
        "parallelism_decision": dict(data.get("parallelism_decision") or {}),
        "execution_strategy": str(data.get("execution_strategy", "serial")),
        "context": list(data.get("context") or []),
        "code_map_summary": dict(data.get("code_map_summary") or {}),
        "impact_analysis": dict(data.get("impact_analysis") or {}),
        "resume_target": dict(data.get("resume_target") or {}),
        "human_feedback": dict(data.get("human_feedback") or {}),
        "sandbox_artifacts": dict(data.get("sandbox_artifacts") or {}),
        "failure_reports": list(data.get("failure_reports") or []),
        "outcome_report": dict(data.get("outcome_report") or {}),
        "evidence_timeline": list(data.get("evidence_timeline") or []),
        "git_artifact_proposal": dict(data.get("git_artifact_proposal") or {}),
        "eval_report": dict(data.get("eval_report") or {}),
        "tool_calls": tool_calls,
        "patch_proposal": data.get("patch_proposal"),
        "skill_change_proposal": data.get("skill_change_proposal"),
        "skill_evolution_candidates": list(data.get("skill_evolution_candidates") or []),
        "skill_evolution_proposal": data.get("skill_evolution_proposal"),
        "awaiting_approval": bool(data.get("awaiting_approval", False)),
        "response": str(data.get("response", "")),
        "blocked": bool(data.get("blocked", False)),
        "block_reason": data.get("block_reason"),
        "snapshots": dict(data.get("snapshots") or {}),
        "last_error": dict(data.get("last_error") or {}),
        "retry_count": int(data.get("retry_count", 0)),
        "error_classification": str(data.get("error_classification", "")),
        "compaction_attempts": int(data.get("compaction_attempts", 0)),
        "route_decisions": list(data.get("route_decisions") or []),
        "review_reports": dict(data.get("review_reports") or {}),
        "subagent_dispatches": list(data.get("subagent_dispatches") or []),
        "subagent_results": dict(data.get("subagent_results") or {}),
        "supervisor_report": data.get("supervisor_report"),
        "error_state": dict(data.get("error_state") or {}),
        "approval_state": str(data.get("approval_state", "none")),
        "provider_mode": str(data.get("provider_mode", "complete")),
        "recovery_attempts": int(data.get("recovery_attempts", 0)),
        "current_node": str(data.get("current_node", "")),
        "previous_node": data.get("previous_node"),
    }





def _base_graph_dict(agent_data: dict[str, Any]) -> SoloGraphState:
    return {
        "agent_state": agent_data,
        "events": [],
        "error": None,
        "route_decisions": [],
        "review_reports": {},
        "subagent_dispatches": [],
        "subagent_results": {},
        "supervisor_report": None,
        "error_state": {},
        "approval_state": "none",
        "provider_mode": "complete",
        "recovery_attempts": 0,
        "current_node": "",
        "previous_node": None,
    }


def initial_graph_state(state: AgentState) -> SoloGraphState:
    return _base_graph_dict(agent_state_to_graph_data(state))




def update_from_agent_state(
    graph_state: SoloGraphState,
    state: AgentState,
) -> SoloGraphState:
    graph_state["agent_state"] = agent_state_to_graph_data(state)
    return graph_state
