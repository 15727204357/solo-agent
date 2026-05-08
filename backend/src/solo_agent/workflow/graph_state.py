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
    return AgentState(
        session_id=str(data.get("session_id", "")),
        run_id=str(data.get("run_id", "")),
        user_input=str(data.get("user_input", "")),
        loop_stage=str(data.get("loop_stage", "initialized")),
        run_mode=str(data.get("run_mode", "agent")),
        memory_enabled=bool(data.get("memory_enabled", True)),
        conversation_history_enabled=bool(data.get("conversation_history_enabled", True)),
        memory_budget=dict(data.get("memory_budget") or {}),
        summary_status=str(data.get("summary_status", "not_started")),
        plan=str(data.get("plan", "")),
        deep_plan=str(data.get("deep_plan", "")),
        plan_quality_report=dict(data.get("plan_quality_report") or {}),
        conversation_context=dict(data.get("conversation_context") or {}),
        memory_context_block=str(data.get("memory_context_block", "")),
        memory_warnings=list(data.get("memory_warnings") or []),
        skill_context_block=str(data.get("skill_context_block", "")),
        selected_skills=list(data.get("selected_skills") or []),
        skill_budget=dict(data.get("skill_budget") or {}),
        behavior_policy=dict(data.get("behavior_policy") or {}),
        task_candidates=list(data.get("task_candidates") or []),
        parallelism_decision=dict(data.get("parallelism_decision") or {}),
        execution_strategy=str(data.get("execution_strategy", "serial")),
        context=list(data.get("context") or []),
        tool_calls=tool_calls,
        patch_proposal=data.get("patch_proposal"),
        awaiting_approval=bool(data.get("awaiting_approval", False)),
        response=str(data.get("response", "")),
        blocked=bool(data.get("blocked", False)),
        block_reason=data.get("block_reason"),
        snapshots=dict(data.get("snapshots") or {}),
        last_error=dict(data.get("last_error") or {}),
        retry_count=int(data.get("retry_count", 0)),
        error_classification=str(data.get("error_classification", "")),
        compaction_attempts=int(data.get("compaction_attempts", 0)),
    )


def initial_graph_state(state: AgentState) -> SoloGraphState:
    return {
        "agent_state": agent_state_to_graph_data(state),
        "events": [],
        "error": None,
    }


def update_from_agent_state(
    graph_state: SoloGraphState,
    state: AgentState,
) -> SoloGraphState:
    graph_state["agent_state"] = agent_state_to_graph_data(state)
    return graph_state
