from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCallRecord:
    name: str
    arguments: dict[str, Any]
    result: Any = None
    blocked: bool = False
    reason: str | None = None


@dataclass
class AgentState:
    session_id: str
    run_id: str
    user_input: str
    loop_stage: str = "initialized"
    run_mode: str = "agent"
    memory_enabled: bool = True
    conversation_history_enabled: bool = True
    memory_budget: dict[str, Any] = field(default_factory=dict)
    summary_status: str = "not_started"
    plan: str = ""
    deep_plan: str = ""
    plan_quality_report: dict[str, Any] = field(default_factory=dict)
    conversation_context: dict[str, Any] = field(default_factory=dict)
    memory_context_block: str = ""
    memory_warnings: list[str] = field(default_factory=list)
    skill_context_block: str = ""
    selected_skills: list[dict[str, Any]] = field(default_factory=list)
    skill_budget: dict[str, Any] = field(default_factory=dict)
    behavior_policy: dict[str, Any] = field(default_factory=dict)
    task_candidates: list[dict[str, Any]] = field(default_factory=list)
    parallelism_decision: dict[str, Any] = field(default_factory=dict)
    execution_strategy: str = "serial"
    context: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    patch_proposal: dict[str, Any] | None = None
    awaiting_approval: bool = False
    response: str = ""
    blocked: bool = False
    block_reason: str | None = None
    snapshots: dict[str, Any] = field(default_factory=dict)
    # 错误处理层新增字段
    last_error: dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    error_classification: str = ""
    compaction_attempts: int = 0

    # Graph-level state (final workflow orchestration)
    route_decisions: list[dict[str, Any]] = field(default_factory=list)
    review_reports: dict[str, Any] = field(default_factory=dict)
    subagent_dispatches: list[dict[str, Any]] = field(default_factory=list)
    subagent_results: dict[str, Any] = field(default_factory=dict)
    supervisor_report: dict[str, Any] | None = None
    error_state: dict[str, Any] = field(default_factory=dict)
    approval_state: str = "none"
    provider_mode: str = "complete"
    recovery_attempts: int = 0
    current_node: str = ""
    previous_node: str | None = None

    def snapshot(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "run_id": self.run_id,
            "user_input": self.user_input,
            "loop_stage": self.loop_stage,
            "run_mode": self.run_mode,
            "memory_enabled": self.memory_enabled,
            "conversation_history_enabled": self.conversation_history_enabled,
            "memory_budget": self.memory_budget,
            "summary_status": self.summary_status,
            "plan": self.plan,
            "deep_plan": self.deep_plan,
            "plan_quality_report": self.plan_quality_report,
            "conversation_context": self.conversation_context,
            "memory_context_block": self.memory_context_block,
            "memory_warnings": self.memory_warnings,
            "skill_context_block": self.skill_context_block,
            "selected_skills": self.selected_skills,
            "skill_budget": self.skill_budget,
            "behavior_policy": self.behavior_policy,
            "task_candidates": self.task_candidates,
            "parallelism_decision": self.parallelism_decision,
            "execution_strategy": self.execution_strategy,
            "context": self.context,
            "tool_calls": [
                {
                    "name": call.name,
                    "arguments": call.arguments,
                    "result": call.result,
                    "blocked": call.blocked,
                    "reason": call.reason,
                }
                for call in self.tool_calls
            ],
            "patch_proposal": self.patch_proposal,
            "awaiting_approval": self.awaiting_approval,
            "response": self.response,
            "blocked": self.blocked,
            "block_reason": self.block_reason,
            "snapshots": self.snapshots,
            # 错误处理层新增字段
            "last_error": self.last_error,
            "retry_count": self.retry_count,
            "error_classification": self.error_classification,
            "compaction_attempts": self.compaction_attempts,
            # Graph-level state
            "route_decisions": self.route_decisions,
            "review_reports": self.review_reports,
            "subagent_dispatches": self.subagent_dispatches,
            "subagent_results": self.subagent_results,
            "supervisor_report": self.supervisor_report,
            "error_state": self.error_state,
            "approval_state": self.approval_state,
            "provider_mode": self.provider_mode,
            "recovery_attempts": self.recovery_attempts,
            "current_node": self.current_node,
            "previous_node": self.previous_node,
        }
