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
    tool_loop_mode: str = "heuristic"
    approval_mode: str = "confirm"
    workspace_backend: str = "copy"
    eval_suite_id: str | None = None
    is_plan_mode: bool = False
    subagent_policy: str = "off"
    subagent_enabled: bool = False
    memory_enabled: bool = True
    conversation_history_enabled: bool = True
    memory_budget: dict[str, Any] = field(default_factory=dict)
    summary_status: str = "not_started"
    plan: str = ""
    conversation_context: dict[str, Any] = field(default_factory=dict)
    memory_context_block: str = ""
    memory_warnings: list[str] = field(default_factory=list)
    skills_index_block: str = ""
    skill_recipes_block: str = ""
    skill_context_block: str = ""
    selected_skills: list[dict[str, Any]] = field(default_factory=list)
    selected_recipes: list[dict[str, Any]] = field(default_factory=list)
    recipe_runs: list[dict[str, Any]] = field(default_factory=list)
    active_subflow: dict[str, Any] = field(default_factory=dict)
    recipe_policy_snapshot: dict[str, Any] = field(default_factory=dict)
    skill_budget: dict[str, Any] = field(default_factory=dict)
    behavior_policy: dict[str, Any] = field(default_factory=dict)
    task_list: dict[str, Any] = field(default_factory=dict)
    task_candidates: list[dict[str, Any]] = field(default_factory=list)
    parallelism_decision: dict[str, Any] = field(default_factory=dict)
    execution_strategy: str = "serial"
    context: list[dict[str, Any]] = field(default_factory=list)
    code_map_summary: dict[str, Any] = field(default_factory=dict)
    impact_analysis: dict[str, Any] = field(default_factory=dict)
    resume_target: dict[str, Any] = field(default_factory=dict)
    human_feedback: dict[str, Any] = field(default_factory=dict)
    sandbox_artifacts: dict[str, Any] = field(default_factory=dict)
    failure_reports: list[dict[str, Any]] = field(default_factory=list)
    outcome_report: dict[str, Any] = field(default_factory=dict)
    evidence_timeline: list[dict[str, Any]] = field(default_factory=list)
    git_artifact_proposal: dict[str, Any] = field(default_factory=dict)
    eval_report: dict[str, Any] = field(default_factory=dict)
    intent_route_plan: dict[str, Any] = field(default_factory=dict)
    route_epoch: int = 0
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    patch_proposal: dict[str, Any] | None = None
    skill_change_proposal: dict[str, Any] | None = None
    skill_evolution_candidates: list[dict[str, Any]] = field(default_factory=list)
    skill_evolution_proposal: dict[str, Any] | None = None
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
        return self._base_snapshot()

    def _base_snapshot(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "run_id": self.run_id,
            "user_input": self.user_input,
            "loop_stage": self.loop_stage,
            "run_mode": self.run_mode,
            "tool_loop_mode": self.tool_loop_mode,
            "approval_mode": self.approval_mode,
            "workspace_backend": self.workspace_backend,
            "eval_suite_id": self.eval_suite_id,
            "is_plan_mode": self.is_plan_mode,
            "subagent_policy": self.subagent_policy,
            "subagent_enabled": self.subagent_enabled,
            "memory_enabled": self.memory_enabled,
            "conversation_history_enabled": self.conversation_history_enabled,
            "memory_budget": self.memory_budget,
            "summary_status": self.summary_status,
            "plan": self.plan,
            "conversation_context": self.conversation_context,
            "memory_context_block": self.memory_context_block,
            "memory_warnings": self.memory_warnings,
            "skills_index_block": self.skills_index_block,
            "skill_recipes_block": self.skill_recipes_block,
            "skill_context_block": self.skill_context_block,
            "selected_skills": self.selected_skills,
            "selected_recipes": self.selected_recipes,
            "recipe_runs": self.recipe_runs,
            "active_subflow": self.active_subflow,
            "recipe_policy_snapshot": self.recipe_policy_snapshot,
            "skill_budget": self.skill_budget,
            "behavior_policy": self.behavior_policy,
            "task_list": self.task_list,
            "task_candidates": self.task_candidates,
            "parallelism_decision": self.parallelism_decision,
            "execution_strategy": self.execution_strategy,
            "context": self.context,
            "code_map_summary": self.code_map_summary,
            "impact_analysis": self.impact_analysis,
            "resume_target": self.resume_target,
            "human_feedback": self.human_feedback,
            "sandbox_artifacts": self.sandbox_artifacts,
            "failure_reports": self.failure_reports,
            "outcome_report": self.outcome_report,
            "evidence_timeline": self.evidence_timeline,
            "git_artifact_proposal": self.git_artifact_proposal,
            "eval_report": self.eval_report,
            "intent_route_plan": self.intent_route_plan,
            "route_epoch": self.route_epoch,
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
            "skill_change_proposal": self.skill_change_proposal,
            "skill_evolution_candidates": self.skill_evolution_candidates,
            "skill_evolution_proposal": self.skill_evolution_proposal,
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
