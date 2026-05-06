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
    memory_enabled: bool = True
    conversation_history_enabled: bool = True
    memory_budget: dict[str, Any] = field(default_factory=dict)
    summary_status: str = "not_started"
    plan: str = ""
    conversation_context: dict[str, Any] = field(default_factory=dict)
    memory_context_block: str = ""
    memory_warnings: list[str] = field(default_factory=list)
    skill_context_block: str = ""
    selected_skills: list[dict[str, Any]] = field(default_factory=list)
    skill_budget: dict[str, Any] = field(default_factory=dict)
    context: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    response: str = ""
    blocked: bool = False
    block_reason: str | None = None
    snapshots: dict[str, Any] = field(default_factory=dict)

    def snapshot(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "run_id": self.run_id,
            "user_input": self.user_input,
            "loop_stage": self.loop_stage,
            "memory_enabled": self.memory_enabled,
            "conversation_history_enabled": self.conversation_history_enabled,
            "memory_budget": self.memory_budget,
            "summary_status": self.summary_status,
            "plan": self.plan,
            "conversation_context": self.conversation_context,
            "memory_context_block": self.memory_context_block,
            "memory_warnings": self.memory_warnings,
            "skill_context_block": self.skill_context_block,
            "selected_skills": self.selected_skills,
            "skill_budget": self.skill_budget,
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
            "response": self.response,
            "blocked": self.blocked,
            "block_reason": self.block_reason,
            "snapshots": self.snapshots,
        }
