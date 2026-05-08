from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from solo_agent.agent.state import AgentState


@dataclass
class SubagentRunRecord:
    run_id: str
    subagent_type: str
    description: str
    status: str = "pending"
    result: str = ""
    error: str = ""
    started_at: str = ""
    completed_at: str = ""


@dataclass
class WorkflowState:
    agent_state: AgentState
    messages: list[Any] = field(default_factory=list)
    thread_data: dict[str, Any] = field(default_factory=dict)
    sandbox: Any = None
    artifacts: dict[str, Any] = field(default_factory=dict)
    todos: list[dict[str, Any]] = field(default_factory=list)
    subagent_runs: dict[str, SubagentRunRecord] = field(default_factory=dict)
    active_subagent_count: int = 0

    @classmethod
    def from_agent_state(cls, state: AgentState) -> WorkflowState:
        return cls(agent_state=state)

    @property
    def session_id(self) -> str:
        return self.agent_state.session_id

    @property
    def run_id(self) -> str:
        return self.agent_state.run_id

    def snapshot(self) -> dict[str, Any]:
        base = self.agent_state.snapshot()
        base["workflow_artifacts"] = self.artifacts
        base["workflow_todos"] = self.todos
        base["subagent_runs"] = {
            rid: {
                "run_id": r.run_id,
                "subagent_type": r.subagent_type,
                "description": r.description,
                "status": r.status,
                "result": r.result,
                "error": r.error,
            }
            for rid, r in self.subagent_runs.items()
        }
        return base

    def add_subagent_run(self, record: SubagentRunRecord) -> None:
        self.subagent_runs[record.run_id] = record

    def get_active_subagent_count(self) -> int:
        return sum(
            1 for r in self.subagent_runs.values()
            if r.status in ("pending", "running")
        )
