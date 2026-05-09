from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass(frozen=True)
class AgentEvent:
    type: str
    session_id: str
    run_id: str
    node: str
    message: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    agent_source: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "session_id": self.session_id,
            "run_id": self.run_id,
            "node": self.node,
            "message": self.message,
            "data": self.data,
            "agent_source": self.agent_source,
            "created_at": self.created_at,
        }

    @staticmethod
    def _make_event(
        type_: str,
        session_id: str,
        run_id: str,
        node: str,
        message: str = "",
        data: dict[str, Any] | None = None,
        agent_source: str = "",
    ) -> AgentEvent:
        return AgentEvent(
            type=type_,
            session_id=session_id,
            run_id=run_id,
            node=node,
            message=message,
            data=data or {},
            agent_source=agent_source,
        )

    # Workflow 新事件类型工厂方法
    @classmethod
    def task_started(cls, session_id: str, run_id: str, node: str,
                     message: str = "", data: dict[str, Any] | None = None) -> AgentEvent:
        return cls._make_event("task_started", session_id, run_id, node, message, data)

    @classmethod
    def task_running(cls, session_id: str, run_id: str, node: str,
                     message: str = "", data: dict[str, Any] | None = None) -> AgentEvent:
        return cls._make_event("task_running", session_id, run_id, node, message, data)

    @classmethod
    def task_completed(cls, session_id: str, run_id: str, node: str,
                       message: str = "", data: dict[str, Any] | None = None) -> AgentEvent:
        return cls._make_event("task_completed", session_id, run_id, node, message, data)

    @classmethod
    def task_failed(cls, session_id: str, run_id: str, node: str,
                    message: str = "", data: dict[str, Any] | None = None) -> AgentEvent:
        return cls._make_event("task_failed", session_id, run_id, node, message, data)

    @classmethod
    def subagent_limited(cls, session_id: str, run_id: str, node: str,
                         message: str = "", data: dict[str, Any] | None = None) -> AgentEvent:
        return cls._make_event("subagent_limited", session_id, run_id, node, message, data)

    def to_sse(self) -> str:
        import json

        return f"event: {self.type}\ndata: {json.dumps(self.to_dict(), ensure_ascii=False, default=str)}\n\n"
