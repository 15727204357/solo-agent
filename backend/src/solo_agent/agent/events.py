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
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "session_id": self.session_id,
            "run_id": self.run_id,
            "node": self.node,
            "message": self.message,
            "data": self.data,
            "created_at": self.created_at,
        }

    def to_sse(self) -> str:
        import json

        return f"event: {self.type}\ndata: {json.dumps(self.to_dict(), ensure_ascii=False, default=str)}\n\n"
