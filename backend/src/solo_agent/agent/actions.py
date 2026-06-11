"""Typed model-driven action contract for coding-agent tool loops."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

AgentActionKind = Literal[
    "think",
    "read",
    "search",
    "edit_preview",
    "edit_apply_in_sandbox",
    "run_check",
    "ask_user",
    "final",
    "halt",
]
ApprovalMode = Literal["confirm", "manual_only"]

AUTO_CONFIRM_ACTIONS: frozenset[str] = frozenset({"think", "read", "search", "edit_preview", "run_check"})
TERMINAL_ACTIONS: frozenset[str] = frozenset({"final", "halt"})
APPROVAL_GATED_ACTIONS: frozenset[str] = frozenset({"edit_apply_in_sandbox", "ask_user"})


class AgentAction(BaseModel):
    """One strongly typed action proposed by a model-driven tool loop."""

    kind: AgentActionKind
    thought: str = Field(default="", max_length=4000)
    tool_name: str | None = Field(default=None, max_length=120)
    arguments: dict[str, Any] = Field(default_factory=dict)
    final_response: str = Field(default="", max_length=12000)
    reason: str = Field(default="", max_length=1200)

    @field_validator("tool_name")
    @classmethod
    def validate_tool_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            return None
        if not normalized.replace("_", "").replace("-", "").isalnum():
            raise ValueError("tool_name must be a simple tool identifier")
        return normalized

    def requires_approval(self, *, approval_mode: ApprovalMode = "confirm") -> bool:
        return action_requires_approval(self, approval_mode=approval_mode)

    def to_tool_call(self) -> dict[str, Any] | None:
        """Convert tool-like actions into the existing registry call shape."""

        if self.kind in TERMINAL_ACTIONS or self.kind in {"think", "ask_user"}:
            return None
        if self.tool_name:
            return {"name": self.tool_name, "arguments": dict(self.arguments)}
        default_tool = {
            "read": "read_file",
            "search": "search_code",
            "edit_preview": "preview_patch",
            "edit_apply_in_sandbox": "apply_text_edit",
            "run_check": "run_command",
        }.get(self.kind)
        if not default_tool:
            return None
        return {"name": default_tool, "arguments": dict(self.arguments)}


def action_requires_approval(
    action: AgentAction | Mapping[str, Any],
    *,
    approval_mode: ApprovalMode = "confirm",
) -> bool:
    """Return whether an action must stop for human approval before execution."""

    kind = action.kind if isinstance(action, AgentAction) else str(action.get("kind") or "")
    if approval_mode == "manual_only":
        return kind not in {"think", *TERMINAL_ACTIONS}
    if kind in APPROVAL_GATED_ACTIONS:
        return True
    if kind in AUTO_CONFIRM_ACTIONS or kind in TERMINAL_ACTIONS:
        return False
    return True

