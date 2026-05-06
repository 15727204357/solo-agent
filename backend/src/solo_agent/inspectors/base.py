"""Shared types for deterministic safety inspectors."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class InspectionResult:
    """Result returned by inspectors.

    Inspectors are intentionally deterministic for the first MVP: a blocked
    result means the caller should not continue with the request or tool call.
    """

    allowed: bool
    reason: str = ""
    code: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def allow(cls, metadata: Mapping[str, Any] | None = None) -> InspectionResult:
        return cls(allowed=True, metadata=metadata or {})

    @classmethod
    def block(
        cls,
        reason: str,
        *,
        code: str = "blocked",
        metadata: Mapping[str, Any] | None = None,
    ) -> InspectionResult:
        return cls(
            allowed=False,
            reason=reason,
            code=code,
            metadata=metadata or {},
        )


@dataclass(frozen=True)
class ToolCall:
    """Normalized tool call passed through inspectors."""

    name: str
    arguments: Mapping[str, Any] = field(default_factory=dict)


class Inspector(Protocol):
    """Protocol implemented by all safety inspectors."""

    def inspect_text(self, text: str) -> InspectionResult:
        """Inspect a user or model text message."""

    def inspect_tool_call(self, call: ToolCall) -> InspectionResult:
        """Inspect a proposed tool call before execution."""
