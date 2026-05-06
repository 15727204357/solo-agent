"""Detect repeated tool calls that are unlikely to make progress."""

from __future__ import annotations

import json
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .base import InspectionResult, ToolCall


@dataclass
class RepetitionInspector:
    """Block the same tool call after it repeats too many times."""

    max_repetitions: int = 3
    window_size: int = 8
    _recent_calls: deque[str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._recent_calls = deque(maxlen=max(self.window_size, self.max_repetitions))

    def inspect_text(self, text: str) -> InspectionResult:
        return InspectionResult.allow()

    def inspect_tool_call(self, call: ToolCall) -> InspectionResult:
        signature = _signature(call.name, call.arguments)
        repeat_count = sum(1 for existing in self._recent_calls if existing == signature)
        if repeat_count >= self.max_repetitions:
            return InspectionResult.block(
                "Repeated identical tool call was blocked.",
                code="repeated_tool_call",
                metadata={"tool": call.name, "repetitions": repeat_count},
            )

        self._recent_calls.append(signature)
        return InspectionResult.allow()

    def reset(self) -> None:
        self._recent_calls.clear()


def _signature(name: str, arguments: Mapping[str, Any]) -> str:
    return json.dumps(
        {"name": name, "arguments": arguments},
        sort_keys=True,
        default=str,
        separators=(",", ":"),
    )
