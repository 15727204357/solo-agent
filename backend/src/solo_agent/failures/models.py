"""Models for closed-loop failure analysis."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class FailureKind(StrEnum):
    TEST_FAILURE = "test_failure"
    LINT_FAILURE = "lint_failure"
    TYPE_FAILURE = "type_failure"
    DEPENDENCY_MISSING = "dependency_missing"
    ENVIRONMENT_ERROR = "environment_error"
    PATCH_CONFLICT = "patch_conflict"
    POLICY_BLOCKED = "policy_blocked"
    REQUIREMENT_GAP = "requirement_gap"
    UNKNOWN_FAILURE = "unknown_failure"


@dataclass(frozen=True)
class FailureReport:
    kind: FailureKind
    command: str
    summary: str
    returncode: int | None = None
    file: str = ""
    line: int | None = None
    failing_test: str = ""
    rule: str = ""
    snippet: str = ""
    stack_trace: str = ""
    raw_output: str = ""
    retryable: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["kind"] = self.kind.value
        return data
