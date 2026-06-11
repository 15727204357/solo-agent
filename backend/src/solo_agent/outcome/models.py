"""Models for task outcome judging."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

OutcomeStatus = Literal["passed", "needs_fix", "blocked", "inconclusive"]


@dataclass(frozen=True)
class RequirementEvidence:
    requirement: str
    status: str
    evidence: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class RiskItem:
    severity: str
    description: str
    mitigation: str = ""


@dataclass(frozen=True)
class TaskOutcomeReport:
    status: OutcomeStatus
    approval_ready: bool
    summary: str
    requirements_covered: list[RequirementEvidence] = field(default_factory=list)
    missing_evidence: list[str] = field(default_factory=list)
    risks: list[RiskItem] = field(default_factory=list)
    recommended_next_actions: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
