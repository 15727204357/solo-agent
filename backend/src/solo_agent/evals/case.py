"""Data models for local coding-agent eval cases."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class EvalCase:
    id: str
    user_request: str
    initial_files: dict[str, str]
    task_type: str = "repo_bugfix"
    suite_id: str = "smoke"
    public_tests: list[str] = field(default_factory=list)
    hidden_assertions: list[str] = field(default_factory=list)
    expected_changed_files: list[str] = field(default_factory=list)
    forbidden_changed_files: list[str] = field(default_factory=list)
    allowed_commands: list[str] = field(default_factory=list)
    expected_intent: str = ""
    accepted_intents: list[str] = field(default_factory=list)
    required_scopes: list[str] = field(default_factory=list)
    forbidden_tools: list[str] = field(default_factory=list)
    required_tools: list[str] = field(default_factory=list)
    max_risk_level: str = ""
    approval_required: bool | None = None
    expected_reroute: bool | None = None
    route_score_threshold: float = 0.75

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvalResult:
    case_id: str
    passed: bool
    score: float
    changed_files: list[str]
    task_type: str = "repo_bugfix"
    suite_id: str = "smoke"
    patch_applied: bool = False
    unrelated_changed_files: list[str] = field(default_factory=list)
    tests_passed: int = 0
    tests_failed: int = 0
    sandbox_violations: list[str] = field(default_factory=list)
    outcome_status: str = "unknown"
    iterations: int = 0
    tool_calls: int = 0
    human_interventions: int = 0
    failure_class: str = ""
    route_passed: bool = True
    route_score: float = 1.0
    route_notes: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
