"""Outcome aggregation for procedural skills and recipes."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class SkillOutcomeRecord:
    skill_name: str
    selected: bool = True
    recipe_status: str = "none"
    verified: bool = False
    blocked_steps: int = 0
    failed_steps: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SkillOutcomeSummary:
    skill_name: str
    selected_count: int = 0
    verified_count: int = 0
    recipe_completed_count: int = 0
    recipe_blocked_count: int = 0
    recipe_failed_count: int = 0
    blocked_step_count: int = 0
    failed_step_count: int = 0
    runs: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def summarize_skill_outcomes(records: list[SkillOutcomeRecord | dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, dict[str, Any]] = {}
    for raw in records:
        record = raw if isinstance(raw, SkillOutcomeRecord) else SkillOutcomeRecord(**raw)
        bucket = grouped.setdefault(
            record.skill_name,
            {
                "skill_name": record.skill_name,
                "selected_count": 0,
                "verified_count": 0,
                "recipe_completed_count": 0,
                "recipe_blocked_count": 0,
                "recipe_failed_count": 0,
                "blocked_step_count": 0,
                "failed_step_count": 0,
                "runs": [],
            },
        )
        if record.selected:
            bucket["selected_count"] += 1
        if record.verified:
            bucket["verified_count"] += 1
        if record.recipe_status == "completed":
            bucket["recipe_completed_count"] += 1
        elif record.recipe_status == "blocked":
            bucket["recipe_blocked_count"] += 1
        elif record.recipe_status == "failed":
            bucket["recipe_failed_count"] += 1
        bucket["blocked_step_count"] += int(record.blocked_steps)
        bucket["failed_step_count"] += int(record.failed_steps)
        bucket["runs"].append(record.to_dict())
    return {
        "skills": [
            SkillOutcomeSummary(**value).to_dict()
            for value in sorted(grouped.values(), key=lambda item: item["skill_name"])
        ],
        "summary": {
            "skill_count": len(grouped),
            "selected_count": sum(item["selected_count"] for item in grouped.values()),
            "verified_count": sum(item["verified_count"] for item in grouped.values()),
        },
    }
