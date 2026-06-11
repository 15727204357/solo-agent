"""Scoring helpers for local eval results."""

from __future__ import annotations

from .case import EvalCase


def score_eval_case(
    case: EvalCase,
    changed_files: list[str],
    *,
    tests_failed: int,
    outcome_status: str,
) -> tuple[bool, float, list[str]]:
    notes: list[str] = []
    score = 1.0
    missing = sorted(set(case.expected_changed_files) - set(changed_files))
    forbidden = sorted(set(case.forbidden_changed_files).intersection(changed_files))
    unrelated = sorted(set(changed_files) - set(case.expected_changed_files)) if case.expected_changed_files else []
    if missing:
        score -= 0.35
        notes.append(f"missing expected changes: {', '.join(missing)}")
    if forbidden:
        score -= 0.4
        notes.append(f"modified forbidden files: {', '.join(forbidden)}")
    if unrelated:
        score -= 0.1
        notes.append(f"unrelated changes: {', '.join(unrelated)}")
    if tests_failed:
        score -= 0.35
        notes.append(f"{tests_failed} public test(s) failed")
    if outcome_status not in {"passed", "inconclusive"}:
        score -= 0.15
        notes.append(f"outcome status: {outcome_status}")
    score = max(0.0, round(score, 3))
    return score >= 0.75, score, notes
