"""Deterministic remediation policies for classified failures."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def remediation_for_failures(failures: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not failures:
        return {"status": "none", "retryable": False, "developer_feedback": {}, "next_actions": []}
    retryable = any(bool(item.get("retryable", True)) for item in failures)
    actions: list[str] = []
    focus: list[dict[str, Any]] = []
    for failure in failures:
        kind = str(failure.get("kind") or "unknown_failure")
        focus.append(
            {
                "kind": kind,
                "file": failure.get("file") or "",
                "line": failure.get("line"),
                "failing_test": failure.get("failing_test") or "",
                "rule": failure.get("rule") or "",
                "summary": failure.get("summary") or "",
                "snippet": failure.get("snippet") or "",
            }
        )
        if kind == "test_failure":
            actions.append("Fix the failing test with the smallest code change, then rerun the targeted pytest command.")
        elif kind in {"lint_failure", "type_failure"}:
            actions.append("Fix the reported file/line without broad refactors, then rerun the quality command.")
        elif kind == "dependency_missing":
            actions.append(
                "Do not install dependencies automatically; produce a setup/approval note or adjust code to existing deps."
            )
        elif kind == "patch_conflict":
            actions.append("Refresh file hashes and rebuild the patch against the current workspace content.")
        elif kind == "policy_blocked":
            actions.append("Choose an allowed command/tool path or ask for explicit approval.")
        elif kind == "requirement_gap":
            actions.append("Return to the plan and implement the missing requirement evidence.")
        else:
            actions.append("Inspect the command output and produce a narrow corrective edit.")
    return {
        "status": "retryable" if retryable else "blocked",
        "retryable": retryable,
        "developer_feedback": {
            "reason": "verification_failed",
            "failures": focus,
            "instructions": actions[:6],
        },
        "next_actions": actions[:6],
    }
