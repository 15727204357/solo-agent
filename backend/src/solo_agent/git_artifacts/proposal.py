"""Git branch, commit, and PR text proposal generation."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any


def propose_git_artifacts(
    *,
    user_input: str,
    patch_proposal: Mapping[str, Any] | None = None,
    outcome_report: Mapping[str, Any] | None = None,
    evidence: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    patch = dict(patch_proposal or {})
    outcome = dict(outcome_report or {})
    title = _title(user_input, patch)
    branch = f"codex/{_slug(title)}"
    summary = str(patch.get("summary") or title)
    test_lines = _test_lines(evidence or [])
    risk_lines = [
        f"- {item.get('severity', 'risk')}: {item.get('description', '')}"
        for item in outcome.get("risks", [])
        if isinstance(item, Mapping)
    ] or ["- No high-confidence residual risks were identified by the outcome judge."]
    return {
        "branch_name": branch[:80],
        "commit_message": f"{_imperative(title)}\n\n{summary}".strip(),
        "pr_title": title,
        "pr_description": "\n".join(
            [
                "## Summary",
                f"- {summary}",
                "",
                "## Verification",
                *(test_lines or ["- Verification evidence is incomplete."]),
                "",
                "## Risk / Rollback",
                *risk_lines,
                "- Roll back by rejecting the patch proposal or reverting the generated commit.",
            ]
        ),
        "status": "proposal_only",
        "patch_id": patch.get("id"),
        "outcome_status": outcome.get("status"),
    }


def _title(user_input: str, patch: Mapping[str, Any]) -> str:
    base = str(patch.get("summary") or user_input or "Update code").strip()
    base = re.sub(r"\s+", " ", base)
    return base[:80].rstrip(" .") or "Update code"


def _slug(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text.lower()).strip("-")
    return slug[:60] or "update-code"


def _imperative(title: str) -> str:
    lowered = title.lower()
    if lowered.startswith(("add ", "fix ", "update ", "refactor ", "remove ")):
        return title[:72]
    return f"Update {title[0].lower()}{title[1:]}"[:72] if title else "Update code"


def _test_lines(evidence: Sequence[Mapping[str, Any]]) -> list[str]:
    lines: list[str] = []
    for item in evidence:
        command = str(item.get("command") or "")
        result = item.get("result")
        if not command:
            continue
        status = "passed" if _ok(result) else "failed"
        lines.append(f"- `{command}`: {status}")
    return lines[:10]


def _ok(result: Any) -> bool:
    if isinstance(result, Mapping) and result.get("ok") is False:
        return False
    payload = result.get("result") if isinstance(result, Mapping) and isinstance(result.get("result"), Mapping) else result
    if isinstance(payload, Mapping) and "returncode" in payload:
        return payload.get("returncode") == 0
    return True
