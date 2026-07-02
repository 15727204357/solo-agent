"""Rules-first outcome judge for closed-loop coding runs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .models import RequirementEvidence, RiskItem, TaskOutcomeReport


def judge_task_outcome(
    *,
    user_input: str,
    plan: str = "",
    impact_analysis: Mapping[str, Any] | None = None,
    sandbox_diff: str = "",
    patch_proposal: Mapping[str, Any] | None = None,
    test_report: Mapping[str, Any] | None = None,
    failure_reports: Sequence[Mapping[str, Any]] | None = None,
    command_evidence: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    failures = [dict(item) for item in (failure_reports or [])]
    evidence = [dict(item) for item in (command_evidence or [])]
    proposal = dict(patch_proposal or {})
    evidence.extend(_command_evidence_from_patch(proposal))
    diff = sandbox_diff or str(proposal.get("diff") or "")
    test_status = str((test_report or {}).get("status") or "")
    stop_gate = _stop_gate(proposal)
    stop_gate_status = str(stop_gate.get("status") or "")
    stop_gate_waived = stop_gate_status == "waived" and bool(str(stop_gate.get("reason") or "").strip())
    has_changes = bool(diff.strip()) or bool(proposal.get("edits"))
    passed_commands = _passed_command_count(evidence)
    failed_commands = _failed_command_count(evidence)
    missing: list[str] = []
    risks: list[RiskItem] = []
    next_actions: list[str] = []

    if not has_changes:
        missing.append("No code diff or patch proposal evidence was produced.")
        next_actions.append("Produce a concrete patch before requesting approval.")
    if failures:
        next_actions.append("Resolve classified verification failures before approval.")
    if failed_commands:
        risks.append(RiskItem("high", f"{failed_commands} verification command(s) failed.", "Use remediation feedback."))
    if stop_gate_status == "failed":
        risks.append(RiskItem("high", "Patch stop gate failed.", "Fix the patch or rerun the planned verification."))
    if not passed_commands and test_status not in {"passed", "accepted_with_failures"} and not stop_gate_waived:
        missing.append("No passing verification command evidence is available.")
        risks.append(RiskItem("medium", "Patch has limited verification evidence.", "Run targeted pytest/ruff where possible."))
    if has_changes and stop_gate_status == "missing":
        next_actions.append("Run the patch verification plan before approval.")
    if impact_analysis and not impact_analysis.get("related_tests"):
        risks.append(RiskItem("low", "Impact analysis did not identify related tests.", "Consider broader pytest coverage."))

    requirements = [
        RequirementEvidence(
            requirement=_compact_requirement(user_input or plan),
            status="covered" if has_changes and not failures else "needs_evidence",
            evidence=_requirement_evidence(diff, passed_commands, proposal),
        )
    ]
    blocked = any(str(item.get("kind")) in {"dependency_missing", "policy_blocked"} for item in failures)
    if blocked:
        status = "blocked"
        approval_ready = False
    elif failures or failed_commands or stop_gate_status == "failed":
        status = "needs_fix"
        approval_ready = False
    elif has_changes and (passed_commands or test_status == "passed" or stop_gate_status == "passed"):
        status = "passed"
        approval_ready = True
    elif has_changes and stop_gate_waived:
        status = "inconclusive"
        approval_ready = True
        next_actions.append("Approval can proceed under the recorded stop-gate waiver.")
    elif has_changes and not failures:
        status = "inconclusive"
        approval_ready = False
        next_actions.append("Approval should wait for passing verification evidence or an explicit waiver.")
    else:
        status = "inconclusive"
        approval_ready = False

    return TaskOutcomeReport(
        status=status,
        approval_ready=approval_ready,
        summary=_summary(status, has_changes, failures, passed_commands),
        requirements_covered=requirements,
        missing_evidence=missing,
        risks=risks,
        recommended_next_actions=next_actions,
        metadata={
            "has_changes": has_changes,
            "passed_command_count": passed_commands,
            "failed_command_count": failed_commands,
            "failure_count": len(failures),
            "test_status": test_status,
            "stop_gate_status": stop_gate_status,
            "stop_gate_approval_ready": bool(stop_gate.get("approval_ready", False)),
        },
    ).to_dict()


def _stop_gate(proposal: Mapping[str, Any]) -> dict[str, Any]:
    gate = proposal.get("stop_gate")
    return dict(gate) if isinstance(gate, Mapping) else {}


def _command_evidence_from_patch(proposal: Mapping[str, Any]) -> list[dict[str, Any]]:
    verification = proposal.get("verification")
    if not isinstance(verification, Mapping):
        return []
    commands = verification.get("commands")
    if not isinstance(commands, Sequence) or isinstance(commands, (str, bytes)):
        return []
    evidence: list[dict[str, Any]] = []
    for item in commands:
        if not isinstance(item, Mapping):
            continue
        evidence.append({"command": item.get("command"), "result": item.get("result")})
    return evidence


def _passed_command_count(evidence: Sequence[Mapping[str, Any]]) -> int:
    return sum(1 for item in evidence if _result_ok(item.get("result")))


def _failed_command_count(evidence: Sequence[Mapping[str, Any]]) -> int:
    return sum(1 for item in evidence if item.get("result") is not None and not _result_ok(item.get("result")))


def _result_ok(result: Any) -> bool:
    if isinstance(result, Mapping) and result.get("ok") is False:
        return False
    payload = result.get("result") if isinstance(result, Mapping) and isinstance(result.get("result"), Mapping) else result
    if isinstance(payload, Mapping) and "returncode" in payload:
        return payload.get("returncode") == 0
    return bool(payload is None or not isinstance(payload, Mapping) or payload.get("ok", True))


def _compact_requirement(text: str) -> str:
    normalized = " ".join(str(text or "").split())
    return normalized[:220] or "Complete the requested coding task."


def _requirement_evidence(diff: str, passed_commands: int, proposal: Mapping[str, Any]) -> list[str]:
    evidence: list[str] = []
    if diff.strip():
        evidence.append("code_diff_present")
    if proposal.get("id"):
        evidence.append(f"patch_proposal:{proposal.get('id')}")
    if passed_commands:
        evidence.append(f"passing_commands:{passed_commands}")
    return evidence


def _summary(status: str, has_changes: bool, failures: Sequence[Mapping[str, Any]], passed_commands: int) -> str:
    if status == "passed":
        return f"Task has code changes and {passed_commands} passing verification command(s)."
    if status == "needs_fix":
        return f"Task still has {len(failures)} classified failure(s) or failed verification command(s)."
    if status == "blocked":
        return "Task is blocked by environment, dependency, or policy constraints."
    if has_changes:
        return "Task has code changes but verification evidence is incomplete."
    return "Task does not yet have enough implementation evidence."
