"""Build a compact evidence timeline from an AgentState-like object."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def build_evidence_timeline(state: Any) -> list[dict[str, Any]]:
    snapshots = dict(getattr(state, "snapshots", {}) or {})
    timeline: list[dict[str, Any]] = []
    _append(timeline, "user_request", "User request captured", {"text": getattr(state, "user_input", "")})
    if getattr(state, "plan", ""):
        _append(timeline, "plan", "Plan generated", {"text": getattr(state, "plan", "")[:4000]})
    code_map_summary = getattr(state, "code_map_summary", None)
    if code_map_summary:
        _append(timeline, "code_index", "Code intelligence summary built", code_map_summary)
    impact_analysis = getattr(state, "impact_analysis", None)
    if impact_analysis:
        _append(timeline, "impact_analysis", "Impact analysis completed", impact_analysis)
    for checkpoint in snapshots.get("sandbox_checkpoints", []) or []:
        if isinstance(checkpoint, Mapping):
            _append(timeline, "sandbox_checkpoint", "Sandbox checkpoint created", checkpoint)
    sandbox = getattr(state, "sandbox_artifacts", {}) or {}
    if sandbox:
        _append(timeline, "sandbox_artifacts", "Sandbox artifacts captured", sandbox)
    failure_reports = list(getattr(state, "failure_reports", []) or snapshots.get("failure_reports") or [])
    for failure in failure_reports:
        if isinstance(failure, Mapping):
            _append(timeline, "failure_report", str(failure.get("summary") or "Failure classified"), failure)
    outcome = getattr(state, "outcome_report", None) or snapshots.get("outcome_report")
    if isinstance(outcome, Mapping) and outcome:
        _append(timeline, "outcome_report", str(outcome.get("summary") or "Outcome judged"), outcome)
    patch = getattr(state, "patch_proposal", None) or snapshots.get("patch_proposal")
    if isinstance(patch, Mapping) and patch:
        _append(timeline, "patch_proposal", "Patch proposal created", patch)
    git_artifact = getattr(state, "git_artifact_proposal", None) or snapshots.get("git_artifact_proposal")
    if isinstance(git_artifact, Mapping) and git_artifact:
        _append(timeline, "git_artifact_proposal", "Git artifact proposal created", git_artifact)
    return timeline


def _append(timeline: list[dict[str, Any]], kind: str, title: str, data: Any) -> None:
    timeline.append({"order": len(timeline) + 1, "kind": kind, "title": title, "data": _bounded(data)})


def _bounded(value: Any) -> Any:
    if isinstance(value, str):
        return value[:6000]
    if isinstance(value, Mapping):
        return {str(key): _bounded(item) for key, item in list(value.items())[:80]}
    if isinstance(value, list):
        return [_bounded(item) for item in value[:80]]
    return value
