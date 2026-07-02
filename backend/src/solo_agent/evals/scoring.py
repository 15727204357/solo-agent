"""Scoring helpers for local eval results."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from .case import EvalCase


def score_eval_case(
    case: EvalCase,
    changed_files: list[str],
    *,
    tests_failed: int,
    outcome_status: str,
    route_plan: Mapping[str, Any] | None = None,
    route_events: Iterable[Mapping[str, Any]] | None = None,
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
    route_passed, route_score, route_notes = score_route_case(case, route_plan or {}, route_events or [])
    if route_notes:
        score = min(score, route_score)
        notes.extend(route_notes)
    score = max(0.0, round(score, 3))
    return score >= 0.75 and route_passed, score, notes


def score_route_case(
    case: EvalCase,
    route_plan: Mapping[str, Any],
    route_events: Iterable[Mapping[str, Any]] = (),
) -> tuple[bool, float, list[str]]:
    notes: list[str] = []
    score = 1.0
    expected_intents = set(case.accepted_intents)
    if case.expected_intent:
        expected_intents.add(case.expected_intent)
    intent = str(route_plan.get("intent") or "")
    if expected_intents and intent not in expected_intents:
        score -= 0.3
        notes.append(f"route intent {intent or 'missing'} not in expected intents: {', '.join(sorted(expected_intents))}")

    scopes = _route_scopes(route_plan)
    missing_scopes = sorted(set(case.required_scopes) - scopes)
    if missing_scopes:
        score -= 0.2
        notes.append(f"route missing required scopes: {', '.join(missing_scopes)}")

    tools = _route_tools(route_plan)
    missing_tools = sorted(set(case.required_tools) - tools)
    forbidden_tools = sorted(set(case.forbidden_tools).intersection(tools))
    if missing_tools:
        score -= 0.25
        notes.append(f"route missing required tools: {', '.join(missing_tools)}")
    if forbidden_tools:
        score -= 0.35
        notes.append(f"route selected forbidden tools: {', '.join(forbidden_tools)}")

    if case.max_risk_level and _risk_rank(_route_risk(route_plan)) > _risk_rank(case.max_risk_level):
        score -= 0.25
        notes.append(f"route risk {_route_risk(route_plan)} exceeds max {case.max_risk_level}")

    if case.approval_required is not None and _route_approval(route_plan) != case.approval_required:
        score -= 0.2
        notes.append(f"route approval_required expected {case.approval_required}")

    if case.expected_reroute is not None and _has_reroute(route_events) != case.expected_reroute:
        score -= 0.2
        notes.append(f"route expected_reroute expected {case.expected_reroute}")

    score = max(0.0, round(score, 3))
    return score >= float(case.route_score_threshold), score, notes


def _route_scopes(route_plan: Mapping[str, Any]) -> set[str]:
    scopes = {str(scope) for scope in route_plan.get("searched_scopes") or []}
    context_plan = route_plan.get("context_plan")
    if isinstance(context_plan, Mapping):
        for item in context_plan.get("scopes") or []:
            if isinstance(item, Mapping) and item.get("scope"):
                scopes.add(str(item["scope"]))
    return scopes


def _route_tools(route_plan: Mapping[str, Any]) -> set[str]:
    tools = set()
    for call in route_plan.get("proposed_tool_calls") or []:
        if isinstance(call, Mapping) and call.get("name"):
            tools.add(str(call["name"]))
    tool_plan = route_plan.get("tool_plan")
    if isinstance(tool_plan, Mapping):
        for item in tool_plan.get("selected_tools") or []:
            if isinstance(item, Mapping) and item.get("name"):
                tools.add(str(item["name"]))
    return tools


def _route_risk(route_plan: Mapping[str, Any]) -> str:
    risk_summary = route_plan.get("risk_summary")
    if isinstance(risk_summary, Mapping):
        return str(risk_summary.get("max_risk_level") or "none")
    return "none"


def _route_approval(route_plan: Mapping[str, Any]) -> bool:
    approval_plan = route_plan.get("approval_plan")
    if isinstance(approval_plan, Mapping):
        return bool(approval_plan.get("requires_approval"))
    risk_summary = route_plan.get("risk_summary")
    if isinstance(risk_summary, Mapping):
        return bool(risk_summary.get("requires_approval"))
    return False


def _has_reroute(route_events: Iterable[Mapping[str, Any]]) -> bool:
    reroute_types = {"intent_route_reroute_requested", "intent_route_reroute_completed"}
    return any(str(event.get("type") or "") in reroute_types for event in route_events)


def _risk_rank(level: str) -> int:
    return {"none": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}.get(level.casefold(), 2)
