from __future__ import annotations

import asyncio
import inspect
import json
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from solo_agent.providers import ChatMessage

ROUTE_PLAN_SCHEMA_VERSION = "2"


class IntentKind:
    ANSWER_QUESTION = "answer_question"
    INSPECT_CODE = "inspect_code"
    MODIFY_CODE = "modify_code"
    DEBUG_TEST_FAILURE = "debug_test_failure"
    RUN_QUALITY_CHECKS = "run_quality_checks"
    REVIEW_DIFF = "review_diff"
    MANAGE_SKILL = "manage_skill"
    PLAN_REFACTOR = "plan_refactor"
    UNKNOWN = "unknown"


class RouteScope:
    WORKSPACE = "workspace"
    FILES = "files"
    SEARCH = "search"
    CODE_INDEX = "code_index"
    IMPACT = "impact"
    TESTS = "tests"
    QUALITY = "quality"
    GIT = "git"
    SKILLS = "skills"
    RECIPES = "recipes"
    SUBAGENTS = "subagents"


@dataclass(frozen=True)
class RouteCandidate:
    name: str
    category: str = "context"
    capability: str = "context"
    arguments: dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    evidence: list[str] = field(default_factory=list)
    risk_level: str = "low"
    read_only: bool = True
    requires_approval: bool = False
    confidence: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ContextScopePlan:
    scope: str
    reason: str
    query: str = ""
    expected_evidence: list[str] = field(default_factory=list)
    fallback_if_empty: str = ""
    budget: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class IntentRoutePlan:
    intent: str
    confidence: float
    matched_terms: list[str]
    searched_scopes: list[str]
    tool_candidates: list[dict[str, Any]]
    proposed_tool_calls: list[dict[str, Any]]
    skill_candidates: list[dict[str, Any]]
    recipe_candidates: list[dict[str, Any]]
    evidence: list[dict[str, Any]]
    risk_summary: dict[str, Any]
    next_actions: list[str]
    route_plan_schema_version: str = ROUTE_PLAN_SCHEMA_VERSION
    route_id: str = ""
    route_epoch: int = 0
    intent_alternatives: list[dict[str, Any]] = field(default_factory=list)
    constraints: dict[str, Any] = field(default_factory=dict)
    context_plan: dict[str, Any] = field(default_factory=dict)
    tool_plan: dict[str, Any] = field(default_factory=dict)
    skill_plan: dict[str, Any] = field(default_factory=dict)
    recipe_plan: dict[str, Any] = field(default_factory=dict)
    approval_plan: dict[str, Any] = field(default_factory=dict)
    verification_plan: dict[str, Any] = field(default_factory=dict)
    decision_trace: list[dict[str, Any]] = field(default_factory=list)
    reroute_triggers: list[dict[str, Any]] = field(default_factory=list)
    model_advisor: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


async def plan_intent_route(
    tool_registry: Any | None,
    state: Any,
    settings: Any,
    *,
    provider: Any | None = None,
    reroute_reason: str = "",
) -> IntentRoutePlan:
    """Plan explainable tool routing for the current turn without executing tools."""

    if tool_registry is None:
        return _empty_plan(state, "No tool registry is configured.")

    tool_specs = await _list_tool_metadata(tool_registry)
    available = set(tool_specs)
    max_calls = int(_setting(settings, "max_tool_calls", 3))
    route_epoch = int(getattr(state, "route_epoch", 0) or 0)
    task = str(getattr(state, "user_input", "") or "").strip()
    plan = str(getattr(state, "plan", "") or "")
    task_lc = task.casefold()
    plan_lc = plan.casefold()
    intent, intent_confidence, matched_terms = _classify_intent(task, plan)
    calls: list[dict[str, Any]] = []
    candidates: list[RouteCandidate] = []
    evidence: list[dict[str, Any]] = [
        {
            "kind": "intent_classifier",
            "intent": intent,
            "confidence": intent_confidence,
            "matched_terms": matched_terms,
            "source": "user_input+plan",
        }
    ]

    def add_call(
        call: dict[str, Any],
        *,
        reason: str,
        evidence_terms: list[str] | None = None,
        confidence: float = 0.65,
    ) -> None:
        if call.get("name") not in available:
            return
        calls.append(call)
        candidates.append(
            _candidate_from_call(
                call,
                tool_specs.get(str(call.get("name")), {}),
                reason=reason,
                evidence=evidence_terms or matched_terms,
                confidence=confidence,
            )
        )

    if (
        "select_relevant_skills" in available
        and not getattr(state, "selected_skills", [])
        and not getattr(state, "snapshots", {}).get("skill_selection_attempted")
    ):
        add_call(
            {
                "name": "select_relevant_skills",
                "arguments": {"task": task, "plan": plan, "max_skills": 3},
                "category": "skill",
            },
            reason="Route compact Skill metadata before loading full Skill content.",
            evidence_terms=["skill_metadata"],
            confidence=0.72,
        )
    elif "list_skills" in available and _mentions_skill(task_lc, plan_lc):
        add_call(
            {"name": "list_skills", "arguments": {}, "category": "skill"},
            reason="User or plan mentions Skill/SOP discovery.",
            evidence_terms=["skill"],
            confidence=0.62,
        )

    if _mentions_code_task(task, plan):
        if "code_map" in available and not getattr(state, "code_map_summary", {}):
            add_call(
                {
                    "name": "code_map",
                    "arguments": {"path": ".", "max_files": int(_setting(settings, "context_file_limit", 80))},
                    "category": "code_intelligence",
                },
                reason="Code task needs a repository map before choosing implementation context.",
                evidence_terms=_matched_markers(task, plan, _CODE_MARKERS),
                confidence=0.78,
            )
        if "analyze_impact" in available and not getattr(state, "impact_analysis", {}):
            path_hint = _extract_path_hint(task)
            add_call(
                {
                    "name": "analyze_impact",
                    "arguments": {
                        "paths": [path_hint] if path_hint else [],
                        "symbols": _extract_symbol_hints(task),
                        "include_tests": True,
                    },
                    "category": "code_intelligence",
                },
                reason="Code task should identify impacted files and tests before action.",
                evidence_terms=[term for term in [path_hint, *_extract_symbol_hints(task)] if term],
                confidence=0.76,
            )

    for call in _subagent_calls(state, settings, available, max_calls=max_calls, current_call_count=len(calls)):
        add_call(
            call,
            reason="Parallelism gate selected scoped read-only subagent context collection.",
            evidence_terms=["parallelism_decision"],
            confidence=0.7,
        )

    requested_skill_names = _requested_skill_views(state)
    if "skill_view" in available:
        for skill_name in requested_skill_names[: max(0, max_calls - len(calls))]:
            add_call(
                {
                    "name": "skill_view",
                    "arguments": {"name": skill_name},
                    "category": "skill",
                },
                reason="Explicit Skill request needs full Skill content after compact selection.",
                evidence_terms=[skill_name],
                confidence=0.86,
            )
    if "skill_recipe_list" in available:
        for skill_name in requested_skill_names[: max(0, max_calls - len(calls))]:
            add_call(
                {
                    "name": "skill_recipe_list",
                    "arguments": {"skill_name": skill_name, "query": task, "max_entries": 5},
                    "category": "skill",
                },
                reason="Explicit Skill request should expose compact recipe options and policy boundaries.",
                evidence_terms=[skill_name],
                confidence=0.82,
            )

    if "workspace_snapshot" in available:
        add_call(
            {
                "name": "workspace_snapshot",
                "arguments": {
                    "path": ".",
                    "max_entries": int(_setting(settings, "context_file_limit", 80)),
                },
                "category": "context",
            },
            reason="Baseline workspace context is needed for transparent agent routing.",
            evidence_terms=["workspace"],
            confidence=0.66,
        )
    elif "list_files" in available:
        add_call(
            {
                "name": "list_files",
                "arguments": {
                    "path": ".",
                    "max_entries": int(_setting(settings, "context_file_limit", 80)),
                },
                "category": "context",
            },
            reason="Workspace file listing is the fallback baseline context.",
            evidence_terms=["workspace"],
            confidence=0.6,
        )

    path_hint = _extract_path_hint(task)
    if path_hint and "read_file" in available:
        add_call(
            {
                "name": "read_file",
                "arguments": {"path": path_hint},
                "category": "context",
            },
            reason="User input contains a concrete path, so read that file directly.",
            evidence_terms=[path_hint],
            confidence=0.82,
        )
    elif "search_text" in available and task:
        add_call(
            {
                "name": "search_text",
                "arguments": {
                    "query": task[:200],
                    "max_matches": int(_setting(settings, "context_search_limit", 20)),
                },
                "category": "context",
            },
            reason="No concrete path was found; text search is the safest context fallback.",
            evidence_terms=_top_terms(task),
            confidence=0.62,
        )

    if _mentions_pytest(task_lc, plan_lc):
        if "run_command" in available:
            add_call(
                {
                    "name": "run_command",
                    "arguments": {
                        "command": "python",
                        "args": ["-m", "pytest", "-q"],
                        "purpose": "Run the test suite requested by the user.",
                    },
                    "category": "quality",
                },
                reason="User or plan requested pytest/test verification.",
                evidence_terms=_matched_markers(task, plan, _TEST_MARKERS),
                confidence=0.8,
            )
        elif "run_pytest" in available:
            add_call(
                {"name": "run_pytest", "arguments": {}, "category": "quality"},
                reason="User or plan requested pytest/test verification.",
                evidence_terms=_matched_markers(task, plan, _TEST_MARKERS),
                confidence=0.8,
            )
    if _mentions_ruff(task_lc, plan_lc):
        if "run_command" in available:
            add_call(
                {
                    "name": "run_command",
                    "arguments": {
                        "command": "ruff",
                        "args": ["check", "."],
                        "purpose": "Run lint checks requested by the user.",
                    },
                    "category": "quality",
                },
                reason="User or plan requested Ruff/lint quality checks.",
                evidence_terms=_matched_markers(task, plan, _RUFF_MARKERS),
                confidence=0.78,
            )
        elif "run_ruff_check" in available:
            add_call(
                {"name": "run_ruff_check", "arguments": {}, "category": "quality"},
                reason="User or plan requested Ruff/lint quality checks.",
                evidence_terms=_matched_markers(task, plan, _RUFF_MARKERS),
                confidence=0.78,
            )
    if _mentions_format(task_lc, plan_lc):
        if "run_command" in available:
            add_call(
                {
                    "name": "run_command",
                    "arguments": {
                        "command": "ruff",
                        "args": ["format", "--check", "."],
                        "purpose": "Check formatting requested by the user.",
                    },
                    "category": "quality",
                },
                reason="User or plan requested formatting verification.",
                evidence_terms=_matched_markers(task, plan, _FORMAT_MARKERS),
                confidence=0.76,
            )
        elif "run_ruff_format_check" in available:
            add_call(
                {"name": "run_ruff_format_check", "arguments": {}, "category": "quality"},
                reason="User or plan requested formatting verification.",
                evidence_terms=_matched_markers(task, plan, _FORMAT_MARKERS),
                confidence=0.76,
            )

    deduped_calls = _dedupe_tool_calls(calls)[:max_calls]
    deduped_candidates = _align_candidates_to_calls(candidates, deduped_calls)
    scopes = _derive_scopes(deduped_calls, state=state)
    skill_candidates = _skill_candidates(state, task, intent)
    recipe_candidates = _recipe_candidates(state, task)
    risk_summary = _risk_summary(intent=intent, candidates=deduped_candidates, proposed_tool_calls=deduped_calls)
    next_actions = _next_actions(intent, deduped_calls, skill_candidates=skill_candidates, recipe_candidates=recipe_candidates)
    context_plan = _context_plan(deduped_calls, state=state, settings=settings, task=task)
    decision_trace = _decision_trace(
        selected_candidates=deduped_candidates,
        selected_calls=deduped_calls,
        tool_specs=tool_specs,
        available_tools=available,
    )
    constraints = _route_constraints(settings, state)
    approval_plan = _approval_plan(risk_summary, state=state, intent=intent)
    verification_plan = _verification_plan(intent, deduped_calls)
    route_id = _route_id(state, route_epoch)
    reroute_triggers = _reroute_triggers(state, route_epoch=route_epoch, settings=settings, reason=reroute_reason)

    evidence.append(
        {
            "kind": "route_candidates",
            "candidate_count": len(deduped_candidates),
            "selected_count": len(deduped_calls),
            "max_tool_calls": max_calls,
            "source": "tool_registry",
        }
    )
    route_plan = IntentRoutePlan(
        intent=intent,
        confidence=intent_confidence,
        matched_terms=matched_terms,
        searched_scopes=scopes,
        tool_candidates=[candidate.to_dict() for candidate in deduped_candidates],
        proposed_tool_calls=deduped_calls,
        skill_candidates=skill_candidates,
        recipe_candidates=recipe_candidates,
        evidence=evidence,
        risk_summary=risk_summary,
        next_actions=next_actions,
        route_id=route_id,
        route_epoch=route_epoch,
        intent_alternatives=_intent_alternatives(task, plan, primary_intent=intent),
        constraints=constraints,
        context_plan=context_plan,
        tool_plan=_tool_plan(
            selected_candidates=deduped_candidates,
            selected_calls=deduped_calls,
            tool_specs=tool_specs,
            available_tools=available,
        ),
        skill_plan=_skill_plan(skill_candidates, state=state),
        recipe_plan=_recipe_plan(recipe_candidates, state=state),
        approval_plan=approval_plan,
        verification_plan=verification_plan,
        decision_trace=decision_trace,
        reroute_triggers=reroute_triggers,
    )
    return await _apply_model_advisor(route_plan, provider=provider, state=state, settings=settings, tool_specs=tool_specs)


def _empty_plan(state: Any, reason: str) -> IntentRoutePlan:
    intent, confidence, matched_terms = _classify_intent(str(getattr(state, "user_input", "") or ""), "")
    route_epoch = int(getattr(state, "route_epoch", 0) or 0)
    return IntentRoutePlan(
        intent=intent,
        confidence=confidence,
        matched_terms=matched_terms,
        searched_scopes=[],
        tool_candidates=[],
        proposed_tool_calls=[],
        skill_candidates=[],
        recipe_candidates=[],
        evidence=[{"kind": "route_unavailable", "reason": reason}],
        risk_summary={"max_risk_level": "none", "requires_approval": False, "boundary": reason},
        next_actions=["answer_from_existing_context"],
        route_id=_route_id(state, route_epoch),
        route_epoch=route_epoch,
        intent_alternatives=[],
        constraints={},
        context_plan={"scopes": []},
        tool_plan={"selected_tools": [], "rejected_tools": [], "all_candidates": []},
        skill_plan={"candidates": [], "alternatives": []},
        recipe_plan={"candidates": []},
        approval_plan={"requires_approval": False, "approval_boundary": reason},
        verification_plan={"required": False, "commands": []},
        decision_trace=[{"kind": "route_unavailable", "reason": reason}],
        reroute_triggers=[],
    )


async def _list_tool_metadata(tool_registry: Any) -> dict[str, dict[str, Any]]:
    for method_name in ("list_tools", "list_all_tools", "tools"):
        method = getattr(tool_registry, method_name, None)
        if method is None:
            continue
        if method_name == "list_tools" and callable(method):
            try:
                tools = await _maybe_await(method(visibility="model"))
            except TypeError:
                tools = await _maybe_await(method())
        elif method_name == "list_all_tools" and callable(method):
            tools = await _maybe_await(method())
        else:
            tools = await _maybe_await(method() if callable(method) else method)
        return {
            _tool_name(tool): _normalize_tool_spec(tool)
            for tool in tools or []
            if _tool_name(tool)
        }
    return {
        name: {
            "name": name,
            "category": "context",
            "capability": "context",
            "read_only": True,
            "risk_level": "low",
            "requires_approval": False,
        }
        for name in ("list_files", "read_file", "search_text")
    }


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def _normalize_tool_spec(tool: Any) -> dict[str, Any]:
    if isinstance(tool, str):
        return {"name": tool, "category": "context", "capability": "context", "read_only": True}
    if isinstance(tool, Mapping):
        return {
            "name": str(tool.get("name", "")),
            "description": str(tool.get("description", "")),
            "category": str(tool.get("category") or "context"),
            "capability": str(tool.get("capability") or tool.get("category") or "context"),
            "read_only": bool(tool.get("read_only", True)),
            "risk_level": str(tool.get("risk_level") or "low"),
            "requires_approval": bool(tool.get("requires_approval", False)),
        }
    return {
        "name": str(getattr(tool, "name", tool)),
        "category": str(getattr(tool, "category", "context")),
        "capability": str(getattr(tool, "capability", "context")),
        "read_only": bool(getattr(tool, "read_only", True)),
        "risk_level": str(getattr(tool, "risk_level", "low")),
        "requires_approval": bool(getattr(tool, "requires_approval", False)),
    }


def _tool_name(tool: Any) -> str:
    if isinstance(tool, str):
        return tool
    if isinstance(tool, Mapping):
        return str(tool.get("name", ""))
    return str(getattr(tool, "name", tool))


def _candidate_from_call(
    call: Mapping[str, Any],
    spec: Mapping[str, Any],
    *,
    reason: str,
    evidence: list[str],
    confidence: float,
) -> RouteCandidate:
    category = str(call.get("category") or spec.get("category") or "context")
    return RouteCandidate(
        name=str(call.get("name") or spec.get("name") or ""),
        category=category,
        capability=str(spec.get("capability") or category),
        arguments=dict(call.get("arguments") or {}),
        reason=reason,
        evidence=[str(item) for item in evidence if str(item).strip()][:8],
        risk_level=str(spec.get("risk_level") or "low"),
        read_only=bool(spec.get("read_only", True)),
        requires_approval=bool(spec.get("requires_approval", False)),
        confidence=round(confidence, 2),
    )


def _classify_intent(task: str, plan: str) -> tuple[str, float, list[str]]:
    text = f"{task}\n{plan}".casefold()
    matched: list[str] = []

    def has(markers: tuple[str, ...]) -> bool:
        found = _matched_markers(task, plan, markers)
        matched.extend(term for term in found if term not in matched)
        return bool(found)

    skill = has(_SKILL_MARKERS) or bool(_explicit_skill_requests(task))
    tests = has(_TEST_MARKERS)
    failing = has(_FAILURE_MARKERS)
    quality = has(_RUFF_MARKERS) or has(_FORMAT_MARKERS) or tests
    edit = has(_EDIT_MARKERS)
    refactor = has(_REFACTOR_MARKERS)
    inspect_code = _mentions_code_task(task, plan)
    review = has(_REVIEW_MARKERS)
    question = has(_QUESTION_MARKERS)

    if skill:
        return IntentKind.MANAGE_SKILL, _confidence(matched, 0.75), matched
    if tests and failing:
        return IntentKind.DEBUG_TEST_FAILURE, _confidence(matched, 0.82), matched
    if review:
        return IntentKind.REVIEW_DIFF, _confidence(matched, 0.72), matched
    if refactor and not edit:
        return IntentKind.PLAN_REFACTOR, _confidence(matched, 0.68), matched
    if edit:
        return IntentKind.MODIFY_CODE, _confidence(matched, 0.78), matched
    if quality:
        return IntentKind.RUN_QUALITY_CHECKS, _confidence(matched, 0.7), matched
    if inspect_code:
        terms = _matched_markers(task, plan, _CODE_MARKERS)
        return IntentKind.INSPECT_CODE, _confidence([*matched, *terms], 0.66), [*matched, *terms]
    if question or text.strip():
        return IntentKind.ANSWER_QUESTION, _confidence(matched, 0.55), matched
    return IntentKind.UNKNOWN, 0.25, matched


def _confidence(matched: list[str], base: float) -> float:
    return round(min(0.95, base + min(len(set(matched)), 5) * 0.03), 2)


def _mentions_skill(task_lc: str, plan_lc: str) -> bool:
    text = f"{task_lc}\n{plan_lc}"
    return any(marker in text for marker in _SKILL_MARKERS)


def _mentions_code_task(task: str, plan: str = "") -> bool:
    text = f"{task}\n{plan}".casefold()
    return any(marker in text for marker in _CODE_MARKERS)


def _mentions_pytest(task_lc: str, plan_lc: str) -> bool:
    text = f"{task_lc}\n{plan_lc}"
    return any(marker in text for marker in _TEST_MARKERS)


def _mentions_ruff(task_lc: str, plan_lc: str) -> bool:
    text = f"{task_lc}\n{plan_lc}"
    return any(marker in text for marker in _RUFF_MARKERS)


def _mentions_format(task_lc: str, plan_lc: str) -> bool:
    text = f"{task_lc}\n{plan_lc}"
    return any(marker in text for marker in _FORMAT_MARKERS)


def _extract_symbol_hints(text: str) -> list[str]:
    symbols: list[str] = []
    seen: set[str] = set()
    for token in re.findall(r"\b[A-Za-z_][A-Za-z0-9_]{3,}\b", text or ""):
        if token in {"test", "tests", "pytest", "ruff", "code", "file", "function", "class"}:
            continue
        if token not in seen and (token[:1].isupper() or "_" in token):
            symbols.append(token)
            seen.add(token)
        if len(symbols) >= 8:
            break
    return symbols


def _extract_path_hint(text: str) -> str | None:
    for token in text.replace("`", " ").split():
        normalized = token.strip(".,:;()[]{}'\"")
        if "/" in normalized or "\\" in normalized or normalized.endswith((".py", ".md", ".toml", ".ts", ".tsx")):
            return normalized
    return None


def _explicit_skill_requests(text: str) -> list[str]:
    requested: list[str] = []
    seen: set[str] = set()
    for match in re.finditer(r"(?:^|\s)/skill\s+([A-Za-z0-9_-]+)", text or "", re.IGNORECASE):
        name = match.group(1).strip()
        key = name.casefold()
        if name and key not in seen:
            requested.append(name)
            seen.add(key)
    return requested


def _requested_skill_views(state: Any) -> list[str]:
    indexed = [skill for skill in getattr(state, "selected_skills", []) if isinstance(skill, Mapping)]
    explicit_requests = _explicit_skill_requests(str(getattr(state, "user_input", "") or ""))
    if not indexed:
        return explicit_requests
    explicit_slugs = {
        match.group(1).casefold()
        for match in re.finditer(r"/([A-Za-z0-9_-]+)", str(getattr(state, "user_input", "") or ""))
    }
    plan_lc = str(getattr(state, "plan", "") or "").casefold()
    plan_mentions_skills = "skill" in plan_lc or "sop" in plan_lc
    requested: list[str] = []
    seen: set[str] = set()
    for skill in indexed:
        name = str(skill.get("name") or "").strip()
        path = str(skill.get("path") or "")
        slug = Path(path).parent.name if path else name
        candidates = {name.casefold(), slug.casefold()}
        mentioned_in_plan = plan_mentions_skills and any(item and item in plan_lc for item in candidates)
        if explicit_slugs.intersection(candidates) or mentioned_in_plan:
            key = name or slug
            if key and key not in seen:
                requested.append(key)
                seen.add(key)
    for name in explicit_requests:
        key = name.casefold()
        if key not in seen:
            requested.append(name)
            seen.add(key)
    return requested


def _subagent_calls(
    state: Any,
    settings: Any,
    available: set[str],
    *,
    max_calls: int,
    current_call_count: int,
) -> list[dict[str, Any]]:
    decision = getattr(state, "snapshots", {}).get("parallelism_decision") or getattr(state, "parallelism_decision", {}) or {}
    subagent_enabled = bool(decision.get("subagent_enabled", False))
    subagent_policy = str(decision.get("subagent_policy", _setting(settings, "subagent_policy", "off")))
    suitable_for_task = bool(decision.get("suitable", decision.get("allowed", False)))
    strategy = str(decision.get("strategy", "serial"))
    candidates = decision.get("candidates") or decision.get("tasks") or []
    if not (
        "task" in available
        and subagent_enabled
        and subagent_policy == "auto"
        and suitable_for_task
        and strategy == "parallel"
        and len(candidates) >= 2
    ):
        return []
    calls: list[dict[str, Any]] = []
    remaining = max(0, max_calls - current_call_count)
    for candidate in candidates[:remaining]:
        if not isinstance(candidate, Mapping):
            continue
        description = str(candidate.get("title") or candidate.get("description") or candidate.get("id") or "Subtask")
        read_paths = [
            str(path)
            for path in [
                *(candidate.get("read_paths") or []),
                *(candidate.get("write_paths") or []),
            ]
            if str(path).strip()
        ]
        prompt_parts = [
            f"Subtask: {description}",
            f"Parent user task: {getattr(state, 'user_input', '')}",
            f"Parent plan: {getattr(state, 'plan', '') or '(no plan)'}",
            f"Candidate metadata: {candidate}",
            "Return concise structured findings and evidence for the main agent to synthesize. Do not edit files.",
        ]
        calls.append(
            {
                "name": "task",
                "arguments": {
                    "description": description,
                    "prompt": "\n\n".join(prompt_parts),
                    "subagent_type": str(candidate.get("subagent_type") or "general-purpose"),
                    "read_paths": read_paths,
                    "allowed_tools": ["workspace_snapshot", "list_files", "read_file", "search_text"],
                },
                "category": "subagent",
            }
        )
    return calls


def _dedupe_tool_calls(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str]] = set()
    deduped: list[dict[str, Any]] = []
    for call in calls:
        key = (
            str(call.get("name", "")),
            json.dumps(call.get("arguments") or {}, ensure_ascii=False, sort_keys=True, default=str),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(call)
    return deduped


def _align_candidates_to_calls(candidates: list[RouteCandidate], calls: list[dict[str, Any]]) -> list[RouteCandidate]:
    remaining = list(candidates)
    aligned: list[RouteCandidate] = []
    for call in calls:
        key = (
            str(call.get("name", "")),
            json.dumps(call.get("arguments") or {}, ensure_ascii=False, sort_keys=True, default=str),
        )
        for index, candidate in enumerate(remaining):
            candidate_key = (
                candidate.name,
                json.dumps(candidate.arguments, ensure_ascii=False, sort_keys=True, default=str),
            )
            if candidate_key == key:
                aligned.append(candidate)
                remaining.pop(index)
                break
    return aligned


def _derive_scopes(calls: list[dict[str, Any]], *, state: Any) -> list[str]:
    scopes: list[str] = []

    def add(scope: str) -> None:
        if scope not in scopes:
            scopes.append(scope)

    if calls:
        add(RouteScope.WORKSPACE)
    for call in calls:
        name = str(call.get("name") or "")
        category = str(call.get("category") or "")
        if name in {"workspace_snapshot", "list_files", "read_file", "find_files"}:
            add(RouteScope.FILES)
        if name in {"search_text", "search_code", "semantic_code_search"}:
            add(RouteScope.SEARCH)
        if name in {"code_map", "symbol_search", "symbol_definition", "find_references", "call_graph"}:
            add(RouteScope.CODE_INDEX)
        if name in {"analyze_impact", "test_relevance"}:
            add(RouteScope.IMPACT)
        if name in {"run_pytest", "targeted_pytest"} or "pytest" in json.dumps(call.get("arguments") or {}):
            add(RouteScope.TESTS)
        if category == "quality" or name.startswith("run_ruff"):
            add(RouteScope.QUALITY)
        if name.startswith("git_"):
            add(RouteScope.GIT)
        if category == "skill" or name.startswith("skill") or name.endswith("_skills"):
            add(RouteScope.SKILLS)
        if "recipe" in name:
            add(RouteScope.RECIPES)
        if category == "subagent" or name == "task":
            add(RouteScope.SUBAGENTS)
    if getattr(state, "selected_skills", []):
        add(RouteScope.SKILLS)
    if getattr(state, "selected_recipes", []):
        add(RouteScope.RECIPES)
    return scopes


def _skill_candidates(state: Any, task: str, intent: str) -> list[dict[str, Any]]:
    skills = [skill for skill in getattr(state, "selected_skills", []) if isinstance(skill, Mapping)]
    terms = set(_top_terms(task))
    candidates: list[dict[str, Any]] = []
    for skill in skills[:5]:
        haystack = _skill_search_text(skill)
        matched_terms = sorted(term for term in terms if term.casefold() in haystack)
        score = float(skill.get("score") or len(matched_terms) or 1)
        candidates.append(
            {
                "name": skill.get("name"),
                "description": skill.get("description"),
                "category": skill.get("category"),
                "path": skill.get("path"),
                "matched_terms": matched_terms,
                "matched_intent": intent,
                "source_scope": "workspace_skills",
                "confidence": round(min(0.95, 0.55 + score * 0.08), 2),
                "risk_level": "low",
                "required_tools": skill.get("required_tools", []),
                "recommendation_reason": "Selected from compact Skill metadata before loading full content.",
            }
        )
    return candidates


def _recipe_candidates(state: Any, task: str) -> list[dict[str, Any]]:
    recipes = [recipe for recipe in getattr(state, "selected_recipes", []) if isinstance(recipe, Mapping)]
    terms = set(_top_terms(task))
    candidates: list[dict[str, Any]] = []
    for recipe in recipes[:5]:
        haystack = " ".join(
            [
                str(recipe.get("id", "")),
                str(recipe.get("name", "")),
                str(recipe.get("description", "")),
                " ".join(str(item) for item in recipe.get("when", [])),
            ]
        ).casefold()
        matched_terms = sorted(term for term in terms if term.casefold() in haystack)
        manual_count = int(recipe.get("manual_step_count") or 0)
        auto_count = int(recipe.get("auto_step_count") or 0)
        candidates.append(
            {
                "id": recipe.get("id"),
                "name": recipe.get("name"),
                "skill_name": recipe.get("skill_name"),
                "description": recipe.get("description"),
                "matched_terms": matched_terms,
                "confidence": round(0.6 + min(len(matched_terms), 3) * 0.08, 2),
                "auto_step_count": auto_count,
                "manual_step_count": manual_count,
                "blocked_or_manual_reason": "manual_steps_present" if manual_count else "auto_boundary_allows_preview",
                "run_policy": recipe.get("run_policy"),
            }
        )
    return candidates


def _risk_summary(
    *,
    intent: str,
    candidates: list[RouteCandidate],
    proposed_tool_calls: list[dict[str, Any]],
) -> dict[str, Any]:
    risk_rank = {"none": 0, "low": 1, "medium": 2, "medium-safe": 2, "high": 3}
    max_risk = "none"
    approval = False
    non_read_only: list[str] = []
    for candidate in candidates:
        if risk_rank.get(candidate.risk_level, 1) > risk_rank.get(max_risk, 0):
            max_risk = candidate.risk_level
        approval = approval or candidate.requires_approval
        if not candidate.read_only:
            non_read_only.append(candidate.name)
    boundary = "route_is_context_and_verification_only"
    if intent == IntentKind.MODIFY_CODE:
        boundary = "modify intent detected; edits still require verified editing and patch approval"
    elif intent == IntentKind.MANAGE_SKILL:
        boundary = "skill route uses progressive disclosure; skill changes require proposal approval"
    elif not proposed_tool_calls:
        boundary = "no tools selected; answer from existing context or ask for clarification"
    return {
        "max_risk_level": max_risk,
        "requires_approval": approval,
        "non_read_only_tools": non_read_only,
        "proposed_call_count": len(proposed_tool_calls),
        "boundary": boundary,
    }


def _next_actions(
    intent: str,
    calls: list[dict[str, Any]],
    *,
    skill_candidates: list[dict[str, Any]],
    recipe_candidates: list[dict[str, Any]],
) -> list[str]:
    actions: list[str] = []
    if calls:
        actions.append("execute_selected_context_and_quality_tools")
    if intent == IntentKind.MODIFY_CODE:
        actions.append("prepare_verified_patch_after_context")
    if intent == IntentKind.DEBUG_TEST_FAILURE:
        actions.append("diagnose_failure_from_test_and_code_evidence")
    if skill_candidates:
        actions.append("load_full_skill_only_if_selected_or_explicit")
    if recipe_candidates:
        actions.append("preview_recipe_before_any_auto_run")
    if not actions:
        actions.append("answer_from_existing_context")
    return actions


def reroute_triggers_from_state(state: Any, settings: Any, *, reason: str = "") -> list[dict[str, Any]]:
    route_epoch = int(_state_get(state, "route_epoch", 0) or 0)
    return _reroute_triggers(state, route_epoch=route_epoch, settings=settings, reason=reason)


def _route_id(state: Any, route_epoch: int) -> str:
    session_id = str(_state_get(state, "session_id", "session") or "session")
    run_id = str(_state_get(state, "run_id", "run") or "run")
    return f"{session_id}:{run_id}:route:{route_epoch}"


def _route_constraints(settings: Any, state: Any) -> dict[str, Any]:
    return {
        "run_mode": str(_state_get(state, "run_mode", _setting(settings, "run_mode", "agent"))),
        "plan_mode": bool(_state_get(state, "is_plan_mode", _setting(settings, "is_plan_mode", False))),
        "approval_mode": str(_state_get(state, "approval_mode", _setting(settings, "approval_mode", "confirm"))),
        "verified_editing_enabled": bool(_setting(settings, "verified_editing_enabled", False)),
        "intent_router_mode": str(_setting(settings, "intent_router_mode", "shadow_hybrid")),
        "max_tool_calls": int(_setting(settings, "max_tool_calls", 3)),
        "max_route_epochs": int(_setting(settings, "intent_router_max_epochs", 3)),
        "hard_guardrails": [
            "permissions_profile",
            "tool_registry_metadata",
            "command_allowlist",
            "verified_editing",
            "skill_recipe_policy",
        ],
    }


def _context_plan(calls: list[dict[str, Any]], *, state: Any, settings: Any, task: str) -> dict[str, Any]:
    scopes = [
        _context_scope_plan(scope, calls, state=state, settings=settings, task=task).to_dict()
        for scope in _derive_scopes(calls, state=state)
    ]
    return {
        "scopes": scopes,
        "fallback_policy": "reroute_on_no_results_then_fallback_search",
        "confidence_floor": 0.55,
    }


def _context_scope_plan(scope: str, calls: list[dict[str, Any]], *, state: Any, settings: Any, task: str) -> ContextScopePlan:
    path_hint = _extract_path_hint(task) or ""
    query = path_hint if scope == RouteScope.FILES and path_hint else task[:200]
    budget: dict[str, Any] = {}
    if scope in {RouteScope.WORKSPACE, RouteScope.FILES}:
        budget["max_entries"] = int(_setting(settings, "context_file_limit", 80))
    if scope == RouteScope.SEARCH:
        budget["max_matches"] = int(_setting(settings, "context_search_limit", 20))
    reason_by_scope = {
        RouteScope.WORKSPACE: "Establish baseline repository context.",
        RouteScope.FILES: "Read or enumerate concrete files before acting.",
        RouteScope.SEARCH: "Search text when no single path is sufficient.",
        RouteScope.CODE_INDEX: "Use code intelligence for symbol and module structure.",
        RouteScope.IMPACT: "Estimate affected files and tests before edits.",
        RouteScope.TESTS: "Collect verification evidence from test tools.",
        RouteScope.QUALITY: "Collect lint/format quality evidence.",
        RouteScope.GIT: "Read git state without mutating it.",
        RouteScope.SKILLS: "Use compact Skill metadata before loading full content.",
        RouteScope.RECIPES: "Expose recipe policy and preview boundary.",
        RouteScope.SUBAGENTS: "Delegate scoped read-only context collection.",
    }
    expected = {
        RouteScope.WORKSPACE: ["workspace files", "entrypoints"],
        RouteScope.FILES: ["file contents", "path evidence"],
        RouteScope.SEARCH: ["matching text", "candidate files"],
        RouteScope.CODE_INDEX: ["symbols", "modules"],
        RouteScope.IMPACT: ["impacted paths", "related tests"],
        RouteScope.TESTS: ["test status", "failure output"],
        RouteScope.QUALITY: ["lint status", "diagnostics"],
        RouteScope.GIT: ["diff/status"],
        RouteScope.SKILLS: ["matched Skill metadata"],
        RouteScope.RECIPES: ["recipe policy", "runnable/manual steps"],
        RouteScope.SUBAGENTS: ["subtask findings"],
    }
    fallback = {
        RouteScope.FILES: "fallback_to_text_search",
        RouteScope.SEARCH: "fallback_to_code_map_or_symbol_search",
        RouteScope.CODE_INDEX: "fallback_to_workspace_snapshot",
        RouteScope.IMPACT: "fallback_to_related_tests_or_search",
    }.get(scope, "continue_with_available_evidence")
    return ContextScopePlan(
        scope=scope,
        reason=reason_by_scope.get(scope, "Collect route evidence."),
        query=query,
        expected_evidence=expected.get(scope, ["evidence"]),
        fallback_if_empty=fallback,
        budget=budget,
    )


def _tool_plan(
    *,
    selected_candidates: list[RouteCandidate],
    selected_calls: list[dict[str, Any]],
    tool_specs: Mapping[str, Mapping[str, Any]],
    available_tools: set[str],
) -> dict[str, Any]:
    selected_names = {candidate.name for candidate in selected_candidates}
    rejected = []
    for name in sorted(available_tools - selected_names):
        spec = tool_specs.get(name, {})
        rejected.append(
            {
                "name": name,
                "category": spec.get("category", "context"),
                "capability": spec.get("capability", spec.get("category", "context")),
                "risk_level": spec.get("risk_level", "low"),
                "read_only": bool(spec.get("read_only", True)),
                "requires_approval": bool(spec.get("requires_approval", False)),
                "reason": "not_needed_for_current_intent_or_budget",
                "source": "tool_registry",
            }
        )
    return {
        "selected_tools": [candidate.to_dict() for candidate in selected_candidates],
        "selected_calls": selected_calls,
        "rejected_tools": rejected,
        "all_candidates": [candidate.to_dict() for candidate in selected_candidates],
        "policy": "selected tools must remain within registry metadata and guardrail boundaries",
    }


def _skill_plan(skill_candidates: list[dict[str, Any]], *, state: Any) -> dict[str, Any]:
    return {
        "candidates": skill_candidates,
        "alternatives": skill_candidates[1:],
        "progressive_disclosure": True,
        "may_load_full_skill": bool(_requested_skill_views(state) or len(skill_candidates) == 1),
        "disambiguation_required": len(skill_candidates) > 1,
        "disambiguation_reason": "multiple_compact_skill_candidates" if len(skill_candidates) > 1 else "",
    }


def _recipe_plan(recipe_candidates: list[dict[str, Any]], *, state: Any) -> dict[str, Any]:
    return {
        "candidates": recipe_candidates,
        "required_preview": bool(recipe_candidates),
        "approval_boundary": "manual_or_blocked_steps_require_proposal_approval",
        "policy": _state_get(state, "recipe_policy_snapshot", {}) or {},
    }


def _approval_plan(risk_summary: Mapping[str, Any], *, state: Any, intent: str) -> dict[str, Any]:
    requires = bool(risk_summary.get("requires_approval")) or intent in {IntentKind.MODIFY_CODE, IntentKind.MANAGE_SKILL}
    return {
        "requires_approval": requires,
        "approval_mode": str(_state_get(state, "approval_mode", "confirm")),
        "approval_boundary": str(risk_summary.get("boundary") or ""),
        "risk_level": str(risk_summary.get("max_risk_level") or "none"),
    }


def _verification_plan(intent: str, calls: list[dict[str, Any]]) -> dict[str, Any]:
    commands = []
    for call in calls:
        name = str(call.get("name") or "")
        args = dict(call.get("arguments") or {})
        if name in {"run_pytest", "targeted_pytest", "run_ruff_check", "run_ruff_format_check"} or name == "run_command":
            commands.append({"tool": name, "arguments": args, "purpose": args.get("purpose") or "route verification"})
    return {
        "required": intent in {IntentKind.MODIFY_CODE, IntentKind.DEBUG_TEST_FAILURE, IntentKind.RUN_QUALITY_CHECKS},
        "commands": commands,
        "reason": "quality_or_edit_intent_requires_verification" if commands else "no_quality_tool_selected_yet",
    }


def _decision_trace(
    *,
    selected_candidates: list[RouteCandidate],
    selected_calls: list[dict[str, Any]],
    tool_specs: Mapping[str, Mapping[str, Any]],
    available_tools: set[str],
) -> list[dict[str, Any]]:
    trace = [
        {
            "kind": "tool_candidate_selected",
            "name": candidate.name,
            "reason": candidate.reason,
            "evidence": candidate.evidence,
            "confidence": candidate.confidence,
            "risk_level": candidate.risk_level,
        }
        for candidate in selected_candidates
    ]
    selected_names = {candidate.name for candidate in selected_candidates}
    for name in sorted(available_tools - selected_names):
        spec = tool_specs.get(name, {})
        trace.append(
            {
                "kind": "tool_candidate_rejected",
                "name": name,
                "reason": "not_needed_for_current_intent_or_budget",
                "category": spec.get("category", "context"),
                "risk_level": spec.get("risk_level", "low"),
            }
        )
    if not selected_calls:
        trace.append({"kind": "no_tool_selected", "reason": "route will answer from existing context"})
    return trace


def _intent_alternatives(task: str, plan: str, *, primary_intent: str) -> list[dict[str, Any]]:
    alternatives: list[dict[str, Any]] = []
    marker_sets = [
        (IntentKind.DEBUG_TEST_FAILURE, _TEST_MARKERS + _FAILURE_MARKERS, 0.64),
        (IntentKind.MODIFY_CODE, _EDIT_MARKERS, 0.6),
        (IntentKind.INSPECT_CODE, _CODE_MARKERS, 0.55),
        (IntentKind.RUN_QUALITY_CHECKS, _TEST_MARKERS + _RUFF_MARKERS + _FORMAT_MARKERS, 0.52),
        (IntentKind.MANAGE_SKILL, _SKILL_MARKERS, 0.5),
        (IntentKind.REVIEW_DIFF, _REVIEW_MARKERS, 0.48),
    ]
    for intent, markers, base in marker_sets:
        if intent == primary_intent:
            continue
        matched = _matched_markers(task, plan, markers)
        if matched:
            alternatives.append(
                {
                    "intent": intent,
                    "confidence": _confidence(matched, base),
                    "matched_terms": matched[:8],
                    "reason": "secondary marker match",
                }
            )
    return sorted(alternatives, key=lambda item: -float(item["confidence"]))[:4]


def _reroute_triggers(state: Any, *, route_epoch: int, settings: Any, reason: str) -> list[dict[str, Any]]:
    max_epochs = int(_setting(settings, "intent_router_max_epochs", 3))
    if route_epoch >= max_epochs:
        return []
    triggers: list[dict[str, Any]] = []
    if reason:
        triggers.append({"kind": reason, "reason": reason, "route_epoch": route_epoch})
    calls = _tool_calls_from_state(state)
    edit_call_names = {"apply_text_edit", "prepare_edit", "preview_patch", "skill_recipe_run"}
    quality_call_names = {"run_pytest", "targeted_pytest", "run_ruff_check", "run_ruff_format_check", "run_command"}
    for index, call in enumerate(calls):
        name = str(call.get("name") or "")
        result = call.get("result")
        if _tool_result_failed(result) and not (name in quality_call_names and _quality_failed(result)):
            triggers.append({"kind": "tool_failure", "tool": name, "reason": _tool_failure_reason(result)})
        elif (
            name in {"search_text", "search_code", "semantic_code_search", "read_file", "code_map"}
            and _tool_result_empty(result)
        ):
            triggers.append(
                {"kind": "tool_no_results", "tool": name, "reason": "selected context tool returned no useful evidence"}
            )
        elif (
            name in quality_call_names
            and _quality_failed(result)
            and not any(str(next_call.get("name") or "") in edit_call_names for next_call in calls[index + 1 :])
        ):
            triggers.append({"kind": "quality_failure", "tool": name, "reason": "quality command failed"})
    patch = _state_get(state, "patch_proposal", None)
    if isinstance(patch, Mapping):
        stop_gate = patch.get("stop_gate") if isinstance(patch.get("stop_gate"), Mapping) else {}
        if stop_gate and str(stop_gate.get("status") or "") in {"failed", "missing"}:
            triggers.append({"kind": "patch_gate_blocked", "reason": str(stop_gate.get("reason") or "patch gate blocked")})
    route_plan = _state_get(state, "intent_route_plan", {}) or _state_get(state, "snapshots", {}).get("intent_route_plan", {})
    if isinstance(route_plan, Mapping) and float(route_plan.get("confidence") or 1.0) < 0.55:
        triggers.append({"kind": "context_confidence_below_threshold", "reason": "route confidence below 0.55"})
    return _dedupe_triggers(triggers)


async def _apply_model_advisor(
    route_plan: IntentRoutePlan,
    *,
    provider: Any | None,
    state: Any,
    settings: Any,
    tool_specs: Mapping[str, Mapping[str, Any]],
) -> IntentRoutePlan:
    mode = str(_setting(settings, "intent_router_mode", "shadow_hybrid"))
    if mode == "rules" or provider is None:
        return route_plan
    timeout_seconds = float(_setting(settings, "intent_router_model_timeout_seconds", 1.5))
    try:
        advisor = await asyncio.wait_for(
            _call_model_advisor(provider, state=state, route_plan=route_plan, tool_specs=tool_specs),
            timeout=timeout_seconds,
        )
    except Exception as exc:
        return replace(
            route_plan,
            model_advisor={
                "mode": mode,
                "status": "fallback",
                "reason": f"{type(exc).__name__}: {exc}",
                "applied": False,
            },
        )
    sanitized = _sanitize_model_advisor(advisor, tool_specs)
    trace = [
        *route_plan.decision_trace,
        {
            "kind": "model_advisor",
            "mode": mode,
            "status": sanitized["status"],
            "applied": False,
            "reason": sanitized.get("reason", "shadow_only"),
        },
    ]
    if mode == "hybrid" and sanitized["status"] == "valid" and float(sanitized.get("confidence") or 0) >= 0.7:
        suggested_intent = str(sanitized.get("intent") or "")
        if suggested_intent in _INTENT_VALUES:
            trace[-1] = {**trace[-1], "applied": True, "reason": "hybrid_intent_override_with_guardrails"}
            alternatives = [
                {"intent": route_plan.intent, "confidence": route_plan.confidence, "reason": "deterministic_primary"},
                *route_plan.intent_alternatives,
            ]
            return replace(
                route_plan,
                intent=suggested_intent,
                confidence=max(route_plan.confidence, float(sanitized["confidence"])),
                intent_alternatives=alternatives[:5],
                model_advisor={**sanitized, "mode": mode, "applied": True},
                decision_trace=trace,
            )
    return replace(route_plan, model_advisor={**sanitized, "mode": mode, "applied": False}, decision_trace=trace)


async def _call_model_advisor(
    provider: Any,
    *,
    state: Any,
    route_plan: IntentRoutePlan,
    tool_specs: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    available_tools = sorted(tool_specs)
    prompt = {
        "task": _state_get(state, "user_input", ""),
        "plan": _state_get(state, "plan", ""),
        "deterministic_route": route_plan.to_dict(),
        "available_tools": available_tools[:80],
        "instructions": (
            "Return compact JSON only with keys: intent, confidence, suggested_tools, "
            "reason, alternatives. Do not invent tools."
        ),
    }
    raw = await _maybe_await(
        provider.complete(
            [
                ChatMessage(
                    role="system",
                    content="You are a routing advisor. Return JSON only and never request unsafe tools.",
                ),
                ChatMessage(role="user", content=json.dumps(prompt, ensure_ascii=False, default=str)),
            ],
            temperature=0,
            max_tokens=700,
        )
    )
    parsed = _extract_json_object(str(raw or ""))
    if not isinstance(parsed, Mapping):
        raise ValueError("model advisor did not return a JSON object")
    return dict(parsed)


def _sanitize_model_advisor(advisor: Mapping[str, Any], tool_specs: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    suggested_tools = [str(name) for name in advisor.get("suggested_tools") or [] if str(name).strip()]
    rejected_tools = [name for name in suggested_tools if name not in tool_specs]
    accepted_tools = [name for name in suggested_tools if name in tool_specs]
    status = "valid"
    reason = str(advisor.get("reason") or "")
    confidence = float(advisor.get("confidence") or 0)
    if rejected_tools:
        status = "guardrail_rejected"
        reason = f"advisor suggested unknown tools: {', '.join(rejected_tools[:5])}"
    elif confidence < 0.55:
        status = "low_confidence"
        reason = reason or "advisor confidence below routing threshold"
    return {
        "status": status,
        "intent": str(advisor.get("intent") or ""),
        "confidence": round(max(0.0, min(confidence, 1.0)), 2),
        "suggested_tools": accepted_tools,
        "rejected_tools": rejected_tools,
        "alternatives": advisor.get("alternatives") if isinstance(advisor.get("alternatives"), list) else [],
        "reason": reason,
    }


def _extract_json_object(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped, flags=re.IGNORECASE).strip()
        stripped = re.sub(r"```$", "", stripped).strip()
    try:
        parsed = json.loads(stripped)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        try:
            parsed = json.loads(stripped[start : end + 1])
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def _tool_calls_from_state(state: Any) -> list[dict[str, Any]]:
    raw = _state_get(state, "tool_calls", []) or []
    calls = []
    for call in raw:
        if isinstance(call, Mapping):
            calls.append(dict(call))
        else:
            calls.append(
                {
                    "name": getattr(call, "name", ""),
                    "arguments": getattr(call, "arguments", {}),
                    "result": getattr(call, "result", None),
                    "blocked": getattr(call, "blocked", False),
                    "reason": getattr(call, "reason", None),
                }
            )
    return calls


def _tool_result_failed(result: Any) -> bool:
    if isinstance(result, Mapping):
        if result.get("ok") is False:
            return True
        if str(result.get("status") or "").casefold() in {"failed", "blocked", "error"}:
            return True
        if int(result.get("returncode") or 0) != 0:
            return True
    return False


def _tool_failure_reason(result: Any) -> str:
    if isinstance(result, Mapping):
        return str(result.get("error") or result.get("reason") or result.get("status") or "tool failed")
    return "tool failed"


def _tool_result_empty(result: Any) -> bool:
    if result in (None, "", [], {}):
        return True
    if isinstance(result, Mapping):
        for key in ("matches", "files", "entries", "symbols", "modules", "results", "items"):
            if key in result and not result.get(key):
                return True
    return False


def _quality_failed(result: Any) -> bool:
    if isinstance(result, Mapping):
        nested = result.get("result") if isinstance(result.get("result"), Mapping) else {}
        return_code = (
            result.get("returncode")
            or result.get("exit_code")
            or nested.get("returncode")
            or nested.get("exit_code")
            or 0
        )
        return (
            result.get("ok") is False
            or bool(result.get("failed"))
            or bool(nested.get("failed"))
            or int(return_code) != 0
            or str(result.get("status") or nested.get("status") or "").casefold() == "failed"
        )
    return False


def _dedupe_triggers(triggers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str]] = set()
    deduped: list[dict[str, Any]] = []
    for trigger in triggers:
        key = (str(trigger.get("kind") or ""), str(trigger.get("tool") or ""), str(trigger.get("reason") or ""))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(trigger)
    return deduped


def _state_get(state: Any, key: str, default: Any = None) -> Any:
    if isinstance(state, Mapping):
        return state.get(key, default)
    return getattr(state, key, default)


def _skill_search_text(skill: Mapping[str, Any]) -> str:
    return " ".join(
        [
            str(skill.get("name", "")),
            str(skill.get("description", "")),
            str(skill.get("category", "")),
            " ".join(str(item) for item in skill.get("tags", [])),
            " ".join(str(item) for item in skill.get("triggers", [])),
            " ".join(str(item) for item in skill.get("required_tools", [])),
        ]
    ).casefold()


def _matched_markers(task: str, plan: str, markers: tuple[str, ...]) -> list[str]:
    text = f"{task}\n{plan}".casefold()
    return [marker for marker in markers if marker in text]


def _top_terms(text: str, *, limit: int = 8) -> list[str]:
    seen: set[str] = set()
    terms: list[str] = []
    for term in re.findall(r"[\w\u4e00-\u9fff]+", text or ""):
        normalized = term.casefold()
        if len(normalized) < 2 or normalized in seen:
            continue
        seen.add(normalized)
        terms.append(normalized)
        if len(terms) >= limit:
            break
    return terms


def _setting(settings: Any, key: str, default: Any) -> Any:
    aliases = {
        "intent_router_mode": "mode",
        "intent_router_max_epochs": "max_epochs",
        "intent_router_model_timeout_seconds": "model_timeout_seconds",
    }
    if isinstance(settings, Mapping):
        if key in settings:
            return settings.get(key, default)
        alias = aliases.get(key)
        if alias and alias in settings:
            return settings.get(alias, default)
        return default
    if hasattr(settings, key):
        return getattr(settings, key)
    alias = aliases.get(key)
    if alias and hasattr(settings, alias):
        return getattr(settings, alias)
    return default


_CODE_MARKERS = (
    ".py",
    ".ts",
    ".tsx",
    ".js",
    "code",
    "bug",
    "fix",
    "implement",
    "refactor",
    "pytest",
    "ruff",
    "function",
    "class",
    "module",
    "\u4ee3\u7801",
    "\u4fee\u590d",
    "\u5b9e\u73b0",
    "\u91cd\u6784",
)
_EDIT_MARKERS = (
    "edit",
    "modify",
    "fix",
    "implement",
    "change",
    "patch",
    "write code",
    "\u4fee\u6539",
    "\u4fee\u590d",
    "\u5b9e\u73b0",
)
_REFACTOR_MARKERS = ("refactor", "redesign", "architecture", "\u91cd\u6784", "\u67b6\u6784")
_TEST_MARKERS = ("pytest", "test", "tests", "\u6d4b\u8bd5")
_FAILURE_MARKERS = ("fail", "failed", "failing", "failure", "error", "traceback", "\u5931\u8d25", "\u62a5\u9519")
_RUFF_MARKERS = ("ruff", "lint", "\u8d28\u91cf", "\u68c0\u67e5")
_FORMAT_MARKERS = ("ruff format", "format", "\u683c\u5f0f")
_SKILL_MARKERS = ("skill", "sop", "workflow", "\u6280\u80fd", "\u89c4\u8303", "\u6d41\u7a0b")
_REVIEW_MARKERS = ("review", "diff", "pr", "pull request", "git status", "\u8bc4\u5ba1")
_QUESTION_MARKERS = ("what", "why", "how", "explain", "describe", "?", "\u4ec0\u4e48", "\u600e\u4e48", "\u4e3a\u4ec0\u4e48")
_INTENT_VALUES = {
    IntentKind.ANSWER_QUESTION,
    IntentKind.INSPECT_CODE,
    IntentKind.MODIFY_CODE,
    IntentKind.DEBUG_TEST_FAILURE,
    IntentKind.RUN_QUALITY_CHECKS,
    IntentKind.REVIEW_DIFF,
    IntentKind.MANAGE_SKILL,
    IntentKind.PLAN_REFACTOR,
    IntentKind.UNKNOWN,
}
