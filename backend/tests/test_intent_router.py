from __future__ import annotations

import pytest
from solo_agent.agent import AgentSettings
from solo_agent.agent.state import AgentState
from solo_agent.workflow.intent_router import IntentKind, plan_intent_route


class RouteRegistry:
    def __init__(self, tools: list[dict[str, object]]) -> None:
        self._tools = tools

    def list_tools(self) -> list[dict[str, object]]:
        return self._tools


def _registry() -> RouteRegistry:
    return RouteRegistry(
        [
            {"name": "workspace_snapshot", "category": "context", "read_only": True, "risk_level": "low"},
            {"name": "search_text", "category": "context", "read_only": True, "risk_level": "low"},
            {"name": "read_file", "category": "context", "read_only": True, "risk_level": "low"},
            {"name": "code_map", "category": "code_intelligence", "read_only": True, "risk_level": "low"},
            {"name": "analyze_impact", "category": "code_intelligence", "read_only": True, "risk_level": "low"},
            {"name": "run_pytest", "category": "quality", "read_only": True, "risk_level": "medium"},
            {"name": "run_ruff_check", "category": "quality", "read_only": True, "risk_level": "medium"},
            {"name": "skill_view", "category": "skill", "read_only": True, "risk_level": "low"},
            {"name": "skill_recipe_list", "category": "skill", "read_only": True, "risk_level": "low"},
            {"name": "prepare_edit", "category": "edit", "read_only": True, "risk_level": "medium"},
            {"name": "apply_text_edit", "category": "edit", "read_only": False, "risk_level": "high"},
        ]
    )


def _state(user_input: str, *, plan: str = "") -> AgentState:
    return AgentState(session_id="session", run_id="run", user_input=user_input, plan=plan)


class AdvisorProvider:
    def __init__(self, response: str) -> None:
        self.response = response

    async def complete(self, messages, **kwargs) -> str:
        return self.response


@pytest.mark.asyncio
async def test_explain_only_route_does_not_select_edit_tools() -> None:
    route = await plan_intent_route(
        _registry(),
        _state("Explain how AgentState snapshots work."),
        AgentSettings(provider="fake", model="fake", max_tool_calls=5),
    )

    names = {call["name"] for call in route.proposed_tool_calls}
    assert route.intent == IntentKind.ANSWER_QUESTION
    assert "workspace_snapshot" in names
    assert "search_text" in names
    assert "prepare_edit" not in names
    assert "apply_text_edit" not in names
    assert route.risk_summary["boundary"] == "route_is_context_and_verification_only"
    payload = route.to_dict()
    assert payload["route_plan_schema_version"] == "2"
    assert payload["route_id"] == "session:run:route:0"
    assert payload["context_plan"]["scopes"]
    assert payload["tool_plan"]["selected_tools"]
    assert payload["approval_plan"]["requires_approval"] is False
    assert payload["decision_trace"]


@pytest.mark.asyncio
async def test_code_inspection_route_uses_code_context_and_path_read() -> None:
    route = await plan_intent_route(
        _registry(),
        _state("Inspect backend/src/solo_agent/agent/state.py for route fields."),
        AgentSettings(provider="fake", model="fake", max_tool_calls=6),
    )

    names = [call["name"] for call in route.proposed_tool_calls]
    assert route.intent == IntentKind.INSPECT_CODE
    assert "code_map" in names
    assert "analyze_impact" in names
    assert "read_file" in names
    assert "code_index" in route.searched_scopes
    assert "impact" in route.searched_scopes


@pytest.mark.asyncio
async def test_modify_route_identifies_patch_boundary_without_write_tool() -> None:
    route = await plan_intent_route(
        _registry(),
        _state("Implement a fix in backend/src/app.py."),
        AgentSettings(provider="fake", model="fake", max_tool_calls=6),
    )

    names = {call["name"] for call in route.proposed_tool_calls}
    assert route.intent == IntentKind.MODIFY_CODE
    assert "apply_text_edit" not in names
    assert "prepare_edit" not in names
    assert "verified editing" in str(route.risk_summary["boundary"])


@pytest.mark.asyncio
async def test_failing_pytest_route_prefers_debug_and_quality_tool() -> None:
    route = await plan_intent_route(
        _registry(),
        _state("pytest is failing with an AssertionError; debug the failure."),
        AgentSettings(provider="fake", model="fake", max_tool_calls=6),
    )

    names = {call["name"] for call in route.proposed_tool_calls}
    assert route.intent == IntentKind.DEBUG_TEST_FAILURE
    assert "run_pytest" in names
    assert "tests" in route.searched_scopes
    assert route.tool_candidates
    assert all("reason" in candidate for candidate in route.tool_candidates)


@pytest.mark.asyncio
async def test_explicit_skill_route_exposes_skill_and_recipe_candidates() -> None:
    state = _state("/skill python-backend-change run pytest")
    state.selected_skills = [
        {
            "name": "python-backend-change",
            "description": "Python backend workflow.",
            "path": "skills/workflows/python-backend-change/SKILL.md",
            "score": 2,
        }
    ]

    route = await plan_intent_route(
        _registry(),
        state,
        AgentSettings(provider="fake", model="fake", max_tool_calls=8),
    )

    names = {call["name"] for call in route.proposed_tool_calls}
    assert route.intent == IntentKind.MANAGE_SKILL
    assert "skill_view" in names
    assert "skill_recipe_list" in names
    assert route.skill_candidates[0]["matched_intent"] == IntentKind.MANAGE_SKILL
    assert route.skill_candidates[0]["recommendation_reason"]


@pytest.mark.asyncio
async def test_unknown_route_reports_low_confidence_fallback() -> None:
    route = await plan_intent_route(
        _registry(),
        _state(""),
        AgentSettings(provider="fake", model="fake", max_tool_calls=3),
    )

    assert route.intent == IntentKind.UNKNOWN
    assert route.confidence < 0.5
    assert route.evidence[0]["kind"] == "intent_classifier"


@pytest.mark.asyncio
async def test_shadow_model_advisor_records_valid_json_without_changing_route() -> None:
    route = await plan_intent_route(
        _registry(),
        _state("Explain how routing works."),
        AgentSettings(provider="fake", model="fake", max_tool_calls=5, intent_router_mode="shadow_hybrid"),
        provider=AdvisorProvider(
            '{"intent":"modify_code","confidence":0.91,'
            '"suggested_tools":["apply_text_edit"],"reason":"advisor opinion"}'
        ),
    )

    assert route.intent == IntentKind.ANSWER_QUESTION
    assert route.model_advisor["status"] == "valid"
    assert route.model_advisor["applied"] is False
    assert "apply_text_edit" not in {call["name"] for call in route.proposed_tool_calls}


@pytest.mark.asyncio
async def test_model_advisor_invalid_json_falls_back_to_deterministic_route() -> None:
    route = await plan_intent_route(
        _registry(),
        _state("Inspect backend/src/app.py"),
        AgentSettings(provider="fake", model="fake", max_tool_calls=5, intent_router_mode="shadow_hybrid"),
        provider=AdvisorProvider("not-json"),
    )

    assert route.intent == IntentKind.INSPECT_CODE
    assert route.model_advisor["status"] == "fallback"
    assert route.model_advisor["applied"] is False


@pytest.mark.asyncio
async def test_model_advisor_low_confidence_is_not_applied() -> None:
    route = await plan_intent_route(
        _registry(),
        _state("Run pytest"),
        AgentSettings(provider="fake", model="fake", max_tool_calls=5, intent_router_mode="hybrid"),
        provider=AdvisorProvider('{"intent":"review_diff","confidence":0.2,"suggested_tools":["run_pytest"],"reason":"uncertain"}'),
    )

    assert route.intent == IntentKind.RUN_QUALITY_CHECKS
    assert route.model_advisor["status"] == "low_confidence"
    assert route.model_advisor["applied"] is False


@pytest.mark.asyncio
async def test_model_advisor_hallucinated_tool_is_rejected() -> None:
    route = await plan_intent_route(
        _registry(),
        _state("Debug failing pytest"),
        AgentSettings(provider="fake", model="fake", max_tool_calls=5, intent_router_mode="hybrid"),
        provider=AdvisorProvider(
            '{"intent":"debug_test_failure","confidence":0.9,"suggested_tools":["unknown_write_tool"],"reason":"unsafe"}'
        ),
    )

    assert route.intent == IntentKind.DEBUG_TEST_FAILURE
    assert route.model_advisor["status"] == "guardrail_rejected"
    assert route.model_advisor["rejected_tools"] == ["unknown_write_tool"]
