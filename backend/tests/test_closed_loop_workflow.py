from __future__ import annotations

import pytest
from solo_agent.agent import AgentDeps, AgentSettings
from solo_agent.agent.state import AgentState
from solo_agent.workflow.stages import _team_test_stage


class FailingRegistry:
    workspace_root = None
    command_workspace_root = None

    def list_tools(self, visibility: str = "model") -> list[dict[str, str]]:
        return [{"name": "run_command"}]

    def call(self, name: str, arguments: dict[str, object]) -> dict[str, object]:
        assert name == "run_command"
        return {
            "ok": True,
            "tool": name,
            "result": {
                "command": "pytest -q tests/test_service.py",
                "returncode": 1,
                "output": "tests/test_service.py:7: AssertionError\nE assert 1 == 2\nFAILED tests/test_service.py::test_value",
            },
            "metadata": {},
        }


@pytest.mark.asyncio
async def test_team_test_classifies_failures_and_plans_remediation() -> None:
    state = AgentState(session_id="session", run_id="run", user_input="Fix service")
    state.snapshots["team_develop_iteration"] = 1
    state.snapshots["team_plan"] = {"verify_commands": ["pytest -q tests/test_service.py"]}
    state.snapshots["team_developer_output"] = {
        "status": "completed",
        "sandbox_applied": True,
        "sandbox_diff": "--- a/pkg/service.py\n+++ b/pkg/service.py\n",
        "verification": [],
    }

    events = [
        event
        async for event in _team_test_stage(
            state,
            AgentDeps(tool_registry=FailingRegistry()),
            AgentSettings(),
        )
    ]

    event_types = [event.type for event in events]
    assert "failure_classified" in event_types
    assert "remediation_planned" in event_types
    assert state.failure_reports[0]["kind"] == "test_failure"
    assert state.review_reports["team_test"]["feedback"]["reason"] == "verification_failed"
