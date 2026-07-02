from __future__ import annotations

import pytest
from solo_agent.agent.deps import AgentSettings
from solo_agent.agent.prompts import build_subagent_tool_instruction
from solo_agent.agent.state import AgentState
from solo_agent.workflow.stages import _parallelism_gate_stage, _propose_tool_calls


class TaskToolRegistry:
    def list_tools(self) -> list[dict[str, str]]:
        return [{"name": "task"}, {"name": "workspace_snapshot"}, {"name": "search_text"}]


@pytest.mark.asyncio
async def test_parallelism_gate_sets_parallel_strategy_when_all_conditions_pass() -> None:
    state = AgentState(session_id="s1", run_id="r1", user_input="implement independent tasks")
    state.plan = '''
```json
{
  "parallel_tasks": [
    {
      "id": "T1",
      "title": "Provider tests",
      "domain": "providers",
      "read_paths": ["backend/src/solo_agent/providers/"],
      "write_paths": ["backend/tests/test_provider_config.py"],
      "verify_commands": ["pytest backend/tests/test_provider_config.py -q"]
    },
    {
      "id": "T2",
      "title": "Memory tests",
      "domain": "memory",
      "read_paths": ["backend/src/solo_agent/memory/"],
      "write_paths": ["backend/tests/test_memory_inbox.py"],
      "verify_commands": ["pytest backend/tests/test_memory_inbox.py -q"]
    }
  ]
}
```
'''

    settings = AgentSettings(subagent_policy="auto", subagent_enabled=True)
    events = [event async for event in _parallelism_gate_stage(state, settings)]

    assert [event.type for event in events] == [
        "parallelism_gate_started",
        "parallelism_decision_completed",
        "parallelism_gate_completed",
    ]
    assert state.execution_strategy == "parallel"
    assert state.parallelism_decision["allowed"] is True
    assert state.parallelism_decision["suitable"] is True
    assert state.parallelism_decision["subagent_enabled"] is True
    assert state.parallelism_decision["subagent_policy"] == "auto"
    assert state.snapshots["execution_strategy"] == "parallel"
    assert state.snapshots["parallelism_decision"]["strategy"] == "parallel"


@pytest.mark.asyncio
async def test_parallelism_gate_falls_back_to_serial_without_metadata() -> None:
    state = AgentState(session_id="s1", run_id="r1", user_input="fix everything")
    state.plan = "1. Inspect the codebase. 2. Fix the issue. 3. Run pytest."

    events = [event async for event in _parallelism_gate_stage(state, AgentSettings())]

    assert events[-1].type == "parallelism_gate_completed"
    assert state.execution_strategy == "serial"
    assert state.parallelism_decision["allowed"] is False
    assert state.parallelism_decision["suitable"] is False
    assert state.task_candidates[0]["risk_flags"] == ["unstructured_plan"]


@pytest.mark.asyncio
async def test_parallelism_gate_records_suitable_but_serial_when_subagents_disabled() -> None:
    state = AgentState(session_id="s1", run_id="r1", user_input="implement independent tasks")
    state.plan = '''
```json
{
  "parallel_tasks": [
    {
      "id": "T1",
      "title": "Provider tests",
      "domain": "providers",
      "read_paths": ["backend/src/solo_agent/providers/"],
      "write_paths": ["backend/tests/test_provider_config.py"],
      "verify_commands": ["pytest backend/tests/test_provider_config.py -q"]
    },
    {
      "id": "T2",
      "title": "Memory tests",
      "domain": "memory",
      "read_paths": ["backend/src/solo_agent/memory/"],
      "write_paths": ["backend/tests/test_memory_inbox.py"],
      "verify_commands": ["pytest backend/tests/test_memory_inbox.py -q"]
    }
  ]
}
```
'''

    events = [event async for event in _parallelism_gate_stage(state, AgentSettings(subagent_enabled=False))]

    assert any(event.type == "parallelism_decision_completed" for event in events)
    assert state.execution_strategy == "serial"
    assert state.parallelism_decision["suitable"] is True
    assert state.parallelism_decision["strategy"] == "serial"
    assert state.parallelism_decision["reason"] == "subagent_policy_off"
    assert state.parallelism_decision["subagent_enabled"] is False
    assert state.parallelism_decision["subagent_policy"] == "off"


@pytest.mark.asyncio
async def test_parallelism_gate_materializes_team_developer_assignments() -> None:
    state = AgentState(session_id="s1", run_id="r1", user_input="implement team tasks")
    state.snapshots["team_plan"] = {
        "mode": "team",
        "tasks": [
            {
                "id": "T1",
                "title": "Provider tests",
                "write_paths": ["backend/tests/test_provider_config.py"],
                "read_paths": ["backend/src/solo_agent/providers/"],
                "verify_commands": ["pytest backend/tests/test_provider_config.py -q"],
            },
            {
                "id": "T2",
                "title": "Memory tests",
                "write_paths": ["backend/tests/test_memory_inbox.py"],
                "read_paths": ["backend/src/solo_agent/memory/"],
                "verify_commands": ["pytest backend/tests/test_memory_inbox.py -q"],
            },
        ],
        "assignments": [],
    }

    events = [
        event
        async for event in _parallelism_gate_stage(
            state,
            AgentSettings(subagent_policy="auto", subagent_enabled=True, max_concurrent_subagents=2),
        )
    ]

    team_plan = state.snapshots["team_plan"]
    assert [event.type for event in events] == [
        "parallelism_gate_started",
        "parallelism_decision_completed",
        "parallelism_gate_completed",
    ]
    assert state.execution_strategy == "parallel"
    assert state.parallelism_decision["mode"] == "developer_parallelism"
    assert state.parallelism_decision["developer_count"] == 2
    assert len(team_plan["assignments"]) == 2
    assert team_plan["developer_parallelism"] == state.parallelism_decision
    assert team_plan["verify_commands"] == [
        "pytest backend/tests/test_provider_config.py -q",
        "pytest backend/tests/test_memory_inbox.py -q",
    ]


@pytest.mark.asyncio
async def test_parallelism_gate_collapses_team_assignments_when_write_sets_conflict() -> None:
    state = AgentState(session_id="s1", run_id="r1", user_input="implement overlapping team tasks")
    state.snapshots["team_plan"] = {
        "mode": "team",
        "tasks": [
            {"id": "T1", "title": "First edit", "write_paths": ["backend/src/app.py"]},
            {"id": "T2", "title": "Second edit", "write_paths": ["backend/src/app.py"]},
        ],
        "assignments": [],
    }

    events = [
        event
        async for event in _parallelism_gate_stage(
            state,
            AgentSettings(subagent_policy="auto", subagent_enabled=True, max_concurrent_subagents=2),
        )
    ]

    assert events[-1].type == "parallelism_gate_completed"
    assert state.execution_strategy == "serial"
    assert state.parallelism_decision["mode"] == "developer_parallelism"
    assert state.parallelism_decision["suitable"] is False
    assert state.parallelism_decision["reason"] == "insufficient_independent_developer_work"
    assert len(state.snapshots["team_plan"]["assignments"]) == 1
    assert state.snapshots["team_plan"]["assignments"][0]["developer_id"] == "developer-1"

@pytest.mark.asyncio
async def test_select_tools_proposes_task_only_when_subagent_enabled_and_suitable() -> None:
    state = AgentState(session_id="s1", run_id="r1", user_input="inspect independent areas")
    state.parallelism_decision = {
        "strategy": "parallel",
        "suitable": True,
        "subagent_enabled": True,
        "subagent_policy": "auto",
        "task_count": 2,
        "candidates": [
            {"id": "T1", "title": "Inspect app", "read_paths": ["app.py"], "write_paths": []},
            {"id": "T2", "title": "Inspect tests", "read_paths": ["tests"], "write_paths": []},
        ],
    }
    state.snapshots["parallelism_decision"] = state.parallelism_decision

    calls = await _propose_tool_calls(
        TaskToolRegistry(),
        state,
        AgentSettings(max_tool_calls=3, subagent_policy="auto", subagent_enabled=True),
    )

    assert [call["name"] for call in calls[:2]] == ["task", "task"]
    assert "Parent user task" in calls[0]["arguments"]["prompt"]


@pytest.mark.asyncio
async def test_select_tools_does_not_propose_task_when_unsuitable() -> None:
    state = AgentState(session_id="s1", run_id="r1", user_input="simple task")
    state.parallelism_decision = {
        "strategy": "serial",
        "suitable": False,
        "subagent_enabled": True,
        "subagent_policy": "auto",
        "task_count": 1,
        "candidates": [{"id": "T1", "title": "Single task"}],
    }
    state.snapshots["parallelism_decision"] = state.parallelism_decision

    calls = await _propose_tool_calls(
        TaskToolRegistry(),
        state,
        AgentSettings(max_tool_calls=3, subagent_policy="auto", subagent_enabled=True),
    )
    instruction = build_subagent_tool_instruction(state, task_tool_available=True)

    assert "task" not in [call["name"] for call in calls]
    assert "do not call task" in instruction.lower()


def test_subagent_tool_instruction_omits_task_when_disabled() -> None:
    state = AgentState(session_id="s1", run_id="r1", user_input="parallel")
    state.parallelism_decision = {"suitable": True, "subagent_enabled": False}

    assert build_subagent_tool_instruction(state, task_tool_available=True) == ""
