from __future__ import annotations

import pytest
from solo_agent.agent import AgentDeps, AgentSettings
from solo_agent.agent.state import AgentState
from solo_agent.context import WorkspaceTaskStore
from solo_agent.tools import create_default_registry
from solo_agent.workflow.stages import _execute_tools_node, _task_state_stage


async def _collect(stage):
    return [event async for event in stage]


@pytest.mark.asyncio
async def test_agent_mode_skips_task_list_stage_and_write_todos(tmp_path) -> None:
    settings = AgentSettings(workspace_root=tmp_path, run_mode="agent", is_plan_mode=False)
    state = AgentState(session_id="s-agent", run_id="r1", user_input="hello", is_plan_mode=False)
    registry = create_default_registry(tmp_path, is_plan_mode=False)

    events = await _collect(_task_state_stage(state, settings))

    assert events == []
    assert state.task_list == {}
    tool_names = {tool["name"] for tool in registry.list_tools()}
    assert "write_todos" not in tool_names
    assert not {"task_create", "task_get", "task_list", "task_update"} & tool_names
    assert not (tmp_path / ".solo-agent" / "tasks" / "s-agent.json").exists()


@pytest.mark.asyncio
async def test_plan_mode_loads_or_initializes_task_list(tmp_path) -> None:
    settings = AgentSettings(workspace_root=tmp_path, run_mode="plan", is_plan_mode=True)
    state = AgentState(
        session_id="s-plan",
        run_id="r1",
        user_input="ship it",
        is_plan_mode=True,
        plan="- [ ] Inspect workflow\n- [ ] Wire task events",
    )

    events = await _collect(_task_state_stage(state, settings))
    restored = WorkspaceTaskStore(tmp_path).load("s-plan")

    assert [event.type for event in events] == ["task_list_loaded"]
    assert state.task_list["tasks"][0]["subject"] == "Inspect workflow"
    assert isinstance(state.task_list, dict)
    assert any(item.get("source") == "task_list" for item in state.context)
    assert [item.subject for item in restored.items] == ["Inspect workflow", "Wire task events"]


@pytest.mark.asyncio
async def test_write_todos_updates_task_list_and_emits_event(tmp_path) -> None:
    settings = AgentSettings(workspace_root=tmp_path, run_mode="plan", is_plan_mode=True)
    registry = create_default_registry(tmp_path, is_plan_mode=True)
    state = AgentState(
        session_id="s-plan",
        run_id="r2",
        user_input="continue",
        is_plan_mode=True,
    )
    state.snapshots["proposed_tool_calls"] = [
        {
            "name": "write_todos",
            "arguments": {
                "tasks": [
                    {"id": "T-1", "subject": "Inspect workflow", "status": "completed"},
                    {"id": "T-2", "subject": "Update tests", "status": "in_progress"},
                ]
            },
        }
    ]
    deps = AgentDeps(tool_registry=registry, safety_inspector=registry, settings=settings)

    events = await _collect(_execute_tools_node(state, deps, settings))
    restored = WorkspaceTaskStore(tmp_path).load("s-plan")

    assert "task_list_updated" in [event.type for event in events]
    assert state.task_list["tasks"][1]["status"] == "in_progress"
    assert [item.status for item in restored.items] == ["completed", "in_progress"]


@pytest.mark.asyncio
async def test_execute_tools_runs_task_tool_and_updates_subagent_state(tmp_path) -> None:
    (tmp_path / "app.py").write_text("def service():\n    return 'ok'\n", encoding="utf-8")
    settings = AgentSettings(workspace_root=tmp_path, subagent_enabled=True)
    registry = create_default_registry(tmp_path, subagent_enabled=True)
    state = AgentState(session_id="s-subagent", run_id="r1", user_input="inspect app")
    state.snapshots["proposed_tool_calls"] = [
        {
            "name": "task",
            "arguments": {
                "description": "Inspect app",
                "prompt": "Inspect app.py and summarize service.",
                "read_paths": ["app.py"],
            },
        }
    ]
    deps = AgentDeps(tool_registry=registry, safety_inspector=registry, settings=settings)

    events = await _collect(_execute_tools_node(state, deps, settings))
    event_types = [event.type for event in events]
    result = next(iter(state.subagent_results.values()))

    assert "task_started" in event_types
    assert "task_completed" in event_types
    assert state.subagent_dispatches[0]["task_id"].startswith("task_")
    assert result["metadata"]["thread_id"] == "s-subagent"
    assert result["status"] == "completed"
    assert state.snapshots["subagent_results"] == state.subagent_results
    assert any(item["source"] == "tool:task" for item in state.context)


@pytest.mark.asyncio
async def test_execute_tools_emits_task_failed_for_failed_task_result(tmp_path) -> None:
    settings = AgentSettings(workspace_root=tmp_path, subagent_enabled=True)
    registry = create_default_registry(tmp_path, subagent_enabled=True)
    state = AgentState(session_id="s-subagent", run_id="r2", user_input="inspect missing")
    state.snapshots["proposed_tool_calls"] = [
        {
            "name": "task",
            "arguments": {
                "description": "Inspect missing",
                "prompt": "Inspect missing file.",
                "read_paths": ["missing.py"],
            },
        }
    ]
    deps = AgentDeps(tool_registry=registry, safety_inspector=registry, settings=settings)

    events = await _collect(_execute_tools_node(state, deps, settings))
    failed = next(event for event in events if event.type == "task_failed")
    result = next(iter(state.subagent_results.values()))

    assert result["status"] == "failed"
    assert "does not exist" in failed.data["error"]
