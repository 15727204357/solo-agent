from __future__ import annotations

import asyncio

import pytest
from solo_agent.agent.deps import AgentDeps, AgentSettings
from solo_agent.agent.state import AgentState
from solo_agent.workflow.stages import _execute_tools_node


class ReadonlyRegistry:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def list_tools(self) -> list[dict[str, str]]:
        return [
            {"name": "workspace_snapshot"},
            {"name": "code_map"},
        ]

    async def call_tool(self, name: str, arguments: dict):
        await asyncio.sleep(0)
        self.calls.append((name, dict(arguments)))
        return {"ok": True, "tool": name, "result": {"name": name, "arguments": dict(arguments)}}


@pytest.mark.asyncio
async def test_execute_tools_prefetches_and_reuses_initial_readonly_context_tools() -> None:
    registry = ReadonlyRegistry()
    state = AgentState(session_id="s1", run_id="r1", user_input="inspect code")
    state.snapshots["proposed_tool_calls"] = [
        {"name": "workspace_snapshot", "arguments": {"path": ".", "max_entries": 80}},
        {"name": "code_map", "arguments": {"path": ".", "max_files": 80}},
    ]

    events = [
        event
        async for event in _execute_tools_node(
            state,
            AgentDeps(tool_registry=registry),
            AgentSettings(tool_call_cut_off=4),
        )
    ]

    assert [event.type for event in events if event.type.startswith("tool_prefetch_")] == [
        "tool_prefetch_started",
        "tool_prefetch_completed",
    ]
    assert [name for name, _ in registry.calls] == ["workspace_snapshot", "code_map"]
    assert state.snapshots["tool_result_cache_stats"] == {"misses": 2, "hits": 2}
    assert [call.name for call in state.tool_calls] == ["workspace_snapshot", "code_map"]