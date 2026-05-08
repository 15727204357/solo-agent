from __future__ import annotations

import pytest
from solo_agent.agent.state import AgentState
from solo_agent.workflow.graph_nodes import (
    make_persist_snapshot_node,
    make_skip_memory_node,
    make_task_state_node,
)
from solo_agent.workflow.graph_state import initial_graph_state


@pytest.mark.asyncio
async def test_skip_memory_node_collects_events() -> None:
    state = AgentState(session_id="s1", run_id="r1", user_input="test", memory_enabled=False)
    gs = initial_graph_state(state)
    node = make_skip_memory_node()
    result = await node(gs)
    events = result.get("events", [])
    assert len(events) > 0
    assert events[0]["type"] == "memory_skipped"
    restored = result["agent_state"]
    assert restored["conversation_context"]["memory_enabled"] is False


@pytest.mark.asyncio
async def test_persist_snapshot_node_saves_state() -> None:
    state = AgentState(session_id="s2", run_id="r2", user_input="snapshot test")
    gs = initial_graph_state(state)
    node = make_persist_snapshot_node()
    result = await node(gs)
    events = result.get("events", [])
    assert any(e["type"] == "persist_snapshot_completed" for e in events)
    assert "snapshot" in events[-1]["data"]


@pytest.mark.asyncio
async def test_subagent_dispatch_node_serial_fallback() -> None:
    state = AgentState(session_id="s3", run_id="r3", user_input="parallel test", execution_strategy="parallel")
    gs = initial_graph_state(state)
    from solo_agent.workflow.graph_nodes import make_subagent_dispatch_node
    settings = type("Settings", (), {"max_concurrent_subagents": 3})()
    deps = type("Deps", (), {})()
    node = make_subagent_dispatch_node(deps, settings)
    result = await node(gs)
    # Without task candidates, should fall back to serial
    assert result["agent_state"]["execution_strategy"] == "serial"


@pytest.mark.asyncio
async def test_node_preserves_other_state_fields() -> None:
    state = AgentState(session_id="s4", run_id="r4", user_input="preserve test", plan="some plan")
    gs = initial_graph_state(state)
    node = make_task_state_node()
    result = await node(gs)
    assert result["agent_state"]["session_id"] == "s4"
    assert result["agent_state"]["run_id"] == "r4"
    assert result["agent_state"]["user_input"] == "preserve test"
