from __future__ import annotations

import pytest
from langgraph.graph import END
from solo_agent.workflow.graph_state import SoloGraphState
from solo_agent.workflow.graphs import (
    build_text_provider_graph,
    make_route_after_guard,
    route_after_execute_tools,
    route_after_parallelism_gate,
    route_after_patch,
)


class FakeSettings:
    workflow_checkpointer = "memory"
    workflow_checkpoint_path = ".solo-agent/checkpoints/test.sqlite3"
    memory_enabled = True
    conversation_history_enabled = True
    max_concurrent_subagents = 3
    history_message_limit = 12
    memory_search_limit = 5
    max_tool_calls = 3
    tool_call_cut_off = 3
    tool_output_max_bytes = 12_000
    max_selected_skills = 3
    context_file_limit = 80
    context_search_limit = 20
    response_max_tokens = 1400
    temperature = 0.2
    plan_max_tokens = 500
    patch_max_tokens = 1400
    verified_editing_enabled = True
    workspace_root = "."
    summary_trigger_messages = 8
    plan_deep_max_tokens = 6000
    subagent_enabled = False
    workflow_runtime_root = ".solo-agent/runs"
    subagent_timeout_seconds = 900
    sandbox_mode = "local"
    run_mode = "agent"


class FakeProvider:
    supports_tool_calling = False

    async def stream_chat(self, messages, **kwargs):
        yield type("obj", (object,), {"content": ""})()

    async def complete(self, messages, **kwargs):
        return ""


class FakeDeps:
    settings = FakeSettings()
    persistence = None
    tool_registry = None
    safety_inspector = None
    context_provider = None


@pytest.mark.asyncio
async def test_graph_compiles_without_checkpointer() -> None:
    graph = build_text_provider_graph(provider=FakeProvider(), deps=FakeDeps(), settings=FakeSettings())
    compiled = graph.compile(checkpointer=False)
    assert compiled is not None


@pytest.mark.asyncio
async def test_graph_compiles_with_memory_checkpointer() -> None:
    graph = build_text_provider_graph(provider=FakeProvider(), deps=FakeDeps(), settings=FakeSettings())
    from langgraph.checkpoint.memory import InMemorySaver
    compiled = graph.compile(checkpointer=InMemorySaver())
    assert compiled is not None


@pytest.mark.asyncio
async def test_route_after_guard_blocked_returns_end() -> None:
    state: SoloGraphState = {"agent_state": {"blocked": True}, "events": [], "error": None}
    router = make_route_after_guard("select_tools")
    assert router(state) == END


@pytest.mark.asyncio
async def test_route_after_guard_not_blocked_returns_target() -> None:
    state: SoloGraphState = {"agent_state": {"blocked": False}, "events": [], "error": None}
    router = make_route_after_guard("select_tools")
    assert router(state) == "select_tools"


@pytest.mark.asyncio
async def test_route_after_parallelism_gate_blocked_returns_end() -> None:
    state: SoloGraphState = {"agent_state": {"blocked": True}, "events": [], "error": None}
    assert route_after_parallelism_gate(state) == END


@pytest.mark.asyncio
async def test_route_after_parallelism_gate_parallel() -> None:
    state: SoloGraphState = {"agent_state": {"execution_strategy": "parallel", "blocked": False}, "events": [], "error": None}
    assert route_after_parallelism_gate(state) == "parallel_dispatch"


@pytest.mark.asyncio
async def test_route_after_parallelism_gate_serial() -> None:
    state: SoloGraphState = {"agent_state": {"execution_strategy": "serial", "blocked": False}, "events": [], "error": None}
    assert route_after_parallelism_gate(state) == "collect_context"


@pytest.mark.asyncio
async def test_route_after_execute_tools_awaiting_approval() -> None:
    state: SoloGraphState = {"agent_state": {"awaiting_approval": True}, "events": [], "error": None}
    assert route_after_execute_tools(state) == END


@pytest.mark.asyncio
async def test_route_after_execute_tools_continue() -> None:
    state: SoloGraphState = {"agent_state": {"awaiting_approval": False}, "events": [], "error": None}
    assert route_after_execute_tools(state) == "spec_compliance_review"


@pytest.mark.asyncio
async def test_route_after_patch_awaiting_approval() -> None:
    state: SoloGraphState = {"agent_state": {"awaiting_approval": True}, "events": [], "error": None}
    assert route_after_patch(state) == END


@pytest.mark.asyncio
async def test_route_after_patch_continue() -> None:
    state: SoloGraphState = {"agent_state": {"awaiting_approval": False}, "events": [], "error": None}
    assert route_after_patch(state) == "subdirectory_hint"
