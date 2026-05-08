from __future__ import annotations

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from solo_agent.agent.state import AgentState
from solo_agent.workflow.graph_state import initial_graph_state
from solo_agent.workflow.graphs import build_text_provider_graph


class ChatChunk:
    def __init__(self, content: str):
        self.content = content
        self.role = "assistant"


class FakeProvider:
    supports_tool_calling = False

    def __init__(self, plan: str = ""):
        self._plan = plan

    async def stream_chat(self, messages, **kwargs):
        yield ChatChunk(self._plan)

    async def complete(self, messages, **kwargs):
        return self._plan or ""


class FakeSettings:
    workflow_engine = "langgraph"
    workflow_checkpointer = "memory"
    workflow_checkpoint_path = ".solo-agent/checkpoints/test.sqlite3"
    memory_enabled = False
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
    verified_editing_enabled = False
    workspace_root = "."
    summary_trigger_messages = 8
    plan_deep_max_tokens = 6000
    subagent_enabled = False
    workflow_runtime_root = ".solo-agent/runs"
    subagent_timeout_seconds = 900
    sandbox_mode = "local"
    run_mode = "agent"
    context_window_tokens = 128_000
    context_regular_threshold = 0.80
    context_long_task_threshold = 0.50
    context_long_task_after_compressions = 2
    context_tool_output_cutoff = 10
    auxiliary_compression_provider = "ollama"
    auxiliary_compression_model = "qwen3.5:4b"
    auxiliary_compression_base_url = "http://localhost:11434"
    tool_call_cut_off = 3


class FakeDeps:
    settings = FakeSettings()
    persistence = None
    tool_registry = None
    safety_inspector = None
    context_provider = None


def _deduplicated_events(updates) -> list[dict]:
    seen: set[tuple[str, str, str]] = set()
    events: list[dict] = []
    for update in updates:
        if isinstance(update, dict):
            for ed in update.get("events", []):
                if isinstance(ed, dict):
                    key = (ed.get("type", ""), ed.get("session_id", ""), ed.get("created_at", ""))
                    if key not in seen:
                        seen.add(key)
                        events.append(ed)
    return events


@pytest.mark.asyncio
async def test_graph_runtime_emits_parallelism_gate_completed() -> None:
    state = AgentState(
        session_id="s1",
        run_id="r1",
        user_input="implement feature",
    )
    gs = initial_graph_state(state)
    provider = FakeProvider(plan="1. Do step one\n2. Do step two\n3. Run tests")
    graph = build_text_provider_graph(provider=provider, deps=FakeDeps(), settings=FakeSettings())
    compiled = graph.compile(checkpointer=InMemorySaver())

    config = {"configurable": {"thread_id": "r1"}}
    updates = [u async for u in compiled.astream(gs, config=config, stream_mode="values")]
    events = _deduplicated_events(updates)
    types = [e.get("type", "") for e in events]

    assert "parallelism_gate_completed" in types


@pytest.mark.asyncio
async def test_serial_strategy_with_no_metadata() -> None:
    state = AgentState(
        session_id="s2",
        run_id="r2",
        user_input="fix everything",
    )
    gs = initial_graph_state(state)
    provider = FakeProvider(plan="1. Inspect the codebase.\n2. Fix the issue.\n3. Run pytest.")
    graph = build_text_provider_graph(provider=provider, deps=FakeDeps(), settings=FakeSettings())
    compiled = graph.compile(checkpointer=InMemorySaver())

    final_state = None
    config = {"configurable": {"thread_id": "r2"}}
    async for update in compiled.astream(gs, config=config, stream_mode="values"):
        if isinstance(update, dict) and "agent_state" in update:
            final_state = update["agent_state"]

    assert final_state is not None
    assert final_state.get("execution_strategy") == "serial"


@pytest.mark.asyncio
async def test_agent_state_is_restored_after_stream() -> None:
    state = AgentState(
        session_id="s4",
        run_id="r4",
        user_input="test state restoration",
    )
    gs = initial_graph_state(state)
    provider = FakeProvider(plan="Do something")
    graph = build_text_provider_graph(provider=provider, deps=FakeDeps(), settings=FakeSettings())
    compiled = graph.compile(checkpointer=InMemorySaver())

    final_state = None
    config = {"configurable": {"thread_id": "r4"}}
    async for update in compiled.astream(gs, config=config, stream_mode="values"):
        if isinstance(update, dict) and "agent_state" in update:
            final_state = update["agent_state"]

    assert final_state is not None
    assert final_state.get("session_id") == "s4"
    assert final_state.get("run_id") == "r4"
    assert final_state.get("user_input") == "test state restoration"
