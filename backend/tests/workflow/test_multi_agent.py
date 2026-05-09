"""Tests for supervisor LLM routing and subagent integration."""

from __future__ import annotations

import pytest
from solo_agent.workflow.supervisor_nodes import _normalize_routing_key


def test_normalize_basic_keys():
    assert _normalize_routing_key("continue_explore") == "continue_explore"
    assert _normalize_routing_key("summarize_return") == "summarize_return"
    assert _normalize_routing_key("fallback_serial") == "fallback_serial"


def test_normalize_with_punctuation():
    assert _normalize_routing_key('"continue_explore"') == "continue_explore"
    assert _normalize_routing_key("'summarize_return'") == "summarize_return"
    assert _normalize_routing_key("continue_explore.") == "continue_explore"


def test_normalize_whitespace():
    assert _normalize_routing_key("  summarize_return  ") == "summarize_return"
    assert _normalize_routing_key("CONTINUE_EXPLORE") == "continue_explore"


def test_normalize_invalid_falls_back():
    assert _normalize_routing_key("garbage") == "summarize_return"
    assert _normalize_routing_key("") == "summarize_return"
    assert _normalize_routing_key("unknown_decision") == "summarize_return"


@pytest.mark.asyncio
async def test_evaluate_node_loop_limit():
    """Verify the evaluate node enforces loop limit at >= 2 cycles."""
    from solo_agent.agent.state import CoordinatorState
    from solo_agent.workflow.graph_state import (
        SoloGraphState,
        coordinator_state_to_graph_data,
    )
    from solo_agent.workflow.supervisor_nodes import make_evaluate_results_node

    state = CoordinatorState(session_id="s1", run_id="r1", user_input="test")
    state.plan = "research plan"
    state.subagent_results = {"t1": {"findings": "some result"}}

    # Fake provider that records calls
    calls = []
    class FakeProvider:
        async def complete(self, messages, **kwargs):
            calls.append(messages)
            return "summarize_return"

    settings = type("obj", (), {"temperature": 0.1, "max_tokens": 50})()
    node_fn = make_evaluate_results_node(FakeProvider(), settings)

    # Cycle 0: should call LLM
    gs: SoloGraphState = {"agent_state": coordinator_state_to_graph_data(state), "events": [], "error": None}
    gs["agent_state"]["snapshots"]["exploration_loop_count"] = 0
    result = await node_fn(gs)
    assert len(calls) == 1
    assert result["agent_state"]["snapshots"]["exploration_decision"] == "summarize_return"

    # Cycle 2 (>=2): should skip LLM and force return
    gs["agent_state"]["snapshots"]["exploration_loop_count"] = 2
    calls.clear()
    result = await node_fn(gs)
    assert len(calls) == 0  # should not call LLM
    assert result["agent_state"]["snapshots"]["exploration_decision"] == "summarize_return"


@pytest.mark.asyncio
async def test_coordinator_graph_compiles():
    """Verify the coordinator graph compiles successfully."""
    from solo_agent.workflow.coordinator_graph import build_coordinator_graph

    class FakeSettings:
        workflow_checkpointer = "memory"
        workflow_checkpoint_path = ".solo-agent/test.sqlite3"
        memory_enabled = True
        conversation_history_enabled = True
        max_concurrent_subagents = 2
        subagent_timeout_seconds = 10
        history_message_limit = 12
        memory_search_limit = 5
        max_tool_calls = 3
        tool_call_cut_off = 3
        tool_output_max_bytes = 12000
        max_selected_skills = 3
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

    class FakeProvider:
        supports_tool_calling = False
        async def stream_chat(self, messages, **kwargs):
            yield type("obj", (), {"content": ""})()
        async def complete(self, messages, **kwargs):
            return ""

    class FakeDeps:
        settings = FakeSettings()
        persistence = None
        tool_registry = None
        safety_inspector = None
        context_provider = None

    graph = build_coordinator_graph(
        provider=FakeProvider(), deps=FakeDeps(), settings=FakeSettings(),
    )
    compiled = graph.compile(checkpointer=False)
    assert compiled is not None
    assert "receive_user_turn" in graph.nodes
    assert "generate_research_plan" in graph.nodes
    assert "dispatch_researchers" in graph.nodes
    assert "evaluate_results" in graph.nodes
    assert "respond" in graph.nodes
