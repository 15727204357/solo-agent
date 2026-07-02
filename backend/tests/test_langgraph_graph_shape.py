from __future__ import annotations

import pytest
from langgraph.graph import END
from solo_agent.workflow.graph_state import SoloGraphState
from solo_agent.workflow.graphs import (
    _error_aware_route,
    _has_error,
    build_main_workflow_graph,
    make_route_after_guard,
    route_after_execute_tools,
    route_after_parallelism_gate,
    route_after_patch,
    route_after_supervisor_review,
    route_after_task_state,
    route_after_team_supervisor,
    route_after_team_test,
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
    skill_evolution_enabled = True
    skill_evolution_min_confidence = 0.72
    skill_evolution_max_proposals_per_run = 1
    intent_router_mode = "shadow_hybrid"
    intent_router_max_epochs = 3
    intent_router_model_timeout_seconds = 1.5


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
    graph = build_main_workflow_graph(provider=FakeProvider(), deps=FakeDeps(), settings=FakeSettings())
    compiled = graph.compile(checkpointer=False)
    assert compiled is not None


@pytest.mark.asyncio
async def test_graph_compiles_with_memory_checkpointer() -> None:
    graph = build_main_workflow_graph(provider=FakeProvider(), deps=FakeDeps(), settings=FakeSettings())
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
    assert route_after_parallelism_gate(state) == "team_develop"


@pytest.mark.asyncio
async def test_route_after_parallelism_gate_serial() -> None:
    state: SoloGraphState = {"agent_state": {"execution_strategy": "serial", "blocked": False}, "events": [], "error": None}
    assert route_after_parallelism_gate(state) == "team_develop"


@pytest.mark.asyncio
async def test_route_after_execute_tools_awaiting_approval() -> None:
    state: SoloGraphState = {"agent_state": {"awaiting_approval": True}, "events": [], "error": None}
    assert route_after_execute_tools(state) == END


@pytest.mark.asyncio
async def test_route_after_execute_tools_continue() -> None:
    state: SoloGraphState = {"agent_state": {"awaiting_approval": False}, "events": [], "error": None}
    assert route_after_execute_tools(state) == "spec_compliance_review"


@pytest.mark.asyncio
async def test_route_after_execute_tools_reroutes_on_no_results() -> None:
    state: SoloGraphState = {
        "agent_state": {
            "awaiting_approval": False,
            "route_epoch": 0,
            "snapshots": {"intent_router": {"max_epochs": 3}},
            "tool_calls": [{"name": "search_text", "result": {"matches": []}}],
        },
        "events": [],
        "error": None,
    }
    assert route_after_execute_tools(state) == "intent_route"


@pytest.mark.asyncio
async def test_route_after_execute_tools_honors_pending_reroute_request() -> None:
    state: SoloGraphState = {
        "agent_state": {
            "awaiting_approval": False,
            "route_epoch": 3,
            "snapshots": {
                "intent_router": {"max_epochs": 3},
                "pending_reroute": {"reason": "tool_no_results"},
            },
            "tool_calls": [],
        },
        "events": [],
        "error": None,
    }
    assert route_after_execute_tools(state) == "intent_route"


@pytest.mark.asyncio
async def test_route_after_task_state_team_mode_requires_plan_and_subagent() -> None:
    state: SoloGraphState = {
        "agent_state": {"run_mode": "plan", "is_plan_mode": True, "subagent_enabled": True},
        "events": [],
        "error": None,
    }
    assert route_after_task_state(state) == "team_plan"


@pytest.mark.asyncio
async def test_route_after_task_state_serial_without_double_switch() -> None:
    state: SoloGraphState = {
        "agent_state": {"run_mode": "plan", "is_plan_mode": True, "subagent_enabled": False},
        "events": [],
        "error": None,
    }
    assert route_after_task_state(state) == "collect_context"


@pytest.mark.asyncio
async def test_route_after_supervisor_review_passed() -> None:
    state: SoloGraphState = {
        "agent_state": {"supervisor_report": {"status": "passed"}},
        "events": [],
        "error": None,
    }
    assert route_after_supervisor_review(state) == "spec_compliance_review"


@pytest.mark.asyncio
async def test_route_after_supervisor_review_fallbacks_to_serial() -> None:
    state: SoloGraphState = {
        "agent_state": {"supervisor_report": {"status": "fallback_serial"}},
        "events": [],
        "error": None,
    }
    assert route_after_supervisor_review(state) == "collect_context"


@pytest.mark.asyncio
async def test_route_after_team_test_loops_until_max_iterations() -> None:
    state: SoloGraphState = {
        "agent_state": {
            "review_reports": {
                "team_test": {"status": "needs_fix", "iteration": 1, "max_iterations": 2},
            },
        },
        "events": [],
        "error": None,
    }
    assert route_after_team_test(state) == "team_develop"


@pytest.mark.asyncio
async def test_route_after_team_test_supervises_at_max_iterations() -> None:
    state: SoloGraphState = {
        "agent_state": {
            "review_reports": {
                "team_test": {"status": "needs_fix", "iteration": 2, "max_iterations": 2},
            },
        },
        "events": [],
        "error": None,
    }
    assert route_after_team_test(state) == "team_supervisor"


@pytest.mark.asyncio
async def test_route_after_team_supervisor_awaiting_approval_ends() -> None:
    state: SoloGraphState = {"agent_state": {"awaiting_approval": True}, "events": [], "error": None}
    assert route_after_team_supervisor(state) == END


@pytest.mark.asyncio
async def test_route_after_patch_awaiting_approval() -> None:
    state: SoloGraphState = {"agent_state": {"awaiting_approval": True}, "events": [], "error": None}
    assert route_after_patch(state) == END


@pytest.mark.asyncio
async def test_route_after_patch_continue() -> None:
    state: SoloGraphState = {"agent_state": {"awaiting_approval": False}, "events": [], "error": None}
    assert route_after_patch(state) == "subdirectory_hint"


@pytest.mark.asyncio
async def test_has_error_detects_error_state() -> None:
    state: SoloGraphState = {"agent_state": {}, "events": [], "error": {"error_type": "ValueError", "error_message": "test"}}
    assert _has_error(state) is True


@pytest.mark.asyncio
async def test_has_error_none_when_clean() -> None:
    state: SoloGraphState = {"agent_state": {}, "events": [], "error": None}
    assert _has_error(state) is False


@pytest.mark.asyncio
async def test_error_aware_route_diverts_on_error() -> None:
    state: SoloGraphState = {"agent_state": {}, "events": [], "error": {"error_type": "TestError"}}
    assert _error_aware_route(state, "next_node") == "classify_error"


@pytest.mark.asyncio
async def test_error_aware_route_passes_through_when_clean() -> None:
    state: SoloGraphState = {"agent_state": {}, "events": [], "error": None}
    assert _error_aware_route(state, "next_node") == "next_node"


@pytest.mark.asyncio
async def test_graph_contains_error_recovery_nodes() -> None:
    graph = build_main_workflow_graph(provider=FakeProvider(), deps=FakeDeps(), settings=FakeSettings())
    compiled = graph.compile(checkpointer=False)
    assert compiled is not None
    node_names = list(graph.nodes.keys())
    assert "classify_error" in node_names
    assert "recovery_action" in node_names
    assert "repetition_guard" in node_names
    assert "skill_evolution" in node_names


@pytest.mark.asyncio
async def test_graph_uses_single_main_plan_path() -> None:
    graph = build_main_workflow_graph(provider=FakeProvider(), deps=FakeDeps(), settings=FakeSettings())
    compiled = graph.compile(checkpointer=False)
    assert compiled is not None
    node_names = list(graph.nodes.keys())
    assert "plan" in node_names
    assert "load_task_state" in node_names
    assert "task_state" in node_names
    assert "plan_response" not in node_names


@pytest.mark.asyncio
async def test_graph_keeps_legacy_fanout_nodes_unwired() -> None:
    graph = build_main_workflow_graph(provider=FakeProvider(), deps=FakeDeps(), settings=FakeSettings())
    node_names = set(graph.nodes.keys())
    edges = {(str(edge[0]), str(edge[1])) for edge in graph.edges}

    assert "parallel_dispatch" in node_names
    assert "wait_subagents" in node_names
    assert "supervisor_review" in node_names
    assert ("team_plan", "parallelism_gate") in edges
    assert ("parallel_dispatch", "wait_subagents") not in edges
    assert ("wait_subagents", "supervisor_review") not in edges


@pytest.mark.asyncio
async def test_graph_registers_lightweight_team_nodes() -> None:
    graph = build_main_workflow_graph(provider=FakeProvider(), deps=FakeDeps(), settings=FakeSettings())
    node_names = set(graph.nodes.keys())

    assert "team_plan" in node_names
    assert "team_develop" in node_names
    assert "team_test" in node_names
    assert "team_supervisor" in node_names
    assert "intent_route" in node_names
