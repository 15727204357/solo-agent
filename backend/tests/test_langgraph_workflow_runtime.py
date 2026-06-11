from __future__ import annotations

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from solo_agent.agent.state import AgentState
from solo_agent.tools.registry import create_default_registry
from solo_agent.workflow.graph_state import initial_graph_state
from solo_agent.workflow.graphs import build_main_workflow_graph
from solo_agent.workflow.stages import _TEAM_DEVELOPER_TOOL_NAMES, _sandbox_git_diff_result, _team_sandbox_registry_and_ledger


class ChatChunk:
    def __init__(self, content: str):
        self.content = content
        self.role = "assistant"


class FakeProvider:
    supports_tool_calling = False

    def __init__(self, plan: str = ""):
        self._plan = plan
        self.complete_calls = []

    async def stream_chat(self, messages, **kwargs):
        yield ChatChunk(self._plan)

    async def complete(self, messages, **kwargs):
        self.complete_calls.append({"messages": messages, "kwargs": dict(kwargs)})
        assert "tools" not in kwargs
        assert "tool_choice" not in kwargs
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
    subagent_policy = "off"
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


def test_team_sandbox_git_diff_compares_baseline_without_git(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    sandbox = tmp_path / "sandbox"
    workspace.mkdir()
    sandbox.mkdir()
    (workspace / "sample.txt").write_text("old value\n", encoding="utf-8")
    (sandbox / "sample.txt").write_text("new value\n", encoding="utf-8")

    result = _sandbox_git_diff_result(workspace, sandbox)
    payload = result["result"]

    assert result["ok"] is True
    assert payload["changed_files"] == ["sample.txt"]
    assert "-old value" in payload["diff"]
    assert "+new value" in payload["diff"]


def test_team_developer_ledger_allows_only_sandbox_coding_tools(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    sandbox = tmp_path / "sandbox"
    workspace.mkdir()
    sandbox.mkdir()
    (workspace / "sample.txt").write_text("old value\n", encoding="utf-8")
    (sandbox / "sample.txt").write_text("old value\n", encoding="utf-8")
    registry = create_default_registry(
        workspace,
        is_plan_mode=True,
        subagent_enabled=True,
        command_workspace_root=sandbox,
        sandbox_mode="isolated",
    )

    _, ledger = _team_sandbox_registry_and_ledger(registry, {"sandbox_mode": "isolated"})
    assert ledger is not None
    assert {spec.name for spec in ledger.allowed_specs()} == _TEAM_DEVELOPER_TOOL_NAMES

    prepared = ledger.call("prepare_edit", {"path": "sample.txt", "old_text": "old value"})
    expected_hash = prepared["result"]["expected_hash"]
    applied = ledger.call(
        "apply_text_edit",
        {"path": "sample.txt", "old_text": "old value", "new_text": "new value", "expected_hash": expected_hash},
    )
    blocked = ledger.call("create_file", {"path": "extra.txt", "content": "nope"})

    assert applied["ok"] is True
    assert blocked["ok"] is False
    assert (workspace / "sample.txt").read_text(encoding="utf-8") == "old value\n"
    assert (sandbox / "sample.txt").read_text(encoding="utf-8") == "new value\n"
    assert [call["name"] for call in ledger.calls] == ["prepare_edit", "apply_text_edit", "create_file"]


@pytest.mark.asyncio
async def test_graph_runtime_uses_serial_path_by_default() -> None:
    state = AgentState(
        session_id="s1",
        run_id="r1",
        user_input="implement feature",
    )
    gs = initial_graph_state(state)
    provider = FakeProvider(plan="1. Do step one\n2. Do step two\n3. Run tests")
    graph = build_main_workflow_graph(provider=provider, deps=FakeDeps(), settings=FakeSettings())
    compiled = graph.compile(checkpointer=InMemorySaver())

    config = {"configurable": {"thread_id": "r1"}}
    updates = [u async for u in compiled.astream(gs, config=config, stream_mode="values")]
    events = _deduplicated_events(updates)
    types = [e.get("type", "") for e in events]

    assert "plan_completed" in types
    assert "parallelism_gate_completed" not in types
    assert "team_plan_started" not in types


@pytest.mark.asyncio
async def test_serial_strategy_with_no_metadata() -> None:
    state = AgentState(
        session_id="s2",
        run_id="r2",
        user_input="fix everything",
    )
    gs = initial_graph_state(state)
    provider = FakeProvider(plan="1. Inspect the codebase.\n2. Fix the issue.\n3. Run pytest.")
    graph = build_main_workflow_graph(provider=provider, deps=FakeDeps(), settings=FakeSettings())
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
    graph = build_main_workflow_graph(provider=provider, deps=FakeDeps(), settings=FakeSettings())
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


@pytest.mark.asyncio
async def test_plan_subagent_mode_runs_team_workflow_and_proposes_patch(tmp_path) -> None:
    class TeamSettings(FakeSettings):
        run_mode = "plan"
        is_plan_mode = True
        subagent_policy = "auto"
        subagent_enabled = True
        verified_editing_enabled = True

    workspace = tmp_path / "workspace"
    sandbox = tmp_path / "sandbox"
    workspace.mkdir()
    sandbox.mkdir()
    (workspace / "sample.txt").write_text("old value\n", encoding="utf-8")
    (sandbox / "sample.txt").write_text("old value\n", encoding="utf-8")
    settings = TeamSettings()
    settings.workspace_root = workspace
    settings.sandbox_mode = "isolated"

    class TeamDeps(FakeDeps):
        tool_registry = create_default_registry(
            workspace,
            is_plan_mode=True,
            subagent_enabled=True,
            command_workspace_root=sandbox,
            sandbox_mode="isolated",
        )

    patch_json = """
{
  "summary": "Update sample text",
      "edits": [
        {
          "path": "sample.txt",
          "old_text": "old value",
          "new_text": "new value",
          "reason": "exercise team patch proposal"
        }
  ]
}
"""
    state = AgentState(
        session_id="s5",
        run_id="r5",
        user_input="update sample text",
    )
    gs = initial_graph_state(state)
    provider = FakeProvider(plan=patch_json)
    graph = build_main_workflow_graph(provider=provider, deps=TeamDeps(), settings=settings)
    compiled = graph.compile(checkpointer=InMemorySaver())

    config = {"configurable": {"thread_id": "r5"}}
    updates = [u async for u in compiled.astream(gs, config=config, stream_mode="values")]
    events = _deduplicated_events(updates)
    types = [e.get("type", "") for e in events]
    final_state = next(
        update["agent_state"]
        for update in reversed(updates)
        if isinstance(update, dict) and "agent_state" in update
    )

    assert "team_plan_started" in types
    assert "team_developer_completed" in types
    assert "team_tester_completed" in types
    assert "team_supervisor_completed" in types
    assert "parallel_dispatch_started" not in types
    assert final_state["supervisor_report"]["status"] == "patch_proposed"
    assert final_state["patch_proposal"]["status"] == "pending"
    assert "-old value" in final_state["patch_proposal"]["diff"]
    assert "+new value" in final_state["patch_proposal"]["diff"]
    assert provider.complete_calls
    assert all("tools" not in call["kwargs"] for call in provider.complete_calls)
    assert (workspace / "sample.txt").read_text(encoding="utf-8") == "old value\n"
    assert (sandbox / "sample.txt").read_text(encoding="utf-8") == "new value\n"
