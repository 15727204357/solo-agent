"""Tests for subagent registry, task tool, factories, and executor lifecycle."""

from __future__ import annotations

import asyncio

import pytest
from solo_agent.agent.state import AgentState
from solo_agent.tools.registry import ToolSpec
from solo_agent.workflow.state import WorkflowState
from solo_agent.workflow.subagent.executor import SubagentExecutor
from solo_agent.workflow.subagent.factories import (
    TOOL_WHITELISTS,
    get_system_prompt,
    get_tool_whitelist,
    register_builtin_factories,
)
from solo_agent.workflow.subagent.registry import SubagentRegistry, get_builtin_registry
from solo_agent.workflow.subagent.task_tool import (
    TaskToolHandler,
    is_subagent_context,
    set_subagent_context,
)


def test_registry_register_and_get():
    registry = SubagentRegistry()
    registry.register("test-agent", lambda: "test")
    assert registry.get("test-agent") is not None
    assert registry.has("test-agent")
    assert "test-agent" in registry.list_types()


def test_registry_unknown_raises():
    registry = SubagentRegistry()
    with pytest.raises(ValueError, match="Unknown subagent type"):
        registry.get("nonexistent")


def test_builtin_registry_has_default_types():
    registry = register_builtin_factories(get_builtin_registry())
    assert isinstance(registry, SubagentRegistry)
    assert {"general-purpose", "code-review", "quality"}.issubset(registry.list_types())


def test_tool_whitelist_has_all_three_types():
    assert "general-purpose" in TOOL_WHITELISTS
    assert "code-review" in TOOL_WHITELISTS
    assert "quality" in TOOL_WHITELISTS


def test_code_review_whitelist_is_readonly():
    whitelist = get_tool_whitelist("code-review")
    assert "read_file" in whitelist
    assert "search_text" in whitelist
    assert "apply_text_edit" not in whitelist


def test_quality_whitelist_has_readonly_quality_tools():
    whitelist = get_tool_whitelist("quality")
    assert "run_pytest" in whitelist
    assert "run_ruff_check" in whitelist


def test_system_prompts_not_empty_and_block_task():
    for stype in ("general-purpose", "code-review", "quality"):
        prompt = get_system_prompt(stype)
        assert prompt
        assert "task tool" in prompt


@pytest.fixture
def wf_state():
    agent_state = AgentState(session_id="s1", run_id="r1", user_input="test")
    return WorkflowState.from_agent_state(agent_state)


def test_task_tool_blocked_in_subagent_context():
    set_subagent_context(True)
    try:
        handler = TaskToolHandler()
        result = handler.handle(description="test")
        assert result["ok"] is False
        assert result["code"] == "recursive_task_blocked"
    finally:
        set_subagent_context(False)


def test_task_tool_requires_executor_in_lead_context():
    set_subagent_context(False)
    handler = TaskToolHandler()
    result = handler.handle(description="test")
    assert result["ok"] is False
    assert result["code"] == "executor_not_ready"


@pytest.mark.asyncio
async def test_task_tool_dispatches_registered_runner_and_executor_waits(wf_state):
    async def fake_runner(prompt, max_turns=10, **kwargs):
        await asyncio.sleep(0)
        return f"done: {prompt}:{max_turns}"

    registry = SubagentRegistry()
    registry.register("test-agent", lambda **kwargs: fake_runner)
    executor = SubagentExecutor(max_concurrent=3, timeout_seconds=5)
    handler = TaskToolHandler()
    handler.bind(executor, wf_state, registry=registry)

    result = handler.handle(description="check this", subagent_type="test-agent", max_turns=4)
    assert result["ok"] is True
    assert executor.active_count == 1

    await executor.wait_for_all()
    record = wf_state.subagent_runs[result["task_id"]]
    assert record.status == "completed"
    assert record.result == "done: check this:4"
    assert executor.active_count == 0


@pytest.mark.asyncio
async def test_executor_has_active_count():
    executor = SubagentExecutor(max_concurrent=3, timeout_seconds=10)
    assert executor.active_count == 0


@pytest.mark.asyncio
async def test_executor_cancel_nonexistent_does_not_raise():
    executor = SubagentExecutor(max_concurrent=3)
    executor.cancel("nonexistent")


@pytest.mark.asyncio
async def test_executor_runs_subagent_to_completion(wf_state):
    async def fake_runner(prompt, max_turns=10, **kwargs):
        return "subagent result"

    executor = SubagentExecutor(max_concurrent=3, timeout_seconds=5)
    task_id = "sub_1"

    await executor.run_subagent(
        state=wf_state,
        subagent_type="general-purpose",
        task_id=task_id,
        runner=fake_runner,
        prompt="test prompt",
        max_turns=5,
    )

    assert task_id in wf_state.subagent_runs
    assert wf_state.subagent_runs[task_id].status == "completed"
    assert "subagent result" in wf_state.subagent_runs[task_id].result


@pytest.mark.asyncio
async def test_executor_subagent_timeout(wf_state):
    async def slow_runner(prompt, max_turns=10, **kwargs):
        await asyncio.sleep(5)
        return "never reached"

    executor = SubagentExecutor(max_concurrent=3, timeout_seconds=0.1)
    task_id = "sub_timeout"

    await executor.run_subagent(
        state=wf_state,
        subagent_type="general-purpose",
        task_id=task_id,
        runner=slow_runner,
        prompt="slow",
    )

    assert wf_state.subagent_runs[task_id].status == "failed"
    assert wf_state.subagent_runs[task_id].error == "timeout"


@pytest.mark.asyncio
async def test_executor_emits_subagent_events(wf_state):
    async def fake_runner(prompt, max_turns=10, **kwargs):
        return "ok"

    queue = asyncio.Queue()
    executor = SubagentExecutor(max_concurrent=3, timeout_seconds=5, event_queue=queue)

    await executor.run_subagent(
        state=wf_state,
        subagent_type="general-purpose",
        task_id="sub_events",
        runner=fake_runner,
        prompt="event prompt",
    )

    events = []
    while not queue.empty():
        events.append(queue.get_nowait().type)
    assert events == ["task_started", "task_completed"]


@pytest.mark.asyncio
async def test_contextvar_recursion_guard_is_per_task():
    async def marked_context():
        set_subagent_context(True)
        await asyncio.sleep(0)
        return is_subagent_context()

    async def default_context():
        await asyncio.sleep(0)
        return is_subagent_context()

    marked, default = await asyncio.gather(marked_context(), default_context())
    assert marked is True
    assert default is False
    set_subagent_context(False)


def test_builtin_quality_factory_filters_out_write_tools(monkeypatch):
    import solo_agent.workflow.subagent.factories as factories

    seen_tools = []

    def fake_create_react_agent(*, model, tools, prompt):
        seen_tools.extend(tool.name for tool in tools)

        class FakeAgent:
            async def astream(self, *args, **kwargs):
                if False:
                    yield {}

        return FakeAgent()

    monkeypatch.setattr(factories, "create_react_agent", fake_create_react_agent)

    class FakeRegistry:
        _tools = {
            "run_pytest": ToolSpec("run_pytest", "pytest", True, lambda **kw: {}, {}),
            "apply_text_edit": ToolSpec("apply_text_edit", "edit", False, lambda **kw: {}, {}),
        }

    class FakeModel:
        pass

    registry = SubagentRegistry()
    register_builtin_factories(registry)
    runner = registry.get("quality")(model=FakeModel(), tool_registry=FakeRegistry())
    assert callable(runner)
    assert seen_tools == ["run_pytest"]
