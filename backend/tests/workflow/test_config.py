"""Workflow configuration regression tests."""

from __future__ import annotations

import sys
import types
import warnings
from collections.abc import AsyncIterator
from dataclasses import fields as dataclass_fields
from pathlib import Path

from solo_agent.agent import AgentDeps, AgentSettings, run_agent_events
from solo_agent.agent.events import AgentEvent
from solo_agent.providers import ChatMessage, ProviderChunk
from solo_agent.settings import Settings


def test_workflow_engine_defaults():
    settings = Settings()
    assert settings.workflow_engine == "legacy"
    assert settings.workflow_checkpointer == "memory"

def test_workflow_engine_is_not_in_agent_settings():
    assert "workflow_engine" not in {field.name for field in dataclass_fields(AgentSettings)}


def test_workflow_runtime_has_no_old_graph_layers():
    src_root = Path(__file__).resolve().parents[2] / "src" / "solo_agent"
    files = [src_root / "agent" / "graph.py"]
    files.extend((src_root / "workflow").rglob("*.py"))
    combined = "\n".join(path.read_text(encoding="utf-8") for path in files)

    for forbidden in (
        "build_langgraph_topology",
        "_run_graph",
        "CompatibilityRunner",
        "_uses_mature_workflow_path",
        "_run_mature_workflow_path",
    ):
        assert forbidden not in combined


def test_default_subagent_enabled_is_true():
    settings = Settings()
    assert settings.subagent_enabled is True


def test_subagent_disabled():
    settings = Settings(subagent_enabled=False)
    assert settings.subagent_enabled is False


def test_default_max_concurrent_subagents_is_3():
    settings = Settings()
    assert settings.max_concurrent_subagents == 3


def test_max_concurrent_out_of_range_falls_back():
    with warnings.catch_warnings(record=True) as _w:
        warnings.simplefilter("always")
        settings = Settings(max_concurrent_subagents=15)
        assert settings.max_concurrent_subagents == 3


def test_max_concurrent_minimum():
    with warnings.catch_warnings(record=True) as _w:
        warnings.simplefilter("always")
        settings = Settings(max_concurrent_subagents=0)
        assert settings.max_concurrent_subagents == 3


def test_max_concurrent_valid():
    settings = Settings(max_concurrent_subagents=5)
    assert settings.max_concurrent_subagents == 5


def test_default_subagent_timeout_is_900():
    settings = Settings()
    assert settings.subagent_timeout_seconds == 900


def test_subagent_timeout_out_of_range_falls_back():
    with warnings.catch_warnings(record=True) as _w:
        warnings.simplefilter("always")
        settings = Settings(subagent_timeout_seconds=5000)
        assert settings.subagent_timeout_seconds == 900


def test_subagent_timeout_too_low_falls_back():
    with warnings.catch_warnings(record=True) as _w:
        warnings.simplefilter("always")
        settings = Settings(subagent_timeout_seconds=30)
        assert settings.subagent_timeout_seconds == 900


def test_default_sandbox_mode_is_local():
    settings = Settings()
    assert settings.sandbox_mode == "local"


def test_docker_mode_falls_back_to_local():
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        settings = Settings(sandbox_mode="docker")
        assert settings.sandbox_mode == "local"


def test_invalid_sandbox_mode_falls_back():
    with warnings.catch_warnings(record=True) as _w:
        warnings.simplefilter("always")
        settings = Settings(sandbox_mode="kubernetes")
        assert settings.sandbox_mode == "local"


def test_default_workflow_runtime_root():
    settings = Settings()
    assert settings.workflow_runtime_root == ".solo-agent/runs"


def test_custom_workflow_runtime_root():
    settings = Settings(workflow_runtime_root="/tmp/custom-runs")
    assert settings.workflow_runtime_root == "/tmp/custom-runs"


class FakeProvider:
    name = "fake"
    model = "fake"

    async def stream_chat(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[ProviderChunk]:
        if False:
            yield ProviderChunk()


def test_agent_settings_include_workflow_defaults():
    settings = AgentSettings()

    assert settings.subagent_enabled is True
    assert settings.max_concurrent_subagents == 3
    assert settings.subagent_timeout_seconds == 900
    assert settings.sandbox_mode == "local"
    assert settings.workflow_runtime_root == ".solo-agent/runs"


def test_run_agent_events_uses_workflow_runtime_without_workflow_engine(monkeypatch):
    seen: dict[str, object] = {}

    class FakeWorkflowRuntime:
        def __init__(self, *, deps, state, provider, **kwargs):
            seen["deps_settings"] = deps.settings
            seen["state"] = state
            seen["provider"] = provider
            seen["extra_kwargs"] = kwargs

        async def run(self):
            settings = seen["deps_settings"]
            yield AgentEvent(
                type="run_completed",
                session_id="session",
                run_id="run",
                node="workflow",
                data={
                    "run_mode": settings.run_mode,
                    "max_concurrent_subagents": settings.max_concurrent_subagents,
                    "workflow_runtime_root": str(settings.workflow_runtime_root),
                },
            )

    runtime_module = types.ModuleType("solo_agent.workflow.runtime")
    runtime_module.WorkflowRuntime = FakeWorkflowRuntime
    monkeypatch.setitem(sys.modules, "solo_agent.workflow.runtime", runtime_module)

    deps = AgentDeps(provider=FakeProvider())
    raw_settings = {
        "run_mode": "plan",
        "max_concurrent_subagents": 5,
        "workflow_runtime_root": ".tmp/workflow",
    }

    events = []

    async def collect() -> None:
        async for event in run_agent_events(
            "session",
            "run",
            "hello",
            deps=deps,
            settings=raw_settings,
        ):
            events.append(event)

    import asyncio

    asyncio.run(collect())

    assert isinstance(deps.settings, AgentSettings)
    assert seen["deps_settings"] is deps.settings
    assert seen["extra_kwargs"] == {}
    assert seen["state"].run_mode == "plan"
    assert "workflow_engine" not in raw_settings
    assert events[-1].data == {
        "run_mode": "plan",
        "max_concurrent_subagents": 5,
        "workflow_runtime_root": ".tmp/workflow",
    }
