from __future__ import annotations

import pytest
from solo_agent.agent.deps import AgentSettings
from solo_agent.providers import ChatMessage
from solo_agent.tools import create_default_registry
from solo_agent.workflow.sandbox.command_workspace import prepare_command_workspace
from solo_agent.workflow.subagent_runner import SubagentRunner


class FakeProvider:
    name = "fake"
    model = "fake-model"

    async def complete(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        assert "read-only subagent" in messages[0].content
        assert "app.py" in messages[1].content
        return "- Service returns ok\n- Evidence came from app.py"


class FailingProvider:
    name = "fake"
    model = "fake-model"

    async def complete(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        raise RuntimeError("provider failed")


class QualityProvider:
    name = "fake"
    model = "fake-model"

    async def complete(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        assert "run_pytest" in messages[1].content
        return "- Quality checks passed"


@pytest.mark.asyncio
async def test_subagent_runner_returns_readonly_findings(tmp_path) -> None:
    (tmp_path / "app.py").write_text("def service():\n    return 'ok'\n", encoding="utf-8")
    registry = create_default_registry(tmp_path, subagent_enabled=True)
    runner = SubagentRunner(FakeProvider(), registry, AgentSettings(workspace_root=tmp_path))

    result = await runner.run(
        task_id="task_1",
        description="Inspect app",
        prompt="Inspect app.py and summarize service.",
        subagent_type="general-purpose",
        read_paths=["app.py"],
        allowed_tools=["read_file", "workspace_snapshot", "apply_text_edit", "task"],
        timeout_seconds=30,
        parent_session_id="s1",
        parent_run_id="r1",
    )

    assert result["status"] == "completed"
    assert result["task_id"] == "task_1"
    assert result["metadata"]["mode"] == "sync_child_agent"
    assert "apply_text_edit" not in result["metadata"]["allowed_tools"]
    assert "task" not in result["metadata"]["allowed_tools"]
    assert result["metadata"]["blocked_tools"] == ["apply_text_edit", "task"]
    assert result["findings"]
    assert result["evidence"]


@pytest.mark.asyncio
async def test_subagent_runner_returns_failed_result_on_provider_error(tmp_path) -> None:
    registry = create_default_registry(tmp_path, subagent_enabled=True)
    runner = SubagentRunner(FailingProvider(), registry, AgentSettings(workspace_root=tmp_path))

    result = await runner.run(
        task_id="task_2",
        description="Inspect app",
        prompt="Inspect app.py.",
        subagent_type="general-purpose",
        read_paths=["missing.py"],
        allowed_tools=["read_file"],
        timeout_seconds=30,
        parent_session_id="s1",
        parent_run_id="r1",
    )

    assert result["status"] == "failed"
    assert result["error"] == "provider failed"
    assert result["metadata"]["mode"] == "sync_child_agent"


@pytest.mark.asyncio
async def test_quality_subagent_runs_checks_in_isolated_command_workspace(tmp_path) -> None:
    (tmp_path / "test_demo.py").write_text("def test_ok():\n    assert True\n", encoding="utf-8")
    command_workspace = prepare_command_workspace(
        tmp_path,
        session_id="session-1",
        run_id="run-1",
        sandbox_mode="isolated",
    )
    registry = create_default_registry(
        tmp_path,
        subagent_enabled=True,
        command_workspace_root=command_workspace.command_workspace_root,
        sandbox_mode=command_workspace.mode,
    )
    runner = SubagentRunner(QualityProvider(), registry, AgentSettings(workspace_root=tmp_path, sandbox_mode="isolated"))

    result = await runner.run(
        task_id="task_quality",
        description="Run checks",
        prompt="Run pytest for this workspace.",
        subagent_type="quality",
        read_paths=[],
        allowed_tools=["run_pytest", "apply_text_edit", "task"],
        timeout_seconds=30,
        parent_session_id="s1",
        parent_run_id="r1",
    )

    assert result["status"] == "completed"
    assert result["allowed_tools_effective"] == ["run_pytest"]
    assert result["blocked_tools"] == ["apply_text_edit", "task"]
    assert result["sandbox"]["commands"][0]["mode"] == "isolated"
    assert result["sandbox"]["commands"][0]["workspace_root"] == str(command_workspace.command_workspace_root)
    assert not (tmp_path / ".pytest_cache").exists()
    command_workspace.cleanup()
