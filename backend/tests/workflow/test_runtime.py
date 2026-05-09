from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from pathlib import Path

import pytest
from solo_agent.agent import AgentDeps, AgentSettings, AgentState
from solo_agent.providers import ChatMessage, ProviderChunk, ProviderResponse, ProviderToolCall
from solo_agent.tools import create_default_registry
from solo_agent.workflow.runtime import WorkflowRuntime


class ToolCallingFakeProvider:
    name = "fake-tools"
    model = "fake-model"
    supports_tool_calling = True

    async def complete(self, messages: list[ChatMessage], *, temperature=None, max_tokens=None) -> str:
        return "plain response"

    async def complete_message(
        self,
        messages: list[ChatMessage],
        *,
        temperature=None,
        max_tokens=None,
        tools: Sequence[dict] | None = None,
        tool_choice=None,
    ) -> ProviderResponse:
        tool_names = {
            str((tool.get("function") or {}).get("name"))
            for tool in tools or ()
            if isinstance(tool, dict)
        }
        if "task" not in tool_names:
            return ProviderResponse(content="Subagent result")
        if any(message.role == "tool" for message in messages):
            return ProviderResponse(content="Lead final response")
        return ProviderResponse(
            content="",
            tool_calls=(
                ProviderToolCall(
                    id="call_task_1",
                    name="task",
                    arguments={
                        "description": "Inspect project",
                        "prompt": "Summarize the project read-only.",
                        "subagent_type": "general-purpose",
                        "max_turns": 2,
                    },
                ),
            ),
            finish_reason="tool_calls",
        )


class NonToolCallingFakeProvider:
    name = "fake-no-tools"
    model = "fake-model"
    supports_tool_calling = False

    async def stream_chat(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[ProviderChunk]:
        system = messages[0].content if messages else ""
        if system.startswith("You are Solo Agent, a transparent"):
            yield ProviderChunk(content="1. Inspect project\n2. Answer from context")
        else:
            yield ProviderChunk(content="Text-provider final response")

    async def complete(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        return "memory summary"


class PlanModeFakeProvider(NonToolCallingFakeProvider):
    async def stream_chat(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[ProviderChunk]:
        system = messages[0].content if messages else ""
        if "deep planning mode" in system:
            yield ProviderChunk(
                content=(
                    "## Summary\nCreate a focused runtime compatibility plan.\n\n"
                    "## File Map\n"
                    "| File Path | Action | Purpose |\n"
                    "|-----------|--------|---------|\n"
                    "| backend/src/solo_agent/workflow/runtime.py | MODIFY | Route runtime modes. |\n\n"
                    "## Steps\n"
                    "1. Command: `python -m pytest backend/tests/workflow/test_runtime.py -q`\n"
                    "   Expected Output: Runtime compatibility tests pass.\n"
                    "   Success Criteria: Plan and runtime strategy events remain stable.\n"
                    "   Files Affected: backend/src/solo_agent/workflow/runtime.py.\n\n"
                    "## Verification\n"
                    "`python -m pytest backend/tests/workflow/test_runtime.py -q`\n\n"
                    "## Execution Options\n"
                    "Single Agent is recommended because the runtime change is bounded.\n\n"
                    "## Self-Review\n"
                    "No placeholders; every path and command is concrete.\n"
                )
            )
        else:
            yield ProviderChunk(content="Text-provider final response")

    async def complete(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        system = messages[0].content if messages else ""
        if "plan quality reviewer" in system:
            return '{"passed":true,"issues":[],"summary":"All checks passed."}'
        return "memory summary"


class PatchApprovalFakeProvider(NonToolCallingFakeProvider):
    async def complete(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        return (
            '{"summary":"Update greeting","edits":[{"path":"app.py",'
            '"old_text":"hello","new_text":"hi","reason":"demo"}]}'
        )


@pytest.mark.asyncio
async def test_workflow_runtime_plan_mode_uses_unified_plan_strategy(tmp_path: Path) -> None:
    settings = AgentSettings(
        provider="ollama",
        model="fake-model",
        run_mode="plan",
        workflow_runtime_root=tmp_path / "runs",
        memory_enabled=False,
    )
    provider = PlanModeFakeProvider()
    deps = AgentDeps(
        provider=provider,
        tool_registry=create_default_registry(tmp_path),
        settings=settings,
    )

    events = [
        event
        async for event in WorkflowRuntime(
            deps=deps,
            state=AgentState("session-1", "run-1", "Plan runtime compatibility."),
            provider=provider,
        ).run()
    ]

    event_types = [event.type for event in events]
    assert "deep_plan_started" in event_types
    assert "plan_self_review_completed" in event_types
    assert "response_completed" in event_types
    assert "task_started" not in event_types
    assert events[-1].type == "run_completed"


@pytest.mark.asyncio
async def test_workflow_runtime_text_provider_strategy_keeps_patch_approval_event(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("hello\n", encoding="utf-8")
    settings = AgentSettings(
        provider="ollama",
        model="fake-model",
        workflow_runtime_root=tmp_path / "runs",
        memory_enabled=False,
        verified_editing_enabled=True,
    )
    provider = PatchApprovalFakeProvider()
    deps = AgentDeps(
        provider=provider,
        tool_registry=create_default_registry(tmp_path),
        settings=settings,
    )

    events = [
        event
        async for event in WorkflowRuntime(
            deps=deps,
            state=AgentState("session-1", "run-1", "fix app.py greeting"),
            provider=provider,
        ).run()
    ]

    assert (tmp_path / "app.py").read_text(encoding="utf-8") == "hello\n"
    assert any(event.type == "patch_proposed" for event in events)
    assert events[-1].type == "patch_approval_required"


@pytest.mark.asyncio
async def test_workflow_runtime_non_tool_calling_provider_uses_text_strategy(tmp_path: Path) -> None:
    settings = AgentSettings(
        provider="ollama",
        model="fake-model",
        workflow_runtime_root=tmp_path / "runs",
        memory_enabled=False,
    )
    provider = NonToolCallingFakeProvider()
    deps = AgentDeps(
        provider=provider,
        tool_registry=create_default_registry(tmp_path),
        settings=settings,
    )

    events = [
        event
        async for event in WorkflowRuntime(
            deps=deps,
            state=AgentState("session-1", "run-1", "Summarize this project."),
            provider=provider,
        ).run()
    ]

    event_types = [event.type for event in events]
    assert "receive_user_turn" in event_types
    assert "plan_completed" in event_types
    assert "response_completed" in event_types
    assert "task_started" not in event_types
    assert events[-1].type == "run_completed"
