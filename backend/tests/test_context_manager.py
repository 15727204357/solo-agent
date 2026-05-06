from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import pytest
from solo_agent.agent.state import ToolCallRecord
from solo_agent.context import AuxiliaryClient, ContextManager
from solo_agent.providers import ChatMessage, ProviderChunk


class FakeProvider:
    name = "fake"

    def __init__(self, model: str) -> None:
        self.model = model
        self.complete_calls: list[list[ChatMessage]] = []
        self.stream_calls = 0
        self.registry_touched = False

    async def stream_chat(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[ProviderChunk]:
        self.stream_calls += 1
        yield ProviderChunk(content="stream should not be used")

    async def complete(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        self.complete_calls.append(messages)
        return "中文压缩摘要"


class FixedEstimator:
    def __init__(self, tokens: int) -> None:
        self.tokens = tokens

    def estimate_state(self, state: Any) -> int:
        return self.tokens


@dataclass
class FakeState:
    user_input: str = "fix tests"
    plan: str = "inspect failure"
    context: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    compression_count: int = 0
    context_summary: str = ""
    snapshots: dict[str, Any] = field(default_factory=dict)


def test_evaluate_uses_main_threshold_until_third_compression() -> None:
    manager = ContextManager(settings={"context_token_budget": 1000}, main_provider=FakeProvider("main-model"))
    state = FakeState(compression_count=2)

    report = manager.evaluate(state, estimator=FixedEstimator(799))
    assert report.threshold_ratio == 0.80
    assert report.provider_role == "main"
    assert report.provider_model == "main-model"
    assert report.should_compress is False

    report = manager.evaluate(state, estimator=FixedEstimator(800))
    assert report.should_compress is True


def test_evaluate_switches_to_auxiliary_after_two_compressions() -> None:
    manager = ContextManager(
        settings={"context_token_budget": 1000},
        main_provider=FakeProvider("main-model"),
        auxiliary_provider=FakeProvider("qwen3.5:4b"),
    )
    state = FakeState(compression_count=3)

    report = manager.evaluate(state, estimator=FixedEstimator(500))

    assert report.threshold_ratio == 0.50
    assert report.provider_role == "auxiliary"
    assert report.provider_model == "qwen3.5:4b"
    assert report.should_compress is True


@pytest.mark.asyncio
async def test_maybe_compress_calls_complete_only_and_increments_count() -> None:
    provider = FakeProvider("main-model")
    manager = ContextManager(
        settings={"context_token_budget": 1000, "summary_max_tokens": 200},
        main_provider=provider,
    )
    state = FakeState(compression_count=1)

    result = await manager.maybe_compress(state, estimator=FixedEstimator(900))

    assert result.compressed is True
    assert result.compression_count == 2
    assert state.compression_count == 2
    assert state.context_summary == "中文压缩摘要"
    assert len(provider.complete_calls) == 1
    assert provider.stream_calls == 0
    assert "Write the final summary in Chinese" in provider.complete_calls[0][0].content
    assert "Chinese summary" in provider.complete_calls[0][1].content


@pytest.mark.asyncio
async def test_maybe_compress_uses_auxiliary_provider_after_two_compressions() -> None:
    main = FakeProvider("main-model")
    auxiliary = FakeProvider("qwen3.5:4b")
    manager = ContextManager(
        settings={"context_token_budget": 1000},
        main_provider=main,
        auxiliary_provider=auxiliary,
    )
    state = FakeState(compression_count=3)

    result = await manager.maybe_compress(state, estimator=FixedEstimator(900))

    assert result.provider_role == "auxiliary"
    assert result.provider_model == "qwen3.5:4b"
    assert len(auxiliary.complete_calls) == 1
    assert main.complete_calls == []
    assert result.compression_count == 4


def test_old_tool_outputs_are_summarized_and_recent_outputs_stay_full() -> None:
    state = FakeState(
        tool_calls=[
            ToolCallRecord(name="old", arguments={"secret": "drop"}, result="x" * 700, blocked=True, reason="blocked"),
            ToolCallRecord(name="recent", arguments={"path": "README.md"}, result={"ok": True}),
        ]
    )
    manager = ContextManager(settings={"context_tool_output_cutoff": 1})

    result = manager.evaluate(state, estimator=FixedEstimator(1))
    outputs = manager.maybe_compress
    summarized = __import__("solo_agent.context.manager", fromlist=["summarize_tool_outputs"]).summarize_tool_outputs(
        state,
        {"context_tool_output_cutoff": 1},
    )

    assert result.should_compress is False
    assert outputs is not None
    assert summarized[0]["compressed"] is True
    assert summarized[0]["name"] == "old"
    assert summarized[0]["blocked"] is True
    assert summarized[0]["reason"] == "blocked"
    assert "arguments" not in summarized[0]
    assert summarized[0]["result"].endswith("...[truncated]")
    assert summarized[1]["compressed"] is False
    assert summarized[1]["arguments"] == {"path": "README.md"}
    assert summarized[1]["result"] == {"ok": True}


def test_auxiliary_client_defaults_compression_to_ollama_qwen() -> None:
    provider = AuxiliaryClient.for_task("compression", {"base_url": None})

    assert provider.name == "ollama"
    assert provider.model == "qwen3.5:4b"
