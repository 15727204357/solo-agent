from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from solo_agent.providers import ChatProvider


@dataclass(frozen=True)
class AgentSettings:
    provider: str = "openai"
    workspace_root: str | Path | None = None
    model: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    timeout_seconds: float = 60.0
    temperature: float = 0.2
    plan_max_tokens: int = 500
    response_max_tokens: int = 1400
    max_tool_calls: int = 3
    tool_call_cut_off: int = 3
    tool_output_max_bytes: int = 12_000
    context_file_limit: int = 80
    context_search_limit: int = 20
    history_message_limit: int = 12
    memory_search_limit: int = 5
    summary_trigger_messages: int = 8
    summary_max_tokens: int = 700
    context_window_tokens: int = 128_000
    context_regular_threshold: float = 0.80
    context_long_task_threshold: float = 0.50
    context_long_task_after_compressions: int = 2
    context_tool_output_cutoff: int = 10
    auxiliary_compression_provider: str = "ollama"
    auxiliary_compression_model: str = "qwen3.5:4b"
    auxiliary_compression_base_url: str = "http://localhost:11434"
    memory_enabled: bool = True
    conversation_history_enabled: bool = True
    verified_editing_enabled: bool = False
    patch_max_tokens: int = 1400
    use_multi_agent: bool = False
    run_mode: str = "agent"
    plan_deep_max_tokens: int = 6000
    subagent_enabled: bool = True
    max_concurrent_subagents: int = 3
    subagent_timeout_seconds: int = 900
    sandbox_mode: str = "local"
    workflow_runtime_root: str | Path = ".solo-agent/runs"
    extra_headers: dict[str, str] = field(default_factory=dict)
    extra_body: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentDeps:
    provider: ChatProvider | None = None
    tool_registry: Any | None = None
    safety_inspector: Any | None = None
    persistence: Any | None = None
    context_provider: Any | None = None
    settings: Any | None = None
