from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from solo_agent.providers import ChatProvider


@dataclass(frozen=True)
class AgentSettings:
    provider: str = "openai"
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
    memory_enabled: bool = True
    conversation_history_enabled: bool = True
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
