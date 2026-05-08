from __future__ import annotations

from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

JsonObject = dict[str, Any]
ToolChoice = str | JsonObject | None
ProviderToolSpec = JsonObject


@dataclass(frozen=True)
class ProviderToolCall:
    id: str
    name: str
    arguments: JsonObject = field(default_factory=dict)
    raw_arguments: str | None = None
    index: int | None = None
    raw: JsonObject | None = None

    def to_openai(self) -> JsonObject:
        import json

        arguments = self.raw_arguments
        if arguments is None:
            arguments = json.dumps(self.arguments)
        result: JsonObject = {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": arguments},
        }
        if self.index is not None:
            result["index"] = self.index
        return result


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str
    tool_call_id: str | None = None
    tool_calls: tuple[ProviderToolCall, ...] = ()

    def to_dict(self) -> JsonObject:
        result: JsonObject = {"role": self.role, "content": self.content}
        if self.tool_call_id:
            result["tool_call_id"] = self.tool_call_id
        if self.tool_calls:
            result["tool_calls"] = [tool_call.to_openai() for tool_call in self.tool_calls]
        return result


@dataclass(frozen=True)
class ProviderChunk:
    content: str = ""
    tool_calls: tuple[ProviderToolCall, ...] = ()
    finish_reason: str | None = None
    raw: JsonObject | None = None


@dataclass(frozen=True)
class ProviderResponse:
    content: str = ""
    tool_calls: tuple[ProviderToolCall, ...] = ()
    finish_reason: str | None = None
    raw: JsonObject | None = None


@dataclass(frozen=True)
class ProviderConfig:
    provider: str
    model: str
    api_key: str | None = None
    base_url: str | None = None
    timeout_seconds: float = 60.0
    extra_headers: Mapping[str, str] = field(default_factory=dict)
    extra_body: Mapping[str, Any] = field(default_factory=dict)


class ProviderError(RuntimeError):
    """Raised when an upstream model provider cannot complete a request."""


class ChatProvider(Protocol):
    name: str
    model: str
    supports_tool_calling: bool

    async def stream_chat(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: Sequence[ProviderToolSpec] | None = None,
        tool_choice: ToolChoice = None,
    ) -> AsyncIterator[ProviderChunk]:
        ...

    async def complete_message(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: Sequence[ProviderToolSpec] | None = None,
        tool_choice: ToolChoice = None,
    ) -> ProviderResponse:
        parts: list[str] = []
        tool_calls: list[ProviderToolCall] = []
        finish_reason: str | None = None
        raw: JsonObject | None = None
        async for chunk in self.stream_chat(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice=tool_choice,
        ):
            if chunk.content:
                parts.append(chunk.content)
            if chunk.tool_calls:
                tool_calls.extend(chunk.tool_calls)
            finish_reason = chunk.finish_reason or finish_reason
            raw = chunk.raw or raw
        return ProviderResponse(
            content="".join(parts),
            tool_calls=tuple(tool_calls),
            finish_reason=finish_reason,
            raw=raw,
        )

    async def complete(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        response = await self.complete_message(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.content
