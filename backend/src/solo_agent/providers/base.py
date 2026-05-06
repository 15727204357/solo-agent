from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

JsonObject = dict[str, Any]


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str

    def to_dict(self) -> JsonObject:
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True)
class ProviderChunk:
    content: str = ""
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

    async def stream_chat(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[ProviderChunk]:
        ...

    async def complete(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        parts: list[str] = []
        async for chunk in self.stream_chat(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
        ):
            if chunk.content:
                parts.append(chunk.content)
        return "".join(parts)
