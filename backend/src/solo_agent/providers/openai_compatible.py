from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence
from typing import Any

from .base import (
    ChatMessage,
    ProviderChunk,
    ProviderConfig,
    ProviderError,
    ProviderResponse,
    ProviderToolCall,
    ProviderToolSpec,
    ToolChoice,
)


class OpenAICompatibleProvider:
    """Streams chat completions from providers that implement OpenAI's API."""

    name = "openai-compatible"
    supports_tool_calling = True

    def __init__(self, config: ProviderConfig) -> None:
        if not config.api_key:
            raise ProviderError(f"{config.provider} provider requires an API key")
        self.config = config
        self.name = config.provider
        self.model = config.model
        self.base_url = (config.base_url or "https://api.openai.com/v1").rstrip("/")

    async def stream_chat(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: Sequence[ProviderToolSpec] | None = None,
        tool_choice: ToolChoice = None,
    ) -> AsyncIterator[ProviderChunk]:
        try:
            import httpx
        except ImportError as exc:
            raise ProviderError("httpx is required for OpenAI-compatible providers") from exc

        body: dict[str, Any] = {
            "model": self.model,
            "messages": [message.to_dict() for message in messages],
            "stream": True,
        }
        if temperature is not None:
            body["temperature"] = temperature
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if tools:
            body["tools"] = list(tools)
        if tool_choice is not None:
            body["tool_choice"] = tool_choice
        body.update(dict(self.config.extra_body))

        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
            **dict(self.config.extra_headers),
        }

        timeout = httpx.Timeout(self.config.timeout_seconds)
        async with httpx.AsyncClient(timeout=timeout) as client:
            try:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=body,
                ) as response:
                    if response.status_code >= 400:
                        detail = await response.aread()
                        raise ProviderError(
                            f"{self.name} returned HTTP {response.status_code}: "
                            f"{detail.decode('utf-8', errors='replace')}"
                        )
                    async for line in response.aiter_lines():
                        chunk = self._parse_sse_line(line)
                        if chunk is not None:
                            yield chunk
            except httpx.HTTPError as exc:
                raise ProviderError(f"{self.name} request failed: {exc}") from exc

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

    async def complete_message(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: Sequence[ProviderToolSpec] | None = None,
        tool_choice: ToolChoice = None,
    ) -> ProviderResponse:
        if not tools:
            parts: list[str] = []
            finish_reason: str | None = None
            raw: dict[str, Any] | None = None
            async for chunk in self.stream_chat(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
            ):
                if chunk.content:
                    parts.append(chunk.content)
                finish_reason = chunk.finish_reason or finish_reason
                raw = chunk.raw or raw
            return ProviderResponse(content="".join(parts), finish_reason=finish_reason, raw=raw)

        try:
            import httpx
        except ImportError as exc:
            raise ProviderError("httpx is required for OpenAI-compatible providers") from exc

        body: dict[str, Any] = {
            "model": self.model,
            "messages": [message.to_dict() for message in messages],
            "stream": False,
            "tools": list(tools),
        }
        if temperature is not None:
            body["temperature"] = temperature
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if tool_choice is not None:
            body["tool_choice"] = tool_choice
        body.update(dict(self.config.extra_body))

        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
            **dict(self.config.extra_headers),
        }

        timeout = httpx.Timeout(self.config.timeout_seconds)
        async with httpx.AsyncClient(timeout=timeout) as client:
            try:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=body,
                )
                if response.status_code >= 400:
                    raise ProviderError(
                        f"{self.name} returned HTTP {response.status_code}: {response.text}"
                    )
            except httpx.HTTPError as exc:
                raise ProviderError(f"{self.name} request failed: {exc}") from exc

        try:
            data = response.json()
        except json.JSONDecodeError as exc:
            raise ProviderError(f"{self.name} sent invalid completion JSON") from exc
        return self._parse_completion_response(data)

    def _parse_sse_line(self, line: str) -> ProviderChunk | None:
        line = line.strip()
        if not line or not line.startswith("data:"):
            return None

        payload = line.removeprefix("data:").strip()
        if payload == "[DONE]":
            return ProviderChunk(finish_reason="stop")

        try:
            data = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ProviderError(f"{self.name} sent invalid SSE JSON") from exc

        choice = (data.get("choices") or [{}])[0]
        delta = choice.get("delta") or {}
        content = delta.get("content") or ""
        tool_calls = tuple(self._parse_tool_call(item) for item in delta.get("tool_calls") or ())
        finish_reason = choice.get("finish_reason")
        return ProviderChunk(content=content, tool_calls=tool_calls, finish_reason=finish_reason, raw=data)

    def _parse_completion_response(self, data: dict[str, Any]) -> ProviderResponse:
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        content = message.get("content") or ""
        tool_calls = tuple(self._parse_tool_call(item) for item in message.get("tool_calls") or ())
        return ProviderResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=choice.get("finish_reason"),
            raw=data,
        )

    def _parse_tool_call(self, item: dict[str, Any]) -> ProviderToolCall:
        function = item.get("function") or {}
        raw_arguments = function.get("arguments")
        arguments: dict[str, Any] = {}
        if isinstance(raw_arguments, str) and raw_arguments:
            try:
                loaded = json.loads(raw_arguments)
                if isinstance(loaded, dict):
                    arguments = loaded
            except json.JSONDecodeError:
                arguments = {}
        return ProviderToolCall(
            id=str(item.get("id") or ""),
            name=str(function.get("name") or ""),
            arguments=arguments,
            raw_arguments=raw_arguments if isinstance(raw_arguments, str) else None,
            index=item.get("index") if isinstance(item.get("index"), int) else None,
            raw=item,
        )
