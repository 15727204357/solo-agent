from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from .base import ChatMessage, ProviderChunk, ProviderConfig, ProviderError


class OpenAICompatibleProvider:
    """Streams chat completions from providers that implement OpenAI's API."""

    name = "openai-compatible"

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
        parts: list[str] = []
        async for chunk in self.stream_chat(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
        ):
            if chunk.content:
                parts.append(chunk.content)
        return "".join(parts)

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
        finish_reason = choice.get("finish_reason")
        return ProviderChunk(content=content, finish_reason=finish_reason, raw=data)
