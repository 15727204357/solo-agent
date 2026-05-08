from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence
from typing import Any

from .base import ChatMessage, ProviderChunk, ProviderConfig, ProviderError, ProviderToolSpec, ToolChoice


class OllamaProvider:
    name = "ollama"
    supports_tool_calling = False

    def __init__(self, config: ProviderConfig) -> None:
        self.config = config
        self.model = config.model
        self.base_url = (config.base_url or "http://localhost:11434").rstrip("/")

    async def stream_chat(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: Sequence[ProviderToolSpec] | None = None,
        tool_choice: ToolChoice = None,
    ) -> AsyncIterator[ProviderChunk]:
        if tools or tool_choice is not None:
            raise ProviderError("ollama provider does not support tool calling through this adapter")
        try:
            import httpx
        except ImportError as exc:
            raise ProviderError("httpx is required for Ollama provider") from exc

        options: dict[str, Any] = {}
        if temperature is not None:
            options["temperature"] = temperature
        if max_tokens is not None:
            options["num_predict"] = max_tokens

        body: dict[str, Any] = {
            "model": self.model,
            "messages": [message.to_dict() for message in messages],
            "stream": True,
        }
        if options:
            body["options"] = options
        body.update(dict(self.config.extra_body))

        timeout = httpx.Timeout(self.config.timeout_seconds)
        async with httpx.AsyncClient(timeout=timeout) as client:
            try:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/api/chat",
                    json=body,
                    headers=dict(self.config.extra_headers),
                ) as response:
                    if response.status_code >= 400:
                        detail = await response.aread()
                        raise ProviderError(
                            f"ollama returned HTTP {response.status_code}: "
                            f"{detail.decode('utf-8', errors='replace')}"
                        )
                    async for line in response.aiter_lines():
                        chunk = self._parse_json_line(line)
                        if chunk is not None:
                            yield chunk
            except httpx.HTTPError as exc:
                raise ProviderError(f"ollama request failed: {exc}") from exc

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

    def _parse_json_line(self, line: str) -> ProviderChunk | None:
        line = line.strip()
        if not line:
            return None
        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ProviderError("ollama sent invalid JSONL") from exc

        message = data.get("message") or {}
        content = message.get("content") or ""
        finish_reason = "stop" if data.get("done") else None
        return ProviderChunk(content=content, finish_reason=finish_reason, raw=data)
