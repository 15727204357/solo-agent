from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from .base import ChatProvider, ProviderConfig, ProviderError
from .ollama import OllamaProvider
from .openai_compatible import OpenAICompatibleProvider

OPENAI_DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEEPSEEK_DEFAULT_BASE_URL = "https://api.deepseek.com/v1"


def create_provider(config: ProviderConfig) -> ChatProvider:
    provider = config.provider.lower().strip()
    if provider == "openai":
        base = config.base_url or OPENAI_DEFAULT_BASE_URL
        return OpenAICompatibleProvider(_replace_base(config, base))
    if provider == "deepseek":
        base = config.base_url or DEEPSEEK_DEFAULT_BASE_URL
        return OpenAICompatibleProvider(_replace_base(config, base))
    if provider == "ollama":
        return OllamaProvider(config)
    raise ProviderError(f"Unsupported provider: {config.provider}")


def create_provider_from_settings(settings: Any | None = None) -> ChatProvider:
    """Build a real provider from object/dict settings plus environment fallback."""

    provider = str(_get(settings, "provider", os.getenv("SOLO_AGENT_PROVIDER", "openai")))
    provider_key = provider.upper().replace("-", "_")

    model = _get(
        settings,
        "model",
        os.getenv(f"{provider_key}_MODEL") or os.getenv("SOLO_AGENT_MODEL"),
    )
    if not model:
        model = "gpt-4.1-mini" if provider.lower() == "openai" else "deepseek-chat"
    if provider.lower() == "ollama" and not _get(settings, "model", None):
        model = os.getenv("OLLAMA_MODEL", "llama3.1")

    api_key = _get(
        settings,
        "api_key",
        os.getenv(f"{provider_key}_API_KEY") or os.getenv("SOLO_AGENT_API_KEY"),
    )
    base_url = _get(
        settings,
        "base_url",
        os.getenv(f"{provider_key}_BASE_URL") or os.getenv("SOLO_AGENT_BASE_URL"),
    )
    timeout = float(_get(settings, "timeout_seconds", os.getenv("SOLO_AGENT_TIMEOUT", "60")))
    extra_headers = _get(settings, "extra_headers", {}) or {}
    extra_body = _get(settings, "extra_body", {}) or {}

    return create_provider(
        ProviderConfig(
            provider=provider,
            model=str(model),
            api_key=str(api_key) if api_key else None,
            base_url=str(base_url) if base_url else None,
            timeout_seconds=timeout,
            extra_headers=dict(extra_headers),
            extra_body=dict(extra_body),
        )
    )


def _replace_base(config: ProviderConfig, base_url: str) -> ProviderConfig:
    return ProviderConfig(
        provider=config.provider,
        model=config.model,
        api_key=config.api_key,
        base_url=base_url,
        timeout_seconds=config.timeout_seconds,
        extra_headers=config.extra_headers,
        extra_body=config.extra_body,
    )


def _get(settings: Any | None, key: str, default: Any = None) -> Any:
    if settings is None:
        return default
    if isinstance(settings, Mapping):
        return settings.get(key, default)
    return getattr(settings, key, default)
