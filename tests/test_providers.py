from __future__ import annotations

import pytest

from solo_agent.providers import (
    OllamaProvider,
    OpenAICompatibleProvider,
    ProviderConfig,
    create_provider,
)


def test_create_openai_provider() -> None:
    provider = create_provider(
        ProviderConfig(provider="openai", model="gpt-4.1-mini", api_key="test-key")
    )

    assert isinstance(provider, OpenAICompatibleProvider)
    assert provider.base_url == "https://api.openai.com/v1"


def test_create_deepseek_provider() -> None:
    provider = create_provider(
        ProviderConfig(provider="deepseek", model="deepseek-chat", api_key="test-key")
    )

    assert isinstance(provider, OpenAICompatibleProvider)
    assert provider.base_url == "https://api.deepseek.com/v1"


def test_create_ollama_provider_without_api_key() -> None:
    provider = create_provider(ProviderConfig(provider="ollama", model="llama3.1"))

    assert isinstance(provider, OllamaProvider)
    assert provider.base_url == "http://localhost:11434"


def test_unknown_provider_is_rejected() -> None:
    with pytest.raises(RuntimeError):
        create_provider(ProviderConfig(provider="unknown", model="x"))
