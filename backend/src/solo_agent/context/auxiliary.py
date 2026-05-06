from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from solo_agent.providers import ChatProvider, ProviderConfig, create_provider


class AuxiliaryClient:
    """按任务创建辅助模型客户端。"""

    @classmethod
    def for_task(cls, task: str, settings: Any | None = None) -> ChatProvider:
        if task != "compression":
            raise ValueError(f"Unsupported auxiliary task: {task}")

        provider = str(
            _get(
                settings,
                "auxiliary_compression_provider",
                _get(settings, "compression_provider", _get(settings, "auxiliary_provider", "ollama")),
            )
        )
        model = str(
            _get(
                settings,
                "auxiliary_compression_model",
                _get(settings, "compression_model", _get(settings, "auxiliary_model", "qwen3.5:4b")),
            )
        )
        api_key = _get(settings, "compression_api_key", _get(settings, "auxiliary_api_key", _get(settings, "api_key", None)))
        base_url = _get(
            settings,
            "auxiliary_compression_base_url",
            _get(settings, "auxiliary_base_url", _get(settings, "base_url", None)),
        )
        timeout_seconds = float(
            _get(settings, "compression_timeout_seconds", _get(settings, "timeout_seconds", 60.0))
        )
        extra_headers = _get(settings, "compression_extra_headers", _get(settings, "extra_headers", {})) or {}
        extra_body = _get(settings, "compression_extra_body", _get(settings, "extra_body", {})) or {}

        return create_provider(
            ProviderConfig(
                provider=provider,
                model=model,
                api_key=str(api_key) if api_key else None,
                base_url=str(base_url) if base_url else None,
                timeout_seconds=timeout_seconds,
                extra_headers=dict(extra_headers),
                extra_body=dict(extra_body),
            )
        )


def _get(settings: Any | None, key: str, default: Any = None) -> Any:
    if settings is None:
        return default
    if isinstance(settings, Mapping):
        return settings.get(key, default)
    return getattr(settings, key, default)
