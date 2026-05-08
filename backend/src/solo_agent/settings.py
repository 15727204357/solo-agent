"""Application settings loaded from environment variables."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the Web app, providers, and persistence."""

    model_config = SettingsConfigDict(
        env_prefix="SOLO_AGENT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Solo Agent"
    environment: str = "development"
    host: str = "127.0.0.1"
    port: int = 8000
    reload: bool = True
    log_level: str = "info"
    workspace_root: Path = Field(default_factory=Path.cwd)
    database_url: str = "sqlite+aiosqlite:///./data/solo_agent.sqlite3"
    event_heartbeat_seconds: int = 15

    provider: str = "ollama"
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
    verified_editing_enabled: bool = True
    patch_max_tokens: int = 1400
    plan_deep_max_tokens: int = 6000

    # DeerFlow-style workflow runtime settings.
    subagent_enabled: bool = True
    max_concurrent_subagents: int = 3
    subagent_timeout_seconds: int = 900
    sandbox_mode: str = "local"
    workflow_runtime_root: str = ".solo-agent/runs"

    @field_validator("max_concurrent_subagents")
    @classmethod
    def validate_max_concurrent_subagents(cls, value: int) -> int:
        if not 1 <= value <= 10:
            import warnings
            warnings.warn(f"max_concurrent_subagents {value} out of range, using default 3", stacklevel=2)
            return 3
        return value

    @field_validator("subagent_timeout_seconds")
    @classmethod
    def validate_subagent_timeout(cls, value: int) -> int:
        if not 60 <= value <= 3600:
            import warnings
            warnings.warn(f"subagent_timeout_seconds {value} out of range, using default 900", stacklevel=2)
            return 900
        return value

    @field_validator("sandbox_mode")
    @classmethod
    def validate_sandbox_mode(cls, value: str) -> str:
        if value not in ("local", "docker"):
            import warnings
            warnings.warn(f"Invalid sandbox_mode '{value}', falling back to 'local'", stacklevel=2)
            return "local"
        if value == "docker":
            import warnings
            warnings.warn("Docker sandbox not yet implemented, falling back to 'local'", stacklevel=2)
            return "local"
        return value

    @field_validator("workspace_root")
    @classmethod
    def resolve_workspace_root(cls, value: Path) -> Path:
        return value.expanduser().resolve()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return process-wide settings."""

    return Settings()
