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
    memory_enabled: bool = True
    conversation_history_enabled: bool = True

    @field_validator("workspace_root")
    @classmethod
    def resolve_workspace_root(cls, value: Path) -> Path:
        return value.expanduser().resolve()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return process-wide settings."""

    return Settings()
