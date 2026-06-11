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
    max_tool_calls: int = 8
    tool_call_cut_off: int = 8
    tool_output_max_bytes: int = 24_000
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
    tool_loop_mode: str = "heuristic"
    approval_mode: str = "confirm"
    workspace_backend: str = "copy"
    eval_suite_id: str | None = None

    # DeerFlow-style workflow runtime settings.
    subagent_policy: str = "auto"
    subagent_enabled: bool = False
    max_concurrent_subagents: int = 3
    subagent_timeout_seconds: int = 900
    sandbox_mode: str = "auto"
    sandbox_retain_on_failure: bool = True
    sandbox_network_policy: str = "deny"
    sandbox_command_timeout_seconds: int = 60
    sandbox_max_output_bytes: int = 32_000
    sandbox_max_commands_per_run: int = 50
    sandbox_max_changed_files: int = 200
    sandbox_max_workspace_bytes: int = 512_000_000
    codeintel_max_files: int = 2_000
    codeintel_max_file_bytes: int = 512_000
    codeintel_index_ttl_seconds: int = 30
    outcome_judge_enabled: bool = True
    outcome_judge_provider_mode: str = "rules"
    eval_runtime_root: str = ".solo-agent/evals"
    git_artifacts_enabled: bool = True
    workflow_runtime_root: str = ".solo-agent/runs"
    skill_evolution_enabled: bool = True
    skill_evolution_min_confidence: float = 0.72
    skill_evolution_max_proposals_per_run: int = 1

    # Workflow engine settings — LangGraph StateGraph is the only engine.
    workflow_checkpointer: str = "sqlite"
    workflow_checkpoint_path: str = ".solo-agent/checkpoints/solo_agent_graph.sqlite3"

    # Declarative multi-provider routing (optional — falls back to single provider).
    provider_fast: str | None = None
    provider_complete: str | None = None
    provider_compression: str | None = None
    provider_fallback: str | None = None

    @field_validator("workflow_checkpointer")
    @classmethod
    def validate_workflow_checkpointer(cls, value: str) -> str:
        if value not in ("memory", "sqlite", "none"):
            import warnings
            warnings.warn(f"Invalid workflow_checkpointer '{value}', falling back to 'sqlite'", stacklevel=2)
            return "sqlite"
        return value

    @field_validator("max_concurrent_subagents")
    @classmethod
    def validate_max_concurrent_subagents(cls, value: int) -> int:
        if not 1 <= value <= 10:
            import warnings
            warnings.warn(f"max_concurrent_subagents {value} out of range, using default 3", stacklevel=2)
            return 3
        return value

    @field_validator("subagent_policy")
    @classmethod
    def validate_subagent_policy(cls, value: str) -> str:
        if value not in ("off", "auto"):
            import warnings
            warnings.warn(f"Invalid subagent_policy '{value}', falling back to 'auto'", stacklevel=2)
            return "auto"
        return value

    @field_validator("subagent_timeout_seconds")
    @classmethod
    def validate_subagent_timeout(cls, value: int) -> int:
        if not 60 <= value <= 3600:
            import warnings
            warnings.warn(f"subagent_timeout_seconds {value} out of range, using default 900", stacklevel=2)
            return 900
        return value

    @field_validator("tool_loop_mode")
    @classmethod
    def validate_tool_loop_mode(cls, value: str) -> str:
        if value not in ("heuristic", "model"):
            import warnings
            warnings.warn(f"Invalid tool_loop_mode '{value}', falling back to 'heuristic'", stacklevel=2)
            return "heuristic"
        return value

    @field_validator("approval_mode")
    @classmethod
    def validate_approval_mode(cls, value: str) -> str:
        if value not in ("confirm", "manual_only"):
            import warnings
            warnings.warn(f"Invalid approval_mode '{value}', falling back to 'confirm'", stacklevel=2)
            return "confirm"
        return value

    @field_validator("workspace_backend")
    @classmethod
    def validate_workspace_backend(cls, value: str) -> str:
        if value not in ("local", "copy", "docker"):
            import warnings
            warnings.warn(f"Invalid workspace_backend '{value}', falling back to 'copy'", stacklevel=2)
            return "copy"
        return value

    @field_validator("sandbox_mode")
    @classmethod
    def validate_sandbox_mode(cls, value: str) -> str:
        if value not in ("auto", "local", "copy", "isolated", "worktree", "docker"):
            import warnings
            warnings.warn(f"Invalid sandbox_mode '{value}', falling back to 'auto'", stacklevel=2)
            return "auto"
        if value == "docker":
            import warnings
            warnings.warn("Docker sandbox not yet implemented, falling back to 'auto'", stacklevel=2)
            return "auto"
        return value

    @field_validator("sandbox_network_policy")
    @classmethod
    def validate_sandbox_network_policy(cls, value: str) -> str:
        if value not in ("deny", "allow_local", "allow_all"):
            import warnings
            warnings.warn(f"Invalid sandbox_network_policy '{value}', falling back to 'deny'", stacklevel=2)
            return "deny"
        return value

    @field_validator("outcome_judge_provider_mode")
    @classmethod
    def validate_outcome_judge_provider_mode(cls, value: str) -> str:
        if value not in ("rules", "provider", "hybrid"):
            import warnings
            warnings.warn(f"Invalid outcome_judge_provider_mode '{value}', falling back to 'rules'", stacklevel=2)
            return "rules"
        return value

    @field_validator("workspace_root")
    @classmethod
    def resolve_workspace_root(cls, value: Path) -> Path:
        return value.expanduser().resolve()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return process-wide settings."""

    return Settings()
