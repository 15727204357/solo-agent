from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from dataclasses import fields as dataclass_fields
from typing import Any

from solo_agent.providers import create_provider_from_settings

from .deps import AgentDeps, AgentSettings
from .events import AgentEvent
from .state import AgentState


async def run_agent_events(
    session_id: str,
    run_id: str,
    user_input: str,
    deps: AgentDeps | None = None,
    settings: AgentSettings | Mapping[str, Any] | None = None,
) -> AsyncIterator[AgentEvent]:
    """Run the primary workflow runtime and stream visible progress events."""

    deps = deps or AgentDeps()
    settings = _coerce_agent_settings(settings or deps.settings or AgentSettings())
    deps.settings = settings
    provider = deps.provider or create_provider_from_settings(settings)
    state = AgentState(session_id=session_id, run_id=run_id, user_input=user_input)
    state.run_mode = str(_setting(settings, "run_mode", "agent"))
    state.memory_enabled = bool(_setting(settings, "memory_enabled", True))
    state.conversation_history_enabled = bool(
        _setting(settings, "conversation_history_enabled", True)
    )
    from solo_agent.workflow.stages import _BEHAVIOR_POLICY, _event, _persist

    _BEHAVIOR_POLICY.start_error_run(run_id)

    from solo_agent.workflow.runtime import WorkflowRuntime

    runtime = WorkflowRuntime(
        deps=deps,
        state=state,
        provider=provider,
    )

    try:
        await _persist(deps.persistence, "start_run", state)
        async for event in runtime.run():
            await _persist(deps.persistence, "save_event", event, state)
            yield event
    except Exception as exc:
        event = _event(state, "error", "error", str(exc), {"error_type": type(exc).__name__})
        await _persist(deps.persistence, "save_event", event, state)
        await _persist(deps.persistence, "finish_run", state, status="error", error=str(exc))
        yield event
    else:
        await _persist(
            deps.persistence,
            "finish_run",
            state,
            status="awaiting_approval" if state.awaiting_approval else "completed",
        )
    finally:
        _BEHAVIOR_POLICY.finish_error_run(run_id)


def _setting(settings: AgentSettings | Mapping[str, Any], key: str, default: Any) -> Any:
    if isinstance(settings, Mapping):
        return settings.get(key, default)
    return getattr(settings, key, default)


def _coerce_agent_settings(settings: AgentSettings | Mapping[str, Any] | Any) -> AgentSettings:
    if isinstance(settings, AgentSettings):
        return settings
    defaults = AgentSettings()
    values: dict[str, Any] = {}
    for item in dataclass_fields(AgentSettings):
        default = getattr(defaults, item.name)
        values[item.name] = _setting(settings, item.name, default)
    return AgentSettings(**values)
