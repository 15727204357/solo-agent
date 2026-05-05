"""Bridge Web runs to the Solo Agent event graph."""

from __future__ import annotations

from solo_agent.agent import AgentDeps, AgentSettings, run_agent_events
from solo_agent.providers import create_provider_from_settings
from solo_agent.settings import get_settings
from solo_agent.tools import create_default_registry
from solo_agent.web.store import SessionRepository


class AgentRunner:
    def __init__(self, repository: SessionRepository) -> None:
        self._repository = repository

    async def run(self, session_id: str, run_id: str) -> None:
        run = await self._repository.get_run(session_id, run_id)
        if run is None:
            return

        try:
            await self._repository.mark_run_status(session_id, run_id, "running")
            await self._repository.append_event(
                session_id,
                run_id,
                "started",
                "Run accepted by Solo Agent.",
                {"prompt": run.prompt},
            )

            settings = get_settings()
            memory_enabled = bool(run.metadata.get("memory_enabled", settings.memory_enabled))
            conversation_history_enabled = bool(
                run.metadata.get(
                    "conversation_history_enabled",
                    settings.conversation_history_enabled,
                )
            )
            agent_settings = AgentSettings(
                provider=settings.provider,
                model=settings.model,
                api_key=settings.api_key,
                base_url=settings.base_url,
                timeout_seconds=settings.timeout_seconds,
                temperature=settings.temperature,
                plan_max_tokens=settings.plan_max_tokens,
                response_max_tokens=settings.response_max_tokens,
                max_tool_calls=settings.max_tool_calls,
                context_file_limit=settings.context_file_limit,
                context_search_limit=settings.context_search_limit,
                history_message_limit=settings.history_message_limit,
                memory_search_limit=settings.memory_search_limit,
                summary_trigger_messages=settings.summary_trigger_messages,
                summary_max_tokens=settings.summary_max_tokens,
                memory_enabled=memory_enabled,
                conversation_history_enabled=conversation_history_enabled,
            )
            registry = create_default_registry(settings.workspace_root)
            provider = create_provider_from_settings(agent_settings)
            persistence = None
            if hasattr(self._repository, "memory_repository"):
                persistence = await self._repository.memory_repository()
            deps = AgentDeps(
                provider=provider,
                tool_registry=registry,
                safety_inspector=registry,
                persistence=persistence,
                settings=agent_settings,
            )

            async for event in run_agent_events(
                session_id=session_id,
                run_id=run_id,
                user_input=run.prompt,
                deps=deps,
                settings=agent_settings,
            ):
                await self._repository.append_event(
                    session_id,
                    run_id,
                    _to_web_event_type(event.type),
                    event.message,
                    event.to_dict(),
                )

            await self._repository.mark_run_status(session_id, run_id, "completed")
        except Exception as exc:  # pragma: no cover - defensive boundary for background tasks.
            await self._repository.append_event(
                session_id,
                run_id,
                "failed",
                "Run failed before completion.",
                {"error": str(exc)},
            )
            await self._repository.mark_run_status(session_id, run_id, "failed")


def _to_web_event_type(agent_event_type: str) -> str:
    if agent_event_type == "error":
        return "failed"
    return agent_event_type
