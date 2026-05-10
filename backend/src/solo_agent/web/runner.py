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
            run_mode = str(run.metadata.get("run_mode", "agent"))
            use_multi_agent = run_mode == "research"
            agent_settings = AgentSettings(
                provider=settings.provider,
                workspace_root=settings.workspace_root,
                model=settings.model,
                api_key=settings.api_key,
                base_url=settings.base_url,
                timeout_seconds=settings.timeout_seconds,
                temperature=settings.temperature,
                plan_max_tokens=settings.plan_max_tokens,
                response_max_tokens=settings.response_max_tokens,
                max_tool_calls=settings.max_tool_calls,
                tool_call_cut_off=settings.tool_call_cut_off,
                tool_output_max_bytes=settings.tool_output_max_bytes,
                context_file_limit=settings.context_file_limit,
                context_search_limit=settings.context_search_limit,
                history_message_limit=settings.history_message_limit,
                memory_search_limit=settings.memory_search_limit,
                summary_trigger_messages=settings.summary_trigger_messages,
                summary_max_tokens=settings.summary_max_tokens,
                context_window_tokens=settings.context_window_tokens,
                context_regular_threshold=settings.context_regular_threshold,
                context_long_task_threshold=settings.context_long_task_threshold,
                context_long_task_after_compressions=settings.context_long_task_after_compressions,
                context_tool_output_cutoff=settings.context_tool_output_cutoff,
                auxiliary_compression_provider=settings.auxiliary_compression_provider,
                auxiliary_compression_model=settings.auxiliary_compression_model,
                auxiliary_compression_base_url=settings.auxiliary_compression_base_url,
                memory_enabled=memory_enabled,
                conversation_history_enabled=conversation_history_enabled,
                verified_editing_enabled=settings.verified_editing_enabled,
                patch_max_tokens=settings.patch_max_tokens,
                run_mode=run_mode,
                use_multi_agent=use_multi_agent,
                plan_deep_max_tokens=settings.plan_deep_max_tokens,
                subagent_enabled=settings.subagent_enabled,
                max_concurrent_subagents=settings.max_concurrent_subagents,
                subagent_timeout_seconds=settings.subagent_timeout_seconds,
                sandbox_mode=settings.sandbox_mode,
                workflow_runtime_root=settings.workflow_runtime_root,
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

            awaiting_approval = False
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
                if event.type == "patch_approval_required":
                    awaiting_approval = True

            await self._repository.mark_run_status(
                session_id,
                run_id,
                "awaiting_approval" if awaiting_approval else "completed",
            )
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
