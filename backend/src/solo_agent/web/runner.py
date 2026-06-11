"""Bridge Web runs to the Solo Agent event graph."""

from __future__ import annotations

from solo_agent.agent import AgentDeps, AgentSettings, run_agent_events
from solo_agent.providers import create_provider_from_settings
from solo_agent.settings import get_settings
from solo_agent.tools import create_default_registry
from solo_agent.web.store import SessionRepository
from solo_agent.workflow.sandbox.workspace_backend import create_workspace_backend


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
            tool_loop_mode = _bounded_choice(
                str(run.metadata.get("tool_loop_mode", settings.tool_loop_mode)),
                {"heuristic", "model"},
                settings.tool_loop_mode,
            )
            approval_mode = _bounded_choice(
                str(run.metadata.get("approval_mode", settings.approval_mode)),
                {"confirm", "manual_only"},
                settings.approval_mode,
            )
            default_workspace_backend = "local" if settings.sandbox_mode == "local" else settings.workspace_backend
            workspace_backend_kind = _bounded_choice(
                str(run.metadata.get("workspace_backend", default_workspace_backend)),
                {"local", "copy", "docker"},
                default_workspace_backend,
            )
            eval_suite_id = run.metadata.get("eval_suite_id") or settings.eval_suite_id
            is_plan_mode = run_mode == "plan"
            default_policy = "auto" if is_plan_mode else "off"
            subagent_policy = str(run.metadata.get("subagent_policy", default_policy))
            if subagent_policy not in {"off", "auto"}:
                subagent_policy = default_policy
            subagent_enabled = bool(
                run.metadata.get(
                    "subagent_enabled",
                    is_plan_mode and subagent_policy == "auto",
                )
            )
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
                tool_loop_mode=tool_loop_mode,
                approval_mode=approval_mode,
                workspace_backend=workspace_backend_kind,
                eval_suite_id=str(eval_suite_id) if eval_suite_id else None,
                is_plan_mode=is_plan_mode,
                plan_deep_max_tokens=settings.plan_deep_max_tokens,
                subagent_policy=subagent_policy,
                subagent_enabled=subagent_enabled,
                max_concurrent_subagents=settings.max_concurrent_subagents,
                subagent_timeout_seconds=settings.subagent_timeout_seconds,
                sandbox_mode=settings.sandbox_mode,
                sandbox_retain_on_failure=settings.sandbox_retain_on_failure,
                sandbox_network_policy=settings.sandbox_network_policy,
                sandbox_command_timeout_seconds=settings.sandbox_command_timeout_seconds,
                sandbox_max_output_bytes=settings.sandbox_max_output_bytes,
                sandbox_max_commands_per_run=settings.sandbox_max_commands_per_run,
                sandbox_max_changed_files=settings.sandbox_max_changed_files,
                sandbox_max_workspace_bytes=settings.sandbox_max_workspace_bytes,
                codeintel_max_files=settings.codeintel_max_files,
                codeintel_max_file_bytes=settings.codeintel_max_file_bytes,
                codeintel_index_ttl_seconds=settings.codeintel_index_ttl_seconds,
                outcome_judge_enabled=settings.outcome_judge_enabled,
                outcome_judge_provider_mode=settings.outcome_judge_provider_mode,
                eval_runtime_root=settings.eval_runtime_root,
                git_artifacts_enabled=settings.git_artifacts_enabled,
                workflow_runtime_root=settings.workflow_runtime_root,
            )
            workspace_backend = create_workspace_backend(
                workspace_backend_kind,
                settings.workspace_root,
                session_id=session_id,
                run_id=run_id,
                network_policy=agent_settings.sandbox_network_policy,
                resource_limits={
                    "command_timeout_seconds": agent_settings.sandbox_command_timeout_seconds,
                    "max_output_bytes": agent_settings.sandbox_max_output_bytes,
                    "max_changed_files": agent_settings.sandbox_max_changed_files,
                    "max_workspace_bytes": agent_settings.sandbox_max_workspace_bytes,
                },
            )
            command_workspace = workspace_backend.prepare()
            workspace_metadata = {
                **command_workspace.metadata(),
                "workspace_backend": workspace_backend.metadata(),
            }
            if command_workspace.created:
                await self._repository.append_event(
                    session_id,
                    run_id,
                    "sandbox_created",
                    "Created isolated command workspace.",
                    workspace_metadata,
                )
                await self._repository.append_event(
                    session_id,
                    run_id,
                    "sandbox_policy_applied",
                    "Applied sandbox command, network, env, cache, and resource policies.",
                    workspace_metadata,
                )
                checkpoint = command_workspace.create_checkpoint("run_started")
                await self._repository.append_event(
                    session_id,
                    run_id,
                    "sandbox_checkpoint_created",
                    "Created sandbox checkpoint.",
                    checkpoint,
                )
            try:
                registry_kwargs = {"is_plan_mode": is_plan_mode, "subagent_enabled": subagent_enabled}
                registry_kwargs.update(
                    {
                        "codeintel_max_files": agent_settings.codeintel_max_files,
                        "codeintel_max_file_bytes": agent_settings.codeintel_max_file_bytes,
                        "codeintel_index_ttl_seconds": agent_settings.codeintel_index_ttl_seconds,
                    }
                )
                if command_workspace.created:
                    registry_kwargs.update(
                        {
                            "command_workspace_root": command_workspace.command_workspace_root,
                            "sandbox_mode": command_workspace.mode,
                            "sandbox_id": command_workspace.sandbox_id,
                            "cache_root": command_workspace.cache_root,
                            "sandbox_network_policy": agent_settings.sandbox_network_policy,
                            "sandbox_command_timeout_seconds": agent_settings.sandbox_command_timeout_seconds,
                            "sandbox_max_output_bytes": agent_settings.sandbox_max_output_bytes,
                            "sandbox_max_changed_files": agent_settings.sandbox_max_changed_files,
                            "sandbox_max_workspace_bytes": agent_settings.sandbox_max_workspace_bytes,
                        }
                    )
                registry = create_default_registry(settings.workspace_root, **registry_kwargs)
            except TypeError:
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
                current = await self._repository.get_run(session_id, run_id)
                if current is not None and current.status in {"cancelled", "paused", "awaiting_feedback"}:
                    break
                await self._repository.append_event(
                    session_id,
                    run_id,
                    _to_web_event_type(event.type),
                    event.message,
                    event.to_dict(),
                )
                if event.type in {"patch_approval_required", "skill_change_approval_required"}:
                    awaiting_approval = True
                current = await self._repository.get_run(session_id, run_id)
                if current is not None and current.status in {"cancelled", "paused", "awaiting_feedback"}:
                    break

            current = await self._repository.get_run(session_id, run_id)
            if current is None or current.status not in {"cancelled", "paused", "awaiting_feedback"}:
                completed_status = "awaiting_approval" if awaiting_approval else "completed"
                await self._repository.mark_run_status(
                    session_id,
                    run_id,
                    completed_status,
                )
                if command_workspace.created and completed_status == "completed":
                    cleanup = command_workspace.cleanup()
                    await self._repository.append_event(
                        session_id,
                        run_id,
                        "sandbox_cleanup_completed",
                        "Cleaned up isolated command workspace.",
                        cleanup,
                    )
                elif command_workspace.created:
                    await self._repository.append_event(
                        session_id,
                        run_id,
                        "sandbox_retained",
                        "Retained isolated command workspace for approval or resume.",
                        workspace_metadata,
                    )
            elif command_workspace.created and current.status in {"paused", "awaiting_feedback"}:
                await self._repository.append_event(
                    session_id,
                    run_id,
                    "sandbox_retained",
                    "Retained isolated command workspace for resume.",
                    workspace_metadata,
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


def _bounded_choice(value: str, allowed: set[str], fallback: str) -> str:
    return value if value in allowed else fallback
