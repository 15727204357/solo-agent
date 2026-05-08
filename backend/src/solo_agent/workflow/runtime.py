"""Workflow runtime lifecycle orchestration."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from solo_agent.agent.events import AgentEvent
from solo_agent.agent.state import AgentState
from solo_agent.workflow.factory import LeadAgentFactory
from solo_agent.workflow.langchain_adapter import LangChainChatAdapter
from solo_agent.workflow.sandbox.local import LocalSandboxProvider
from solo_agent.workflow.stages import (
    _build_memory_context_stage,
    _collect_context_node,
    _compress_memory_stage,
    _context_guard_stage,
    _execute_tools_node,
    _inspect_node,
    _load_builtin_memory_stage,
    _persist_snapshot_stage,
    _plan_mode_path,
    _plan_node,
    _prefetch_memory_stage,
    _propose_verified_patch_node,
    _queue_prefetch_stage,
    _receive_user_turn_stage,
    _respond_node,
    _select_tools_node,
    _skill_context_stage,
    _skip_memory_stage,
    _subdirectory_hint_stage,
    _sync_memory_stage,
    _task_state_stage,
)
from solo_agent.workflow.state import WorkflowState
from solo_agent.workflow.subagent.executor import SubagentExecutor
from solo_agent.workflow.subagent.factories import register_builtin_factories
from solo_agent.workflow.subagent.task_tool import bind_task_executor


class WorkflowRuntime:
    """Run the single DeerFlow-style workflow and stream AgentEvent values."""

    def __init__(
        self,
        *,
        deps: Any,
        state: AgentState,
        provider: Any,
    ):
        self._deps = deps
        self._agent_state = state
        self._provider = provider
        self._event_queue: asyncio.Queue[AgentEvent] = asyncio.Queue()

    async def _emit(
        self,
        type_: str,
        message: str = "",
        node: str = "workflow",
        data: dict[str, Any] | None = None,
    ) -> None:
        await self._event_queue.put(
            AgentEvent(
                type=type_,
                session_id=self._agent_state.session_id,
                run_id=self._agent_state.run_id,
                node=node,
                message=message,
                data=data or {},
            )
        )

    async def run(self) -> AsyncIterator[AgentEvent]:
        settings = self._deps.settings
        wf_state = WorkflowState.from_agent_state(self._agent_state)

        async for event in self._run_shared_prelude(settings):
            yield event
        if self._agent_state.blocked:
            yield self._run_completed_event(
                "Workflow run blocked by safety inspection",
                {"blocked": True, "reason": self._agent_state.block_reason},
            )
            return

        if self._agent_state.run_mode == "plan":
            async for event in _plan_mode_path(self._agent_state, self._provider, self._deps, settings):
                yield event
        elif self._uses_lead_agent_strategy(settings):
            async for event in self._run_lead_agent_strategy(wf_state, settings):
                yield event
        else:
            async for event in self._run_text_provider_strategy(settings):
                yield event

        if self._agent_state.blocked:
            yield self._run_completed_event(
                "Workflow run blocked by safety inspection",
                {"blocked": True, "reason": self._agent_state.block_reason},
            )
            return
        if self._agent_state.awaiting_approval:
            return

        async for event in self._run_shared_postlude(settings):
            yield event
        yield self._run_completed_event("Workflow run completed", wf_state.snapshot())

    async def _run_text_provider_strategy(self, settings: Any) -> AsyncIterator[AgentEvent]:
        state = self._agent_state
        async for event in _plan_node(state, self._provider, settings):
            yield event
        async for event in _task_state_stage(state):
            yield event
        async for event in _collect_context_node(state, self._deps, settings):
            yield event
        async for event in _inspect_node(state, self._deps):
            yield event
        if state.blocked:
            return
        async for event in _select_tools_node(state, self._deps, settings):
            yield event
        async for event in _execute_tools_node(state, self._deps, settings):
            yield event
        if state.awaiting_approval:
            return

        async for event in _propose_verified_patch_node(state, self._provider, self._deps, settings):
            yield event
        if state.awaiting_approval:
            return

        async for event in _subdirectory_hint_stage(state, settings):
            yield event
        async for event in _context_guard_stage(
            state,
            self._provider,
            self._deps,
            settings,
            phase="before_respond",
        ):
            yield event
        async for event in _respond_node(state, self._provider, settings):
            yield event

    async def _run_lead_agent_strategy(
        self,
        wf_state: WorkflowState,
        settings: Any,
    ) -> AsyncIterator[AgentEvent]:
        async for event in _collect_context_node(self._agent_state, self._deps, settings):
            yield event
        async for event in _inspect_node(self._agent_state, self._deps):
            yield event
        if self._agent_state.blocked:
            return
        async for event in _context_guard_stage(
            self._agent_state,
            self._provider,
            self._deps,
            settings,
            phase="before_respond",
        ):
            yield event

        sandbox = LocalSandboxProvider(runtime_root=settings.workflow_runtime_root)
        await sandbox.setup(wf_state.session_id, wf_state.run_id)
        wf_state.sandbox = sandbox

        adapter = LangChainChatAdapter(
            provider=self._provider,
            temperature=settings.temperature,
            max_tokens=settings.response_max_tokens,
        )
        executor = SubagentExecutor(
            max_concurrent=settings.max_concurrent_subagents,
            timeout_seconds=settings.subagent_timeout_seconds,
            event_queue=self._event_queue,
        )
        bind_task_executor(
            executor,
            wf_state,
            registry=register_builtin_factories(),
            factory_context={
                "model": adapter,
                "tool_registry": self._deps.tool_registry,
            },
        )
        factory = LeadAgentFactory(
            model=adapter,
            tool_registry=self._deps.tool_registry,
            system_prompt=self._build_system_prompt(wf_state),
            event_queue=self._event_queue,
        )

        try:
            await self._emit("run_started", node="workflow")
            agent = factory.create_agent(
                include_task=settings.subagent_enabled,
                allowed_names=self._lead_tool_allowlist(),
                state=wf_state,
            )
            async for event in self._flush_events():
                yield event

            input_messages = [{"role": "user", "content": wf_state.agent_state.user_input}]
            config = {"configurable": {"thread_id": wf_state.run_id}}

            full_response = ""
            async for chunk in agent.astream(
                {"messages": input_messages},
                config=config,
                stream_mode="values",
            ):
                async for event in self._flush_events():
                    yield event
                if "messages" not in chunk:
                    continue
                last_msg = chunk["messages"][-1] if chunk["messages"] else None
                if (
                    last_msg
                    and hasattr(last_msg, "content")
                    and isinstance(getattr(last_msg, "type", ""), str)
                    and last_msg.type == "ai"
                ):
                    content = last_msg.content
                    if isinstance(content, str) and content:
                        delta = content[len(full_response):]
                        full_response = content
                        if delta:
                            yield AgentEvent(
                                type="response_delta",
                                session_id=wf_state.session_id,
                                run_id=wf_state.run_id,
                                node="responder",
                                data={"delta": delta},
                            )

            async for event in self._wait_for_subagents(executor, timeout=settings.subagent_timeout_seconds):
                yield event

            if full_response:
                wf_state.agent_state.response = full_response
                wf_state.agent_state.snapshots["response"] = full_response
                yield AgentEvent(
                    type="response_completed",
                    session_id=wf_state.session_id,
                    run_id=wf_state.run_id,
                    node="responder",
                    data={"response": full_response},
                )
            wf_state.agent_state.snapshots["subagent_runs"] = wf_state.snapshot().get("subagent_runs", {})

        except Exception as exc:
            executor.cancel_all()
            await executor.wait_for_all(timeout=0.1)
            await self._emit("run_failed", message=str(exc), data={"error": str(exc)})
            async for event in self._flush_events():
                yield event
            raise

        finally:
            async for event in self._wait_for_subagents(executor, timeout=0.1):
                yield event
            await sandbox.teardown(wf_state.session_id, wf_state.run_id)

    async def _run_shared_prelude(self, settings: Any) -> AsyncIterator[AgentEvent]:
        state = self._agent_state
        async for event in _receive_user_turn_stage(state, settings):
            yield event

        if state.memory_enabled:
            async for event in _load_builtin_memory_stage(state, self._deps):
                yield event
            async for event in _prefetch_memory_stage(state, self._deps, settings):
                yield event
            async for event in _build_memory_context_stage(state):
                yield event
        else:
            async for event in _skip_memory_stage(state):
                yield event

        async for event in _skill_context_stage(state, self._deps, settings):
            yield event
        async for event in _context_guard_stage(
            state,
            self._provider,
            self._deps,
            settings,
            phase="before_plan",
        ):
            yield event

    async def _run_shared_postlude(self, settings: Any) -> AsyncIterator[AgentEvent]:
        state = self._agent_state
        if state.memory_enabled:
            async for event in _sync_memory_stage(state, self._deps):
                yield event
            async for event in _queue_prefetch_stage(state, self._deps, settings):
                yield event
            async for event in _compress_memory_stage(state, self._provider, self._deps, settings):
                yield event

        async for event in _persist_snapshot_stage(state):
            yield event

    async def _flush_events(self) -> AsyncIterator[AgentEvent]:
        while not self._event_queue.empty():
            yield self._event_queue.get_nowait()

    async def _wait_for_subagents(
        self,
        executor: SubagentExecutor,
        *,
        timeout: float | None,
    ) -> AsyncIterator[AgentEvent]:
        wait_task = asyncio.create_task(executor.wait_for_all(timeout=timeout))
        try:
            while True:
                if wait_task.done() and self._event_queue.empty():
                    break
                try:
                    yield await asyncio.wait_for(self._event_queue.get(), timeout=0.05)
                except TimeoutError:
                    continue
            await wait_task
        finally:
            if not wait_task.done():
                wait_task.cancel()
                await asyncio.gather(wait_task, return_exceptions=True)

    def _build_system_prompt(self, wf_state: WorkflowState) -> str:
        state = wf_state.agent_state
        parts = ["You are solo-agent, a professional software development assistant."]

        if state.skill_context_block:
            parts.append(state.skill_context_block)
        if state.memory_context_block:
            parts.append(state.memory_context_block)
        if state.context:
            parts.append(f"Context items available to this run:\n{state.context}")

        if self._deps.settings.subagent_enabled:
            parts.append(
                "You may use the `task` tool to dispatch read-only subagents for "
                "parallel research, code review, and quality checks. Editing remains "
                "centralized through verified patch proposal and approval."
            )
        return "\n\n".join(parts)

    def _uses_lead_agent_strategy(self, settings: Any) -> bool:
        return (
            bool(getattr(self._provider, "supports_tool_calling", False))
            and self._deps.tool_registry is not None
            and not bool(getattr(settings, "verified_editing_enabled", False))
        )

    def _lead_tool_allowlist(self) -> set[str]:
        allowed: set[str] = set()
        for name, spec in getattr(self._deps.tool_registry, "_tools", {}).items():
            if not bool(getattr(spec, "read_only", False)):
                continue
            if str(getattr(spec, "category", "")) == "edit":
                continue
            allowed.add(str(name))
        return allowed

    def _run_completed_event(self, message: str, data: dict[str, Any]) -> AgentEvent:
        return AgentEvent(
            type="run_completed",
            session_id=self._agent_state.session_id,
            run_id=self._agent_state.run_id,
            node="workflow",
            message=message,
            data=data,
        )
