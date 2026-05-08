from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Any

from solo_agent.agent.events import AgentEvent
from solo_agent.workflow.state import SubagentRunRecord, WorkflowState
from solo_agent.workflow.subagent.task_tool import subagent_context

SubagentRunner = Callable[..., Awaitable[Any]]


class SubagentExecutor:
    """Run subagents as tracked asyncio tasks and emit lifecycle events."""

    def __init__(
        self,
        max_concurrent: int = 3,
        timeout_seconds: float = 900,
        event_queue: asyncio.Queue | None = None,
    ):
        self._max_concurrent = max_concurrent
        self._timeout = timeout_seconds
        self._active_tasks: dict[str, asyncio.Task] = {}
        self._event_queue = event_queue
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def _emit(self, event: AgentEvent) -> None:
        if self._event_queue is not None:
            await self._event_queue.put(event)

    def start_subagent(
        self,
        *,
        state: WorkflowState,
        subagent_type: str,
        task_id: str,
        runner: SubagentRunner,
        prompt: str,
        max_turns: int = 10,
        **kwargs: Any,
    ) -> asyncio.Task:
        """Schedule a subagent and keep it owned by the executor."""
        task = asyncio.create_task(
            self.run_subagent(
                state=state,
                subagent_type=subagent_type,
                task_id=task_id,
                runner=runner,
                prompt=prompt,
                max_turns=max_turns,
                **kwargs,
            )
        )
        self._active_tasks[task_id] = task
        return task

    async def run_subagent(
        self,
        state: WorkflowState,
        subagent_type: str,
        task_id: str,
        runner: SubagentRunner,
        prompt: str,
        max_turns: int = 10,
        **kwargs: Any,
    ) -> None:
        """Run one subagent, update state, and publish lifecycle events."""
        run_record = SubagentRunRecord(
            run_id=task_id,
            subagent_type=subagent_type,
            description=prompt[:200],
            status="pending",
            started_at=str(time.time()),
        )
        state.add_subagent_run(run_record)

        async with self._semaphore:
            run_record.status = "running"
            await self._emit(
                AgentEvent(
                    type="task_started",
                    session_id=state.session_id,
                    run_id=state.run_id,
                    node=subagent_type,
                    message=f"Subagent started: {subagent_type}",
                    data={"task_id": task_id, "subagent_type": subagent_type},
                )
            )

            token = subagent_context.set(True)
            try:
                result = await asyncio.wait_for(
                    runner(prompt=prompt, max_turns=max_turns, **kwargs),
                    timeout=self._timeout,
                )
                run_record.status = "completed"
                run_record.completed_at = str(time.time())
                run_record.result = str(result) if result else ""
                await self._emit(
                    AgentEvent(
                        type="task_completed",
                        session_id=state.session_id,
                        run_id=state.run_id,
                        node=subagent_type,
                        message=f"Subagent completed: {subagent_type}",
                        data={"task_id": task_id, "result_preview": run_record.result[:500]},
                    )
                )

            except TimeoutError:
                run_record.status = "failed"
                run_record.completed_at = str(time.time())
                run_record.error = "timeout"
                await self._emit(
                    AgentEvent(
                        type="task_failed",
                        session_id=state.session_id,
                        run_id=state.run_id,
                        node=subagent_type,
                        message=f"Subagent timed out: {subagent_type}",
                        data={"task_id": task_id, "reason": "timeout"},
                    )
                )

            except asyncio.CancelledError:
                run_record.status = "failed"
                run_record.completed_at = str(time.time())
                run_record.error = "cancelled"
                await self._emit(
                    AgentEvent(
                        type="task_failed",
                        session_id=state.session_id,
                        run_id=state.run_id,
                        node=subagent_type,
                        message=f"Subagent cancelled: {subagent_type}",
                        data={"task_id": task_id, "reason": "cancelled"},
                    )
                )
                raise

            except Exception as exc:
                run_record.status = "failed"
                run_record.completed_at = str(time.time())
                run_record.error = str(exc)
                await self._emit(
                    AgentEvent(
                        type="task_failed",
                        session_id=state.session_id,
                        run_id=state.run_id,
                        node=subagent_type,
                        message=f"Subagent failed: {subagent_type}",
                        data={"task_id": task_id, "reason": str(exc)},
                    )
                )

            finally:
                subagent_context.reset(token)
                self._active_tasks.pop(task_id, None)

    async def wait_for_all(self, timeout: float | None = None) -> None:
        """Wait for all tracked subagents to finish, bounded by timeout."""
        tasks = [task for task in self._active_tasks.values() if not task.done()]
        if not tasks:
            return
        try:
            await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=timeout)
        except TimeoutError:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    @property
    def active_count(self) -> int:
        return len(self._active_tasks)

    def cancel(self, task_id: str) -> None:
        task = self._active_tasks.get(task_id)
        if task and not task.done():
            task.cancel()

    def cancel_all(self) -> None:
        for task in list(self._active_tasks.values()):
            if not task.done():
                task.cancel()
