"""Task tool used by the lead agent to dispatch subagents."""

from __future__ import annotations

import contextvars
import uuid
from typing import Any

from solo_agent.tools.registry import ToolSpec
from solo_agent.workflow.subagent.registry import SubagentRegistry, get_builtin_registry

subagent_context: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "solo_agent_subagent_context",
    default=False,
)


def set_subagent_context(flag: bool) -> None:
    subagent_context.set(flag)


def is_subagent_context() -> bool:
    return subagent_context.get()


TASK_TOOL_PARAMETERS = {
    "description": {
        "type": "string",
        "description": "A short task summary.",
        "default": "",
    },
    "prompt": {
        "type": "string",
        "description": "Detailed subagent instructions and expected output.",
        "default": "",
    },
    "subagent_type": {
        "type": "string",
        "description": "Subagent type: general-purpose, code-review, or quality.",
        "default": "general-purpose",
    },
    "max_turns": {
        "type": "integer",
        "description": "Maximum interaction turns.",
        "default": 10,
    },
}


class TaskToolHandler:
    """Dispatch task tool calls through a bound executor and subagent registry."""

    def __init__(self):
        self._executor = None
        self._state = None
        self._registry: SubagentRegistry = get_builtin_registry()
        self._factory_context: dict[str, Any] = {}

    def bind(
        self,
        executor: Any,
        state: Any,
        *,
        registry: SubagentRegistry | None = None,
        factory_context: dict[str, Any] | None = None,
    ) -> None:
        self._executor = executor
        self._state = state
        self._registry = registry or get_builtin_registry()
        self._factory_context = dict(factory_context or {})

    def handle(
        self,
        description: str = "",
        prompt: str = "",
        subagent_type: str = "general-purpose",
        max_turns: int = 10,
    ) -> dict[str, Any]:
        if is_subagent_context():
            return {
                "ok": False,
                "error": "Recursive task calls are not allowed from subagents.",
                "code": "recursive_task_blocked",
            }
        if self._executor is None or self._state is None:
            return {
                "ok": False,
                "error": "Task executor is not initialized.",
                "code": "executor_not_ready",
            }
        if not self._registry.has(subagent_type):
            return {
                "ok": False,
                "error": f"Unknown subagent type: {subagent_type}",
                "code": "unknown_subagent_type",
            }

        task_id = str(uuid.uuid4())[:8]
        try:
            factory = self._registry.get(subagent_type)
            runner = factory(**self._factory_context)
            self._executor.start_subagent(
                state=self._state,
                subagent_type=subagent_type,
                task_id=task_id,
                runner=runner,
                prompt=prompt or description,
                max_turns=max_turns,
            )
            return {
                "ok": True,
                "task_id": task_id,
                "subagent_type": subagent_type,
                "message": f"Subagent dispatched: {task_id} ({subagent_type})",
            }
        except Exception as exc:
            return {
                "ok": False,
                "error": f"Failed to dispatch subagent: {exc}",
                "code": "dispatch_failed",
            }


_task_handler = TaskToolHandler()


def get_task_tool_spec() -> ToolSpec:
    return ToolSpec(
        name="task",
        description=(
            "Dispatch a read-only subagent for research, code review, or quality checks. "
            "Subagents cannot edit files or dispatch nested task calls."
        ),
        read_only=False,
        handler=_task_handler.handle,
        parameters=TASK_TOOL_PARAMETERS,
        category="task",
        risk_level="medium",
        timeout_seconds=900,
    )


def bind_task_executor(
    executor: Any,
    state: Any,
    *,
    registry: SubagentRegistry | None = None,
    factory_context: dict[str, Any] | None = None,
) -> None:
    _task_handler.bind(
        executor,
        state,
        registry=registry,
        factory_context=factory_context,
    )
