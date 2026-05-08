"""Lead agent factory for the workflow runtime."""

from __future__ import annotations

import asyncio
from typing import Any

from langgraph.prebuilt import create_react_agent

from solo_agent.workflow.langchain_adapter import LangChainChatAdapter
from solo_agent.workflow.sandbox.tool_adapter import build_langchain_tool
from solo_agent.workflow.subagent.task_tool import get_task_tool_spec


class LeadAgentFactory:
    """Create the lead LangGraph react agent from registry-backed tools."""

    def __init__(
        self,
        *,
        model: LangChainChatAdapter,
        tool_registry: Any,
        system_prompt: str = "",
        event_queue: asyncio.Queue | None = None,
    ):
        self._model = model
        self._tool_registry = tool_registry
        self._system_prompt = system_prompt
        self._event_queue = event_queue

    def build_tools(
        self,
        include_task: bool = True,
        allowed_names: set[str] | None = None,
    ) -> list[Any]:
        """从 ToolRegistry 构建 LangChain 工具，统一走 registry.call 安全边界。"""
        tools = []
        for name, spec in self._tool_registry._tools.items():
            if allowed_names is not None and name not in allowed_names:
                continue
            tools.append(
                build_langchain_tool(
                    name=spec.name,
                    description=spec.description,
                    handler=spec.handler,
                    parameters=dict(spec.parameters),
                    registry=self._tool_registry,
                )
            )
        if include_task:
            task_spec = get_task_tool_spec()
            tools.append(
                build_langchain_tool(
                    name=task_spec.name,
                    description=task_spec.description,
                    handler=task_spec.handler,
                    parameters=dict(task_spec.parameters),
                )
            )
        return tools

    def create_agent(
        self,
        include_task: bool = True,
        allowed_names: set[str] | None = None,
        state: Any = None,
    ) -> Any:
        del state
        tools = self.build_tools(include_task=include_task, allowed_names=allowed_names)
        return create_react_agent(
            model=self._model,
            tools=tools,
            prompt=self._system_prompt,
        )
