"""Built-in read-only subagent runner factories."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

from solo_agent.tools.registry import ToolSpec
from solo_agent.workflow.sandbox.tool_adapter import READONLY_TOOL_NAMES, build_langchain_tool
from solo_agent.workflow.subagent.registry import SubagentRegistry, get_builtin_registry

TOOL_WHITELISTS: dict[str, set[str]] = {
    "general-purpose": set(READONLY_TOOL_NAMES),
    "code-review": {
        "read_file",
        "search_text",
        "get_file_hash",
        "inspect_python_symbols",
        "git_status",
        "git_diff",
        "git_recent_changes",
        "list_files",
    },
    "quality": {
        "run_pytest",
        "run_ruff_check",
        "targeted_pytest",
        "read_test_failure",
        "read_file",
        "list_files",
        "search_text",
    },
}

SYSTEM_PROMPTS: dict[str, str] = {
    "general-purpose": (
        "You are a read-only solo-agent subagent for focused research. "
        "Use only read-only tools, do not edit files, and do not call the task tool. "
        "Return a concise structured summary."
    ),
    "code-review": (
        "You are a read-only code-review subagent. Look for correctness, security, "
        "maintainability, and test risks. Do not edit files or call the task tool. "
        "Return findings ordered by severity."
    ),
    "quality": (
        "You are a read-only quality subagent. Run bounded test and lint tools when useful, "
        "summarize results, and do not edit files or call the task tool."
    ),
}


def get_tool_whitelist(subagent_type: str) -> set[str]:
    return set(TOOL_WHITELISTS.get(subagent_type, TOOL_WHITELISTS["general-purpose"]))


def get_system_prompt(subagent_type: str) -> str:
    return SYSTEM_PROMPTS.get(subagent_type, SYSTEM_PROMPTS["general-purpose"])


def create_readonly_subagent_runner(
    subagent_type: str,
    *,
    model: Any,
    tool_registry: Any,
) -> Any:
    allowed_names = get_tool_whitelist(subagent_type)
    tools = [
        build_langchain_tool(
            name=spec.name,
            description=spec.description,
            handler=spec.handler,
            parameters=spec.parameters,
            registry=tool_registry,
        )
        for spec in getattr(tool_registry, "_tools", {}).values()
        if spec.name in allowed_names and _is_read_only_quality_tool(spec)
    ]
    agent = create_react_agent(
        model=model,
        tools=tools,
        prompt=get_system_prompt(subagent_type),
    )

    async def _runner(prompt: str, max_turns: int = 10, **_: Any) -> str:
        response = ""
        config = {"recursion_limit": max(2, int(max_turns) * 2)}
        async for chunk in agent.astream(
            {"messages": [HumanMessage(content=prompt)]},
            config=config,
            stream_mode="values",
        ):
            messages = chunk.get("messages", []) if isinstance(chunk, Mapping) else []
            last_msg = messages[-1] if messages else None
            content = getattr(last_msg, "content", None)
            if isinstance(content, str) and content:
                response = content
        return response

    return _runner


def register_builtin_factories(registry: SubagentRegistry | None = None) -> SubagentRegistry:
    target = registry or get_builtin_registry()
    for subagent_type in TOOL_WHITELISTS:
        if not target.has(subagent_type):
            target.register(
                subagent_type,
                lambda *, _type=subagent_type, **kwargs: create_readonly_subagent_runner(_type, **kwargs),
            )
    return target


def _is_read_only_quality_tool(spec: ToolSpec) -> bool:
    return bool(spec.read_only)


def _normalize_parameters(parameters: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    normalized: dict[str, dict[str, Any]] = {}
    for name, value in parameters.items():
        if isinstance(value, Mapping):
            normalized[name] = dict(value)
        else:
            normalized[name] = {"type": "string", "description": str(value), "default": ""}
    return normalized


register_builtin_factories()
