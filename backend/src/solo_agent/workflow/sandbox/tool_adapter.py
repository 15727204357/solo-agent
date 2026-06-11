from __future__ import annotations

import inspect
import json
from collections.abc import Mapping
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field


def create_tool_schema(
    name: str,
    description: str,
    parameters: Mapping[str, Any],
    handler: Any | None = None,
) -> type[BaseModel]:
    """Create a Pydantic schema from ToolSpec parameter metadata."""
    del description
    normalized = _normalize_parameters(parameters)
    annotations: dict[str, Any] = {}
    namespace: dict[str, Any] = {}
    signature = inspect.signature(handler) if callable(handler) else None
    for param_name, param_info in normalized.items():
        annotations[param_name] = _json_type_to_python(str(param_info.get("type", "string")))
        field_kwargs = {"description": param_info.get("description", "")}
        if "default" in param_info:
            field_kwargs["default"] = param_info["default"]
        elif (
            signature is not None
            and (parameter := signature.parameters.get(param_name)) is not None
            and parameter.default is not inspect.Signature.empty
        ):
            field_kwargs["default"] = parameter.default
        else:
            field_kwargs["default"] = ...
        namespace[param_name] = Field(**field_kwargs)

    namespace["__annotations__"] = annotations
    return type(f"{name}_schema", (BaseModel,), namespace)


def _normalize_parameters(parameters: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    if parameters.get("type") == "object" and isinstance(parameters.get("properties"), Mapping):
        required = set(parameters.get("required") or [])
        normalized: dict[str, dict[str, Any]] = {}
        for param_name, param_info in parameters["properties"].items():
            normalized[param_name] = _normalize_parameter_info(param_info)
            if param_name not in required and "default" not in normalized[param_name]:
                normalized[param_name]["default"] = None
        return normalized
    return {
        param_name: _normalize_parameter_info(param_info)
        for param_name, param_info in parameters.items()
    }


def _normalize_parameter_info(param_info: Any) -> dict[str, Any]:
    if isinstance(param_info, str):
        return {"type": "string", "description": param_info}
    if isinstance(param_info, Mapping):
        return dict(param_info)
    return {"type": "string", "description": str(param_info)}


def _json_type_to_python(param_type: str) -> Any:
    if param_type == "string":
        return str
    if param_type == "integer":
        return int
    if param_type == "number":
        return float
    if param_type == "boolean":
        return bool
    if param_type == "array":
        return list
    if param_type == "object":
        return dict
    return Any


def build_langchain_tool(
    name: str,
    description: str,
    handler: Any,
    parameters: Mapping[str, Any],
    registry: Any | None = None,
) -> StructuredTool:
    """Wrap a registered tool as a LangChain StructuredTool."""
    schema = create_tool_schema(name, description, parameters, handler=handler) if parameters else None

    async def _wrapper(**kwargs: Any) -> str:
        if registry is not None:
            result = registry.call(name, kwargs)
        else:
            result = handler(**kwargs)
        if inspect.isawaitable(result):
            result = await result
        return _format_tool_result(result)

    return StructuredTool(
        name=name,
        description=description,
        args_schema=schema,
        coroutine=_wrapper,
    )


def _format_tool_result(result: Any) -> str:
    if isinstance(result, dict):
        return json.dumps(result, ensure_ascii=False, default=str)
    return str(result)


def filter_tools_by_allowlist(
    all_tools: list[StructuredTool],
    allowed_names: set[str],
) -> list[StructuredTool]:
    """Filter tools by an explicit allowlist."""
    return [t for t in all_tools if t.name in allowed_names]


# Default read-only tool allowlist for subagents. Edit tools remain available to the lead agent only.
READONLY_TOOL_NAMES = {
    "find_files",
    "list_files",
    "read_file",
    "search_code",
    "search_text",
    "get_file_hash",
    "workspace_snapshot",
    "inspect_python_symbols",
    "run_command",
    "git_status",
    "git_diff",
    "git_show",
    "git_recent_changes",
    "read_test_failure",
    "list_skills",
    "load_skill",
}
