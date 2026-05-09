"""测试本地沙箱隔离和工具适配器。"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import pytest
from solo_agent.workflow.sandbox.local import LocalSandboxProvider
from solo_agent.workflow.sandbox.provider import SandboxProvider
from solo_agent.workflow.sandbox.tool_adapter import (
    READONLY_TOOL_NAMES,
    build_langchain_tool,
    create_tool_schema,
    filter_tools_by_allowlist,
)


@pytest.fixture
def temp_root():
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


@pytest.mark.asyncio
async def test_local_sandbox_creates_directories(temp_root):
    provider = LocalSandboxProvider(runtime_root=temp_root)
    path = await provider.setup("session_a", "run_1")

    assert os.path.isdir(path)
    for sub in ("workspace", "uploads", "outputs"):
        assert os.path.isdir(Path(path) / sub)


@pytest.mark.asyncio
async def test_local_sandbox_teardown_cleans_up(temp_root):
    provider = LocalSandboxProvider(runtime_root=temp_root)
    path = await provider.setup("session_b", "run_2")
    assert os.path.isdir(path)

    await provider.teardown("session_b", "run_2")
    assert not os.path.exists(path)


@pytest.mark.asyncio
async def test_local_sandbox_teardown_nonexistent_does_not_raise(temp_root):
    provider = LocalSandboxProvider(runtime_root=temp_root)
    await provider.teardown("nonexistent", "run_x")


def test_local_sandbox_path_mapping(temp_root):
    provider = LocalSandboxProvider(runtime_root=temp_root)
    mapped = provider.map_path("s1", "r1", "file.py")
    expected = str(temp_root / "s1" / "r1" / "workspace" / "file.py")
    assert mapped == expected


def test_local_sandbox_rejects_absolute_paths(temp_root):
    provider = LocalSandboxProvider(runtime_root=temp_root)
    absolute = str((temp_root / "outside.txt").resolve())
    with pytest.raises(ValueError, match="relative"):
        provider.map_path("s1", "r1", absolute)


def test_local_sandbox_rejects_parent_traversal(temp_root):
    provider = LocalSandboxProvider(runtime_root=temp_root)
    with pytest.raises(ValueError, match="escapes"):
        provider.map_path("s1", "r1", "../outside.txt")


def test_local_sandbox_workspace_path(temp_root):
    provider = LocalSandboxProvider(runtime_root=temp_root)
    ws = provider.get_workspace("s1", "r1")
    expected = str(temp_root / "s1" / "r1" / "workspace")
    assert ws == expected


def test_sandbox_provider_is_protocol():
    assert issubclass(LocalSandboxProvider, SandboxProvider)


def test_create_tool_schema():
    params = {
        "path": {"type": "string", "description": "File path"},
        "recursive": {"type": "boolean", "description": "Recurse"},
    }
    schema = create_tool_schema("test_tool", "desc", params)
    assert schema.__name__ == "test_tool_schema"


def test_create_tool_schema_accepts_string_parameters():
    schema = create_tool_schema("test_tool", "desc", {"path": "File path"})

    field = schema.model_fields["path"]
    assert field.annotation is str
    assert field.description == "File path"


def test_create_tool_schema_accepts_json_schema_parameters():
    schema = create_tool_schema(
        "test_tool",
        "desc",
        {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path"},
                "limit": {"type": "integer", "description": "Limit"},
            },
            "required": ["path"],
        },
    )

    assert schema.model_fields["path"].is_required()
    assert schema.model_fields["limit"].default is None


def test_build_langchain_tool():
    def fake_handler(path=None, recursive=None):
        return {"ok": True, "result": f"listed {path}"}
    params = {
        "path": {"type": "string", "description": "Path"},
    }
    tool = build_langchain_tool("my_tool", "desc", fake_handler, params)
    assert tool.name == "my_tool"
    assert tool.description == "desc"
    assert tool.args_schema is not None


@pytest.mark.asyncio
async def test_build_langchain_tool_calls_registry_not_handler():
    class FakeRegistry:
        def __init__(self):
            self.calls: list[tuple[str, dict[str, Any]]] = []

        def call(self, name, arguments=None):
            self.calls.append((name, dict(arguments or {})))
            return {"ok": True, "result": {"value": "from registry"}}

    def forbidden_handler(**kwargs):
        raise AssertionError("handler should not be called")

    registry = FakeRegistry()
    tool = build_langchain_tool(
        "my_tool",
        "desc",
        forbidden_handler,
        {"path": "Path"},
        registry=registry,
    )

    result = await tool.ainvoke({"path": "README.md"})

    assert registry.calls == [("my_tool", {"path": "README.md"})]
    assert "from registry" in result


def test_filter_tools_by_allowlist():
    params = {"path": {"type": "string", "description": "p"}}
    t1 = build_langchain_tool("read_file", "read", lambda **kw: {}, params)
    t2 = build_langchain_tool("apply_text_edit", "write", lambda **kw: {}, params)
    t3 = build_langchain_tool("list_files", "list", lambda **kw: {}, params)

    all_tools = [t1, t2, t3]
    allowed = filter_tools_by_allowlist(all_tools, {"read_file", "list_files"})
    assert len(allowed) == 2
    assert all(t.name != "apply_text_edit" for t in allowed)


def test_readonly_tool_names():
    assert "read_file" in READONLY_TOOL_NAMES
    assert "search_text" in READONLY_TOOL_NAMES
    assert "list_files" in READONLY_TOOL_NAMES
    assert "apply_text_edit" not in READONLY_TOOL_NAMES
    assert "prepare_edit" not in READONLY_TOOL_NAMES


def test_local_sandbox_allows_normalized_child_path(temp_root):
    provider = LocalSandboxProvider(runtime_root=temp_root)
    mapped = provider.map_path("s1", "r1", "pkg/../file.py")
    assert mapped == str(temp_root / "s1" / "r1" / "workspace" / "file.py")
