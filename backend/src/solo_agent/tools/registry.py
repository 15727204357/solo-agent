"""Tool registration and guarded execution."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from solo_agent.inspectors import (
    EgressInspector,
    InspectionResult,
    Inspector,
    RepetitionInspector,
    SecurityInspector,
    ToolCall,
)

from .readonly import WorkspaceTools

ToolHandler = Callable[..., Mapping[str, Any]]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    read_only: bool
    handler: ToolHandler
    parameters: Mapping[str, Any] = field(default_factory=dict)
    category: str = "context"
    risk_level: str = "low"
    requires_approval: bool = False
    timeout_seconds: float | None = None
    max_output_bytes: int | None = None
    progress_labels: tuple[str, ...] = ()


class ToolRegistry:
    """Registry for workspace-bounded tools with pre-execution safety checks."""

    def __init__(
        self,
        *,
        workspace_root: str | Path,
        inspectors: list[Inspector] | None = None,
        is_plan_mode: bool = False,
    ) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        self.is_plan_mode = is_plan_mode
        self._tools: dict[str, ToolSpec] = {}
        self.inspectors: list[Inspector] = inspectors or [
            SecurityInspector(),
            EgressInspector(),
            RepetitionInspector(),
        ]
        self._workspace_tools = WorkspaceTools(self.workspace_root)
        self.register_readonly_tools()

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            raise ValueError(f"Tool already registered: {spec.name}")
        self._tools[spec.name] = spec

    def register_readonly_tools(self) -> None:
        if self._tools:
            return

        self.register(
            ToolSpec(
                name="list_files",
                description="List files under the workspace root.",
                read_only=True,
                handler=self._workspace_tools.list_files,
                category="context",
                progress_labels=("scan_started", "scan_completed"),
                parameters={
                    "path": "Directory or file path relative to workspace root.",
                    "recursive": "Whether to recursively list nested files.",
                    "max_entries": "Maximum entries to return.",
                    "include_dirs": "Whether directory entries are included.",
                },
            )
        )
        self.register(
            ToolSpec(
                name="read_file",
                description="Read a UTF-8 text file inside the workspace root.",
                read_only=True,
                handler=self._workspace_tools.read_file,
                category="context",
                progress_labels=("read_started", "read_completed"),
                parameters={
                    "path": "File path relative to workspace root.",
                    "encoding": "Text encoding, defaults to utf-8.",
                    "max_bytes": "Maximum bytes to read.",
                },
            )
        )
        self.register(
            ToolSpec(
                name="search_text",
                description="Search text within files inside the workspace root.",
                read_only=True,
                handler=self._workspace_tools.search_text,
                category="context",
                progress_labels=("search_started", "search_completed"),
                parameters={
                    "query": "Literal text to search for.",
                    "path": "Directory or file path relative to workspace root.",
                    "glob": "Filename glob to include.",
                    "max_matches": "Maximum matches to return.",
                },
            )
        )
        self.register(
            ToolSpec(
                name="get_file_hash",
                description="Return sha256, size, and modified time for a workspace file.",
                read_only=True,
                handler=self._workspace_tools.get_file_hash,
                category="context",
                parameters={"path": "File path relative to workspace root."},
            )
        )
        self.register(
            ToolSpec(
                name="workspace_snapshot",
                description="Summarize workspace files, language mix, and recent changes.",
                read_only=True,
                handler=self._workspace_tools.workspace_snapshot,
                category="context",
                parameters={
                    "path": "Directory or file path relative to workspace root.",
                    "max_entries": "Maximum files to inspect.",
                },
            )
        )
        self.register(
            ToolSpec(
                name="inspect_python_symbols",
                description="Inspect imports, classes, functions, and methods in a Python file.",
                read_only=True,
                handler=self._workspace_tools.inspect_python_symbols,
                category="code_intelligence",
                parameters={"path": "Python file path relative to workspace root."},
            )
        )
        self.register(
            ToolSpec(
                name="prepare_edit",
                description="Prepare a hash-anchored edit package for a workspace file.",
                read_only=True,
                handler=self._workspace_tools.prepare_edit,
                category="edit",
                risk_level="medium",
                parameters={
                    "path": "File path relative to workspace root.",
                    "old_text": "Optional exact anchor text.",
                    "line_start": "Optional 1-based start line.",
                    "line_end": "Optional 1-based end line.",
                },
            )
        )
        self.register(
            ToolSpec(
                name="preview_patch",
                description="Preview a hash-anchored text edit as a unified diff without writing.",
                read_only=True,
                handler=self._workspace_tools.preview_patch,
                category="edit",
                risk_level="medium",
                parameters={
                    "path": "File path relative to workspace root.",
                    "expected_hash": "Current sha256 from get_file_hash or prepare_edit.",
                    "new_text": "Replacement text.",
                    "old_text": "Optional exact text to replace.",
                    "line_start": "Optional 1-based start line.",
                    "line_end": "Optional 1-based end line.",
                },
            )
        )
        self.register(
            ToolSpec(
                name="apply_text_edit",
                description="Apply a hash-anchored workspace edit when anchors still match.",
                read_only=False,
                handler=self._workspace_tools.apply_text_edit,
                category="edit",
                risk_level="high",
                requires_approval=True,
                parameters={
                    "path": "File path relative to workspace root.",
                    "expected_hash": "Current sha256 from get_file_hash or prepare_edit.",
                    "new_text": "Replacement text.",
                    "old_text": "Optional exact text to replace.",
                    "line_start": "Optional 1-based start line.",
                    "line_end": "Optional 1-based end line.",
                },
            )
        )
        self.register(
            ToolSpec(
                name="run_pytest",
                description="Run the fixed pytest command in the workspace.",
                read_only=True,
                handler=self._workspace_tools.run_pytest,
                category="quality",
                risk_level="medium",
                timeout_seconds=60,
                max_output_bytes=24_000,
                progress_labels=("command_started", "command_completed"),
                parameters={"target": "Optional workspace path to test."},
            )
        )
        self.register(
            ToolSpec(
                name="run_ruff_check",
                description="Run the fixed ruff check command in the workspace.",
                read_only=True,
                handler=self._workspace_tools.run_ruff_check,
                category="quality",
                risk_level="medium",
                timeout_seconds=60,
                max_output_bytes=24_000,
                progress_labels=("command_started", "command_completed"),
                parameters={"target": "Optional workspace path to check."},
            )
        )
        self.register(
            ToolSpec(
                name="run_ruff_format_check",
                description="Run the fixed ruff format --check command in the workspace.",
                read_only=True,
                handler=self._workspace_tools.run_ruff_format_check,
                category="quality",
                risk_level="medium",
                timeout_seconds=60,
                max_output_bytes=24_000,
                progress_labels=("command_started", "command_completed"),
                parameters={"target": "Optional workspace path to check formatting."},
            )
        )
        self.register(
            ToolSpec(
                name="git_status",
                description="Run git status --short --branch in the workspace.",
                read_only=True,
                handler=self._workspace_tools.git_status,
                category="vcs",
                risk_level="low",
                timeout_seconds=30,
                max_output_bytes=12_000,
                progress_labels=("command_started", "command_completed"),
            )
        )
        self.register(
            ToolSpec(
                name="git_diff",
                description="Run git diff for the workspace or a single workspace path.",
                read_only=True,
                handler=self._workspace_tools.git_diff,
                category="vcs",
                risk_level="medium",
                timeout_seconds=30,
                max_output_bytes=24_000,
                progress_labels=("command_started", "command_completed"),
                parameters={"path": "Optional workspace path to diff."},
            )
        )
        self.register(
            ToolSpec(
                name="git_recent_changes",
                description="Show recent git commits with a short oneline log.",
                read_only=True,
                handler=self._workspace_tools.git_recent_changes,
                category="vcs",
                risk_level="low",
                timeout_seconds=30,
                max_output_bytes=16_000,
                progress_labels=("command_started", "command_completed"),
                parameters={
                    "limit": "Maximum commits to show, capped at 50.",
                    "path": "Optional workspace path to filter recent commits.",
                },
            )
        )
        self.register(
            ToolSpec(
                name="read_test_failure",
                description="Extract failed pytest files, test names, and assertion snippets from pytest output.",
                read_only=True,
                handler=self._workspace_tools.read_test_failure,
                category="quality",
                risk_level="low",
                parameters={
                    "output": "Pytest stdout/stderr text to parse.",
                    "max_failures": "Maximum failures to return.",
                },
            )
        )
        self.register(
            ToolSpec(
                name="targeted_pytest",
                description="Run pytest -q against one validated workspace test path or node id.",
                read_only=True,
                handler=self._workspace_tools.targeted_pytest,
                category="quality",
                risk_level="medium",
                timeout_seconds=60,
                max_output_bytes=24_000,
                progress_labels=("command_started", "command_completed"),
                parameters={"target": "Workspace test path or path::test_name node id."},
            )
        )
        self.register(
            ToolSpec(
                name="list_skills",
                description="List project SKILL.md files available to the agent.",
                read_only=True,
                handler=self._workspace_tools.list_skills,
                category="skill",
                parameters={"max_entries": "Maximum skills to return."},
            )
        )
        self.register(
            ToolSpec(
                name="load_skill",
                description="Load and sanitize a project SKILL.md file.",
                read_only=True,
                handler=self._workspace_tools.load_skill,
                category="skill",
                parameters={"path": "SKILL.md path or skill name."},
            )
        )
        self.register(
            ToolSpec(
                name="select_relevant_skills",
                description="Select relevant project skills for a task.",
                read_only=True,
                handler=self._workspace_tools.select_relevant_skills,
                category="skill",
                parameters={
                    "task": "Current user task.",
                    "max_skills": "Maximum skills to select.",
                },
            )
        )
        self.register(
            ToolSpec(
                name="task_create",
                description="Create a structured TaskList item for a LangGraph thread.",
                read_only=False,
                handler=self._workspace_tools.task_create,
                category="task",
                risk_level="low",
                parameters={
                    "thread_id": "Current LangGraph thread/session id.",
                    "subject": "Imperative task subject.",
                    "description": "Optional task details.",
                    "status": "pending, in_progress, completed, blocked, or deleted.",
                    "active_form": "Present-continuous form for resume prompts.",
                    "blocked_by": "Optional blocking task ids or reasons.",
                    "blocks": "Optional task ids this task blocks.",
                },
            )
        )
        self.register(
            ToolSpec(
                name="task_get",
                description="Get a structured TaskList item by id.",
                read_only=True,
                handler=self._workspace_tools.task_get,
                category="task",
                risk_level="low",
                parameters={"thread_id": "Current LangGraph thread/session id.", "task_id": "Task id."},
            )
        )
        self.register(
            ToolSpec(
                name="task_list",
                description="List structured TaskList items for a LangGraph thread.",
                read_only=True,
                handler=self._workspace_tools.task_list,
                category="task",
                risk_level="low",
                parameters={
                    "thread_id": "Current LangGraph thread/session id.",
                    "include_deleted": "Whether deleted tasks should be returned.",
                },
            )
        )
        self.register(
            ToolSpec(
                name="task_update",
                description="Update a structured TaskList item.",
                read_only=False,
                handler=self._workspace_tools.task_update,
                category="task",
                risk_level="low",
                parameters={
                    "thread_id": "Current LangGraph thread/session id.",
                    "task_id": "Task id.",
                    "subject": "Optional replacement subject.",
                    "description": "Optional replacement description.",
                    "status": "Optional status update.",
                    "active_form": "Optional resume wording.",
                    "blocked_by": "Optional blockers.",
                    "blocks": "Optional task ids this task blocks.",
                },
            )
        )
        if self.is_plan_mode:
            self.register(
                ToolSpec(
                    name="write_todos",
                    description=(
                        "Plan mode only: replace or merge the current session TaskList. "
                        "Use for complex tasks that need explicit progress tracking."
                    ),
                    read_only=False,
                    handler=self._workspace_tools.write_todos,
                    category="task",
                    risk_level="low",
                    parameters={
                        "tasks": (
                            "List of task objects with optional id, subject, description, status, "
                            "active_form, blocked_by, blocks, and metadata."
                        ),
                        "merge": "Whether to merge into the existing TaskList instead of replacing it.",
                    },
                )
            )
        else:
            for name in ("task_create", "task_get", "task_list", "task_update"):
                self._tools.pop(name, None)

    def list_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": spec.name,
                "description": spec.description,
                "read_only": spec.read_only,
                "category": spec.category,
                "risk_level": spec.risk_level,
                "requires_approval": spec.requires_approval,
                "timeout_seconds": spec.timeout_seconds,
                "max_output_bytes": spec.max_output_bytes,
                "progress_labels": list(spec.progress_labels),
                "parameters": dict(spec.parameters),
            }
            for spec in self._tools.values()
        ]

    def get(self, name: str) -> ToolSpec:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise KeyError(f"Unknown tool: {name}") from exc

    def inspect_text(self, text: str) -> InspectionResult:
        for inspector in self.inspectors:
            result = inspector.inspect_text(text)
            if not result.allowed:
                return result
        return InspectionResult.allow()

    def call(self, name: str, arguments: Mapping[str, Any] | None = None) -> dict[str, Any]:
        args = dict(arguments or {})
        try:
            spec = self.get(name)
        except KeyError as exc:
            return {
                "ok": False,
                "tool": name,
                "error": str(exc),
                "code": "tool_not_found",
                "metadata": {},
            }

        call = ToolCall(name=name, arguments=args)

        for inspector in self.inspectors:
            result = self._inspect_with_registered_tool(inspector, spec, call)
            if not result.allowed:
                return {
                    "ok": False,
                    "tool": name,
                    "error": result.reason,
                    "code": result.code,
                    "metadata": dict(result.metadata),
                }

        try:
            result = spec.handler(**args)
        except (KeyError, PermissionError, ValueError, OSError) as exc:
            return {
                "ok": False,
                "tool": name,
                "error": str(exc),
                "code": "tool_error",
                "metadata": {},
            }
        except TimeoutError as exc:
            return {
                "ok": False,
                "tool": name,
                "error": str(exc),
                "code": "tool_timeout",
                "metadata": self._metadata(spec, truncated=True),
            }

        output = dict(result)
        metadata = self._metadata(spec, truncated=bool(output.get("truncated", False)))
        metadata.update(dict(output.pop("metadata", {})))
        return {"ok": True, "tool": name, "result": output, "metadata": metadata}

    def _inspect_with_registered_tool(
        self,
        inspector: Inspector,
        spec: ToolSpec,
        call: ToolCall,
    ) -> InspectionResult:
        result = inspector.inspect_tool_call(call)
        if result.allowed:
            return result
        if result.code != "tool_not_allowed" or call.name not in self._tools:
            return result

        combined = " ".join(_flatten_values(call.arguments))
        text_result = inspector.inspect_text(combined)
        if spec.category == "task" and text_result.allowed:
            return InspectionResult.allow({"tool": call.name, "task_state_tool": True})
        if text_result.code == "write_not_allowed" and spec.category == "edit":
            return InspectionResult.allow({"tool": call.name, "safe_edit_tool": True})
        return text_result

    def _metadata(self, spec: ToolSpec, *, truncated: bool = False) -> dict[str, Any]:
        return {
            "category": spec.category,
            "risk_level": spec.risk_level,
            "read_only": spec.read_only,
            "requires_approval": spec.requires_approval,
            "timeout_seconds": spec.timeout_seconds,
            "max_output_bytes": spec.max_output_bytes,
            "progress_labels": list(spec.progress_labels),
            "truncated": truncated,
        }


def _flatten_values(value: Any) -> list[str]:
    if isinstance(value, str):
        return [_redact_large_text(value)]
    if isinstance(value, Mapping):
        values: list[str] = []
        for item in value.values():
            values.extend(_flatten_values(item))
        return values
    if isinstance(value, (list, tuple, set, frozenset)):
        values = []
        for item in value:
            values.extend(_flatten_values(item))
        return values
    if value is None:
        return []
    return [str(value)]


def _redact_large_text(value: str) -> str:
    """Keep inspector input bounded while preserving dangerous command markers."""

    return re.sub(r"\s+", " ", value[:4_000])


def create_default_registry(workspace_root: str | Path, *, is_plan_mode: bool = False) -> ToolRegistry:
    return ToolRegistry(workspace_root=workspace_root, is_plan_mode=is_plan_mode)
