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
    capability: str = "context"
    visibility: str = "model"
    risk_level: str = "low"
    requires_approval: bool = False
    timeout_seconds: float | None = None
    max_output_bytes: int | None = None
    command_policy: Mapping[str, Any] = field(default_factory=dict)
    progress_labels: tuple[str, ...] = ()


class ToolRegistry:
    """Registry for workspace-bounded tools with pre-execution safety checks."""

    def __init__(
        self,
        *,
        workspace_root: str | Path,
        inspectors: list[Inspector] | None = None,
        is_plan_mode: bool = False,
        subagent_enabled: bool = False,
        command_workspace_root: str | Path | None = None,
        sandbox_mode: str = "local",
        sandbox_id: str = "",
        cache_root: str | Path | None = None,
        sandbox_network_policy: str = "deny",
        sandbox_command_timeout_seconds: int = 60,
        sandbox_max_output_bytes: int = 32_000,
        sandbox_max_changed_files: int = 200,
        sandbox_max_workspace_bytes: int = 512_000_000,
        codeintel_max_files: int = 2_000,
        codeintel_max_file_bytes: int = 512_000,
        codeintel_index_ttl_seconds: int = 30,
    ) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        self.command_workspace_root = Path(command_workspace_root or self.workspace_root).resolve()
        self.sandbox_mode = sandbox_mode
        self.sandbox_id = sandbox_id
        self.cache_root = Path(cache_root).resolve() if cache_root is not None else None
        self.sandbox_network_policy = sandbox_network_policy
        self.sandbox_command_timeout_seconds = sandbox_command_timeout_seconds
        self.sandbox_max_output_bytes = sandbox_max_output_bytes
        self.sandbox_max_changed_files = sandbox_max_changed_files
        self.sandbox_max_workspace_bytes = sandbox_max_workspace_bytes
        self.codeintel_max_files = codeintel_max_files
        self.codeintel_max_file_bytes = codeintel_max_file_bytes
        self.codeintel_index_ttl_seconds = codeintel_index_ttl_seconds
        self.is_plan_mode = is_plan_mode
        self.subagent_enabled = subagent_enabled
        self._tools: dict[str, ToolSpec] = {}
        self.inspectors: list[Inspector] = inspectors or [
            SecurityInspector(),
            EgressInspector(),
            RepetitionInspector(),
        ]
        self._workspace_tools = WorkspaceTools(
            self.workspace_root,
            command_workspace_root=self.command_workspace_root,
            sandbox_mode=self.sandbox_mode,
            sandbox_id=self.sandbox_id,
            cache_root=self.cache_root,
            sandbox_network_policy=self.sandbox_network_policy,
            sandbox_command_timeout_seconds=self.sandbox_command_timeout_seconds,
            sandbox_max_output_bytes=self.sandbox_max_output_bytes,
            sandbox_max_changed_files=self.sandbox_max_changed_files,
            sandbox_max_workspace_bytes=self.sandbox_max_workspace_bytes,
            codeintel_max_files=self.codeintel_max_files,
            codeintel_max_file_bytes=self.codeintel_max_file_bytes,
            codeintel_index_ttl_seconds=self.codeintel_index_ttl_seconds,
        )
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
                visibility="compat",
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
                name="find_files",
                description="Find files by glob under a workspace path.",
                read_only=True,
                handler=self._workspace_tools.find_files,
                category="context",
                capability="filesystem",
                progress_labels=("scan_started", "scan_completed"),
                parameters={
                    "path": {"type": "string", "description": "Directory or file path relative to workspace root."},
                    "glob": {"type": "string", "description": "Filename glob to match, such as *.py.", "default": "*"},
                    "recursive": {
                        "type": "boolean",
                        "description": "Whether to recursively search nested files.",
                        "default": True,
                    },
                    "max_entries": {"type": "integer", "description": "Maximum entries to return.", "default": 200},
                    "include_dirs": {
                        "type": "boolean",
                        "description": "Whether directory entries are included.",
                        "default": False,
                    },
                },
            )
        )
        self.register(
            ToolSpec(
                name="read_file",
                description="Read a UTF-8 text file inside the workspace root, optionally by line range.",
                read_only=True,
                handler=self._workspace_tools.read_file,
                category="context",
                capability="filesystem",
                progress_labels=("read_started", "read_completed"),
                parameters={
                    "path": {"type": "string", "description": "File path relative to workspace root."},
                    "encoding": {"type": "string", "description": "Text encoding, defaults to utf-8.", "default": "utf-8"},
                    "max_bytes": {"type": "integer", "description": "Maximum bytes to read.", "default": 64000},
                    "line_start": {"type": "integer", "description": "Optional 1-based start line.", "default": None},
                    "line_end": {"type": "integer", "description": "Optional 1-based end line.", "default": None},
                    "include_line_numbers": {
                        "type": "boolean",
                        "description": "Whether returned content includes line numbers.",
                        "default": True,
                    },
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
                visibility="compat",
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
                name="search_code",
                description="Search code text with literal or regex matching, glob filters, and context lines.",
                read_only=True,
                handler=self._workspace_tools.search_code,
                category="context",
                capability="code_search",
                progress_labels=("search_started", "search_completed"),
                parameters={
                    "query": {"type": "string", "description": "Text or regex pattern to search for."},
                    "path": {
                        "type": "string",
                        "description": "Directory or file path relative to workspace root.",
                        "default": ".",
                    },
                    "glob": {"type": "string", "description": "Filename glob to include.", "default": "*"},
                    "regex": {"type": "boolean", "description": "Whether query is a regular expression.", "default": False},
                    "context_lines": {
                        "type": "integer",
                        "description": "Number of surrounding lines to include.",
                        "default": 0,
                    },
                    "max_matches": {"type": "integer", "description": "Maximum matches to return.", "default": 100},
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
                visibility="compat",
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
                capability="workspace",
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
                visibility="compat",
                parameters={"path": "Python file path relative to workspace root."},
            )
        )
        self.register(
            ToolSpec(
                name="code_map",
                description="Build a lightweight code map with Python symbols, imports, calls, tests, and entrypoints.",
                read_only=True,
                handler=self._workspace_tools.code_map,
                category="code_intelligence",
                capability="code_map",
                parameters={
                    "path": {"type": "string", "description": "Workspace path to map.", "default": "."},
                    "max_files": {"type": "integer", "description": "Maximum files to scan.", "default": 500},
                },
            )
        )
        self.register(
            ToolSpec(
                name="find_references",
                description="Find definitions, imports, and references for a symbol using AST and token search.",
                read_only=True,
                handler=self._workspace_tools.find_references,
                category="code_intelligence",
                capability="references",
                parameters={
                    "symbol": {"type": "string", "description": "Symbol name to find."},
                    "path": {"type": "string", "description": "Workspace path to search.", "default": "."},
                    "max_matches": {"type": "integer", "description": "Maximum matches to return.", "default": 100},
                },
            )
        )
        self.register(
            ToolSpec(
                name="analyze_impact",
                description="Estimate affected files, references, related tests, and verification commands.",
                read_only=True,
                handler=self._workspace_tools.analyze_impact,
                category="code_intelligence",
                capability="impact_analysis",
                parameters={
                    "paths": {"type": "array", "description": "Changed or target paths.", "default": []},
                    "symbols": {"type": "array", "description": "Changed or target symbols.", "default": []},
                    "include_tests": {
                        "type": "boolean",
                        "description": "Whether to include related tests and pytest commands.",
                        "default": True,
                    },
                },
            )
        )
        self.register(
            ToolSpec(
                name="semantic_code_search",
                description="Search code semantically with local token scoring over paths, symbols, comments, and text.",
                read_only=True,
                handler=self._workspace_tools.semantic_code_search,
                category="code_intelligence",
                capability="semantic_search",
                parameters={
                    "query": {"type": "string", "description": "Natural language or code search query."},
                    "path": {"type": "string", "description": "Workspace path to search.", "default": "."},
                    "max_matches": {"type": "integer", "description": "Maximum matches to return.", "default": 20},
                },
            )
        )
        self.register(
            ToolSpec(
                name="code_index_status",
                description="Inspect or refresh the persistent Python LSP-like code intelligence index.",
                read_only=True,
                handler=self._workspace_tools.code_index_status,
                category="code_intelligence",
                capability="code_index",
                parameters={
                    "path": {"type": "string", "description": "Workspace path to inspect.", "default": "."},
                    "refresh": {"type": "boolean", "description": "Force an index refresh.", "default": False},
                },
            )
        )
        self.register(
            ToolSpec(
                name="symbol_search",
                description="Search indexed Python symbols by name or qualified name.",
                read_only=True,
                handler=self._workspace_tools.symbol_search,
                category="code_intelligence",
                capability="symbols",
                parameters={
                    "query": {"type": "string", "description": "Symbol name or qualified-name fragment."},
                    "kind": {"type": "string", "description": "Optional symbol kind filter.", "default": None},
                    "max_results": {"type": "integer", "description": "Maximum symbols to return.", "default": 50},
                },
            )
        )
        self.register(
            ToolSpec(
                name="symbol_definition",
                description="Find indexed definitions for a symbol or qualified name.",
                read_only=True,
                handler=self._workspace_tools.symbol_definition,
                category="code_intelligence",
                capability="definitions",
                parameters={
                    "symbol": {"type": "string", "description": "Simple symbol name.", "default": None},
                    "qualified_name": {"type": "string", "description": "Fully qualified symbol name.", "default": None},
                    "path": {"type": "string", "description": "Optional file path filter.", "default": None},
                },
            )
        )
        self.register(
            ToolSpec(
                name="call_graph",
                description="Return indexed Python call graph edges for a symbol or path.",
                read_only=True,
                handler=self._workspace_tools.call_graph,
                category="code_intelligence",
                capability="call_graph",
                parameters={
                    "symbol": {"type": "string", "description": "Symbol or call name.", "default": None},
                    "path": {"type": "string", "description": "Optional file path.", "default": None},
                    "direction": {"type": "string", "description": "incoming, outgoing, or both.", "default": "both"},
                    "depth": {"type": "integer", "description": "Traversal depth hint.", "default": 1},
                },
            )
        )
        self.register(
            ToolSpec(
                name="test_relevance",
                description="Score pytest files relevant to changed paths or symbols.",
                read_only=True,
                handler=self._workspace_tools.test_relevance,
                category="code_intelligence",
                capability="test_relevance",
                parameters={
                    "paths": {"type": "array", "description": "Changed or target paths.", "default": []},
                    "symbols": {"type": "array", "description": "Changed or target symbols.", "default": []},
                    "max_tests": {"type": "integer", "description": "Maximum tests to return.", "default": 20},
                },
            )
        )
        self.register(
            ToolSpec(
                name="prepare_edit",
                description="Prepare a hash-anchored edit package for a workspace file.",
                read_only=True,
                handler=self._workspace_tools.prepare_edit,
                category="edit",
                capability="edit",
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
                capability="edit",
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
                capability="edit",
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
                name="create_file",
                description="Create a new workspace file only when it is expected to be absent.",
                read_only=False,
                handler=self._workspace_tools.create_file,
                category="filesystem",
                capability="filesystem_write",
                risk_level="medium",
                requires_approval=True,
                parameters={
                    "path": {"type": "string", "description": "File path relative to workspace root."},
                    "content": {"type": "string", "description": "File content to write.", "default": ""},
                    "parents": {
                        "type": "boolean",
                        "description": "Whether missing parent directories may be created.",
                        "default": False,
                    },
                    "expected_absent": {
                        "type": "boolean",
                        "description": "Must be true to prove the caller expects no existing file.",
                        "default": True,
                    },
                },
            )
        )
        self.register(
            ToolSpec(
                name="mkdir",
                description="Create a workspace directory.",
                read_only=False,
                handler=self._workspace_tools.mkdir,
                category="filesystem",
                capability="filesystem_write",
                risk_level="medium",
                requires_approval=True,
                parameters={
                    "path": {"type": "string", "description": "Directory path relative to workspace root."},
                    "parents": {"type": "boolean", "description": "Whether missing parents may be created.", "default": True},
                    "exist_ok": {
                        "type": "boolean",
                        "description": "Whether an existing directory is accepted.",
                        "default": True,
                    },
                },
            )
        )
        self.register(
            ToolSpec(
                name="move_path",
                description="Move a workspace file or small directory with hash/absence safeguards.",
                read_only=False,
                handler=self._workspace_tools.move_path,
                category="filesystem",
                capability="filesystem_write",
                risk_level="high",
                requires_approval=True,
                parameters={
                    "source": {"type": "string", "description": "Source path relative to workspace root."},
                    "destination": {"type": "string", "description": "Destination path relative to workspace root."},
                    "expected_hash": {"type": "string", "description": "Required sha256 for file sources.", "default": None},
                    "expected_absent": {
                        "type": "boolean",
                        "description": "Whether destination must not exist.",
                        "default": True,
                    },
                    "parents": {
                        "type": "boolean",
                        "description": "Whether missing destination parents may be created.",
                        "default": False,
                    },
                },
            )
        )
        self.register(
            ToolSpec(
                name="delete_path",
                description="Delete a workspace file or small directory with hash safeguards.",
                read_only=False,
                handler=self._workspace_tools.delete_path,
                category="filesystem",
                capability="filesystem_write",
                risk_level="high",
                requires_approval=True,
                parameters={
                    "path": {"type": "string", "description": "Path relative to workspace root."},
                    "expected_hash": {"type": "string", "description": "Required sha256 for files.", "default": None},
                    "recursive": {
                        "type": "boolean",
                        "description": "Whether small directories may be deleted recursively.",
                        "default": False,
                    },
                    "max_entries": {
                        "type": "integer",
                        "description": "Maximum directory entries allowed for recursive delete.",
                        "default": 20,
                    },
                },
            )
        )
        self.register(
            ToolSpec(
                name="run_command",
                description="Run an allowlisted programming command with structured argv inside the workspace.",
                read_only=True,
                handler=self._workspace_tools.run_command,
                category="quality",
                capability="command",
                risk_level="medium",
                timeout_seconds=60,
                max_output_bytes=24_000,
                progress_labels=("command_started", "command_completed"),
                command_policy={"shell": False, "allowlist": "programming_quality_commands"},
                parameters={
                    "command": {
                        "type": "string",
                        "description": "Executable name, such as python, pytest, uv, npm, cargo, go, ruff, git.",
                    },
                    "args": {
                        "type": "array",
                        "description": "Argument list. Shell metacharacters are rejected.",
                        "default": [],
                    },
                    "cwd": {"type": "string", "description": "Workspace-relative working directory.", "default": "."},
                    "timeout_seconds": {
                        "type": "integer",
                        "description": "Timeout budget, capped at 300 seconds.",
                        "default": 60,
                    },
                    "max_output_bytes": {
                        "type": "integer",
                        "description": "Maximum stdout/stderr bytes to return.",
                        "default": 32000,
                    },
                    "purpose": {"type": "string", "description": "Short reason for running the command.", "default": ""},
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
                visibility="compat",
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
                visibility="compat",
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
                visibility="compat",
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
                capability="git",
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
                capability="git",
                risk_level="medium",
                timeout_seconds=30,
                max_output_bytes=24_000,
                progress_labels=("command_started", "command_completed"),
                parameters={"path": "Optional workspace path to diff."},
            )
        )
        self.register(
            ToolSpec(
                name="git_show",
                description="Run git show --stat --patch for a safe ref, optionally scoped to a path.",
                read_only=True,
                handler=self._workspace_tools.git_show,
                category="vcs",
                capability="git",
                risk_level="medium",
                timeout_seconds=30,
                max_output_bytes=24_000,
                progress_labels=("command_started", "command_completed"),
                parameters={
                    "ref": "Git ref to inspect, defaults to HEAD.",
                    "path": "Optional workspace path to scope the output.",
                },
            )
        )
        self.register(
            ToolSpec(
                name="git_recent_changes",
                description="Show recent git commits with a short oneline log.",
                read_only=True,
                handler=self._workspace_tools.git_recent_changes,
                category="vcs",
                visibility="compat",
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
                visibility="compat",
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
                visibility="compat",
                risk_level="medium",
                timeout_seconds=60,
                max_output_bytes=24_000,
                progress_labels=("command_started", "command_completed"),
                parameters={"target": "Workspace test path or path::test_name node id."},
            )
        )
        self.register(
            ToolSpec(
                name="skills_list",
                description="List compact Hermes-style skill index entries from workspace skills/.",
                read_only=True,
                handler=self._workspace_tools.skills_list,
                category="skill",
                capability="skill",
                risk_level="low",
                parameters={
                    "query": "Optional search query over skill metadata.",
                    "category": "Optional skill category filter.",
                    "max_entries": "Maximum compact skills to return.",
                },
            )
        )
        self.register(
            ToolSpec(
                name="skill_view",
                description="Load one workspace skill's SKILL.md or a supporting references/templates/scripts/assets file.",
                read_only=True,
                handler=self._workspace_tools.skill_view,
                category="skill",
                capability="skill",
                risk_level="low",
                parameters={
                    "name": "Skill name, path, or slug.",
                    "file_path": "Optional file path relative to the skill directory. Defaults to SKILL.md.",
                    "max_bytes": "Maximum bytes to read before truncating.",
                },
            )
        )
        self.register(
            ToolSpec(
                name="skill_manage",
                description=(
                    "Propose a Hermes-style skill change. Supports create, patch, edit, delete, write_file, "
                    "and remove_file; returns a pending SkillChangeProposal and never writes directly."
                ),
                read_only=True,
                handler=self._workspace_tools.skill_manage,
                category="skill",
                capability="skill_manage",
                risk_level="medium",
                parameters={
                    "action": "create, patch, edit, delete, write_file, or remove_file.",
                    "skill_name": "Skill display name or existing skill name/slug.",
                    "content": "Full content for create/edit/write_file.",
                    "category": "Category folder for create; defaults to workflows.",
                    "old_string": "Exact old string for patch.",
                    "new_string": "Replacement string for patch.",
                    "file_path": "Supporting file path for write_file/remove_file.",
                },
            )
        )
        self.register(
            ToolSpec(
                name="skill_recipe_list",
                description="List compact declarative recipes declared by workspace skills.",
                read_only=True,
                handler=self._workspace_tools.skill_recipe_list,
                category="skill",
                capability="skill_recipe",
                risk_level="low",
                parameters={
                    "skill_name": "Optional skill name, path, or slug to list recipes for.",
                    "query": "Optional task query for matching recipe when clauses.",
                    "max_entries": "Maximum compact recipes to return.",
                },
            )
        )
        self.register(
            ToolSpec(
                name="skill_recipe_view",
                description="Load one declarative recipe's full compiled definition and optional recipe file content.",
                read_only=True,
                handler=self._workspace_tools.skill_recipe_view,
                category="skill",
                capability="skill_recipe",
                risk_level="low",
                parameters={
                    "skill_name": "Skill name, path, or slug.",
                    "recipe_id": "Recipe id.",
                    "max_bytes": "Maximum bytes to read before truncating recipe file content.",
                },
            )
        )
        self.register(
            ToolSpec(
                name="skill_recipe_preview",
                description="Compile a declarative recipe into an auditable subflow plan without executing it.",
                read_only=True,
                handler=self._workspace_tools.skill_recipe_preview,
                category="skill",
                capability="skill_recipe",
                risk_level="low",
                parameters={
                    "skill_name": "Skill name, path, or slug.",
                    "recipe_id": "Recipe id.",
                    "user_input": "Current user task for template rendering.",
                    "plan": "Current plan for template rendering.",
                },
            )
        )
        self.register(
            ToolSpec(
                name="skill_recipe_run",
                description="Run auto-executable read/search/git-read/test/build/lint/check steps from a declarative recipe.",
                read_only=True,
                handler=self._workspace_tools.skill_recipe_run,
                category="skill",
                capability="skill_recipe",
                risk_level="medium",
                timeout_seconds=120,
                max_output_bytes=24_000,
                parameters={
                    "skill_name": "Skill name, path, or slug.",
                    "recipe_id": "Recipe id.",
                    "user_input": "Current user task for template rendering.",
                    "plan": "Current plan for template rendering.",
                },
            )
        )
        self.register(
            ToolSpec(
                name="skill_script_run",
                description="Run a declared read-only or quality skill script with structured argv and no shell.",
                read_only=True,
                handler=self._workspace_tools.skill_script_run,
                category="skill",
                capability="skill_script",
                risk_level="medium",
                timeout_seconds=60,
                max_output_bytes=24_000,
                command_policy={
                    "shell": False,
                    "allowlist": "declared_skill_scripts",
                    "auto_boundary": "read-only/quality scripts only",
                },
                parameters={
                    "skill_name": "Skill name, path, or slug.",
                    "script_id": "Declared script id from the skill contract.",
                    "args": {
                        "type": "array",
                        "description": "Extra structured argv items. Shell metacharacters and write-like flags are rejected.",
                        "default": [],
                    },
                    "timeout_seconds": {
                        "type": "integer",
                        "description": "Timeout budget, capped at 300 seconds.",
                        "default": 60,
                    },
                    "max_output_bytes": {"type": "integer", "description": "Maximum output bytes.", "default": 24000},
                },
            )
        )
        self.register(
            ToolSpec(
                name="list_skills",
                description="List project SKILL.md files available to the agent.",
                read_only=True,
                handler=self._workspace_tools.list_skills,
                category="skill",
                visibility="internal",
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
                visibility="internal",
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
                visibility="internal",
                parameters={
                    "task": "Current user task.",
                    "max_skills": "Maximum skills to select.",
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
        if self.subagent_enabled:
            self.register(
                ToolSpec(
                    name="task",
                    description="Run a focused subagent task with scoped context and return structured findings.",
                    read_only=True,
                    handler=self._workspace_tools.task,
                    category="subagent",
                    capability="subagent",
                    risk_level="medium",
                    parameters={
                        "description": "Short subtask description for trace events.",
                        "prompt": "Complete self-contained subagent instructions.",
                        "subagent_type": "Subagent kind, such as general-purpose, code-review, research, or quality.",
                        "task_id": "Optional stable task id. Generated when omitted.",
                        "read_paths": "Optional workspace-relative paths the subagent should focus on.",
                        "allowed_tools": "Optional read-only tool names available to the subagent.",
                        "timeout_seconds": "Optional timeout budget for the subtask.",
                    },
                )
            )

    def list_tools(self, visibility: str = "model") -> list[dict[str, Any]]:
        if visibility not in {"model", "all", "internal", "compat"}:
            raise ValueError(f"Unknown tool visibility: {visibility}")
        return [
            {
                "name": spec.name,
                "description": spec.description,
                "read_only": spec.read_only,
                "category": spec.category,
                "capability": spec.capability,
                "visibility": spec.visibility,
                "risk_level": spec.risk_level,
                "requires_approval": spec.requires_approval,
                "timeout_seconds": spec.timeout_seconds,
                "max_output_bytes": spec.max_output_bytes,
                "command_policy": dict(spec.command_policy),
                "progress_labels": list(spec.progress_labels),
                "parameters": dict(spec.parameters),
            }
            for spec in self._tools.values()
            if visibility == "all" or spec.visibility == visibility
        ]

    def list_all_tools(self) -> list[dict[str, Any]]:
        return self.list_tools(visibility="all")

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
        if spec.category == "subagent":
            metadata.update(dict(output.get("metadata", {})))
        else:
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
        if (
            spec.category in {"task", "subagent", "context", "code_intelligence", "quality", "vcs", "filesystem"}
            and text_result.allowed
        ):
            return InspectionResult.allow({"tool": call.name, "task_state_tool": True})
        if text_result.code == "write_not_allowed" and spec.category == "edit":
            return InspectionResult.allow({"tool": call.name, "safe_edit_tool": True})
        return text_result

    def _metadata(self, spec: ToolSpec, *, truncated: bool = False) -> dict[str, Any]:
        return {
            "category": spec.category,
            "capability": spec.capability,
            "visibility": spec.visibility,
            "risk_level": spec.risk_level,
            "read_only": spec.read_only,
            "requires_approval": spec.requires_approval,
            "timeout_seconds": spec.timeout_seconds,
            "max_output_bytes": spec.max_output_bytes,
            "command_policy": dict(spec.command_policy),
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


def create_default_registry(
    workspace_root: str | Path,
    *,
    is_plan_mode: bool = False,
    subagent_enabled: bool = False,
    command_workspace_root: str | Path | None = None,
    sandbox_mode: str = "local",
    sandbox_id: str = "",
    cache_root: str | Path | None = None,
    sandbox_network_policy: str = "deny",
    sandbox_command_timeout_seconds: int = 60,
    sandbox_max_output_bytes: int = 32_000,
    sandbox_max_changed_files: int = 200,
    sandbox_max_workspace_bytes: int = 512_000_000,
    codeintel_max_files: int = 2_000,
    codeintel_max_file_bytes: int = 512_000,
    codeintel_index_ttl_seconds: int = 30,
) -> ToolRegistry:
    return ToolRegistry(
        workspace_root=workspace_root,
        is_plan_mode=is_plan_mode,
        subagent_enabled=subagent_enabled,
        command_workspace_root=command_workspace_root,
        sandbox_mode=sandbox_mode,
        sandbox_id=sandbox_id,
        cache_root=cache_root,
        sandbox_network_policy=sandbox_network_policy,
        sandbox_command_timeout_seconds=sandbox_command_timeout_seconds,
        sandbox_max_output_bytes=sandbox_max_output_bytes,
        sandbox_max_changed_files=sandbox_max_changed_files,
        sandbox_max_workspace_bytes=sandbox_max_workspace_bytes,
        codeintel_max_files=codeintel_max_files,
        codeintel_max_file_bytes=codeintel_max_file_bytes,
        codeintel_index_ttl_seconds=codeintel_index_ttl_seconds,
    )
