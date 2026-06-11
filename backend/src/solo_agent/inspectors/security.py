"""Security-focused deterministic checks."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from .base import InspectionResult, ToolCall


class SecurityInspector:
    """Block obviously dangerous local actions and secret exfiltration.

    These checks are intentionally conservative. The registry owns the full
    tool contract, while this inspector blocks dangerous text, secret exposure,
    and unknown ad hoc tool calls before execution.
    """

    _dangerous_delete_patterns = (
        re.compile(r"\brm\s+(?:-[^\s]*[rf][^\s]*|-[^\s]*[fr][^\s]*)\b", re.I),
        re.compile(r"\bRemove-Item\b[^\n\r;|&]*\s-(?:Recurse|Force)\b", re.I),
        re.compile(r"\brmdir\s+(?:/s|/q)\b", re.I),
        re.compile(r"\bdel\s+(?:/s|/q|/f)\b", re.I),
        re.compile(r"\bgit\s+reset\s+--hard\b", re.I),
        re.compile(r"\bgit\s+clean\s+-[^\s]*[fdx][^\s]*\b", re.I),
    )
    _secret_patterns = (
        re.compile(r"\b[A-Z][A-Z0-9_]*(?:TOKEN|SECRET|KEY|PASSWORD|PASSWD)\b"),
        re.compile(r"\b(?:api[_-]?key|access[_-]?token|refresh[_-]?token)\b", re.I),
        re.compile(r"\b(?:OPENAI|DEEPSEEK|ANTHROPIC|GITHUB)_[A-Z0-9_]*KEY\b"),
        re.compile(r"\b(?:print|echo|cat|type)\s+(?:\$env:|\$|%)[A-Z0-9_]*(?:TOKEN|SECRET|KEY|PASSWORD|PASSWD)%?\b", re.I),
    )
    _write_intent_patterns = (
        re.compile(r"\b(?:write|modify|overwrite|delete|remove|rename|move)\s+file\b", re.I),
        re.compile(r"\b(?:touch|mkdir|New-Item|Set-Content|Add-Content|Out-File)\b", re.I),
    )

    def inspect_text(self, text: str) -> InspectionResult:
        normalized = text or ""

        for pattern in self._dangerous_delete_patterns:
            if pattern.search(normalized):
                return InspectionResult.block(
                    "Dangerous deletion or destructive git command was blocked.",
                    code="dangerous_deletion",
                    metadata={"pattern": pattern.pattern},
                )

        for pattern in self._secret_patterns:
            if pattern.search(normalized):
                return InspectionResult.block(
                    "Request appears to expose environment variables or secrets.",
                    code="secret_exposure",
                    metadata={"pattern": pattern.pattern},
                )

        for pattern in self._write_intent_patterns:
            if pattern.search(normalized):
                return InspectionResult.block(
                    "Write-like operations must go through registered guarded tools.",
                    code="write_not_allowed",
                    metadata={"pattern": pattern.pattern},
                )

        return InspectionResult.allow()

    def inspect_tool_call(self, call: ToolCall) -> InspectionResult:
        tool_name = call.name.strip()
        allowed_tools = {
            "workspace_snapshot",
            "find_files",
            "list_files",
            "read_file",
            "search_code",
            "search_text",
            "get_file_hash",
            "inspect_python_symbols",
            "code_map",
            "find_references",
            "analyze_impact",
            "semantic_code_search",
            "prepare_edit",
            "preview_patch",
            "apply_text_edit",
            "create_file",
            "mkdir",
            "move_path",
            "delete_path",
            "run_command",
            "run_pytest",
            "run_ruff_check",
            "run_ruff_format_check",
            "targeted_pytest",
            "read_test_failure",
            "git_status",
            "git_diff",
            "git_show",
            "git_recent_changes",
            "skills_list",
            "skill_view",
            "skill_manage",
            "skill_recipe_list",
            "skill_recipe_view",
            "skill_recipe_preview",
            "skill_recipe_run",
            "list_skills",
            "load_skill",
            "select_relevant_skills",
            "write_todos",
            "task",
        }
        if tool_name not in allowed_tools:
            return InspectionResult.block(
                f"Tool '{tool_name}' is not registered in the guarded tool layer.",
                code="tool_not_allowed",
                metadata={"tool": tool_name},
            )

        path = str(call.arguments.get("path", ""))
        if _looks_like_secret_path(path):
            return InspectionResult.block(
                "Reading environment or secret files is not allowed.",
                code="secret_file_access",
                metadata={"path": path},
            )

        combined = " ".join(_flatten_values(call.arguments))
        return self.inspect_text(combined)


def _flatten_values(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
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


def _looks_like_secret_path(path: str) -> bool:
    parts = re.split(r"[\\/]+", path)
    secret_names = {
        ".env",
        ".env.local",
        ".env.production",
        ".env.development",
        "id_rsa",
        "id_ed25519",
    }
    return any(part in secret_names or part.endswith(".pem") for part in parts)
