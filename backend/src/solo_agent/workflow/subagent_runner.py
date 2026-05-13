from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Mapping
from typing import Any

from solo_agent.agent.deps import AgentSettings
from solo_agent.providers import ChatMessage, ChatProvider

DEFAULT_SUBAGENT_TOOLS = {
    "workspace_snapshot",
    "read_file",
    "search_text",
    "inspect_python_symbols",
    "get_file_hash",
}


class SubagentRunner:
    """Synchronous read-only child task runner.

    This is intentionally not a background child graph. It gathers bounded
    scoped evidence and makes one provider call to produce structured findings.
    """

    def __init__(
        self,
        provider: ChatProvider,
        tool_registry: Any,
        settings: AgentSettings,
    ) -> None:
        self.provider = provider
        self.tool_registry = tool_registry
        self.settings = settings

    async def run(
        self,
        *,
        task_id: str,
        description: str,
        prompt: str,
        subagent_type: str,
        read_paths: list[str],
        allowed_tools: list[str],
        timeout_seconds: int,
        parent_session_id: str,
        parent_run_id: str | None = None,
    ) -> dict[str, Any]:
        bounded_timeout = max(1, min(int(timeout_seconds or self.settings.subagent_timeout_seconds), 3600))
        sanitized_tools = _sanitize_allowed_tools(allowed_tools)
        base = {
            "task_id": str(task_id),
            "subagent_type": str(subagent_type or "general-purpose"),
            "description": str(description or ""),
            "status": "failed",
            "result": "",
            "findings": [],
            "evidence": [],
            "read_paths": [str(path) for path in read_paths],
            "metadata": {
                "mode": "sync_child_agent",
                "provider": getattr(self.provider, "name", ""),
                "model": getattr(self.provider, "model", ""),
                "timeout_seconds": bounded_timeout,
                "parent_session_id": parent_session_id,
                "parent_run_id": parent_run_id,
                "allowed_tools": sorted(sanitized_tools),
            },
        }
        try:
            evidence = await self._collect_evidence(read_paths, sanitized_tools)
            response = await asyncio.wait_for(
                self.provider.complete(
                    _messages(
                        description=description,
                        prompt=prompt,
                        subagent_type=subagent_type,
                        read_paths=read_paths,
                        evidence=evidence,
                    ),
                    temperature=float(self.settings.temperature),
                    max_tokens=int(self.settings.response_max_tokens),
                ),
                timeout=bounded_timeout,
            )
            findings = _parse_findings(response, evidence)
            return _json_safe(
                {
                    **base,
                    "status": "completed",
                    "result": str(response).strip(),
                    "findings": findings,
                    "evidence": evidence,
                }
            )
        except Exception as exc:
            return _json_safe({**base, "status": "failed", "error": str(exc)})

    async def _collect_evidence(self, read_paths: list[str], allowed_tools: set[str]) -> list[dict[str, Any]]:
        evidence: list[dict[str, Any]] = []
        if "workspace_snapshot" in allowed_tools:
            snapshot = await _call_registered_tool(self.tool_registry, "workspace_snapshot", {"path": ".", "max_entries": 80})
            evidence.append(_evidence_item("workspace_snapshot", ".", snapshot))

        for path in [str(item).strip() for item in read_paths if str(item).strip()][:8]:
            if "read_file" in allowed_tools:
                result = await _call_registered_tool(self.tool_registry, "read_file", {"path": path, "max_bytes": 24_000})
                if _tool_ok(result):
                    evidence.append(_evidence_item("read_file", path, result))
                    if path.endswith(".py") and "inspect_python_symbols" in allowed_tools:
                        symbols = await _call_registered_tool(self.tool_registry, "inspect_python_symbols", {"path": path})
                        evidence.append(_evidence_item("inspect_python_symbols", path, symbols))
                    continue
            if "workspace_snapshot" in allowed_tools:
                result = await _call_registered_tool(
                    self.tool_registry,
                    "workspace_snapshot",
                    {"path": path, "max_entries": 80},
                )
                evidence.append(_evidence_item("workspace_snapshot", path, result))

        if "search_text" in allowed_tools:
            for keyword in _keywords(" ".join([*read_paths, *[str(item.get("content", "")) for item in evidence], ""])):
                result = await _call_registered_tool(
                    self.tool_registry,
                    "search_text",
                    {"query": keyword, "path": ".", "max_matches": 20},
                )
                evidence.append(_evidence_item("search_text", keyword, result))
        return evidence[:20]


async def _call_registered_tool(tool_registry: Any, name: str, arguments: dict[str, Any]) -> Any:
    if tool_registry is None:
        return {"ok": False, "tool": name, "error": "no tool registry"}
    if hasattr(tool_registry, "call_tool"):
        return await _maybe_await(tool_registry.call_tool(name, arguments))
    if hasattr(tool_registry, "call"):
        return await _maybe_await(tool_registry.call(name, arguments))
    return {"ok": False, "tool": name, "error": "tool registry cannot call tools"}


def _sanitize_allowed_tools(allowed_tools: list[str]) -> set[str]:
    requested = {str(tool).strip() for tool in allowed_tools if str(tool).strip()}
    return (requested or DEFAULT_SUBAGENT_TOOLS) & DEFAULT_SUBAGENT_TOOLS


def _messages(
    *,
    description: str,
    prompt: str,
    subagent_type: str,
    read_paths: list[str],
    evidence: list[dict[str, Any]],
) -> list[ChatMessage]:
    system = (
        "You are Solo Agent's synchronous read-only subagent. "
        "Analyze only the supplied scoped evidence. Do not write files, run tests, "
        "modify state, or call tools. Return concise structured findings for the parent agent."
    )
    user = (
        f"Subtask description: {description}\n"
        f"Subagent type: {subagent_type or 'general-purpose'}\n"
        f"Allowed read paths: {json.dumps(read_paths, ensure_ascii=False)}\n\n"
        f"Full subtask prompt:\n{prompt}\n\n"
        f"Scoped evidence JSON:\n{json.dumps(evidence, ensure_ascii=False, default=str)[:60_000]}\n\n"
        "Output requirements:\n"
        "- Summarize the answer first.\n"
        "- Include bullet findings with evidence references when possible.\n"
        "- Do not claim edits or command execution.\n"
    )
    return [ChatMessage(role="system", content=system), ChatMessage(role="user", content=user)]


def _parse_findings(response: str, evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lines = [line.strip(" -\t") for line in response.splitlines() if line.strip()]
    findings = [
        {
            "title": line[:80],
            "summary": line,
            "evidence": _evidence_refs(evidence),
            "confidence": 0.6,
        }
        for line in lines[:5]
    ]
    if findings:
        return findings
    return [
        {
            "title": "Subagent analysis",
            "summary": response.strip(),
            "evidence": _evidence_refs(evidence),
            "confidence": 0.5,
        }
    ]


def _evidence_item(tool: str, target: str, result: Any) -> dict[str, Any]:
    return _json_safe({"tool": tool, "target": target, "result": result})


def _evidence_refs(evidence: list[dict[str, Any]]) -> list[str]:
    return [f"{item.get('tool')}:{item.get('target')}" for item in evidence[:5]]


def _keywords(text: str) -> list[str]:
    seen: set[str] = set()
    words: list[str] = []
    for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]{3,}|[\u4e00-\u9fff]{2,}", text):
        key = token.casefold()
        if key in seen:
            continue
        seen.add(key)
        words.append(token)
        if len(words) >= 2:
            return words
    return words


def _tool_ok(result: Any) -> bool:
    return isinstance(result, Mapping) and result.get("ok") is not False


async def _maybe_await(value: Any) -> Any:
    if hasattr(value, "__await__"):
        return await value
    return value


def _json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))
