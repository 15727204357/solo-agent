"""Parse command/tool output into structured failure reports."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from .models import FailureKind, FailureReport

_PYTEST_FAIL_RE = re.compile(r"FAILED\s+([^\s]+)")
_PYTEST_FILE_LINE_RE = re.compile(r"([A-Za-z0-9_./\\-]+\.py):(\d+)")
_RUFF_RE = re.compile(r"([A-Z]+[0-9]+)\s+([^\n]+)")
_RUFF_FILE_LINE_RE = re.compile(r"([A-Za-z0-9_./\\-]+\.py):(\d+):(\d+):\s*([A-Z]+[0-9]+)")
_MISSING_DEP_RE = re.compile(r"No module named ['\"]([^'\"]+)['\"]|ModuleNotFoundError|ImportError", re.IGNORECASE)
_TYPE_RE = re.compile(r"\b(mypy|pyright|type error|Argument .* incompatible|has incompatible type)\b", re.IGNORECASE)
_POLICY_RE = re.compile(r"\b(policy|not allowed|blocked|approval|required|forbidden|violates)\b", re.IGNORECASE)
_PATCH_CONFLICT_RE = re.compile(r"\b(hash mismatch|anchor|stale|conflict|patch .* failed)\b", re.IGNORECASE)


def classify_failures(evidence: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for item in evidence:
        report = classify_command_failure(item)
        if report is not None:
            reports.append(report.to_dict())
    return reports


def classify_command_failure(evidence_item: Mapping[str, Any]) -> FailureReport | None:
    result = evidence_item.get("result")
    payload = _payload(result)
    if _command_ok(result):
        return None
    command = str(evidence_item.get("command") or payload.get("command") or payload.get("tool") or "")
    output = _output_text(payload)
    returncode = _returncode(payload)
    lower = f"{command}\n{output}".casefold()

    if _PATCH_CONFLICT_RE.search(output):
        return FailureReport(
            kind=FailureKind.PATCH_CONFLICT,
            command=command,
            returncode=returncode,
            summary="Patch or edit anchors no longer match the workspace.",
            snippet=_best_snippet(output),
            raw_output=output[:4000],
        )
    if _POLICY_RE.search(str(payload.get("error") or "")) or str(payload.get("code") or "").endswith("not_allowed"):
        return FailureReport(
            kind=FailureKind.POLICY_BLOCKED,
            command=command,
            returncode=returncode,
            summary=str(payload.get("error") or "Command was blocked by policy."),
            snippet=_best_snippet(output or str(payload)),
            raw_output=output[:4000],
            retryable=False,
        )
    if _MISSING_DEP_RE.search(output):
        missing = _MISSING_DEP_RE.search(output)
        module = missing.group(1) if missing and missing.lastindex else ""
        return FailureReport(
            kind=FailureKind.DEPENDENCY_MISSING,
            command=command,
            returncode=returncode,
            summary=f"Missing dependency{f': {module}' if module else ''}.",
            snippet=_best_snippet(output),
            raw_output=output[:4000],
            retryable=False,
            metadata={"missing_module": module} if module else {},
        )
    if "pytest" in lower or "failed" in lower:
        return _pytest_failure(command, returncode, output)
    if "ruff" in lower or _RUFF_FILE_LINE_RE.search(output):
        return _ruff_failure(command, returncode, output)
    if _TYPE_RE.search(output):
        file, line = _first_file_line(output)
        return FailureReport(
            kind=FailureKind.TYPE_FAILURE,
            command=command,
            returncode=returncode,
            summary="Type checker reported an incompatible type or type error.",
            file=file,
            line=line,
            snippet=_best_snippet(output),
            raw_output=output[:4000],
        )
    if payload.get("timed_out") or "timed out" in lower or returncode is None:
        return FailureReport(
            kind=FailureKind.ENVIRONMENT_ERROR,
            command=command,
            returncode=returncode,
            summary="Command timed out or environment execution failed.",
            snippet=_best_snippet(output),
            raw_output=output[:4000],
            retryable=False,
        )
    return FailureReport(
        kind=FailureKind.UNKNOWN_FAILURE,
        command=command,
        returncode=returncode,
        summary="Command failed but no specific failure taxonomy matched.",
        snippet=_best_snippet(output),
        raw_output=output[:4000],
    )


def _pytest_failure(command: str, returncode: int | None, output: str) -> FailureReport:
    failed = _PYTEST_FAIL_RE.search(output)
    failing_test = failed.group(1) if failed else ""
    file, line = _first_file_line(output)
    assertion = next((line_text.strip() for line_text in output.splitlines() if line_text.strip().startswith("E ")), "")
    return FailureReport(
        kind=FailureKind.TEST_FAILURE,
        command=command,
        returncode=returncode,
        summary=f"Pytest failed{f': {failing_test}' if failing_test else ''}.",
        file=file,
        line=line,
        failing_test=failing_test,
        snippet=assertion or _best_snippet(output),
        stack_trace=_trim_trace(output),
        raw_output=output[:4000],
    )


def _ruff_failure(command: str, returncode: int | None, output: str) -> FailureReport:
    file, line = _first_file_line(output)
    rule_match = _RUFF_FILE_LINE_RE.search(output) or _RUFF_RE.search(output)
    rule = rule_match.group(4) if rule_match and rule_match.lastindex and rule_match.lastindex >= 4 else ""
    if not rule and rule_match:
        rule = rule_match.group(1)
    return FailureReport(
        kind=FailureKind.LINT_FAILURE,
        command=command,
        returncode=returncode,
        summary=f"Ruff/lint check failed{f' with {rule}' if rule else ''}.",
        file=file,
        line=line,
        rule=rule,
        snippet=_best_snippet(output),
        raw_output=output[:4000],
    )


def _payload(result: Any) -> dict[str, Any]:
    if isinstance(result, Mapping):
        if "result" in result and isinstance(result.get("result"), Mapping):
            return dict(result.get("result") or {}) | {
                "ok": result.get("ok", True),
                "error": result.get("error", ""),
                "code": result.get("code", ""),
            }
        return dict(result)
    return {"output": str(result)}


def _command_ok(result: Any) -> bool:
    if isinstance(result, Mapping) and result.get("ok") is False:
        return False
    payload = _payload(result)
    if "returncode" in payload:
        return payload.get("returncode") == 0
    return bool(payload.get("ok", True))


def _output_text(payload: Mapping[str, Any]) -> str:
    return str(payload.get("output") or payload.get("stdout") or payload.get("stderr") or payload.get("error") or "")


def _returncode(payload: Mapping[str, Any]) -> int | None:
    value = payload.get("returncode")
    return value if isinstance(value, int) else None


def _first_file_line(output: str) -> tuple[str, int | None]:
    match = _RUFF_FILE_LINE_RE.search(output) or _PYTEST_FILE_LINE_RE.search(output)
    if not match:
        return "", None
    return match.group(1).replace("\\", "/"), int(match.group(2))


def _best_snippet(output: str) -> str:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    interesting = [
        line
        for line in lines
        if line.startswith(("E ", "FAILED", "ERROR", "ImportError", "ModuleNotFoundError")) or ".py:" in line
    ]
    return (interesting[0] if interesting else lines[0] if lines else "")[:600]


def _trim_trace(output: str) -> str:
    lines = output.splitlines()
    selected = [line for line in lines if line.strip().startswith(("E ", ">", "FAILED")) or ".py:" in line]
    return "\n".join(selected[:30])[:2000]
