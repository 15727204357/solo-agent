from __future__ import annotations

import json
import re
import shlex
from collections.abc import Awaitable, Callable, Mapping
from pathlib import PurePosixPath
from typing import Any
from uuid import uuid4

from pydantic import ValidationError

from .models import PatchEdit, PatchProposal, PatchRequest, StopGate, VerificationCommand, VerificationPlan, VerificationResult

ToolCaller = Callable[[str, dict[str, Any]], Awaitable[Mapping[str, Any]]]


class PatchProposalError(ValueError):
    pass


def extract_patch_request(raw: str) -> PatchRequest:
    payload = _extract_json(raw)
    try:
        if isinstance(payload, list):
            payload = {"edits": payload}
        if not isinstance(payload, dict):
            raise PatchProposalError("patch response must be a JSON object or edit list")
        return PatchRequest.model_validate(_normalize_patch_payload(payload))
    except ValidationError as exc:
        raise PatchProposalError(str(exc)) from exc


async def build_patch_proposal(
    request: PatchRequest,
    *,
    session_id: str,
    run_id: str,
    call_tool: ToolCaller,
    patch_id: str | None = None,
    impact_analysis: Mapping[str, Any] | None = None,
) -> PatchProposal:
    prepared: list[PatchEdit] = []
    diffs: list[str] = []

    for edit in request.edits:
        expected_hash = edit.expected_hash
        if not expected_hash:
            prepare_args: dict[str, Any] = {"path": edit.path}
            if edit.old_text is not None:
                prepare_args["old_text"] = edit.old_text
            if edit.line_start is not None and edit.line_end is not None:
                prepare_args["line_start"] = edit.line_start
                prepare_args["line_end"] = edit.line_end
            prepared_result = await call_tool("prepare_edit", prepare_args)
            prepared_payload = _tool_payload(prepared_result)
            expected_hash = str(prepared_payload.get("expected_hash") or prepared_payload.get("sha256") or "")
            if not expected_hash:
                raise PatchProposalError(f"prepare_edit did not return an expected hash for {edit.path}")

        preview_edit = edit.model_copy(update={"expected_hash": expected_hash})
        preview_result = await call_tool("preview_patch", preview_edit.preview_arguments())
        preview_payload = _tool_payload(preview_result)
        if not bool(preview_payload.get("changed", True)):
            continue
        diff = str(preview_payload.get("diff") or "")
        if not diff:
            raise PatchProposalError(f"preview_patch did not return a diff for {edit.path}")
        final_edit = preview_edit.model_copy(
            update={
                "diff": diff,
                "changed": True,
                "new_sha256": preview_payload.get("new_sha256"),
            }
        )
        prepared.append(final_edit)
        diffs.append(diff)

    if not prepared:
        raise PatchProposalError("patch contains no effective changes")

    verification_plan = _build_verification_plan(prepared, explicit=request.verification_plan, impact_analysis=impact_analysis)
    stop_gate = _initial_stop_gate(verification_plan)

    return PatchProposal(
        id=patch_id or f"patch_{uuid4().hex[:16]}",
        session_id=session_id,
        run_id=run_id,
        status="pending",
        summary=request.summary,
        diff="\n".join(diffs),
        edits=prepared,
        verification_plan=verification_plan,
        stop_gate=stop_gate,
    )


async def apply_approved_patch(
    proposal: PatchProposal,
    *,
    call_tool: ToolCaller,
) -> PatchProposal:
    apply_results: list[dict[str, Any]] = []
    for edit in proposal.edits:
        result = dict(await call_tool("apply_text_edit", edit.apply_arguments()))
        apply_results.append(result)
        if not _tool_ok(result):
            return proposal.model_copy(
                update={
                    "status": "failed",
                    "apply_results": apply_results,
                    "error": str(result.get("error") or "apply_text_edit failed"),
                }
            )

    verification_plan = _ensure_verification_plan(proposal)
    verification = await _run_verification_plan(verification_plan, call_tool=call_tool)
    stop_gate = _stop_gate_from_verification(verification_plan, verification)
    return proposal.model_copy(
        update={
            "status": "applied" if stop_gate.approval_ready else "verification_failed",
            "apply_results": apply_results,
            "verification_plan": verification_plan,
            "verification": verification,
            "stop_gate": stop_gate,
            "error": None if stop_gate.approval_ready else stop_gate.reason or "verification failed",
        }
    )


def _normalize_patch_payload(payload: dict[str, Any]) -> dict[str, Any]:
    edits = payload.get("edits", [])
    normalized_edits = []
    for edit in edits:
        if not isinstance(edit, dict):
            continue
        normalized = dict(edit)
        if "old_text" not in normalized and "old" in normalized:
            normalized["old_text"] = normalized["old"]
        if "new_text" not in normalized and "new" in normalized:
            normalized["new_text"] = normalized["new"]
        normalized_edits.append(normalized)
    normalized: dict[str, Any] = {"summary": str(payload.get("summary") or ""), "edits": normalized_edits}
    if isinstance(payload.get("verification_plan"), Mapping):
        normalized["verification_plan"] = payload["verification_plan"]
    return normalized


def _build_verification_plan(
    edits: list[PatchEdit],
    *,
    explicit: VerificationPlan | Mapping[str, Any] | None,
    impact_analysis: Mapping[str, Any] | None,
) -> VerificationPlan:
    if explicit is not None:
        plan = explicit if isinstance(explicit, VerificationPlan) else VerificationPlan.model_validate(explicit)
        if not plan.required and not plan.reason.strip():
            return plan.model_copy(update={"reason": "Verification explicitly waived for a non-code or low-risk patch."})
        return plan

    required = _requires_verification(edits)
    if not required:
        return VerificationPlan(
            commands=[],
            required=False,
            reason="Documentation/text-only patch; verification is not required by the stop gate.",
        )

    command_texts = _impact_verify_commands(impact_analysis)
    commands = [_verification_command_from_text(command, purpose="impact analysis") for command in command_texts]
    if not commands:
        commands = [
            VerificationCommand(
                command="pytest -q",
                args=["-q"],
                target="",
                tool="run_pytest",
                purpose="default Python regression check",
            ),
            VerificationCommand(
                command="ruff check .",
                args=["check", "."],
                target=".",
                tool="run_ruff_check",
                purpose="default lint check",
            ),
        ]
    return VerificationPlan(
        commands=commands,
        required=True,
        reason="Code/config/test patch requires passing verification before approval readiness.",
    )


def _ensure_verification_plan(proposal: PatchProposal) -> VerificationPlan:
    plan = proposal.verification_plan
    if plan.commands or not plan.required:
        return plan
    return _build_verification_plan(proposal.edits, explicit=None, impact_analysis=None)


def _requires_verification(edits: list[PatchEdit]) -> bool:
    return not all(_is_documentation_path(edit.path) for edit in edits)


def _is_documentation_path(path: str) -> bool:
    normalized = path.replace("\\", "/").strip().lower()
    if not normalized:
        return False
    pure = PurePosixPath(normalized)
    if pure.name in {"readme", "readme.md", "changelog", "changelog.md", "license", "license.md"}:
        return True
    if pure.suffix in {".md", ".mdx", ".rst", ".txt"}:
        return True
    return normalized.startswith("docs/")


def _impact_verify_commands(impact_analysis: Mapping[str, Any] | None) -> list[str]:
    if not impact_analysis:
        return []
    commands = impact_analysis.get("verify_commands")
    if not isinstance(commands, list):
        return []
    return [str(command).strip() for command in commands if str(command).strip()]


def _verification_command_from_text(command: str, *, purpose: str) -> VerificationCommand:
    try:
        tokens = shlex.split(command)
    except ValueError:
        tokens = command.split()
    tool: str | None = None
    target: str | None = None
    if tokens:
        executable = tokens[0].lower()
        args = tokens[1:]
        if executable in {"pytest", "py.test"}:
            target = _pytest_target(args)
            tool = "targeted_pytest" if target else "run_pytest"
        elif executable == "ruff":
            tool = "run_ruff_check"
            target = _ruff_target(args)
        else:
            args = tokens[1:]
    else:
        args = []
    return VerificationCommand(command=command, args=args, target=target, tool=tool, purpose=purpose)


def _pytest_target(args: list[str]) -> str:
    targets = [arg for arg in args if arg and not arg.startswith("-")]
    return targets[0] if targets else ""


def _ruff_target(args: list[str]) -> str:
    targets = [arg for arg in args if arg and not arg.startswith("-") and arg != "check"]
    return targets[0] if targets else "."


def _initial_stop_gate(plan: VerificationPlan) -> StopGate:
    if not plan.required:
        return StopGate(status="waived", approval_ready=True, reason=plan.reason)
    commands = ", ".join(command.command for command in plan.commands) or "verification command"
    return StopGate(
        status="missing",
        approval_ready=False,
        reason="Verification has not run for this patch proposal.",
        missing_evidence=[f"Passing result for: {commands}"],
    )


async def _run_verification_plan(plan: VerificationPlan, *, call_tool: ToolCaller) -> VerificationResult:
    if not plan.required:
        return VerificationResult(commands=[], ok=True)
    command_results: list[dict[str, Any]] = []
    pytest_result: dict[str, Any] | None = None
    ruff_result: dict[str, Any] | None = None
    for command in plan.commands:
        entry = await _run_verification_command(command, call_tool=call_tool)
        command_results.append(entry)
        payload = entry["result"] if isinstance(entry.get("result"), Mapping) else {}
        if entry.get("tool") in {"run_pytest", "targeted_pytest"} and pytest_result is None:
            pytest_result = dict(payload)
        if entry.get("tool") == "run_ruff_check" and ruff_result is None:
            ruff_result = dict(payload)
    ok = bool(command_results) and all(bool(entry.get("ok")) for entry in command_results)
    return VerificationResult(pytest=pytest_result, ruff=ruff_result, commands=command_results, ok=ok)


async def _run_verification_command(command: VerificationCommand, *, call_tool: ToolCaller) -> dict[str, Any]:
    tool = command.tool or "run_command"
    arguments = _verification_arguments(command, tool)
    result = dict(await call_tool(tool, arguments))
    payload = dict(result.get("result") or result) if isinstance(result.get("result", result), Mapping) else {}
    return {
        "command": command.command,
        "tool": tool,
        "arguments": arguments,
        "result": payload,
        "ok": _command_ok(result),
    }


def _verification_arguments(command: VerificationCommand, tool: str) -> dict[str, Any]:
    if tool in {"run_pytest", "targeted_pytest"}:
        return {"target": command.target or _pytest_target(command.args)}
    if tool == "run_ruff_check":
        return {"target": command.target or _ruff_target(command.args)}
    return {"command": command.command, "args": command.args, "purpose": command.purpose}


def _stop_gate_from_verification(plan: VerificationPlan, verification: VerificationResult) -> StopGate:
    if not plan.required:
        return StopGate(status="waived", approval_ready=True, reason=plan.reason)
    if verification.ok:
        return StopGate(
            status="passed",
            approval_ready=True,
            reason="All planned verification commands passed.",
            missing_evidence=[],
        )
    failed = [entry.get("command") for entry in verification.commands if not entry.get("ok")]
    if failed:
        return StopGate(
            status="failed",
            approval_ready=False,
            reason="One or more planned verification commands failed.",
            missing_evidence=[f"Passing result for: {command}" for command in failed if command],
        )
    return StopGate(
        status="missing",
        approval_ready=False,
        reason="Verification plan did not produce passing command evidence.",
        missing_evidence=["Passing verification command evidence is required."],
    )


def _extract_json(raw: str) -> Any:
    text = raw.strip()
    if not text:
        raise PatchProposalError("empty patch response")
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start_candidates = [index for index in (text.find("{"), text.find("[")) if index >= 0]
        if not start_candidates:
            raise PatchProposalError("patch response did not contain JSON") from None
        start = min(start_candidates)
        end = max(text.rfind("}"), text.rfind("]"))
        if end < start:
            raise PatchProposalError("patch response did not contain complete JSON") from None
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise PatchProposalError(str(exc)) from exc


def _tool_payload(result: Mapping[str, Any]) -> dict[str, Any]:
    if not _tool_ok(result):
        raise PatchProposalError(str(result.get("error") or "tool call failed"))
    payload = result.get("result", result)
    if not isinstance(payload, Mapping):
        raise PatchProposalError("tool result must be a JSON object")
    return dict(payload)


def _tool_ok(result: Mapping[str, Any]) -> bool:
    return bool(result.get("ok", True))


def _command_ok(result: Mapping[str, Any]) -> bool:
    if not _tool_ok(result):
        return False
    payload = result.get("result", result)
    if isinstance(payload, Mapping) and "returncode" in payload:
        return payload.get("returncode") == 0
    return True
