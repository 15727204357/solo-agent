from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable, Mapping
from typing import Any
from uuid import uuid4

from pydantic import ValidationError

from .models import PatchEdit, PatchProposal, PatchRequest, VerificationResult

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

    return PatchProposal(
        id=patch_id or f"patch_{uuid4().hex[:16]}",
        session_id=session_id,
        run_id=run_id,
        status="pending",
        summary=request.summary,
        diff="\n".join(diffs),
        edits=prepared,
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

    pytest_result = dict(await call_tool("run_pytest", {"args": ["-q"]}))
    ruff_result = dict(await call_tool("run_ruff_check", {"args": ["."]}))
    verification = VerificationResult(
        pytest=dict(pytest_result.get("result") or pytest_result),
        ruff=dict(ruff_result.get("result") or ruff_result),
        ok=_command_ok(pytest_result) and _command_ok(ruff_result),
    )
    return proposal.model_copy(
        update={
            "status": "applied" if verification.ok else "verification_failed",
            "apply_results": apply_results,
            "verification": verification,
            "error": None if verification.ok else "verification failed",
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
    return {"summary": str(payload.get("summary") or ""), "edits": normalized_edits}


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
