from __future__ import annotations

import pytest
from solo_agent.verified_editing import (
    PatchProposalError,
    apply_approved_patch,
    build_patch_proposal,
    extract_patch_request,
)


class FakeToolCaller:
    def __init__(self, *, apply_ok: bool = True, pytest_code: int = 0, ruff_code: int = 0) -> None:
        self.apply_ok = apply_ok
        self.pytest_code = pytest_code
        self.ruff_code = ruff_code
        self.calls: list[tuple[str, dict]] = []

    async def __call__(self, name: str, arguments: dict) -> dict:
        self.calls.append((name, arguments))
        if name == "prepare_edit":
            return {"ok": True, "result": {"path": arguments["path"], "expected_hash": "hash-1"}}
        if name == "preview_patch":
            changed = arguments.get("new_text") != arguments.get("old_text")
            return {
                "ok": True,
                "result": {
                    "path": arguments["path"],
                    "changed": changed,
                    "diff": "--- app.py\n+++ app.py\n@@\n-old\n+new" if changed else "",
                    "new_sha256": "hash-2",
                },
            }
        if name == "apply_text_edit":
            if not self.apply_ok:
                return {"ok": False, "error": "Hash mismatch"}
            return {"ok": True, "result": {"path": arguments["path"], "changed": True}}
        if name == "run_pytest":
            return {"ok": True, "result": {"returncode": self.pytest_code, "output": "pytest"}}
        if name == "run_ruff_check":
            return {"ok": True, "result": {"returncode": self.ruff_code, "output": "ruff"}}
        raise AssertionError(name)


def test_extract_patch_request_accepts_structured_json_and_aliases() -> None:
    request = extract_patch_request(
        '{"summary":"Fix greeting","edits":[{"path":"app.py","old":"old","new":"new","reason":"demo"}]}'
    )

    assert request.summary == "Fix greeting"
    assert request.edits[0].old_text == "old"
    assert request.edits[0].new_text == "new"


def test_extract_patch_request_rejects_invalid_json() -> None:
    with pytest.raises(PatchProposalError):
        extract_patch_request("not json")


@pytest.mark.asyncio
async def test_build_patch_proposal_previews_diff_and_skips_noop() -> None:
    request = extract_patch_request(
        '{"summary":"Fix greeting","edits":[{"path":"app.py","old_text":"old","new_text":"new"}]}'
    )
    caller = FakeToolCaller()

    proposal = await build_patch_proposal(request, session_id="s1", run_id="r1", call_tool=caller)

    assert proposal.status == "pending"
    assert proposal.edits[0].expected_hash == "hash-1"
    assert "--- app.py" in proposal.diff
    assert [name for name, _ in caller.calls] == ["prepare_edit", "preview_patch"]


@pytest.mark.asyncio
async def test_build_patch_proposal_rejects_no_effective_changes() -> None:
    request = extract_patch_request(
        '{"summary":"Noop","edits":[{"path":"app.py","old_text":"same","new_text":"same"}]}'
    )

    with pytest.raises(PatchProposalError):
        await build_patch_proposal(request, session_id="s1", run_id="r1", call_tool=FakeToolCaller())


@pytest.mark.asyncio
async def test_apply_approved_patch_stops_before_verification_on_apply_failure() -> None:
    request = extract_patch_request(
        '{"summary":"Fix","edits":[{"path":"app.py","old_text":"old","new_text":"new"}]}'
    )
    proposal = await build_patch_proposal(request, session_id="s1", run_id="r1", call_tool=FakeToolCaller())
    caller = FakeToolCaller(apply_ok=False)

    applied = await apply_approved_patch(proposal, call_tool=caller)

    assert applied.status == "failed"
    assert applied.verification is None
    assert [name for name, _ in caller.calls] == ["apply_text_edit"]


@pytest.mark.asyncio
async def test_apply_approved_patch_runs_pytest_and_ruff() -> None:
    request = extract_patch_request(
        '{"summary":"Fix","edits":[{"path":"app.py","old_text":"old","new_text":"new"}]}'
    )
    proposal = await build_patch_proposal(request, session_id="s1", run_id="r1", call_tool=FakeToolCaller())
    caller = FakeToolCaller(pytest_code=0, ruff_code=1)

    applied = await apply_approved_patch(proposal, call_tool=caller)

    assert applied.status == "verification_failed"
    assert applied.verification is not None
    assert applied.verification.pytest["returncode"] == 0
    assert applied.verification.ruff["returncode"] == 1
    assert [name for name, _ in caller.calls] == ["apply_text_edit", "run_pytest", "run_ruff_check"]
