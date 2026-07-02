from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

PatchStatus = Literal["pending", "approved", "rejected", "applied", "verification_failed", "failed"]
StopGateStatus = Literal["passed", "failed", "missing", "waived"]


class PatchEdit(BaseModel):
    path: str = Field(min_length=1)
    new_text: str
    old_text: str | None = None
    line_start: int | None = Field(default=None, ge=1)
    line_end: int | None = Field(default=None, ge=1)
    expected_hash: str | None = None
    reason: str = ""
    diff: str = ""
    changed: bool = True
    new_sha256: str | None = None

    @model_validator(mode="after")
    def require_anchor(self) -> PatchEdit:
        has_text_anchor = self.old_text is not None
        has_line_anchor = self.line_start is not None and self.line_end is not None
        if not has_text_anchor and not has_line_anchor:
            raise ValueError("edit requires old_text or line_start/line_end")
        if self.line_start is not None and self.line_end is not None and self.line_end < self.line_start:
            raise ValueError("line_end must be greater than or equal to line_start")
        return self

    def preview_arguments(self) -> dict[str, Any]:
        args: dict[str, Any] = {
            "path": self.path,
            "expected_hash": self.expected_hash,
            "new_text": self.new_text,
        }
        if self.old_text is not None:
            args["old_text"] = self.old_text
        if self.line_start is not None and self.line_end is not None:
            args["line_start"] = self.line_start
            args["line_end"] = self.line_end
        return args

    def apply_arguments(self) -> dict[str, Any]:
        return self.preview_arguments()


class VerificationCommand(BaseModel):
    command: str = Field(min_length=1)
    args: list[str] = Field(default_factory=list)
    target: str | None = None
    tool: str | None = None
    purpose: str = ""


class VerificationPlan(BaseModel):
    commands: list[VerificationCommand] = Field(default_factory=list)
    required: bool = True
    reason: str = ""


class PatchRequest(BaseModel):
    summary: str = ""
    edits: list[PatchEdit]
    verification_plan: VerificationPlan | None = None

    @model_validator(mode="after")
    def require_edits(self) -> PatchRequest:
        if not self.edits:
            raise ValueError("patch request must include at least one edit")
        return self


class StopGate(BaseModel):
    status: StopGateStatus = "missing"
    approval_ready: bool = False
    reason: str = ""
    missing_evidence: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def require_waiver_reason(self) -> StopGate:
        if self.status == "waived" and not self.reason.strip():
            raise ValueError("waived stop gate requires a reason")
        return self


class VerificationResult(BaseModel):
    pytest: dict[str, Any] | None = None
    ruff: dict[str, Any] | None = None
    commands: list[dict[str, Any]] = Field(default_factory=list)
    ok: bool = False


class PatchProposal(BaseModel):
    id: str
    session_id: str
    run_id: str
    status: PatchStatus = "pending"
    summary: str = ""
    diff: str = ""
    edits: list[PatchEdit]
    verification_plan: VerificationPlan = Field(default_factory=VerificationPlan)
    stop_gate: StopGate = Field(default_factory=StopGate)
    apply_results: list[dict[str, Any]] = Field(default_factory=list)
    verification: VerificationResult | None = None
    error: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    decided_at: str | None = None

    def to_public_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")
