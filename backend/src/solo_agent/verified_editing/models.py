from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

PatchStatus = Literal["pending", "approved", "rejected", "applied", "verification_failed", "failed"]


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


class PatchRequest(BaseModel):
    summary: str = ""
    edits: list[PatchEdit]

    @model_validator(mode="after")
    def require_edits(self) -> PatchRequest:
        if not self.edits:
            raise ValueError("patch request must include at least one edit")
        return self


class VerificationResult(BaseModel):
    pytest: dict[str, Any] | None = None
    ruff: dict[str, Any] | None = None
    ok: bool = False


class PatchProposal(BaseModel):
    id: str
    session_id: str
    run_id: str
    status: PatchStatus = "pending"
    summary: str = ""
    diff: str = ""
    edits: list[PatchEdit]
    apply_results: list[dict[str, Any]] = Field(default_factory=list)
    verification: VerificationResult | None = None
    error: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    decided_at: str | None = None

    def to_public_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")
