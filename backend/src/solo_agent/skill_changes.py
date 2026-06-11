from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

SkillChangeAction = Literal["create", "patch", "edit", "delete", "write_file", "remove_file"]
SkillChangeStatus = Literal["pending", "approved", "rejected", "applied", "failed"]


class SkillChangeOperation(BaseModel):
    action: SkillChangeAction
    path: str
    content: str | None = None
    old_string: str | None = None
    new_string: str | None = None


class SkillChangeProposal(BaseModel):
    id: str = Field(default_factory=lambda: f"skillchg_{uuid4().hex[:16]}")
    session_id: str
    run_id: str
    action: SkillChangeAction
    skill_name: str
    target_paths: list[str] = Field(default_factory=list)
    diff: str = ""
    operations: list[SkillChangeOperation] = Field(default_factory=list)
    status: SkillChangeStatus = "pending"
    apply_results: list[dict[str, Any]] = Field(default_factory=list)
    error: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    decided_at: str | None = None

    def to_public_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


def apply_skill_change_proposal(proposal: SkillChangeProposal, workspace_root: str | Path) -> SkillChangeProposal:
    apply_results: list[dict[str, Any]] = []
    root = Path(workspace_root).resolve()
    skills_root = (root / "skills").resolve()

    try:
        for operation in proposal.operations:
            target = _resolve_skill_target(skills_root, operation.path)
            result = _apply_operation(operation, target)
            apply_results.append(result)
    except Exception as exc:
        return proposal.model_copy(
            update={
                "status": "failed",
                "apply_results": apply_results,
                "error": str(exc),
            }
        )

    return proposal.model_copy(
        update={
            "status": "applied",
            "apply_results": apply_results,
            "error": None,
        }
    )


def _apply_operation(operation: SkillChangeOperation, target: Path) -> dict[str, Any]:
    action = operation.action
    if action == "create":
        if target.exists():
            raise FileExistsError(f"Skill file already exists: {operation.path}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(operation.content or "", encoding="utf-8")
        return {"action": action, "path": operation.path, "ok": True}

    if action == "patch":
        if not target.is_file():
            raise FileNotFoundError(f"Skill file does not exist: {operation.path}")
        old_string = operation.old_string or ""
        if not old_string:
            raise ValueError("patch requires old_string")
        current = target.read_text(encoding="utf-8", errors="replace")
        count = current.count(old_string)
        if count != 1:
            raise ValueError(f"patch requires old_string to appear exactly once; found {count}")
        target.write_text(current.replace(old_string, operation.new_string or "", 1), encoding="utf-8")
        return {"action": action, "path": operation.path, "ok": True}

    if action == "edit":
        if not target.is_file():
            raise FileNotFoundError(f"Skill file does not exist: {operation.path}")
        target.write_text(operation.content or "", encoding="utf-8")
        return {"action": action, "path": operation.path, "ok": True}

    if action == "write_file":
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(operation.content or "", encoding="utf-8")
        return {"action": action, "path": operation.path, "ok": True}

    if action == "remove_file":
        if not target.is_file():
            raise FileNotFoundError(f"Skill support file does not exist: {operation.path}")
        target.unlink()
        return {"action": action, "path": operation.path, "ok": True}

    if action == "delete":
        if target.is_dir():
            shutil.rmtree(target)
            return {"action": action, "path": operation.path, "ok": True}
        if target.is_file():
            target.unlink()
            return {"action": action, "path": operation.path, "ok": True}
        raise FileNotFoundError(f"Skill path does not exist: {operation.path}")

    raise ValueError(f"Unsupported skill change action: {action}")


def _resolve_skill_target(skills_root: Path, relative_path: str) -> Path:
    target = (skills_root / relative_path).resolve()
    try:
        target.relative_to(skills_root)
    except ValueError as exc:
        raise PermissionError(f"Skill change path escapes skills root: {relative_path}") from exc
    return target
