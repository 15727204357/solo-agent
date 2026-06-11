from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from solo_agent.skill_changes import SkillChangeOperation
from solo_agent.skill_recipes import parse_structured_recipe_text, recipe_from_payload

SkillEvolutionChangeKind = Literal["patch_skill", "add_recipe", "add_script", "update_recovery"]

SAFE_RECIPE_TOOLS = {
    "workspace_snapshot",
    "search_text",
    "search_code",
    "find_files",
    "read_file",
    "git_status",
    "git_diff",
    "git_show",
    "run_pytest",
    "run_ruff_check",
    "run_ruff_format_check",
    "run_command",
    "skill_recipe_preview",
    "skill_recipe_run",
    "skill_script_run",
}
QUALITY_TOOLS = {"run_pytest", "run_ruff_check", "run_ruff_format_check", "run_command"}
RECIPE_TOOL_MAP = {
    "workspace_snapshot",
    "find_files",
    "search_code",
    "read_file",
    "git_status",
    "git_diff",
    "git_show",
    "run_command",
    "skill_script_run",
}
SECRET_RE = re.compile(r"(?i)(api[_-]?key|authorization|bearer\s+[a-z0-9._-]+|password|secret|token)")
EXTERNAL_PATH_RE = re.compile(r"(?i)(\.\./|\.\.\\|\.env\b|[a-z]:\\|/(etc|home|users|var|tmp)/)")
MAX_EVIDENCE_BYTES = 4_000


class SkillEvolutionCandidate(BaseModel):
    target_skill: str
    change_kind: SkillEvolutionChangeKind
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    confidence: float
    proposed_operations: list[SkillChangeOperation] = Field(default_factory=list)
    reason: str = ""
    target_skill_path: str | None = None

    def to_public_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class SkillEvolutionProposal(BaseModel):
    candidates: list[SkillEvolutionCandidate] = Field(default_factory=list)
    proposal_payload: dict[str, Any] | None = None
    skipped_reason: str | None = None

    def to_public_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


def analyze_skill_evolution(
    snapshot: Mapping[str, Any],
    *,
    min_confidence: float = 0.72,
    max_proposals: int = 1,
    workspace_root: str | Path | None = None,
) -> SkillEvolutionProposal:
    if max_proposals <= 0:
        return SkillEvolutionProposal(skipped_reason="max_proposals_is_zero")
    if not _snapshot_is_safe(snapshot):
        return SkillEvolutionProposal(skipped_reason="unsafe_snapshot")

    target = _select_target_skill(snapshot)
    if target is None:
        return SkillEvolutionProposal(skipped_reason="no_target_skill")

    candidates: list[SkillEvolutionCandidate] = []
    blocked = _blocked_recipe_candidate(snapshot, target, workspace_root=workspace_root)
    if blocked is not None:
        candidates.append(blocked)

    sequence = _safe_tool_sequence_candidate(snapshot, target, workspace_root=workspace_root)
    if sequence is not None:
        candidates.append(sequence)

    candidates = sorted(candidates, key=lambda candidate: candidate.confidence, reverse=True)
    candidates = candidates[:max_proposals]
    eligible = [candidate for candidate in candidates if candidate.confidence >= min_confidence]
    if not eligible:
        return SkillEvolutionProposal(candidates=candidates, skipped_reason="below_confidence_threshold")

    top = eligible[0]
    if not top.proposed_operations:
        return SkillEvolutionProposal(
            candidates=candidates,
            skipped_reason="promotion_validation_failed_schema_or_metadata_patch",
        )

    recipe_id = _recipe_id_from_operations(top.proposed_operations)

    payload = {
        "action": top.proposed_operations[0].action,
        "skill_name": top.target_skill,
        "target_paths": [operation.path for operation in top.proposed_operations],
        "diff": _operation_diff(top.proposed_operations),
        "operations": [operation.model_dump(mode="json") for operation in top.proposed_operations],
        "evolution": {
            "change_kind": top.change_kind,
            "confidence": top.confidence,
            "reason": top.reason,
            "evidence": top.evidence,
            "recipe_id": recipe_id,
            "target_skill_path": top.target_skill_path,
            "validated": True,
        },
    }
    return SkillEvolutionProposal(candidates=candidates, proposal_payload=payload)


def _blocked_recipe_candidate(
    snapshot: Mapping[str, Any],
    target: Mapping[str, str],
    *,
    workspace_root: str | Path | None,
) -> SkillEvolutionCandidate | None:
    blocked_steps: list[dict[str, Any]] = []
    for run in snapshot.get("recipe_runs") or []:
        if not isinstance(run, Mapping):
            continue
        for step in run.get("steps") or []:
            if isinstance(step, Mapping) and str(step.get("status") or "") == "blocked":
                blocked_steps.append(_summarize_mapping(step))
    if not blocked_steps:
        return None

    run_id = _safe_slug(str(snapshot.get("run_id") or "run"))
    recipe_id = f"evolution-recovery-{run_id}"
    path = _recipe_path(target, f"{recipe_id}.yaml")
    content = _recovery_recipe_yaml(snapshot, target, blocked_steps, recipe_id=recipe_id)
    operations = _promotion_operations(target, recipe_id, path, content, workspace_root=workspace_root)
    return SkillEvolutionCandidate(
        target_skill=target["name"],
        target_skill_path=target.get("path"),
        change_kind="update_recovery",
        confidence=0.86,
        reason="A declarative recipe step was blocked and can be captured as an approved recovery recipe.",
        evidence=[{"type": "blocked_recipe_step", "steps": blocked_steps}],
        proposed_operations=operations,
    )


def _safe_tool_sequence_candidate(
    snapshot: Mapping[str, Any],
    target: Mapping[str, str],
    *,
    workspace_root: str | Path | None,
) -> SkillEvolutionCandidate | None:
    tool_calls = [
        call for call in snapshot.get("tool_calls") or [] if isinstance(call, Mapping) and _tool_call_is_safe(call)
    ]
    sequence = [call for call in tool_calls if str(call.get("name") or "") in SAFE_RECIPE_TOOLS]
    if len(sequence) < 3:
        return None
    if not _has_successful_quality_check(sequence):
        return None
    recipe_steps = _recipe_steps_from_tool_sequence(sequence[:8])
    if len(recipe_steps) < 2:
        return None

    run_id = _safe_slug(str(snapshot.get("run_id") or "run"))
    recipe_id = f"evolution-{run_id}"
    path = _recipe_path(target, f"{recipe_id}.yaml")
    content = _safe_sequence_recipe_yaml(snapshot, target, recipe_steps, recipe_id=recipe_id)
    operations = _promotion_operations(target, recipe_id, path, content, workspace_root=workspace_root)
    evidence = [
        {
            "type": "safe_tool_sequence",
            "tools": [str(call.get("name") or "") for call in sequence[:8]],
            "explicit_skill_requests": list((snapshot.get("snapshots") or {}).get("explicit_skill_requests") or []),
        }
    ]
    return SkillEvolutionCandidate(
        target_skill=target["name"],
        target_skill_path=target.get("path"),
        change_kind="add_recipe",
        confidence=0.78,
        reason="A successful safe tool sequence under a selected skill can be proposed as a reusable recipe.",
        evidence=evidence,
        proposed_operations=operations,
    )


def _select_target_skill(snapshot: Mapping[str, Any]) -> dict[str, str] | None:
    selected = [skill for skill in snapshot.get("selected_skills") or [] if isinstance(skill, Mapping)]
    if not selected:
        return None
    explicit = {
        str(name).lower()
        for name in (snapshot.get("snapshots") or {}).get("explicit_skill_requests") or []
    }
    for skill in selected:
        name = str(skill.get("name") or "")
        if name.lower() in explicit:
            return {"name": name, "path": str(skill.get("path") or "")}
    first = selected[0]
    return {"name": str(first.get("name") or ""), "path": str(first.get("path") or "")}


def _snapshot_is_safe(snapshot: Mapping[str, Any]) -> bool:
    payload = _json_dumps(snapshot)
    if len(payload.encode("utf-8", errors="ignore")) > MAX_EVIDENCE_BYTES * 12:
        payload = payload[: MAX_EVIDENCE_BYTES * 12]
    return not SECRET_RE.search(payload) and not EXTERNAL_PATH_RE.search(payload)


def _tool_call_is_safe(call: Mapping[str, Any]) -> bool:
    if bool(call.get("blocked", False)):
        return False
    name = str(call.get("name") or "")
    if name not in SAFE_RECIPE_TOOLS:
        return False
    payload = _json_dumps({"arguments": call.get("arguments"), "result": call.get("result")})
    if len(payload.encode("utf-8", errors="ignore")) > MAX_EVIDENCE_BYTES:
        return False
    if SECRET_RE.search(payload) or EXTERNAL_PATH_RE.search(payload):
        return False
    if name == "run_command" and _command_is_write_like(call.get("arguments")):
        return False
    return True


def _has_successful_quality_check(sequence: Sequence[Mapping[str, Any]]) -> bool:
    for call in sequence:
        name = str(call.get("name") or "")
        if name not in QUALITY_TOOLS:
            continue
        result = call.get("result")
        if isinstance(result, Mapping):
            if result.get("ok") is False:
                continue
            if result.get("returncode") not in (None, 0):
                continue
        text = _json_dumps(result).lower()
        if "failed" in text or "error" in text:
            continue
        return True
    return False


def _command_is_write_like(arguments: Any) -> bool:
    payload = _json_dumps(arguments).lower()
    blocked_terms = (
        " rm ",
        " del ",
        " remove-item",
        "git commit",
        "git push",
        "npm install",
        "pip install",
        ">",
        ">>",
    )
    return any(term in f" {payload} " for term in blocked_terms)


def _recipe_path(target: Mapping[str, str], filename: str) -> str:
    skill_path = str(target.get("path") or "")
    if skill_path.startswith("skills/"):
        skill_path = skill_path[len("skills/") :]
    if skill_path.endswith("/SKILL.md"):
        skill_dir = skill_path[: -len("/SKILL.md")]
    elif skill_path:
        skill_dir = skill_path.rsplit("/", 1)[0]
    else:
        skill_dir = _safe_slug(target.get("name") or "unknown-skill")
    return f"{skill_dir}/references/recipes/{filename}"


def _promotion_operations(
    target: Mapping[str, str],
    recipe_id: str,
    recipe_path: str,
    recipe_content: str,
    *,
    workspace_root: str | Path | None,
) -> list[SkillChangeOperation]:
    skill_path = _skill_operation_path(target)
    if not skill_path:
        return []
    if not _recipe_content_valid(recipe_content, target, recipe_path):
        return []
    skill_text = _read_skill_text(workspace_root, skill_path)
    if skill_text is None:
        return []
    metadata_patch = _metadata_recipe_patch(skill_text, skill_path, recipe_id, recipe_path)
    if metadata_patch is None:
        return []
    if workspace_root is not None and _skill_target_exists(workspace_root, recipe_path):
        return []
    return [
        SkillChangeOperation(action="write_file", path=recipe_path, content=recipe_content),
        metadata_patch,
    ]


def _skill_operation_path(target: Mapping[str, str]) -> str:
    skill_path = str(target.get("path") or "")
    if skill_path.startswith("skills/"):
        skill_path = skill_path[len("skills/") :]
    return skill_path if skill_path.endswith("SKILL.md") else ""


def _read_skill_text(workspace_root: str | Path | None, skill_path: str) -> str | None:
    if workspace_root is None:
        return None
    root = Path(workspace_root).resolve()
    skills_root = (root / "skills").resolve()
    target = (skills_root / skill_path).resolve()
    try:
        target.relative_to(skills_root)
    except ValueError:
        return None
    if not target.is_file():
        return None
    return target.read_text(encoding="utf-8", errors="replace")


def _skill_target_exists(workspace_root: str | Path, operation_path: str) -> bool:
    root = Path(workspace_root).resolve()
    skills_root = (root / "skills").resolve()
    target = (skills_root / operation_path).resolve()
    try:
        target.relative_to(skills_root)
    except ValueError:
        return True
    return target.exists()


def _recipe_content_valid(recipe_content: str, target: Mapping[str, str], recipe_path: str) -> bool:
    try:
        payload = parse_structured_recipe_text(recipe_content)
        recipe_from_payload(
            payload,
            skill={"name": target.get("name") or "", "path": target.get("path") or ""},
            source_file=recipe_path,
        )
    except Exception:
        return False
    return True


def _metadata_recipe_patch(
    skill_text: str,
    skill_path: str,
    recipe_id: str,
    recipe_path: str,
) -> SkillChangeOperation | None:
    frontmatter = _frontmatter_text(skill_text)
    if frontmatter is None:
        return None
    lines = frontmatter.splitlines()
    for line in lines:
        if not line.startswith("metadata: "):
            continue
        raw_metadata = line.removeprefix("metadata: ").strip()
        try:
            metadata = json.loads(raw_metadata)
        except json.JSONDecodeError:
            return None
        if not isinstance(metadata, dict):
            return None
        hermes = metadata.setdefault("hermes", {})
        if not isinstance(hermes, dict):
            return None
        recipes = hermes.setdefault("recipes", [])
        if not isinstance(recipes, list):
            return None
        recipe_file = _recipe_file_relative_to_skill(skill_path, recipe_path)
        for recipe in recipes:
            if isinstance(recipe, Mapping) and str(recipe.get("id") or "") == recipe_id:
                return None
        recipes.append({"id": recipe_id, "file": recipe_file})
        new_line = "metadata: " + json.dumps(metadata, ensure_ascii=True, separators=(", ", ": "))
        return SkillChangeOperation(action="patch", path=skill_path, old_string=line, new_string=new_line)
    return None


def _frontmatter_text(skill_text: str) -> str | None:
    lines = skill_text.splitlines()
    if not lines or lines[0].strip() != "---":
        return None
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            return "\n".join(lines[1:index])
    return None


def _recipe_file_relative_to_skill(skill_path: str, recipe_path: str) -> str:
    skill_dir = skill_path.rsplit("/", 1)[0] if "/" in skill_path else ""
    if skill_dir and recipe_path.startswith(f"{skill_dir}/"):
        return recipe_path[len(skill_dir) + 1 :]
    return recipe_path


def _recipe_steps_from_tool_sequence(sequence: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    for call in sequence:
        step = _recipe_step_from_tool_call(call, len(steps) + 1)
        if step is not None:
            steps.append(step)
    return steps


def _recipe_step_from_tool_call(call: Mapping[str, Any], index: int) -> dict[str, Any] | None:
    name = str(call.get("name") or "")
    arguments = call.get("arguments") if isinstance(call.get("arguments"), Mapping) else {}
    if name in RECIPE_TOOL_MAP:
        return {
            "id": f"step-{index}-{_safe_slug(name)}",
            "tool": name,
            "arguments": dict(arguments),
        }
    if name == "search_text":
        return {
            "id": f"step-{index}-search-code",
            "tool": "search_code",
            "arguments": dict(arguments),
        }
    if name == "run_pytest":
        args = ["-m", "pytest"]
        target = str(arguments.get("target") or "").strip()
        if target:
            args.append(target)
        explicit_args = arguments.get("args")
        if isinstance(explicit_args, list):
            args.extend(str(arg) for arg in explicit_args)
        elif not target:
            args.append("-q")
        return {
            "id": f"step-{index}-run-pytest",
            "tool": "run_command",
            "arguments": {
                "command": "python",
                "args": args,
                "purpose": "Run the focused pytest command observed in a successful skill workflow.",
            },
        }
    if name == "run_ruff_check":
        args = ["run", "--extra", "dev", "ruff", "check"]
        target = str(arguments.get("target") or "").strip()
        if target:
            args.append(target)
        else:
            args.append(".")
        return {
            "id": f"step-{index}-run-ruff-check",
            "tool": "run_command",
            "arguments": {
                "command": "uv",
                "args": args,
                "purpose": "Run ruff check observed in a successful skill workflow.",
            },
        }
    if name == "run_ruff_format_check":
        args = ["run", "--extra", "dev", "ruff", "format", "--check"]
        target = str(arguments.get("target") or "").strip()
        args.append(target or ".")
        return {
            "id": f"step-{index}-run-ruff-format-check",
            "tool": "run_command",
            "arguments": {
                "command": "uv",
                "args": args,
                "purpose": "Run ruff format check observed in a successful skill workflow.",
            },
        }
    return None


def _safe_sequence_recipe_yaml(
    snapshot: Mapping[str, Any],
    target: Mapping[str, str],
    steps: Sequence[Mapping[str, Any]],
    *,
    recipe_id: str,
) -> str:
    payload = {
        "id": recipe_id,
        "name": f"Observed workflow for {target['name']}",
        "description": "Proposed from a successful skill run; review before approving.",
        "when": _recipe_when_terms(snapshot),
        "mode": "assist",
        "priority": 20,
        "run_policy": "auto",
        "failure_policy": "stop",
        "steps": list(steps),
        "success_criteria": ["The observed quality check still passes."],
    }
    return json.dumps(payload, ensure_ascii=True, indent=2) + "\n"


def _recovery_recipe_yaml(
    snapshot: Mapping[str, Any],
    target: Mapping[str, str],
    blocked_steps: Sequence[Mapping[str, Any]],
    *,
    recipe_id: str,
) -> str:
    steps = [
        {
            "id": f"recovery-{index}-{_safe_slug(str(step.get('tool') or 'manual'))}",
            "tool": "run_command",
            "description": f"Manual recovery note for blocked step: {_json_dumps(step)}",
            "run_policy": "manual",
            "risk_level": "high",
            "arguments": {"command": "python", "args": ["-m", "pytest", "-q"]},
        }
        for index, step in enumerate(blocked_steps, start=1)
    ]
    payload = {
        "id": recipe_id,
        "name": f"Recovery note for {target['name']}",
        "description": "Proposed from a blocked recipe step; review before approving.",
        "when": _recipe_when_terms(snapshot),
        "mode": "assist",
        "priority": 10,
        "run_policy": "manual",
        "failure_policy": "stop",
        "steps": steps,
        "success_criteria": ["The blocked step has a documented recovery path."],
    }
    return json.dumps(payload, ensure_ascii=True, indent=2) + "\n"


def _recipe_when_terms(snapshot: Mapping[str, Any]) -> list[str]:
    text = str(snapshot.get("user_input") or "")
    terms = re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,30}", text.lower())
    return terms[:8] or ["observed-workflow"]


def _operation_diff(operations: Sequence[SkillChangeOperation]) -> str:
    parts: list[str] = []
    for operation in operations:
        if operation.action == "patch":
            parts.append(f"--- a/skills/{operation.path}\n+++ b/skills/{operation.path}\n@@\n")
            if operation.old_string:
                parts.append(f"-{operation.old_string}\n")
            if operation.new_string:
                parts.append(f"+{operation.new_string}\n")
            continue
        content = operation.content or ""
        parts.append(f"--- /dev/null\n+++ b/skills/{operation.path}\n@@\n")
        parts.extend(f"+{line}\n" for line in content.splitlines())
    return "".join(parts)


def _recipe_id_from_operations(operations: Sequence[SkillChangeOperation]) -> str | None:
    for operation in operations:
        if operation.action != "write_file" or not operation.content:
            continue
        try:
            payload = parse_structured_recipe_text(operation.content)
        except Exception:
            continue
        if isinstance(payload, Mapping):
            return str(payload.get("id") or "") or None
    return None


def _summarize_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key in ("id", "tool", "status", "reason", "description", "run_policy"):
        if key in value:
            summary[key] = value.get(key)
    return summary


def _json_dumps(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=True, sort_keys=True, default=str)
    except TypeError:
        return json.dumps(str(value), ensure_ascii=True)


def _safe_slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", value.strip().lower()).strip("-")
    return slug[:80] or "item"
